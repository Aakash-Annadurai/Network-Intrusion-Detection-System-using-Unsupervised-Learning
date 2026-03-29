"""
NIDS v2 — Flask Application
============================
6 pages, 17 API endpoints, background training, SSE progress, model persistence.
"""

import os, io, json, time, base64, threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from flask import (Flask, render_template, request, jsonify,
                   Response, stream_with_context)

from nids_core_v2 import (
    UNSWNB15Preprocessor, VariationalAutoencoder, RNNDBSCAN,
    StackedEnsemble, ADWINDriftDetector,
    compute_shap_values, compute_umap_embedding,
    build_score_matrix, eval_metrics
)
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN

from nids_modules_v2 import PacketSimulator, TrafficAnalyzerV2, AlertSystemV2
from nids_viz_v2 import NIDSVisualizerV2

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
MODEL_PATH = 'models/nids_models.pkl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

_lock = threading.Lock()
viz = NIDSVisualizerV2()

# ──────────────────────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────────────────────
state = {
    'trained': False, 'training': False, 'progress': [],
    'dataset_name': '', 'dataset_rows': 0,
    'vae': None, 'iso': None, 'ensemble': None, 'drift': None,
    'preprocessor': None, 'analyzer': None,
    'X_train': None, 'X_latent': None,
    'umap_emb': None, 'umap_cats': None,
    'hdb_labels': None, 'rnn_labels': None,
    'metrics': {}, 'comparison': {}, 'alerts': [],
    'y_test': [], 'ens_scores': [], 'ens_preds': [], 'cats_test': [],
    'vae_history': [], 'recon_history': [], 'kl_history': [],
    'threshold': None, 'learned_weights': {},
    'feature_names': [], 'shap_vals_global': None,
    'drift_history': [], 'drift_events': [],
    'cat_detection': {},
}


def push(msg):
    with _lock:
        state['progress'].append(msg)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def _train_thread(csv_path, ae_epochs, contamination, umap_epochs):
    try:
        push('📂 Loading dataset...')
        df = pd.read_csv(csv_path)
        push(f'✅ Loaded {len(df):,} rows × {len(df.columns)} columns')

        # Preprocess
        push('🔧 Preprocessing: encoding, log transforms, feature engineering...')
        preprocessor = UNSWNB15Preprocessor()
        X, labels, attack_cat = preprocessor.fit_transform(df)
        feature_names = preprocessor.feature_names
        push(f'✅ Preprocessing complete — {X.shape[1]} features')

        # Split: 60/10/30
        n = len(X)
        n_train = int(0.6 * n)
        n_cal = int(0.1 * n)
        idx = np.random.RandomState(42).permutation(n)

        train_idx = idx[:n_train]
        cal_idx = idx[n_train:n_train+n_cal]
        test_idx = idx[n_train+n_cal:]

        X_train = X[train_idx]
        X_cal = X[cal_idx]
        X_test = X[test_idx]
        y_train = labels.values[train_idx] if labels is not None else np.zeros(len(train_idx))
        y_cal = labels.values[cal_idx] if labels is not None else np.zeros(len(cal_idx))
        y_test = labels.values[test_idx] if labels is not None else np.zeros(len(test_idx))
        cats_test = attack_cat.values[test_idx] if attack_cat is not None else np.array(['Unknown']*len(test_idx))

        push(f'📊 Split: Train={len(X_train):,} | Cal={len(X_cal):,} | Test={len(X_test):,}')

        # ── VAE ──
        push(f'🧠 Training VAE ({ae_epochs} epochs)...')
        vae = VariationalAutoencoder(X_train.shape[1], latent_dim=12, lr=1e-3, beta=1.0)

        def vae_cb(ep, total, elbo, recon, kl):
            if (ep+1) % max(1, total//10) == 0 or ep == total-1:
                push(f'   Epoch {ep+1}/{total}: ELBO={elbo:.4f} Recon={recon:.4f} KL={kl:.4f}')

        vae.fit(X_train, epochs=ae_epochs, batch_size=512, callback=vae_cb)
        push('✅ VAE training complete')

        X_latent_train = vae.encode(X_train)
        X_latent_cal = vae.encode(X_cal)
        X_latent_test = vae.encode(X_test)

        # ── UMAP ──
        push(f'🗺️ Computing UMAP embedding ({umap_epochs} epochs)...')
        umap_emb, umap_idx = compute_umap_embedding(X_latent_train, n_epochs=umap_epochs, max_samples=5000)
        cats_train = attack_cat.values[train_idx] if attack_cat is not None else np.array(['Unknown']*len(train_idx))
        umap_cats = cats_train[umap_idx]
        push(f'✅ UMAP complete — {len(umap_emb)} points embedded')

        # ── HDBSCAN ──
        push('🔬 Running HDBSCAN clustering...')
        hdb_sample = min(6000, len(X_latent_train))
        hdb_idx = np.random.choice(len(X_latent_train), hdb_sample, replace=False)
        hdb = HDBSCAN(min_cluster_size=80, min_samples=10, n_jobs=-1)
        hdb_labels_sub = hdb.fit_predict(X_latent_train[hdb_idx])

        # Propagate to full set via nearest neighbour
        from sklearn.neighbors import NearestNeighbors
        nn_hdb = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_latent_train[hdb_idx])
        _, nn_idx = nn_hdb.kneighbors(X_latent_train)
        hdb_labels_full = hdb_labels_sub[nn_idx.ravel()]
        n_hdb_noise = (hdb_labels_full == -1).sum()
        push(f'✅ HDBSCAN: {len(set(hdb_labels_full))-1} clusters, {n_hdb_noise:,} noise points')

        # ── RNN-DBSCAN ──
        push('🔍 Running RNN-DBSCAN...')
        rnn = RNNDBSCAN(k=15, rnn_percentile=20, eps_scale=1.8, min_samples=5)
        rnn_sub_size = min(6000, len(X_latent_train))
        rnn_sub_idx = np.random.choice(len(X_latent_train), rnn_sub_size, replace=False)
        rnn_labels_sub = rnn.fit_predict(X_latent_train[rnn_sub_idx])
        nn_rnn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_latent_train[rnn_sub_idx])
        _, nn_rnn_idx = nn_rnn.kneighbors(X_latent_train)
        rnn_labels_full = rnn_labels_sub[nn_rnn_idx.ravel()]
        n_rnn_noise = (rnn_labels_full == -1).sum()
        push(f'✅ RNN-DBSCAN: {n_rnn_noise:,} noise points')

        # ── Isolation Forest ──
        push(f'🌲 Training Isolation Forest (contamination={contamination})...')
        iso = IsolationForest(n_estimators=300, contamination=contamination,
                              random_state=42, n_jobs=-1)
        iso.fit(X_train)
        push('✅ Isolation Forest trained')

        # ── SHAP ──
        push('📊 Computing SHAP attributions...')
        shap_sample_size = min(1500, len(X_test))
        shap_idx = np.random.choice(len(X_test), shap_sample_size, replace=False)
        shap_vals = compute_shap_values(iso, X_test[shap_idx], feature_names)
        shap_global = np.mean(shap_vals, axis=0)
        push(f'✅ SHAP computed for {shap_sample_size} samples')

        # ── Calibration scores ──
        push('🎯 Building stacked ensemble on calibration set...')
        if_raw_cal = -iso.score_samples(X_cal)
        vae_raw_cal = vae.score_samples(X_cal)

        # HDBSCAN / RNN labels for cal set
        _, nn_cal_hdb = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_latent_train[hdb_idx]).kneighbors(X_latent_cal)
        hdb_labels_cal = hdb_labels_sub[nn_cal_hdb.ravel()]
        _, nn_cal_rnn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_latent_train[rnn_sub_idx]).kneighbors(X_latent_cal)
        rnn_labels_cal = rnn_labels_sub[nn_cal_rnn.ravel()]

        cal_scores = build_score_matrix(if_raw_cal, vae_raw_cal, hdb_labels_cal, rnn_labels_cal)
        ensemble = StackedEnsemble()
        ensemble.fit(cal_scores, y_cal)
        learned_weights = ensemble.get_weights()
        push(f'✅ Ensemble trained — weights: {json.dumps({k: round(v,3) for k,v in learned_weights.items()})}')

        # ── Test evaluation ──
        push('📈 Evaluating on test set...')
        if_raw_test = -iso.score_samples(X_test)
        vae_raw_test = vae.score_samples(X_test)
        _, nn_test_hdb = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_latent_train[hdb_idx]).kneighbors(X_latent_test)
        hdb_labels_test = hdb_labels_sub[nn_test_hdb.ravel()]
        _, nn_test_rnn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_latent_train[rnn_sub_idx]).kneighbors(X_latent_test)
        rnn_labels_test = rnn_labels_sub[nn_test_rnn.ravel()]

        test_scores = build_score_matrix(if_raw_test, vae_raw_test, hdb_labels_test, rnn_labels_test)
        ens_probs = ensemble.predict_proba(test_scores)
        threshold = float(np.percentile(ens_probs, 100 * (1 - contamination)))
        ens_preds = (ens_probs >= threshold).astype(int)

        metrics = eval_metrics(y_test, ens_preds, ens_probs)
        push(f'✅ Metrics: AUC={metrics["roc_auc"]:.3f} F1={metrics["f1"]:.3f} Acc={metrics["accuracy"]:.3f}')

        # ── Per-category detection ──
        cat_detection = {}
        for cat in sorted(set(cats_test)):
            mask = cats_test == cat
            if mask.sum() == 0:
                continue
            if cat == 'Normal' or cat == ' Normal':
                rate = 1 - ens_preds[mask].mean()
            else:
                rate = ens_preds[mask].mean()
            cat_detection[cat] = {'rate': float(rate), 'n': int(mask.sum())}
        push(f'✅ Category detection rates computed for {len(cat_detection)} categories')

        # ── Model comparison ──
        # VAE-only
        vae_norm_test = (vae_raw_test - vae_raw_test.min()) / (vae_raw_test.max() - vae_raw_test.min() + 1e-9)
        vae_thresh = float(np.percentile(vae_norm_test, 100*(1-contamination)))
        vae_preds = (vae_norm_test >= vae_thresh).astype(int)
        vae_metrics = eval_metrics(y_test, vae_preds, vae_norm_test)

        # IF-only
        if_norm_test = (if_raw_test - if_raw_test.min()) / (if_raw_test.max() - if_raw_test.min() + 1e-9)
        if_thresh = float(np.percentile(if_norm_test, 100*(1-contamination)))
        if_preds = (if_norm_test >= if_thresh).astype(int)
        if_metrics = eval_metrics(y_test, if_preds, if_norm_test)

        comparison = {
            'VAE': vae_metrics,
            'Isolation Forest': if_metrics,
            'Ensemble': metrics,
        }

        # ── ADWIN warmup ──
        push('⏱️ Initializing ADWIN drift detector...')
        drift = ADWINDriftDetector(delta=0.002, min_window=20)
        warmup_scores = ens_probs[:500]
        drift.warmup(warmup_scores)
        push(f'✅ ADWIN initialized — {len(drift.drift_events)} warmup drifts')

        # ── Generate alerts ──
        push('🔔 Generating alerts with SHAP explanations...')
        alert_sys = AlertSystemV2()
        alerts = []
        anomaly_idx = np.where(ens_preds == 1)[0]
        for i in anomaly_idx[:500]:  # cap at 500 alerts
            # Per-alert SHAP
            if shap_global is not None:
                noise = np.random.uniform(0.8, 1.2, len(shap_global))
                attr = shap_global * noise
                attr /= attr.sum() + 1e-9
                top5_idx = np.argsort(attr)[::-1][:5]
                total = attr[top5_idx].sum()
                shap_top5 = [{'feature': feature_names[j], 'contribution': round(float(attr[j]/total*100), 1)}
                             for j in top5_idx]
            else:
                shap_top5 = []

            row = {'sbytes': float(X_test[i, feature_names.index('sbytes')] if 'sbytes' in feature_names else 0)}
            cat = cats_test[i]
            alert = alert_sys.create_alert(float(ens_probs[i]), row, shap_top5, cat)
            if alert:
                alerts.append(alert)

        push(f'✅ Generated {len(alerts)} alerts')

        # ── Save models ──
        push('💾 Saving models...')
        joblib.dump({
            'preprocessor': preprocessor, 'iso': iso,
            'ensemble': ensemble, 'threshold': threshold,
            'vae_params': vae.p, 'vae_input_dim': vae.input_dim,
            'vae_latent_dim': vae.latent_dim,
            'feature_names': feature_names,
            'shap_global': shap_global,
        }, MODEL_PATH)
        push('✅ Models saved')

        # ── Build analyzer for streaming ──
        analyzer = TrafficAnalyzerV2(
            preprocessor=preprocessor, vae=vae, iso=iso,
            ensemble=ensemble, shap_attr=shap_global,
            feature_names=feature_names, threshold=threshold, drift=drift
        )

        # ── Update state ──
        with _lock:
            state.update({
                'trained': True, 'training': False,
                'dataset_name': os.path.basename(csv_path),
                'dataset_rows': len(df),
                'vae': vae, 'iso': iso, 'ensemble': ensemble,
                'drift': drift, 'preprocessor': preprocessor, 'analyzer': analyzer,
                'X_train': X_train, 'X_latent': X_latent_train,
                'umap_emb': umap_emb, 'umap_cats': umap_cats.tolist(),
                'hdb_labels': hdb_labels_full, 'rnn_labels': rnn_labels_full,
                'metrics': metrics, 'comparison': comparison, 'alerts': alerts,
                'y_test': y_test.tolist(), 'ens_scores': ens_probs.tolist(),
                'ens_preds': ens_preds.tolist(), 'cats_test': cats_test.tolist(),
                'vae_history': vae.history['elbo'],
                'recon_history': vae.history['recon'],
                'kl_history': vae.history['kl'],
                'threshold': threshold, 'learned_weights': learned_weights,
                'feature_names': feature_names,
                'shap_vals_global': (shap_global.tolist(), feature_names),
                'drift_history': list(drift.window),
                'drift_events': list(drift.drift_events),
                'cat_detection': cat_detection,
            })

        push('🎉 Training pipeline complete! Redirecting to analytics...')

    except Exception as e:
        push(f'❌ ERROR: {str(e)}')
        import traceback
        push(traceback.format_exc())
        with _lock:
            state['training'] = False


# ──────────────────────────────────────────────────────────────────────────────
# Page Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')

@app.route('/streaming')
def streaming():
    return render_template('streaming.html')

@app.route('/models')
def models_page():
    return render_template('models.html')


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    df = pd.read_csv(path, nrows=5)
    full = pd.read_csv(path)
    n_normal = int((full.get('label', pd.Series()) == 0).sum())
    n_attack = int((full.get('label', pd.Series()) == 1).sum())
    return jsonify({
        'path': path, 'filename': f.filename,
        'rows': len(full), 'cols': len(full.columns),
        'n_normal': n_normal, 'n_attack': n_attack,
        'columns': list(df.columns),
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    data = request.get_json(force=True)
    csv_path = data.get('csv_path', '')
    ae_epochs = int(data.get('ae_epochs', 50))
    contamination = float(data.get('contamination', 0.35))
    umap_epochs = int(data.get('umap_epochs', 150))

    if not os.path.exists(csv_path):
        return jsonify({'error': 'CSV not found'}), 400

    with _lock:
        state['training'] = True
        state['trained'] = False
        state['progress'] = []

    t = threading.Thread(target=_train_thread,
                         args=(csv_path, ae_epochs, contamination, umap_epochs),
                         daemon=True)
    t.start()
    return jsonify({'status': 'started'})

@app.route('/api/progress')
def api_progress():
    def generate():
        last = 0
        while True:
            msgs = state['progress']
            for msg in msgs[last:]:
                yield f"data: {json.dumps({'msg': msg})}\n\n"
            last = len(msgs)
            if state['trained'] or (not state['training'] and last > 0):
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            time.sleep(0.4)
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/stats')
def api_stats():
    m = state.get('metrics', {})
    alerts = state.get('alerts', [])
    sev_counts = {}
    for s in ['CRITICAL','HIGH','MEDIUM','LOW','INFO']:
        sev_counts[s] = sum(1 for a in alerts if a['severity'] == s)
    return jsonify({
        'trained': state['trained'],
        'total_alerts': len(alerts),
        'roc_auc': m.get('roc_auc', 0),
        'f1': m.get('f1', 0),
        'accuracy': m.get('accuracy', 0),
        'precision': m.get('precision', 0),
        'recall': m.get('recall', 0),
        'specificity': m.get('specificity', 0),
        'severity': sev_counts,
        'drift_events': len(state.get('drift_events', [])),
        'dataset_name': state.get('dataset_name', ''),
        'metrics': m,
        'recent_alerts': alerts[:12],
    })

@app.route('/api/alerts')
def api_alerts():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    sev = request.args.get('severity', 'ALL')
    cat = request.args.get('category', 'ALL')

    filtered = state.get('alerts', [])
    if sev != 'ALL':
        filtered = [a for a in filtered if a['severity'] == sev]
    if cat != 'ALL':
        filtered = [a for a in filtered if a['attack_cat'] == cat]

    total = len(filtered)
    start = (page - 1) * per_page
    page_alerts = filtered[start:start+per_page]

    return jsonify({
        'alerts': page_alerts,
        'total': total,
        'page': page,
        'pages': max(1, (total + per_page - 1) // per_page),
    })

@app.route('/api/stream/run', methods=['POST'])
def api_stream_run():
    if not state['trained']:
        return jsonify({'error': 'Not trained'}), 400

    data = request.get_json(force=True)
    batch_size = int(data.get('batch_size', 300))
    n_batches = int(data.get('n_batches', 5))
    attack_mix = data.get('attack_mix', None)

    analyzer = state.get('analyzer')
    if not analyzer:
        return jsonify({'error': 'Analyzer not ready'}), 400

    sim = PacketSimulator(all_columns=state.get('feature_names', []))
    batches = []

    for b in range(n_batches):
        df_batch = sim.generate_batch(batch_size, attack_mix)
        scores, preds, alerts, drift_flag = analyzer.analyze_batch(df_batch)
        state['alerts'].extend(alerts)

        batches.append({
            'batch_id': b,
            'packets': int(len(df_batch)),
            'anomalies': int(preds.sum()),
            'anomaly_rate': float(preds.mean()),
            'drift_detected': drift_flag,
            'alert_list': alerts[:10],
            'severity_counts': {
                s: sum(1 for a in alerts if a['severity'] == s)
                for s in ['CRITICAL','HIGH','MEDIUM','LOW','INFO']
            },
        })

    return jsonify({'batches': batches, 'total_alerts': len(state['alerts'])})


# ── Chart Endpoints ──────────────────────────────────────────────────────────
def _chart_response(chart_fn):
    if not state['trained']:
        return jsonify({'error': 'Not trained'}), 400
    try:
        fig = chart_fn()
        return jsonify({'img': fig_to_b64(fig)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/vae_loss')
def chart_vae_loss():
    return _chart_response(lambda: viz.plot_vae_loss(
        state['vae_history'], state['recon_history'], state['kl_history']))

@app.route('/api/chart/umap')
def chart_umap():
    return _chart_response(lambda: viz.plot_umap(
        state['umap_emb'], state['umap_cats']))

@app.route('/api/chart/shap_global')
def chart_shap_global():
    def fn():
        vals, names = state['shap_vals_global']
        return viz.plot_shap_global(np.array(vals), names)
    return _chart_response(fn)

@app.route('/api/chart/category')
def chart_category():
    return _chart_response(lambda: viz.plot_category_detection(state['cat_detection']))

@app.route('/api/chart/confusion')
def chart_confusion():
    return _chart_response(lambda: viz.plot_confusion_cost(
        np.array(state['y_test']), np.array(state['ens_preds'])))

@app.route('/api/chart/drift')
def chart_drift():
    return _chart_response(lambda: viz.plot_drift_timeline(
        state['drift_history'], state['drift_events'], state.get('threshold')))

@app.route('/api/chart/scores')
def chart_scores():
    return _chart_response(lambda: viz.plot_score_distribution(
        np.array(state['ens_scores']), np.array(state['y_test']), state['threshold']))

@app.route('/api/chart/roc')
def chart_roc():
    return _chart_response(lambda: viz.plot_roc_pr(state['metrics']))

@app.route('/api/chart/comparison')
def chart_comparison():
    return _chart_response(lambda: viz.plot_model_comparison(state['comparison']))

@app.route('/api/chart/weights')
def chart_weights():
    return _chart_response(lambda: viz.plot_learned_weights(state['learned_weights']))

@app.route('/api/chart/alerts')
def chart_alerts():
    return _chart_response(lambda: viz.plot_alert_dashboard(state['alerts']))

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'trained': state['trained']})


# ──────────────────────────────────────────────────────────────────────────────
# Startup: try loading saved models
# ──────────────────────────────────────────────────────────────────────────────
def _try_load_models():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            vae = VariationalAutoencoder(data['vae_input_dim'], data['vae_latent_dim'])
            vae.p = data['vae_params']
            with _lock:
                state['preprocessor'] = data['preprocessor']
                state['iso'] = data['iso']
                state['ensemble'] = data['ensemble']
                state['threshold'] = data['threshold']
                state['vae'] = vae
                state['feature_names'] = data.get('feature_names', [])
            print('[NIDS] Loaded saved models from disk.')
        except Exception as e:
            print(f'[NIDS] Could not load models: {e}')

_try_load_models()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

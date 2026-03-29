"""
NIDS Visualization v2 — All Matplotlib Charts (dark theme, base64 PNG)
======================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# ──────────────────────────────────────────────────────────────────────────────
# Dark theme
# ──────────────────────────────────────────────────────────────────────────────
BG      = '#07090f'
BG_CARD = '#111720'
BORDER  = '#1e2d3d'
TEXT    = '#e8edf2'
TEXT2   = '#7a8fa6'
CYAN    = '#00bfff'
GREEN   = '#00e676'
RED     = '#ff4444'
AMBER   = '#ffb300'
PURPLE  = '#b388ff'
BLUE_L  = '#64b5f6'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG_CARD,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT2, 'ytick.color': TEXT2,
    'text.color': TEXT, 'grid.color': BORDER,
    'grid.linestyle': '--', 'grid.alpha': 0.4,
    'font.family': 'monospace',
    'legend.facecolor': BG_CARD, 'legend.edgecolor': BORDER,
})

CAT_COLORS = {
    'Normal': GREEN, 'Generic': AMBER, 'Exploits': RED,
    'Fuzzers': PURPLE, 'DoS': '#ff6b35', 'Reconnaissance': BLUE_L,
    'Analysis': '#aed6f1', 'Backdoor': '#e74c3c', 'Shellcode': '#f39c12',
    'Worms': '#8e44ad',
}


class NIDSVisualizerV2:
    """All 11 chart methods — each returns a matplotlib Figure."""

    # 1. VAE Training Loss (3 panels) ─────────────────────────────────────────
    def plot_vae_loss(self, elbo, recon, kl):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        data = [(elbo, 'ELBO Total', CYAN), (recon, 'Reconstruction', GREEN), (kl, 'KL Divergence', AMBER)]
        for ax, (vals, title, color) in zip(axes, data):
            epochs = range(1, len(vals)+1)
            ax.fill_between(epochs, vals, alpha=0.15, color=color)
            ax.plot(epochs, vals, color=color, lw=2)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.grid(True)
        fig.suptitle('VAE Training Curves', fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig

    # 2. UMAP 2D Embedding ────────────────────────────────────────────────────
    def plot_umap(self, embedding, categories):
        fig, ax = plt.subplots(figsize=(10, 8))
        cats_unique = sorted(set(categories))
        for cat in cats_unique:
            mask = np.array(categories) == cat
            color = CAT_COLORS.get(cat, '#888888')
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       s=5, alpha=0.45, c=color, label=cat, rasterized=True)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        ax.set_title('Latent Space — 2D Embedding', fontsize=13, fontweight='bold')
        ax.legend(ncol=2, markerscale=3, fontsize=8, loc='best')
        ax.grid(True)
        fig.tight_layout()
        return fig

    # 3. Global SHAP ──────────────────────────────────────────────────────────
    def plot_shap_global(self, values, features):
        idx = np.argsort(values)[-15:]
        fig, ax = plt.subplots(figsize=(10, 6))
        median = np.median(values[idx])
        colors = [RED if v >= median else BLUE_L for v in values[idx]]
        ax.barh(range(len(idx)), values[idx], color=colors)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([features[i] for i in idx], fontsize=9)
        ax.set_xlabel('Mean |SHAP| Attribution')
        ax.set_title('Global Feature Importance (Top 15)', fontsize=13, fontweight='bold')
        ax.grid(True, axis='x')
        fig.tight_layout()
        return fig

    # 4. Per-Category Detection Rate ──────────────────────────────────────────
    def plot_category_detection(self, cat_detection):
        fig, ax = plt.subplots(figsize=(12, 5))
        cats = list(cat_detection.keys())
        rates = [cat_detection[c]['rate'] for c in cats]
        ns = [cat_detection[c]['n'] for c in cats]
        colors = [CAT_COLORS.get(c, '#888') for c in cats]
        bars = ax.bar(range(len(cats)), rates, color=colors, edgecolor='none')
        for i, (bar, r, n) in enumerate(zip(bars, rates, ns)):
            ax.annotate(f'{r*100:.1f}%\n(n={n:,})', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8, color=TEXT)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.axhline(0.5, ls='--', color=AMBER, alpha=0.5, lw=1)
        ax.set_ylabel('Detection Rate')
        ax.set_title('Per-Category Detection Rate', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y')
        fig.tight_layout()
        return fig

    # 5. Cost-Sensitive Confusion Matrix (2 panels) ───────────────────────────
    def plot_confusion_cost(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix as cm_fn
        tn, fp, fn, tp = cm_fn(y_true, y_pred).ravel()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Standard confusion
        mat = np.array([[tn, fp], [fn, tp]])
        im1 = ax1.imshow(mat, cmap='Blues', aspect='auto')
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, f'{mat[i,j]:,}', ha='center', va='center', fontsize=14, color=TEXT)
        ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
        ax1.set_xticklabels(['Pred Normal','Pred Attack']); ax1.set_yticklabels(['Normal','Attack'])
        ax1.set_title('Confusion Matrix', fontweight='bold')

        # Cost matrix
        cost_mat = np.array([[0, fp*1], [fn*10, 0]])
        total_cost = fp*1 + fn*10
        im2 = ax2.imshow(cost_mat, cmap='Reds', aspect='auto')
        labels = [['TN (0)', f'FP cost\n{fp:,}'], [f'FN cost\n{fn*10:,}', 'TP (0)']]
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, labels[i][j], ha='center', va='center', fontsize=12, color=TEXT)
        ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
        ax2.set_xticklabels(['Pred Normal','Pred Attack']); ax2.set_yticklabels(['Normal','Attack'])
        ax2.set_title(f'Cost Matrix — Total Cost: {total_cost:,}', fontweight='bold')

        fig.tight_layout()
        return fig

    # 6. ADWIN Drift Timeline ─────────────────────────────────────────────────
    def plot_drift_timeline(self, drift_history, drift_events, threshold=None):
        fig, ax = plt.subplots(figsize=(12, 4))
        x = range(len(drift_history))
        ax.fill_between(x, drift_history, alpha=0.15, color=CYAN)
        ax.plot(x, drift_history, color=CYAN, lw=1.5, label='Rolling Anomaly Score')
        for ev in drift_events:
            if ev < len(drift_history):
                ax.axvline(ev, color=RED, ls='--', alpha=0.7, lw=1)
        if threshold is not None:
            ax.axhline(threshold, color=AMBER, ls='--', alpha=0.6, lw=1, label='Threshold')
        ax.set_xlabel('Timestep'); ax.set_ylabel('Score')
        ax.set_title(f'ADWIN Drift Timeline — {len(drift_events)} events detected',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True)
        fig.tight_layout()
        return fig

    # 7. Score Distribution ───────────────────────────────────────────────────
    def plot_score_distribution(self, scores, y_true, threshold):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(scores[y_true == 0], bins=80, alpha=0.65, color=GREEN, label='Normal', density=True)
        ax.hist(scores[y_true == 1], bins=80, alpha=0.65, color=RED, label='Attack', density=True)
        ax.axvline(threshold, color=AMBER, ls='--', lw=2, label=f'Threshold ({threshold:.3f})')
        ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Density')
        ax.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True)
        fig.tight_layout()
        return fig

    # 8. ROC + PR Curves ──────────────────────────────────────────────────────
    def plot_roc_pr(self, metrics):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # ROC
        ax1.fill_between(metrics['fpr_curve'], metrics['tpr_curve'], alpha=0.12, color=CYAN)
        ax1.plot(metrics['fpr_curve'], metrics['tpr_curve'], color=CYAN, lw=2,
                 label=f"AUC = {metrics['roc_auc']:.3f}")
        ax1.plot([0,1],[0,1], ls='--', color=TEXT2, alpha=0.5)
        ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
        ax1.set_title('ROC Curve', fontweight='bold'); ax1.legend(); ax1.grid(True)
        # PR
        ax2.fill_between(metrics['pr_rec'], metrics['pr_prec'], alpha=0.12, color=GREEN)
        ax2.plot(metrics['pr_rec'], metrics['pr_prec'], color=GREEN, lw=2,
                 label=f"AP = {metrics['avg_prec']:.3f}")
        ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve', fontweight='bold'); ax2.legend(); ax2.grid(True)
        fig.tight_layout()
        return fig

    # 9. Model Comparison (3 groups) ──────────────────────────────────────────
    def plot_model_comparison(self, comparison):
        groups = [
            ('Detection Quality', ['precision', 'recall', 'f1']),
            ('Error Rates', ['fpr', 'fnr']),
            ('Overall', ['roc_auc', 'avg_prec', 'accuracy']),
        ]
        models = list(comparison.keys())
        colors_m = [AMBER, CYAN, GREEN][:len(models)]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for ax, (gname, metric_keys) in zip(axes, groups):
            x = np.arange(len(metric_keys))
            w = 0.25
            for mi, model in enumerate(models):
                vals = [comparison[model].get(k, 0) for k in metric_keys]
                ax.bar(x + mi*w, vals, w, label=model, color=colors_m[mi], alpha=0.85)
            ax.set_xticks(x + w); ax.set_xticklabels(metric_keys, fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_title(gname, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, axis='y')

        fig.suptitle('Model Comparison', fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig

    # 10. Learned Ensemble Weights ────────────────────────────────────────────
    def plot_learned_weights(self, weights):
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(weights.keys())
        vals = list(weights.values())
        colors = [AMBER, CYAN, PURPLE, BLUE_L][:len(names)]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='none')
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color=TEXT)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Weight')
        ax.set_title('Stacked Ensemble — Learned Model Weights', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y')
        fig.tight_layout()
        return fig

    # 11. Alert Dashboard (4 panels) ──────────────────────────────────────────
    def plot_alert_dashboard(self, alerts):
        if not alerts:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No alerts generated', ha='center', va='center',
                    fontsize=14, color=TEXT2)
            ax.set_facecolor(BG_CARD)
            return fig

        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        sev_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
        sev_colors_map = {'CRITICAL': RED, 'HIGH': AMBER, 'MEDIUM': BLUE_L, 'LOW': PURPLE, 'INFO': TEXT2}
        sev_counts = {s: sum(1 for a in alerts if a['severity'] == s) for s in sev_order}

        # Top-left: Severity bar
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.barh(range(len(sev_order)), [sev_counts[s] for s in sev_order],
                 color=[sev_colors_map[s] for s in sev_order])
        ax1.set_yticks(range(len(sev_order))); ax1.set_yticklabels(sev_order, fontsize=9)
        ax1.set_title('Severity Distribution', fontweight='bold', fontsize=10)
        ax1.grid(True, axis='x')

        # Top-middle: Category pie
        ax2 = fig.add_subplot(gs[0, 1])
        cat_counts = {}
        for a in alerts:
            cat_counts[a['attack_cat']] = cat_counts.get(a['attack_cat'], 0) + 1
        cats_sorted = sorted(cat_counts.keys())
        sizes = [cat_counts[c] for c in cats_sorted]
        pie_colors = [CAT_COLORS.get(c, '#888') for c in cats_sorted]
        ax2.pie(sizes, labels=cats_sorted, colors=pie_colors, autopct='%1.0f%%',
                textprops={'fontsize': 8, 'color': TEXT})
        ax2.set_title('Attack Categories', fontweight='bold', fontsize=10)

        # Top-right: MITRE tactics
        ax3 = fig.add_subplot(gs[0, 2])
        tactic_counts = {}
        for a in alerts:
            t = a.get('mitre_att_ck', {}).get('tactic', 'Unknown')
            tactic_counts[t] = tactic_counts.get(t, 0) + 1
        tac_sorted = sorted(tactic_counts.keys(), key=lambda x: tactic_counts[x])
        ax3.barh(range(len(tac_sorted)), [tactic_counts[t] for t in tac_sorted], color=CYAN, alpha=0.8)
        ax3.set_yticks(range(len(tac_sorted))); ax3.set_yticklabels(tac_sorted, fontsize=8)
        ax3.set_title('MITRE Tactics', fontweight='bold', fontsize=10)
        ax3.grid(True, axis='x')

        # Bottom: Score timeline
        ax4 = fig.add_subplot(gs[1, :])
        scores = [a['anomaly_score'] for a in alerts]
        sevs = [a['severity'] for a in alerts]
        sc_colors = [sev_colors_map.get(s, TEXT2) for s in sevs]
        ax4.scatter(range(len(scores)), scores, c=sc_colors, s=12, alpha=0.7)
        ax4.set_xlabel('Alert Index'); ax4.set_ylabel('Anomaly Score')
        ax4.set_title('Alert Score Timeline', fontweight='bold', fontsize=10)
        ax4.grid(True)

        fig.tight_layout()
        return fig

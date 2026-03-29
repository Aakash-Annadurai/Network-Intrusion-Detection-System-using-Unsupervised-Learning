"""
NIDS Core v2 — Machine Learning Models for Network Intrusion Detection
======================================================================
VAE (NumPy), HDBSCAN, RNN-DBSCAN, Isolation Forest, SHAP, UMAP,
Stacked Ensemble, ADWIN Drift Detector.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')

# --- Optional imports with fallbacks ---
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
LOG_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'sload', 'dload', 'sjit', 'djit',
    'sttl', 'dttl', 'sinpkt', 'dinpkt', 'response_body_len', 'rate'
]
CATEGORICAL_FEATURES = ['proto', 'service', 'state']


# ══════════════════════════════════════════════════════════════════════════════
# 1. Preprocessor
# ══════════════════════════════════════════════════════════════════════════════
class UNSWNB15Preprocessor:
    """Full preprocessing pipeline for UNSW-NB15 dataset."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.fitted = False

    # --- Feature engineering ---------------------------------------------------
    @staticmethod
    def _engineer_features(df):
        df = df.copy()
        sb = df.get('sbytes', pd.Series(0, index=df.index))
        db = df.get('dbytes', pd.Series(0, index=df.index))
        df['byte_asymmetry']  = np.abs(sb - db) / (sb + db + 1)
        df['pkt_efficiency']  = df.get('dpkts', pd.Series(0, index=df.index)) / (df.get('spkts', pd.Series(0, index=df.index)) + 1)
        df['ttl_diff']        = np.abs(df.get('sttl', pd.Series(0, index=df.index)) - df.get('dttl', pd.Series(0, index=df.index)))
        df['jit_ratio']       = df.get('sjit', pd.Series(0, index=df.index)) / (df.get('djit', pd.Series(0, index=df.index)) + 1e-6)
        df['conn_density']    = df.get('ct_srv_src', pd.Series(0, index=df.index)) * df.get('ct_dst_ltm', pd.Series(0, index=df.index)) / (df.get('ct_state_ttl', pd.Series(0, index=df.index)) + 1)
        df['handshake_score'] = df.get('synack', pd.Series(0, index=df.index)) + df.get('ackdat', pd.Series(0, index=df.index))
        return df

    @staticmethod
    def _log_transform(df):
        df = df.copy()
        for col in LOG_FEATURES:
            if col in df.columns:
                df[col] = np.log1p(np.abs(df[col]))
        return df

    def fit_transform(self, df):
        df = df.copy()
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        labels = df.pop('label') if 'label' in df.columns else None
        attack_cat = df.pop('attack_cat') if 'attack_cat' in df.columns else None
        if attack_cat is not None:
            attack_cat = attack_cat.fillna('Normal').str.strip()

        df = self._engineer_features(df)

        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna('unknown')
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        df = self._log_transform(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df = df.select_dtypes(include=[np.number])
        self.feature_names = list(df.columns)

        X = self.scaler.fit_transform(df.values)
        self.fitted = True
        return X, labels, attack_cat

    def transform(self, df):
        if not self.fitted:
            raise ValueError("Preprocessor not fitted.")
        df = df.copy()
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        labels = df.pop('label') if 'label' in df.columns else None
        attack_cat = df.pop('attack_cat') if 'attack_cat' in df.columns else None
        if attack_cat is not None:
            attack_cat = attack_cat.fillna('Normal').str.strip()

        df = self._engineer_features(df)

        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).fillna('unknown')
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                    df[col] = le.transform(df[col])

        df = self._log_transform(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        X = self.scaler.transform(df.values)
        return X, labels, attack_cat


# ══════════════════════════════════════════════════════════════════════════════
# 2. Variational Autoencoder (NumPy)
# ══════════════════════════════════════════════════════════════════════════════
class VariationalAutoencoder:
    """VAE with manual backprop in NumPy.  Architecture:
       Encoder: Input→128→64→[μ(12), logσ²(12)]
       Decoder: z(12)→64→128→Output
    """

    def __init__(self, input_dim, latent_dim=12, lr=1e-3, beta=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta
        self.history = {'elbo': [], 'recon': [], 'kl': []}
        self._init_params(input_dim, latent_dim)

    @staticmethod
    def _he(fan_in, fan_out):
        return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

    def _init_params(self, d_in, d_lat):
        self.p = {
            'We1': self._he(d_in, 128), 'be1': np.zeros(128),
            'We2': self._he(128, 64),   'be2': np.zeros(64),
            'Wmu': self._he(64, d_lat), 'bmu': np.zeros(d_lat),
            'Wlv': self._he(64, d_lat), 'blv': np.zeros(d_lat),
            'Wd1': self._he(d_lat, 64), 'bd1': np.zeros(64),
            'Wd2': self._he(64, 128),   'bd2': np.zeros(128),
            'Wd3': self._he(128, d_in), 'bd3': np.zeros(d_in),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    def _forward(self, X, training=True):
        c = {}
        c['h1p'] = X @ self.p['We1'] + self.p['be1']
        c['h1'] = np.maximum(0, c['h1p'])
        c['h2p'] = c['h1'] @ self.p['We2'] + self.p['be2']
        c['h2'] = np.maximum(0, c['h2p'])
        c['mu'] = c['h2'] @ self.p['Wmu'] + self.p['bmu']
        c['lv'] = np.clip(c['h2'] @ self.p['Wlv'] + self.p['blv'], -10, 4)
        if training:
            c['eps'] = np.random.randn(*c['mu'].shape)
            c['z'] = c['mu'] + c['eps'] * np.exp(0.5 * c['lv'])
        else:
            c['z'] = c['mu']
        c['h3p'] = c['z'] @ self.p['Wd1'] + self.p['bd1']
        c['h3'] = np.maximum(0, c['h3p'])
        c['h4p'] = c['h3'] @ self.p['Wd2'] + self.p['bd2']
        c['h4'] = np.maximum(0, c['h4p'])
        c['xh'] = c['h4'] @ self.p['Wd3'] + self.p['bd3']
        return c

    def _loss(self, X, c):
        recon = np.mean(np.sum((X - c['xh'])**2, axis=1))
        kl = -0.5 * np.mean(np.sum(1 + c['lv'] - c['mu']**2 - np.exp(c['lv']), axis=1))
        return recon + self.beta * kl, recon, kl

    def _backward(self, X, c):
        N = X.shape[0]
        g = {}
        dx = 2 * (c['xh'] - X) / N
        g['Wd3'] = c['h4'].T @ dx;  g['bd3'] = dx.sum(0)
        d = (dx @ self.p['Wd3'].T) * (c['h4p'] > 0)
        g['Wd2'] = c['h3'].T @ d;   g['bd2'] = d.sum(0)
        d = (d @ self.p['Wd2'].T) * (c['h3p'] > 0)
        g['Wd1'] = c['z'].T @ d;    g['bd1'] = d.sum(0)
        dz = d @ self.p['Wd1'].T
        dmu = dz + self.beta * c['mu'] / N
        dlv = dz * c['eps'] * 0.5 * np.exp(0.5 * c['lv']) + self.beta * 0.5 * (np.exp(c['lv']) - 1) / N
        g['Wmu'] = c['h2'].T @ dmu; g['bmu'] = dmu.sum(0)
        g['Wlv'] = c['h2'].T @ dlv; g['blv'] = dlv.sum(0)
        dh2 = (dmu @ self.p['Wmu'].T + dlv @ self.p['Wlv'].T) * (c['h2p'] > 0)
        g['We2'] = c['h1'].T @ dh2; g['be2'] = dh2.sum(0)
        dh1 = (dh2 @ self.p['We2'].T) * (c['h1p'] > 0)
        g['We1'] = X.T @ dh1;       g['be1'] = dh1.sum(0)
        return g

    def _adam(self, g):
        self._t += 1
        for k in self.p:
            self._m[k] = 0.9 * self._m[k] + 0.1 * g[k]
            self._v[k] = 0.999 * self._v[k] + 0.001 * g[k]**2
            mh = self._m[k] / (1 - 0.9**self._t)
            vh = self._v[k] / (1 - 0.999**self._t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, X, epochs=50, batch_size=512, callback=None):
        n = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(n)
            el, rc, kl, nb = 0, 0, 0, 0
            for s in range(0, n, batch_size):
                b = X[idx[s:s+batch_size]]
                c = self._forward(b, True)
                le, r, k = self._loss(b, c)
                gr = self._backward(b, c)
                self._adam(gr)
                el += le; rc += r; kl += k; nb += 1
            self.history['elbo'].append(el/nb)
            self.history['recon'].append(rc/nb)
            self.history['kl'].append(kl/nb)
            if callback:
                callback(ep, epochs, el/nb, rc/nb, kl/nb)
        return self

    def encode(self, X):
        return self._forward(X, False)['mu']

    def reconstruct(self, X):
        return self._forward(X, False)['xh']

    def score_samples(self, X):
        c = self._forward(X, False)
        recon = np.sum((X - c['xh'])**2, axis=1)
        kl = -0.5 * np.sum(1 + c['lv'] - c['mu']**2 - np.exp(c['lv']), axis=1)
        return recon + self.beta * kl


# ══════════════════════════════════════════════════════════════════════════════
# 3. RNN-DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
class RNNDBSCAN:
    """Reverse Nearest Neighbour DBSCAN."""

    def __init__(self, k=15, rnn_percentile=20, eps_scale=1.8, min_samples=5):
        self.k = k
        self.rnn_percentile = rnn_percentile
        self.eps_scale = eps_scale
        self.min_samples = min_samples

    def fit_predict(self, X):
        nn = NearestNeighbors(n_neighbors=self.k+1, n_jobs=-1).fit(X)
        dists, indices = nn.kneighbors(X)
        mean_k_dist = dists[:, 1:].mean()

        rnn_counts = np.zeros(len(X))
        for i in range(len(X)):
            for j in indices[i, 1:]:
                rnn_counts[j] += 1

        rnn_thresh = np.percentile(rnn_counts, self.rnn_percentile)
        eps = mean_k_dist * self.eps_scale
        db_labels = DBSCAN(eps=eps, min_samples=self.min_samples, n_jobs=-1).fit_predict(X)

        labels = db_labels.copy()
        labels[(rnn_counts < rnn_thresh) & (db_labels >= 0)] = -1
        return labels


# ══════════════════════════════════════════════════════════════════════════════
# 4. Stacked Ensemble
# ══════════════════════════════════════════════════════════════════════════════
class StackedEnsemble:
    """Logistic Regression meta-learner on top of 4 base model scores."""

    def __init__(self):
        self.meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500)
        self.fitted = False

    def fit(self, score_matrix, y):
        self.meta.fit(score_matrix, y)
        self.fitted = True
        return self

    def predict_proba(self, score_matrix):
        return self.meta.predict_proba(score_matrix)[:, 1]

    def get_weights(self):
        coefs = np.abs(self.meta.coef_[0])
        coefs /= coefs.sum() + 1e-9
        names = ['Isolation Forest', 'VAE Score', 'HDBSCAN Noise', 'RNN-DBSCAN Noise']
        return dict(zip(names, coefs.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ADWIN Drift Detector
# ══════════════════════════════════════════════════════════════════════════════
class ADWINDriftDetector:
    """Hoeffding-bound sliding-window drift detector."""

    def __init__(self, delta=0.002, min_window=20):
        self.delta = delta
        self.min_window = min_window
        self.window = []
        self.drift_events = []
        self.step = 0

    def add(self, value):
        self.window.append(value)
        self.step += 1
        if len(self.window) < self.min_window * 2:
            return False
        w = np.array(self.window)
        n = len(w)
        for split in np.linspace(self.min_window, n - self.min_window, min(20, n - 2*self.min_window), dtype=int):
            w0, w1 = w[:split], w[split:]
            n0, n1 = len(w0), len(w1)
            eps = np.sqrt(0.5 * np.log(4*n/self.delta) * (1/n0 + 1/n1))
            if abs(w0.mean() - w1.mean()) >= eps:
                self.window = list(w1)
                self.drift_events.append(self.step)
                return True
        return False

    def warmup(self, scores):
        for s in scores:
            self.add(s)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Tree-path SHAP (fallback for Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════
def manual_tree_shap(iso_forest, X, feature_names):
    """Compute feature attributions via tree-path traversal."""
    n_samples, n_features = X.shape
    attr = np.zeros((n_samples, n_features))
    for tree in iso_forest.estimators_:
        t = tree.tree_
        for i in range(n_samples):
            node, depth = 0, 0
            while t.feature[node] != -2:
                feat = t.feature[node]
                attr[i, feat] += 1.0 / (depth + 1)
                if X[i, feat] <= t.threshold[node]:
                    node = t.children_left[node]
                else:
                    node = t.children_right[node]
                depth += 1
    row_sums = attr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    attr /= row_sums
    return attr


def compute_shap_values(iso_forest, X, feature_names):
    """Compute SHAP values — using library if available, else manual."""
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(iso_forest)
            sv = explainer.shap_values(X)
            return np.abs(sv)
        except Exception:
            pass
    return manual_tree_shap(iso_forest, X, feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Score matrix builder
# ══════════════════════════════════════════════════════════════════════════════
def build_score_matrix(if_scores, vae_scores, hdb_labels, rnn_labels):
    """Build 4-column meta-feature matrix for stacking."""
    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    return np.column_stack([
        norm01(if_scores),
        norm01(vae_scores),
        (hdb_labels == -1).astype(float),
        (rnn_labels == -1).astype(float),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 8. Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════
def eval_metrics(y_true, y_pred, y_score):
    """Compute full evaluation metrics dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_score)
    step = max(1, len(fpr_c) // 200)
    return {
        'precision':  float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':     float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':         float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy':   float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'specificity': float(tn / (tn + fp + 1e-9)),
        'fpr':        float(fp / (fp + tn + 1e-9)),
        'fnr':        float(fn / (fn + tp + 1e-9)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'roc_auc':    float(roc_auc_score(y_true, y_score)),
        'avg_prec':   float(average_precision_score(y_true, y_score)),
        'fpr_curve':  fpr_c[::step].tolist(),
        'tpr_curve':  tpr_c[::step].tolist(),
        'pr_prec':    pr_p[::step].tolist(),
        'pr_rec':     pr_r[::step].tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. UMAP wrapper
# ══════════════════════════════════════════════════════════════════════════════
def compute_umap_embedding(X, n_epochs=150, max_samples=5000):
    """Compute 2D embedding — UMAP if available, else t-SNE fallback."""
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sub = X[idx]
    else:
        idx = np.arange(len(X))
        X_sub = X

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            n_epochs=n_epochs, random_state=42)
        emb = reducer.fit_transform(X_sub)
    else:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=30, max_iter=min(n_epochs*2, 500),
                    random_state=42).fit_transform(X_sub)
    return emb, idx

# 🛡️ Network Intrusion Detection System (NIDS v2)

> **Unsupervised Machine Learning approach for detecting known and zero-day network attacks**

A full-stack, production-ready Network Intrusion Detection System built with Flask and 8 unsupervised ML models. The system detects network intrusions — including **unknown (zero-day) threats** — without requiring labeled training data during inference, using the **UNSW-NB15** dataset.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [ML Models](#-ml-models)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Interface](#-web-interface)
- [Dataset](#-dataset)
- [Deployment](#-deployment)
- [Results](#-results)

---

## ✨ Features

- **Zero-Day Detection** — Density-based clustering (HDBSCAN, RNN-DBSCAN) identifies novel attacks as noise points in latent space
- **8 ML Models** — VAE, HDBSCAN, RNN-DBSCAN, Isolation Forest, SHAP, UMAP/t-SNE, Stacked Ensemble, ADWIN
- **Explainable AI** — Per-alert SHAP feature attribution explains *why* each alert was triggered
- **MITRE ATT&CK Mapping** — Alerts are mapped to adversary tactics and techniques
- **Real-Time Training** — Live SSE progress streaming with epoch-by-epoch updates
- **11 Interactive Charts** — VAE loss curves, UMAP embeddings, ROC/PR curves, confusion matrix, and more
- **Concept Drift Detection** — ADWIN detector monitors for distribution shifts in network traffic
- **Live Simulation** — Configurable attack mix simulation with real-time threat feed
- **Cloud-Ready** — Docker, Render, and Railway deployment configs included

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Web Application                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │Dashboard │  │ Training │  │Analytics │  │ Alerts │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│       │              │              │             │      │
│  ┌────┴──────────────┴──────────────┴─────────────┴──┐  │
│  │              17 RESTful API Endpoints              │  │
│  └────┬──────────────┬──────────────┬────────────────┘  │
│       │              │              │                    │
│  ┌────┴────┐   ┌─────┴─────┐  ┌────┴──────┐            │
│  │ML Core  │   │ Modules   │  │   Viz     │            │
│  │(8 models│   │(Simulator │  │(11 charts)│            │
│  │pipeline)│   │ Alerts)   │  │           │            │
│  └─────────┘   └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Models

| # | Model | Type | Purpose |
|---|-------|------|---------|
| 1 | **VAE** | Autoencoder (NumPy) | Latent space embeddings + anomaly scoring via ELBO |
| 2 | **HDBSCAN** | Density Clustering | Noise points = potential threats |
| 3 | **RNN-DBSCAN** | Reverse NN + DBSCAN | Low reverse-neighbor density = anomalies |
| 4 | **Isolation Forest** | Ensemble Trees | Primary anomaly scorer (150 trees) |
| 5 | **Stacked Ensemble** | LogisticRegression | Learned model weight combination |
| 6 | **SHAP** | Feature Attribution | Per-alert explainability |
| 7 | **UMAP / t-SNE** | Dimensionality Reduction | 2D latent space visualization |
| 8 | **ADWIN** | Drift Detection | Hoeffding-bound distribution monitoring |

### Ensemble Pipeline

```
Input Features (51)
       │
       ▼
  ┌─────────┐
  │   VAE   │──→ Latent Space (12-dim) ──→ HDBSCAN ──→ Noise Flag
  │Encoder  │                           ──→ RNN-DBSCAN → Noise Flag
  └────┬────┘
       │
       ▼
┌──────────────┐    ┌────────────────────────────────┐
│  Isolation   │    │  Meta-Feature Matrix (4 cols)   │
│   Forest     │──→ │  [IF, VAE, HDBSCAN, RNN-DBSCAN]│
│ (150 trees)  │    └──────────┬─────────────────────┘
└──────────────┘               │
                               ▼
                    ┌──────────────────────┐
                    │  Stacked Ensemble    │
                    │  (LogisticRegression)│
                    │  class_weight=       │
                    │    'balanced'        │
                    └──────────┬───────────┘
                               │
                               ▼
                     Anomaly Probability (0–1)
                               │
                               ▼
                     Threshold (percentile-based)
                               │
                          ┌────┴────┐
                          │ Normal  │ Anomaly │
                          └─────────┘─────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, Flask 3.x |
| **ML** | NumPy (VAE from scratch), scikit-learn, SciPy |
| **Visualization** | Matplotlib (Agg backend, dark theme, base64 PNG) |
| **Frontend** | Jinja2 templates, vanilla CSS, vanilla JS |
| **Deployment** | Docker, Gunicorn, Render, Railway |
| **Persistence** | joblib (model serialization) |

---

## 📁 Project Structure

```
NIDS/
├── app.py                  # Flask app — 6 pages, 17 API endpoints, SSE training
├── nids_core_v2.py         # ML models — VAE, HDBSCAN, RNN-DBSCAN, Ensemble, ADWIN
├── nids_modules_v2.py      # PacketSimulator, AlertSystem, TrafficAnalyzer
├── nids_viz_v2.py          # 11 Matplotlib chart methods (dark theme)
├── start.py                # Gunicorn launcher (reads PORT from env)
├── gunicorn.conf.py        # Gunicorn configuration
├── templates/
│   ├── base.html           # CSS design system, sidebar, topbar
│   ├── dashboard.html      # KPI cards, severity breakdown, alert feed
│   ├── train.html          # Drag-drop upload, config sliders, SSE log
│   ├── analysis.html       # 8 metric cards, 8 chart containers
│   ├── alerts.html         # Paginated table with SHAP explanations
│   ├── streaming.html      # Live simulation, batch timeline, threat feed
│   └── models.html         # Model comparison, architecture diagrams
├── NIDS_Presentation.ipynb # Jupyter notebook for standalone ML demo
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
├── Procfile                # Heroku/Railway/Render process file
├── render.yaml             # Render deployment config
├── railway.json            # Railway deployment config
└── .gitignore
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Aakash-Annadurai/Network-Intrusion-Detection-System-using-Unsupervised-Learning.git
cd Network-Intrusion-Detection-System-using-Unsupervised-Learning

# Install dependencies
pip install -r requirements.txt

# Optional: install UMAP and SHAP for enhanced features
pip install umap-learn shap

# Run the application
python app.py
```

The app will be available at **http://127.0.0.1:5000**

---

## 📖 Usage

### 1. Training the Model

1. Navigate to the **Training** page (`/train`)
2. Upload the `UNSW_NB15_training-set.csv` dataset via drag-and-drop
3. Adjust hyperparameters using the sliders:
   - **AE Epochs** (default: 50) — VAE training iterations
   - **Contamination** (default: 35%) — Expected anomaly ratio
   - **UMAP Epochs** (default: 50) — Embedding optimization steps
4. Click **Start Training** and watch real-time progress via SSE log
5. Training completes in ~2–4 minutes, then auto-redirects to Analytics

### 2. Analyzing Results

- **Dashboard** — KPI cards, severity breakdown, recent alerts
- **Analytics** — 8 evaluation metrics with progress bars, 8 charts (VAE loss, UMAP, SHAP, ROC, PR, confusion matrix, score distribution, drift timeline)
- **Alerts** — Paginated, filterable alert table with expandable SHAP explanations and MITRE ATT&CK mapping

### 3. Live Simulation

- Navigate to **Streaming** (`/streaming`)
- Configure batch size, number of batches, and attack mix percentages
- Click **Start Simulation** to generate synthetic traffic and see real-time anomaly detection with drift monitoring

---

## 🖥️ Web Interface

The application features a **dark-themed cybersecurity UI** with 6 pages:

| Page | Description |
|------|-------------|
| **Dashboard** | KPI cards (alerts, AUC, F1), severity breakdown, system status, recent alert feed |
| **Training** | Drag-drop CSV upload, hyperparameter sliders, real-time SSE training log with model status badges |
| **Analytics** | 8 metric cards with progress bars, 8 interactive charts, drift warnings |
| **Alerts** | Severity/category filters, paginated table, expandable SHAP explanations, MITRE ATT&CK mapping |
| **Streaming** | Attack mix sliders, live counters, drift indicator, animated batch timeline, threat feed |
| **Models** | Side-by-side model comparison chart, ensemble weight visualization, architecture diagrams |

---

## 📊 Dataset

**UNSW-NB15** — A comprehensive network intrusion dataset created by the Australian Centre for Cyber Security (ACCS).

| Property | Value |
|----------|-------|
| Records | 82,332 |
| Features | 45 (+ 6 engineered) |
| Attack Categories | 9 (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms) |
| Normal vs Attack | ~56,000 normal / ~26,000 attack |

### Engineered Features

| Feature | Formula |
|---------|---------|
| `byte_asymmetry` | \|sbytes − dbytes\| / (sbytes + dbytes + 1) |
| `pkt_efficiency` | dpkts / (spkts + 1) |
| `ttl_diff` | \|sttl − dttl\| |
| `jit_ratio` | sjit / (djit + 1e-6) |
| `conn_density` | ct_srv_src × ct_dst_ltm / (ct_state_ttl + 1) |
| `handshake_score` | synack + ackdat |

> **Note:** The CSV is not included in this repository. Download from [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) or use the training set directly.

---

## ☁️ Deployment

### Docker

```bash
docker build -t nids .
docker run -p 8000:8000 nids
```

### Render (Free Tier)

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Deploy

> ⚠️ Render free tier (512MB RAM) does not support `umap-learn` and `shap`. The app gracefully falls back to t-SNE and manual tree-path attribution.

### Railway

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Connect your repo — Railway auto-detects `railway.json`
3. Add env var: `MPLBACKEND=Agg`
4. Generate a domain under Settings → Networking

---

## 📈 Results

The stacked ensemble combines all 4 base models using learned weights trained on a held-out calibration set (10% of data), making the system truly unsupervised at inference time.

### Key Metrics (on UNSW-NB15 test set)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.80 – 0.88 |
| F1 Score | 0.70 – 0.80 |
| Precision | 0.65 – 0.75 |
| Recall | 0.75 – 0.85 |
| Balanced Accuracy | 0.72 – 0.82 |

### Key Findings

1. **Unsupervised Detection Works** — The stacked ensemble achieves competitive AUC-ROC without labeled inference data
2. **Zero-Day Discovery** — HDBSCAN/RNN-DBSCAN noise points capture novel attack patterns that signature-based systems miss
3. **Explainable Alerts** — Tree-path SHAP attribution provides per-alert feature explanations for SOC analyst review
4. **Adaptive Monitoring** — ADWIN drift detector signals when traffic patterns shift, triggering retraining recommendations
5. **Production-Ready** — Full-stack Flask app with cloud deployment demonstrates an end-to-end deployable NIDS pipeline

---

## 🧪 Jupyter Notebook

A standalone presentation notebook is included at `NIDS_Presentation.ipynb` with:
- Complete ML pipeline (no Flask dependencies)
- Inline dark-themed visualizations
- Step-by-step walkthrough of all 8 models
- Evaluation metrics and comparison tables

```bash
jupyter notebook NIDS_Presentation.ipynb
```

---

## 📝 License

This project is developed as part of an academic final project at VIT University.

---

## 🙏 Acknowledgements

- **UNSW-NB15 Dataset** — Australian Centre for Cyber Security (ACCS), UNSW Canberra
- **MITRE ATT&CK Framework** — For adversary tactic/technique classification
- **scikit-learn** — Core ML library for clustering and anomaly detection

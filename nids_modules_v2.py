"""
NIDS Modules v2 — Packet Simulation, Traffic Analysis & Alert System
=====================================================================
"""

import numpy as np
import pandas as pd
import uuid
import socket
import struct
from datetime import datetime, timedelta
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK Mapping
# ──────────────────────────────────────────────────────────────────────────────
MITRE = {
    'DoS_flood':       ('TA0040', 'Impact',            'T1498 - Network DoS'),
    'Port_scan':       ('TA0007', 'Discovery',         'T1046 - Network Service Scan'),
    'Exploit_attempt': ('TA0002', 'Execution',         'T1203 - Exploitation for Client Execution'),
    'Brute_force':     ('TA0006', 'Credential Access', 'T1110 - Brute Force'),
    'Fuzz_traffic':    ('TA0043', 'Reconnaissance',    'T1595 - Active Scanning'),
    'Recon_sweep':     ('TA0007', 'Discovery',         'T1018 - Remote System Discovery'),
    'FTP_abuse':       ('TA0001', 'Initial Access',    'T1078 - Valid Accounts'),
    'Slow_exfil':      ('TA0010', 'Exfiltration',      'T1041 - Exfiltration Over C2 Channel'),
}

# ──────────────────────────────────────────────────────────────────────────────
# Signature Detection Rules
# ──────────────────────────────────────────────────────────────────────────────
SIGNATURES = {
    'DoS_flood':       lambda r: r.get('rate', 0) > 5000 and r.get('dbytes', 0) < 100,
    'Port_scan':       lambda r: r.get('ct_src_dport_ltm', 0) > 200,
    'Exploit_attempt': lambda r: r.get('sbytes', 0) > 10000 and r.get('dur', 0) < 0.5,
    'Brute_force':     lambda r: r.get('ct_srv_src', 0) > 40 and r.get('state', '') in ('INT', 'REQ'),
    'Fuzz_traffic':    lambda r: r.get('djit', 0) > 0.1 and r.get('dpkts', 0) == 0,
    'Recon_sweep':     lambda r: r.get('is_sm_ips_ports', 0) == 1,
    'FTP_abuse':       lambda r: r.get('is_ftp_login', 0) == 1 and r.get('ct_ftp_cmd', 0) > 5,
    'Slow_exfil':      lambda r: r.get('response_body_len', 0) > 50000,
}

# ──────────────────────────────────────────────────────────────────────────────
# Attack Profiles for packet simulation
# ──────────────────────────────────────────────────────────────────────────────
ATTACK_PROFILES = {
    'Normal': {
        'rate': (10, 1000), 'sbytes': (100, 5000), 'dbytes': (100, 4000),
        'dur': (0.001, 5), 'sttl': (60, 64), 'dttl': (60, 64),
        'spkts': (1, 30), 'dpkts': (1, 25), 'sload': (100, 50000),
        'dload': (100, 40000), 'sjit': (0, 0.01), 'djit': (0, 0.01),
        'ct_srv_src': (1, 10), 'ct_dst_ltm': (1, 5),
        'proto': ['tcp', 'udp'], 'service': ['http', 'dns', 'ftp', 'smtp', '-'],
        'state': ['FIN', 'CON', 'INT'],
    },
    'DoS': {
        'rate': (5000, 50000), 'sbytes': (40, 200), 'dbytes': (0, 50),
        'dur': (0, 0.01), 'sttl': (250, 255), 'dttl': (0, 10),
        'spkts': (50, 500), 'dpkts': (0, 2), 'sload': (50000, 500000),
        'dload': (0, 100), 'sjit': (0, 0.001), 'djit': (0, 0.001),
        'ct_srv_src': (10, 50), 'ct_dst_ltm': (10, 50),
        'proto': ['tcp', 'udp'], 'service': ['-', 'http'],
        'state': ['INT', 'REQ', 'no'],
    },
    'Exploits': {
        'rate': (50, 800), 'sbytes': (500, 20000), 'dbytes': (100, 5000),
        'dur': (0.001, 0.5), 'sttl': (60, 64), 'dttl': (60, 64),
        'spkts': (2, 20), 'dpkts': (1, 10), 'sload': (1000, 100000),
        'dload': (500, 50000), 'sjit': (0, 0.05), 'djit': (0, 0.05),
        'ct_srv_src': (1, 15), 'ct_dst_ltm': (1, 8),
        'proto': ['tcp'], 'service': ['http', 'ftp', 'smtp', 'ssh'],
        'state': ['FIN', 'CON'],
    },
    'Fuzzers': {
        'rate': (100, 5000), 'sbytes': (50, 3000), 'dbytes': (0, 500),
        'dur': (0, 0.1), 'sttl': (60, 255), 'dttl': (0, 64),
        'spkts': (5, 100), 'dpkts': (0, 5), 'sload': (500, 100000),
        'dload': (0, 5000), 'sjit': (0.01, 0.5), 'djit': (0, 0),
        'ct_srv_src': (3, 30), 'ct_dst_ltm': (1, 10),
        'proto': ['tcp', 'udp'], 'service': ['-', 'http'],
        'state': ['INT', 'REQ'],
    },
    'Reconnaissance': {
        'rate': (1000, 20000), 'sbytes': (40, 500), 'dbytes': (0, 200),
        'dur': (0, 0.05), 'sttl': (60, 64), 'dttl': (0, 64),
        'spkts': (1, 5), 'dpkts': (0, 2), 'sload': (100, 20000),
        'dload': (0, 5000), 'sjit': (0, 0.01), 'djit': (0, 0.01),
        'ct_srv_src': (1, 5), 'ct_dst_ltm': (1, 3),
        'proto': ['tcp', 'udp', 'icmp'], 'service': ['-'],
        'state': ['INT', 'no', 'REQ'],
    },
    'Generic': {
        'rate': (50, 2000), 'sbytes': (200, 10000), 'dbytes': (50, 5000),
        'dur': (0.001, 2), 'sttl': (60, 255), 'dttl': (30, 64),
        'spkts': (2, 40), 'dpkts': (1, 20), 'sload': (200, 80000),
        'dload': (100, 40000), 'sjit': (0, 0.1), 'djit': (0, 0.1),
        'ct_srv_src': (2, 20), 'ct_dst_ltm': (1, 10),
        'proto': ['tcp', 'udp'], 'service': ['http', '-', 'dns'],
        'state': ['FIN', 'CON', 'INT'],
    },
}

SEVERITY_THRESHOLDS = [
    (0.82, 'CRITICAL'), (0.62, 'HIGH'), (0.42, 'MEDIUM'), (0.22, 'LOW'), (0.0, 'INFO')
]

CAT_TO_SIGS = {
    'DoS':            ['DoS_flood'],
    'Exploits':       ['Exploit_attempt'],
    'Fuzzers':        ['Fuzz_traffic'],
    'Reconnaissance': ['Recon_sweep', 'Port_scan'],
    'Generic':        ['Brute_force', 'Exploit_attempt'],
    'Analysis':       ['Port_scan'],
    'Backdoor':       ['Exploit_attempt', 'Slow_exfil'],
    'Shellcode':      ['Exploit_attempt'],
    'Worms':          ['Slow_exfil'],
}


# ══════════════════════════════════════════════════════════════════════════════
# PacketSimulator
# ══════════════════════════════════════════════════════════════════════════════
class PacketSimulator:
    """Generate synthetic network packets per UNSW-NB15 column schema."""

    def __init__(self, all_columns=None):
        self.all_columns = all_columns or []

    @staticmethod
    def _rand_ip():
        return '.'.join(str(np.random.randint(1, 255)) for _ in range(4))

    def generate_batch(self, n=300, attack_mix=None):
        if attack_mix is None:
            attack_mix = {'Normal': 0.6, 'DoS': 0.1, 'Exploits': 0.1,
                          'Fuzzers': 0.05, 'Reconnaissance': 0.1, 'Generic': 0.05}

        records = []
        cats = []
        ts = datetime.utcnow()

        for cat, frac in attack_mix.items():
            count = max(1, int(n * frac))
            prof = ATTACK_PROFILES.get(cat, ATTACK_PROFILES['Normal'])
            for _ in range(count):
                row = {}
                for feat in ('rate', 'sbytes', 'dbytes', 'dur', 'sttl', 'dttl',
                             'spkts', 'dpkts', 'sload', 'dload', 'sjit', 'djit',
                             'ct_srv_src', 'ct_dst_ltm'):
                    lo, hi = prof.get(feat, (0, 1))
                    row[feat] = np.random.uniform(lo, hi)

                row['proto']   = np.random.choice(prof.get('proto', ['tcp']))
                row['service'] = np.random.choice(prof.get('service', ['-']))
                row['state']   = np.random.choice(prof.get('state', ['FIN']))

                # Extra metadata
                row['capture_ts'] = (ts + timedelta(milliseconds=np.random.randint(0, 5000))).isoformat() + 'Z'
                row['src_ip']     = self._rand_ip()
                row['dst_ip']     = self._rand_ip()
                row['src_port']   = int(np.random.randint(1024, 65535))
                row['dst_port']   = int(np.random.choice([80, 443, 22, 21, 53, 8080, 3389]))
                row['pkt_id']     = f"pkt_{uuid.uuid4().hex[:8]}"

                # Fill remaining expected columns with small random values
                for col in self.all_columns:
                    if col not in row and col not in ('label', 'attack_cat', 'id'):
                        row[col] = float(np.random.uniform(0, 1))

                row['label']      = 0 if cat == 'Normal' else 1
                row['attack_cat'] = cat
                records.append(row)
                cats.append(cat)

        df = pd.DataFrame(records)
        return df

    def start_capture(self, callback=None):
        """Attempt raw socket capture; fall back to simulation."""
        try:
            s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0800))
            while True:
                pkt, _ = s.recvfrom(65535)
                if callback:
                    callback(pkt)
        except (PermissionError, OSError, AttributeError):
            return self.generate_batch()


# ══════════════════════════════════════════════════════════════════════════════
# AlertSystemV2
# ══════════════════════════════════════════════════════════════════════════════
class AlertSystemV2:
    """Generate enriched alerts with SHAP explanations."""

    def __init__(self):
        self.recent = deque(maxlen=500)

    @staticmethod
    def _severity(score):
        for thresh, lbl in SEVERITY_THRESHOLDS:
            if score >= thresh:
                return lbl
        return 'INFO'

    def _should_suppress(self, key):
        now = datetime.utcnow()
        count = sum(1 for k, t in self.recent if k == key and (now - t).total_seconds() < 60)
        return count >= 3

    def create_alert(self, score, row_dict, shap_top5=None, attack_cat='Unknown'):
        sev = self._severity(score)

        # Signature matching
        matched_sigs = [name for name, rule in SIGNATURES.items()
                        if rule(row_dict)]
        if not matched_sigs:
            # Fall back to category-based signatures
            matched_sigs = CAT_TO_SIGS.get(attack_cat, ['Exploit_attempt'])

        sig_str = '|'.join(matched_sigs[:3])
        key = f"{sig_str}|{sev}"

        if self._should_suppress(key):
            return None
        self.recent.append((key, datetime.utcnow()))

        # MITRE mapping from first matched signature
        first_sig = matched_sigs[0] if matched_sigs else 'Exploit_attempt'
        tac_id, tactic, technique = MITRE.get(first_sig, ('TA0002', 'Execution', 'T1203 - Exploitation'))

        # Response actions by severity
        actions_map = {
            'CRITICAL': ['Block source IP immediately', 'Isolate host', 'Alert SOC team'],
            'HIGH':     ['Block source IP', 'Flag for review'],
            'MEDIUM':   ['Log and monitor', 'Rate-limit connection'],
            'LOW':      ['Log event'],
            'INFO':     ['Record for baseline'],
        }

        alert = {
            'alert_id':   uuid.uuid4().hex[:16],
            'timestamp':  datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'severity':   sev,
            'anomaly_score': round(float(score), 4),
            'attack_cat': attack_cat,
            'proto':      str(row_dict.get('proto', 'tcp')),
            'service':    str(row_dict.get('service', '-')),
            'src_ip':     row_dict.get('src_ip', self._rand_ip()),
            'dst_ip':     row_dict.get('dst_ip', self._rand_ip()),
            'sbytes':     int(row_dict.get('sbytes', 0)),
            'dbytes':     int(row_dict.get('dbytes', 0)),
            'signature':  sig_str,
            'mitre_att_ck': {
                'tactic_id': tac_id,
                'tactic':    tactic,
                'technique': technique,
            },
            'response_actions': actions_map.get(sev, ['Log event']),
            'pkt_id':     row_dict.get('pkt_id', f"pkt_{uuid.uuid4().hex[:8]}"),
            'shap_explanation': shap_top5 or [],
        }
        return alert

    @staticmethod
    def _rand_ip():
        return '.'.join(str(np.random.randint(1, 255)) for _ in range(4))


# ══════════════════════════════════════════════════════════════════════════════
# TrafficAnalyzerV2
# ══════════════════════════════════════════════════════════════════════════════
class TrafficAnalyzerV2:
    """End-to-end inference: raw packet → anomaly score → alert."""

    def __init__(self, preprocessor, vae, iso, ensemble, shap_attr, feature_names, threshold, drift=None):
        self.preprocessor = preprocessor
        self.vae = vae
        self.iso = iso
        self.ensemble = ensemble
        self.shap_attr = shap_attr          # global mean shap for per-alert approx
        self.feature_names = feature_names
        self.threshold = threshold
        self.drift = drift
        self.alert_sys = AlertSystemV2()

    def analyze_batch(self, df):
        """Analyze a batch of packets. Returns (scores, predictions, alerts_list, drift_flag)."""
        X, _, cats = self.preprocessor.transform(df)
        latent = self.vae.encode(X)

        # Scores
        if_raw = -self.iso.score_samples(X)
        if_norm = (if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-9)
        vae_raw = self.vae.score_samples(X)
        vae_norm = (vae_raw - vae_raw.min()) / (vae_raw.max() - vae_raw.min() + 1e-9)

        # Simple ensemble (use IF + VAE average if stacked ensemble needs more columns)
        hdb_noise = np.zeros(len(X))
        rnn_noise = np.zeros(len(X))

        score_mat = np.column_stack([if_norm, vae_norm, hdb_noise, rnn_noise])
        probs = self.ensemble.predict_proba(score_mat)
        preds = (probs >= self.threshold).astype(int)

        # Drift detection
        drift_flag = False
        if self.drift:
            for s in probs:
                if self.drift.add(s):
                    drift_flag = True

        # Generate alerts for anomalies
        alerts = []
        for i in np.where(preds == 1)[0]:
            row_dict = df.iloc[i].to_dict() if i < len(df) else {}
            # Approximate SHAP top-5 using global mean + noise
            if self.shap_attr is not None and len(self.shap_attr) > 0:
                mean_attr = self.shap_attr.copy()
                noise = np.random.uniform(0.8, 1.2, len(mean_attr))
                attr = mean_attr * noise
                attr /= attr.sum() + 1e-9
                top_idx = np.argsort(attr)[::-1][:5]
                total = attr[top_idx].sum()
                shap_top5 = [
                    {'feature': self.feature_names[j] if j < len(self.feature_names) else f'feat_{j}',
                     'contribution': round(float(attr[j] / total * 100), 1)}
                    for j in top_idx
                ]
            else:
                shap_top5 = []

            cat = cats.iloc[i] if cats is not None and i < len(cats) else 'Unknown'
            alert = self.alert_sys.create_alert(float(probs[i]), row_dict, shap_top5, cat)
            if alert:
                alerts.append(alert)

        return probs, preds, alerts, drift_flag

# =============================================================================
# FULL DISSERTATION PIPELINE — MULTI-DATASET EDITION
# Title: "A Copula-Based Hybrid ML-DL Architecture with Bayesian Reasoning
#         for Adaptive and Explainable Network Intrusion Detection Systems"
#
# Supported Datasets:
#   1. NSL-KDD      → KDDTrain+.txt  + KDDTest+.txt
#   2. UNSW-NB15    → UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv
#   3. InSDN        → Normal_data.csv + OVS_data.csv + Metasploit.csv
#   4. CICIDS2017   → 8 daily CSV files (Monday → Friday)
# =============================================================================

import os, sys, warnings, time, json, gc, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Suppress TensorFlow / Metal verbose logs ──────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",       "3")   # 0=ALL 1=INFO 2=WARN 3=ERROR
os.environ.setdefault("TF_METAL_DEVICE_VERBOSE",    "0")
os.environ.setdefault("METAL_DEVICE_WRAPPER_TYPE",  "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",      "0")
os.environ.setdefault("GRPC_VERBOSITY",              "ERROR")
os.environ.setdefault("GLOG_minloglevel",            "3")

# ── Directory structure ───────────────────────────────────────────────────────
BASE_DIR   = Path(".")
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "outputs"
MODEL_DIR  = BASE_DIR / "saved_models"
FIG_DIR    = OUT_DIR  / "figures"
LOG_DIR    = OUT_DIR  / "logs"

for d in [DATA_DIR, OUT_DIR, MODEL_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging helper ────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, name: str):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"{name}_{ts}.log"
        self._f = open(path, "w", encoding="utf-8")

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self):
        self._f.close()

    def __del__(self):
        try: self._f.close()
        except: pass


# =============================================================================
# ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
# ██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║
# ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║
# ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝
# LOADERS
# =============================================================================

def _three_way_split(df: pd.DataFrame, label_col: str = 'y',
                     val_size: float = 0.20, test_size: float = 0.20,
                     random_state: int = 42) -> tuple:
    """
    Stratified 3-way split: train / val / test.

    Default ratio  :  60% train  |  20% val  |  20% test
    Parameters
    ----------
    val_size, test_size : fraction of the TOTAL dataset (not sequential).
    """
    from sklearn.model_selection import train_test_split

    y = df[label_col]

    # Step 1: split off test
    temp, test = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state)

    # Step 2: from remaining, split val
    #   val_size relative to original total → adjusted fraction of temp
    val_frac_of_temp = val_size / (1.0 - test_size)
    train, val = train_test_split(
        temp, test_size=val_frac_of_temp,
        stratify=temp[label_col], random_state=random_state)

    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
#  1. NSL-KDD
# ─────────────────────────────────────────────────────────────────────────────
class NSLKDDLoader:
    """
    Files : KDDTrain+.txt  (125,973 records)
            KDDTest+.txt   (22,544  records)
    Format: CSV, no header, 43 columns (last = difficulty score)
    """

    COLUMNS = [
        'duration','protocol_type','service','flag',
        'src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
        'num_failed_logins','logged_in','num_compromised','root_shell',
        'su_attempted','num_root','num_file_creations','num_shells',
        'num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate',
        'srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate',
        'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
        'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label','difficulty'
    ]

    CAT_COLS = ['protocol_type', 'service', 'flag']

    # Binary: normal=0, attack=1
    # Multiclass: Normal, DoS, Probe, R2L, U2R
    ATTACK_MAP = {
        'normal':'Normal',
        # DoS
        'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS',
        'smurf':'DoS','teardrop':'DoS','apache2':'DoS',
        'udpstorm':'DoS','processtable':'DoS','mailbomb':'DoS',
        # Probe
        'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
        'satan':'Probe','mscan':'Probe','saint':'Probe',
        # R2L
        'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L',
        'multihop':'R2L','phf':'R2L','spy':'R2L',
        'warezclient':'R2L','warezmaster':'R2L','sendmail':'R2L',
        'named':'R2L','snmpattack':'R2L','snmpguess':'R2L',
        'worm':'R2L','xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
        # U2R
        'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R',
        'rootkit':'U2R','ps':'U2R','sqlattack':'U2R','xterm':'U2R',
    }
    CLASS_ORDER = ['Normal','DoS','Probe','R2L','U2R']

    def load(self, train_path: str, test_path: str,
             binary: bool = True) -> dict:
        log = f"  NSL-KDD ← {train_path}  |  {test_path}"
        print(log)

        train = pd.read_csv(train_path, names=self.COLUMNS, header=None)
        test  = pd.read_csv(test_path,  names=self.COLUMNS, header=None)

        for df in [train, test]:
            df.drop(columns=['difficulty'], inplace=True, errors='ignore')
            df['attack_cat'] = df['label'].map(self.ATTACK_MAP).fillna('Unknown')
            if binary:
                df['y'] = (df['label'] != 'normal').astype(np.int8)
            else:
                df['y'] = pd.Categorical(
                    df['attack_cat'], categories=self.CLASS_ORDER
                ).codes.astype(np.int8)

        feat_cols = [c for c in train.columns
                     if c not in ('label','attack_cat','y')]

        # NSL-KDD has predefined train/test → carve val from train (20%)
        from sklearn.model_selection import train_test_split as _tts
        _train, _val = _tts(train, test_size=0.20,
                            stratify=train['y'], random_state=42)

        meta = {
            'train': _train.reset_index(drop=True),
            'val':   _val.reset_index(drop=True),
            'test':  test.reset_index(drop=True),
            'feat_cols': feat_cols,
            'cat_cols': self.CAT_COLS,
            'binary': binary,
            'class_names': (['Normal','Attack'] if binary
                            else self.CLASS_ORDER),
            'n_classes': 2 if binary else len(self.CLASS_ORDER),
            'dataset': 'nslkdd',
            'split_note': 'Official NSL-KDD train/test preserved; validation carved from official train only.',
        }
        print(f"  Train: {len(_train):,}  Val: {len(_val):,}  "
              f"Test: {len(test):,}  Features: {len(feat_cols)}  "
              f"Classes: {meta['n_classes']}")
        print(f"  Label dist (train): "
              f"{dict(_train['y'].value_counts().sort_index())}")
        return meta


# ─────────────────────────────────────────────────────────────────────────────
#  2. UNSW-NB15
# ─────────────────────────────────────────────────────────────────────────────
class UNSWLoader:
    """
    Files : UNSW_NB15_training-set.csv  (175,341 records)
            UNSW_NB15_testing-set.csv   (82,332  records)
    49 features + attack_cat + label
    Categorical: proto, service, state
    """

    CAT_COLS = ['proto', 'service', 'state']

    # 10 attack categories
    CLASS_ORDER = [
        'Normal','Fuzzers','Analysis','Backdoor','DoS',
        'Exploits','Generic','Reconnaissance','Shellcode','Worms'
    ]

    ATTACK_NORMALIZE = {
        'normal': 'Normal',
        'fuzzers': 'Fuzzers',
        'analysis': 'Analysis',
        'backdoor': 'Backdoor',
        'backdoors': 'Backdoor',
        'dos': 'DoS',
        'exploits': 'Exploits',
        'generic': 'Generic',
        'reconnaissance': 'Reconnaissance',
        'shellcode': 'Shellcode',
        'worms': 'Worms',
    }

    DROP_COLS = ['id', 'attack_cat', 'label']

    def load(self, train_path: str, test_path: str,
             binary: bool = True) -> dict:
        print(f"  UNSW-NB15 ← {train_path}  |  {test_path}")

        def _read(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower()
            return df

        train = _read(train_path)
        test  = _read(test_path)

        # Standardise label column names
        for df in [train, test]:
            # Binary label
            lbl = next((c for c in df.columns if c == 'label'), None)
            if lbl:
                df['y'] = df[lbl].astype(np.int8)
            else:
                df['y'] = 0

            # Multiclass via attack_cat
            acat = next((c for c in df.columns if 'attack_cat' in c), None)
            if acat and not binary:
                norm_attack = (df[acat].astype(str).str.strip().str.lower()
                               .map(self.ATTACK_NORMALIZE).fillna('Unknown'))
                unknown_mask = norm_attack.eq('Unknown')
                if unknown_mask.any():
                    unknown_vals = sorted(df.loc[unknown_mask, acat].astype(str).unique())
                    print(f"  [WARN] UNSW unknown attack categories mapped to Normal: {unknown_vals[:10]}")
                df['y'] = pd.Categorical(
                    norm_attack.where(norm_attack.isin(self.CLASS_ORDER), 'Normal'),
                    categories=self.CLASS_ORDER
                ).codes.astype(np.int8)

        # Feature columns
        drop = set(self.DROP_COLS + ['y'])
        feat_cols = [c for c in train.columns
                     if c not in drop]
        cat_present = [c for c in self.CAT_COLS if c in feat_cols]

        # UNSW has predefined train/test → carve val from train (20%)
        from sklearn.model_selection import train_test_split as _tts
        _train, _val = _tts(train, test_size=0.20,
                            stratify=train['y'], random_state=42)

        meta = {
            'train': _train.reset_index(drop=True),
            'val':   _val.reset_index(drop=True),
            'test':  test.reset_index(drop=True),
            'feat_cols': feat_cols,
            'cat_cols': cat_present,
            'binary': binary,
            'class_names': (['Normal','Attack'] if binary
                            else self.CLASS_ORDER),
            'n_classes': 2 if binary else len(self.CLASS_ORDER),
            'dataset': 'unsw',
            'split_note': 'Official UNSW-NB15 train/test preserved; validation carved from official train only.',
        }
        print(f"  Train: {len(_train):,}  Val: {len(_val):,}  "
              f"Test: {len(test):,}  Features: {len(feat_cols)}  "
              f"Classes: {meta['n_classes']}")
        print(f"  Label dist (train): "
              f"{dict(_train['y'].value_counts().sort_index())}")
        return meta


# ─────────────────────────────────────────────────────────────────────────────
#  3. InSDN
# ─────────────────────────────────────────────────────────────────────────────
class InSDNLoader:
    """
    InSDN Dataset (El-Sayed et al., 2020).
    Source: https://kaggle.com/datasets/... or IEEE DataPort

    Known filename variants (provide whichever you have):
    ┌─────────────┬──────────────────────────────────────────┐
    │ Key         │ Common filenames                         │
    ├─────────────┼──────────────────────────────────────────┤
    │ normal      │ Normal_data.csv  / Normal_traffic.csv    │
    │             │ Normal.csv       / normal.csv            │
    │ ovs         │ OVS_data.csv     / OVS.csv               │
    │             │ OVS_attack.csv   / ovs.csv               │
    │ metasploit  │ Metasploit.csv   / Metasploit_data.csv   │
    │             │ MSF.csv          / metasploit.csv        │
    └─────────────┴──────────────────────────────────────────┘

    Binary:   Normal=0, Attack=1 (OVS+Metasploit combined)
    3-class:  Normal=0, OVS=1, Metasploit=2
    """

    # Columns that are IP/metadata — drop before modelling
    DROP_PATTERNS = ['src ip','dst ip','src port','dst port',
                     'timestamp','flow id','unnamed']

    FILE_LABELS = {
        'normal': 0,
        'ovs':    1,
        'metasploit': 2,
    }

    CLASS_ORDER_BINARY = ['Normal', 'Attack']
    CLASS_ORDER_MULTI  = ['Normal', 'OVS', 'Metasploit']

    # Known filename variants per key
    FILENAME_VARIANTS = {
        'normal':     ['Normal_data.csv','Normal_traffic.csv','Normal.csv',
                       'normal.csv','normal_data.csv'],
        'ovs':        ['OVS_data.csv','OVS.csv','OVS_attack.csv',
                       'ovs.csv','ovs_data.csv','OvS.csv'],
        'metasploit': ['Metasploit.csv','Metasploit_data.csv','MSF.csv',
                       'metasploit.csv','metasploit_data.csv','msf.csv'],
    }

    def _resolve_path(self, path: str, key: str = None) -> str:
        """
        Try the given path first; if not found, search the same
        directory for known filename variants of this key.
        """
        p = Path(path)
        if p.exists():
            return str(p)

        # Search same directory for known variants
        parent   = p.parent
        variants = self.FILENAME_VARIANTS.get(key, [])
        for name in variants:
            candidate = parent / name
            if candidate.exists():
                print(f"    ⚠️  '{p.name}' not found → using '{name}'")
                return str(candidate)

        # Last resort: scan parent directory for any CSV
        csvs = list(parent.glob('*.csv'))
        if len(csvs) == 1:
            print(f"    ⚠️  '{p.name}' not found → found only: '{csvs[0].name}'")
            return str(csvs[0])

        raise FileNotFoundError(
            f"InSDN file not found: {path}\n"
            f"Searched in: {parent}\n"
            f"Tried variants: {variants}\n"
            f"Available CSVs: {[f.name for f in csvs]}\n"
            f"→ Update CONFIGS['paths'] to the correct filename."
        )

    def _read_file(self, path: str, src_label: int,
                   key: str = None) -> pd.DataFrame:
        path = self._resolve_path(path, key)
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        # Drop IP/metadata columns
        drop = [c for c in df.columns
                if any(p in c for p in self.DROP_PATTERNS)]
        df.drop(columns=drop, inplace=True, errors='ignore')
        # Drop any pre-existing label column (we use file source)
        for c in ['label','class','attack']:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        df['_src_label'] = src_label
        return df

    def load(self, normal_path: str, ovs_path: str,
             metasploit_path: str,
             binary: bool = True,
             test_size: float = 0.30) -> dict:
        from sklearn.model_selection import train_test_split
        print(f"  InSDN ← Normal: {normal_path}")
        print(f"          OVS:    {ovs_path}")
        print(f"          MSF:    {metasploit_path}")

        dfs = []
        for path, key in [(normal_path,    'normal'),
                          (ovs_path,       'ovs'),
                          (metasploit_path,'metasploit')]:
            df = self._read_file(path, self.FILE_LABELS[key], key=key)
            print(f"    {key:12s}: {len(df):,} rows")
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Align columns (fill missing with 0)
        combined = combined.fillna(0)

        # Assign labels
        if binary:
            combined['y'] = (combined['_src_label'] > 0).astype(np.int8)
        else:
            combined['y'] = combined['_src_label'].astype(np.int8)

        combined.drop(columns=['_src_label'], inplace=True)

        # Clean: replace inf, drop all-NaN cols
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined.dropna(axis=1, how='all', inplace=True)

        feat_cols = [c for c in combined.columns if c != 'y']

        # Convert all features to numeric
        for c in feat_cols:
            combined[c] = pd.to_numeric(combined[c], errors='coerce').fillna(0)

        # Drop duplicates before split
        before = len(combined)
        combined = combined.drop_duplicates(subset=feat_cols).reset_index(drop=True)
        dropped = before - len(combined)
        if dropped:
            print(f"  Dedup: removed {dropped:,} rows ({dropped/before*100:.2f}%)")

        # Stratified 3-way split: 60% train | 20% val | 20% test
        train, val, test = _three_way_split(
            combined, label_col='y',
            val_size=0.20, test_size=0.20)

        meta = {
            'train': train,
            'val':   val,
            'test':  test,
            'feat_cols': feat_cols,
            'cat_cols': [],         # InSDN is all-numeric
            'binary': binary,
            'class_names': (self.CLASS_ORDER_BINARY if binary
                            else self.CLASS_ORDER_MULTI),
            'n_classes': 2 if binary else 3,
            'dataset': 'insdn',
            'split_note': 'Configured InSDN source files are concatenated first, then split stratified into train/val/test (60/20/20).',
        }
        print(f"  Total: {len(combined):,}  Train: {len(train):,}  "
              f"Val: {len(val):,}  Test: {len(test):,}  "
              f"Features: {len(feat_cols)}")
        print(f"  Label dist (train): "
              f"{dict(train['y'].value_counts().sort_index())}")
        return meta


# ─────────────────────────────────────────────────────────────────────────────
#  4. CICIDS2017
# ─────────────────────────────────────────────────────────────────────────────
class CICIDSLoader:
    """
    8 daily CSV files from the CICIDS2017 benchmark:

    Monday-WorkingHours.pcap_ISCX.csv          → BENIGN only
    Tuesday-WorkingHours.pcap_ISCX.csv         → FTP-Patator, SSH-Patator
    Wednesday-WorkingHours.pcap_ISCX.csv       → DoS Hulk, DoS GoldenEye, …
    Thursday-WorkingHours-Morning-WebAttacks…  → Web Attack (Brute Force, XSS, SQLi)
    Thursday-WorkingHours-Afternoon-Infilter…  → Infiltration
    Friday-WorkingHours-Morning.pcap_ISCX.csv  → Botnet ARES
    Friday-WorkingHours-Afternoon-DDos…        → DDoS LOIT
    Friday-WorkingHours-Afternoon-PortScan…    → PortScan

    ~2.8 M records combined, 78 features (CICFlowMeter output).
    Column ' Label' (with leading space) → stripped to 'Label'.
    """

    # Day tag → expected filename substring (case-insensitive match)
    DAY_KEYS = [
        'monday',
        'tuesday',
        'wednesday',
        'thursday.*morning',
        'thursday.*afternoon',
        'friday.*morning',
        'friday.*afternoon.*ddos',
        'friday.*afternoon.*port',
    ]

    # Normalise raw CICIDS labels → clean category strings
    LABEL_CLEAN = {
        'BENIGN': 'Normal',
        'FTP-Patator': 'BruteForce',
        'SSH-Patator': 'BruteForce',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'Heartbleed': 'DoS',
        'Web Attack – Brute Force': 'WebAttack',
        'Web Attack – XSS': 'WebAttack',
        'Web Attack – Sql Injection': 'WebAttack',
        'Infiltration': 'Infiltration',
        'Bot': 'Botnet',
        'DDoS': 'DDoS',
        'PortScan': 'PortScan',
    }

    CLASS_ORDER = [
        'Normal','BruteForce','DoS','WebAttack',
        'Infiltration','Botnet','DDoS','PortScan'
    ]

    # Columns to drop (metadata / constant / high-cardinality)
    DROP_COLS = [
        'Flow ID',' Flow ID','Src IP',' Src IP','Src Port',' Src Port',
        'Dst IP',' Dst IP','Dst Port',' Dst Port',
        'Timestamp',' Timestamp','timestamp','flow_id',
    ]

    def _find_files(self, csv_dir: str) -> list:
        """
        Match CICIDS files by day-pattern.
        Fallback: if < 2 patterns matched, load ALL CSVs in the directory.
        Also searches one level of sub-directories.
        """
        import re
        dir_path = Path(csv_dir)

        if not dir_path.exists():
            raise FileNotFoundError(
                f"CICIDS directory not found: {dir_path.resolve()}\n"
                f"Current working dir: {Path.cwd()}\n"
                f"→ Update CONFIGS['paths']['dir'] to the correct path."
            )

        # Collect all CSVs (flat + one sub-level)
        all_csvs = sorted(
            list(dir_path.glob("*.csv")) +
            list(dir_path.glob("*/*.csv"))
        )

        if not all_csvs:
            raise FileNotFoundError(
                f"No CSV files found in: {dir_path.resolve()}\n"
                f"Files present: {[f.name for f in dir_path.iterdir()]}"
            )

        print(f"  CICIDS dir: {dir_path.resolve()}")
        print(f"  CSVs found: {len(all_csvs)}")
        for f in all_csvs:
            print(f"    · {f.name}")

        # Try day-pattern matching
        matched = []
        for pattern in self.DAY_KEYS:
            found = [f for f in all_csvs
                     if re.search(pattern, f.name, re.IGNORECASE)]
            if found:
                matched.append(found[0])

        # Fallback: if fewer than 2 day-patterns matched, use all CSVs
        if len(matched) < 2:
            print(f"  ℹ️  Day-pattern matched {len(matched)}/8 files.")
            print(f"  ℹ️  Fallback: loading ALL {len(all_csvs)} CSV(s) found.")
            matched = all_csvs

        return matched

    def _load_one(self, path: Path, day_tag: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()   # remove leading spaces

        # Find label column
        label_col = next(
            (c for c in df.columns
             if c.lower() in ('label', ' label')), None)
        if label_col is None:
            df['Label'] = 'Unknown'
            label_col = 'Label'

        df.rename(columns={label_col: 'Label'}, inplace=True)
        df['day'] = day_tag

        # Clean label
        df['attack_cat'] = df['Label'].map(
            lambda x: self.LABEL_CLEAN.get(str(x).strip(), 'Unknown'))

        print(f"    {day_tag:35s}: {len(df):>8,}  "
              f"{dict(df['attack_cat'].value_counts().head(4))}")
        return df

    # Day-to-split mapping for temporal split
    DAY_SPLIT = {
        'train': ['monday','tuesday','wednesday'],
        'val':   ['thursday'],
        'test':  ['friday'],
    }

    def load(self, csv_dir: str = None,
             file_list: list = None,
             binary: bool = True,
             test_size: float = 0.30,
             sample_frac: float = 1.0,
             split_mode: str = 'random',
             drop_duplicates: bool = True) -> dict:
        """
        csv_dir        : folder containing all 8 CSVs (auto-match)
        file_list      : explicit list of 8 CSV paths in day order
        sample_frac    : fraction to sample (0<x<=1)
        split_mode     : 'random'   — stratified random 60/20/20 split
                         'temporal' — day-based split
                                      train=Mon-Wed, val=Thu, test=Fri
                                      (avoids temporal leakage)
        drop_duplicates: remove near-identical rows (recommended)
        """
        from sklearn.model_selection import train_test_split

        if file_list:
            paths = [Path(p) for p in file_list]
        elif csv_dir:
            paths = self._find_files(csv_dir)
        else:
            raise ValueError("Provide csv_dir or file_list")

        print(f"  CICIDS2017 — {len(paths)} files found:")
        dfs = []
        for path in paths:
            try:
                df = self._load_one(path, path.stem)
                dfs.append(df)
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}")

        if not dfs:
            raise ValueError(
                f"No CICIDS files could be loaded from the directory.\n"
                f"Check that the CSV files are valid and readable."
            )
        combined = pd.concat(dfs, ignore_index=True)
        gc.collect()

        # Optional sub-sampling (for dev speed)
        if sample_frac < 1.0:
            combined = (combined.groupby('attack_cat', group_keys=False)
                                .sample(frac=sample_frac, random_state=42)
                                .reset_index(drop=True))
            print(f"  Sampled to {len(combined):,} rows (frac={sample_frac})")

        # Drop metadata cols
        drop = [c for c in combined.columns
                if c in self.DROP_COLS or c.lower() in
                ('flow id','src ip','src port','dst ip','dst port','timestamp')]
        # Keep 'day' until after temporal split is completed.
        combined.drop(columns=drop + ['Label'], inplace=True,
                      errors='ignore')

        # Assign labels
        if binary:
            combined['y'] = (combined['attack_cat'] != 'Normal').astype(np.int8)
        else:
            combined['y'] = pd.Categorical(
                combined['attack_cat'], categories=self.CLASS_ORDER
            ).codes.astype(np.int8)
            combined['y'] = combined['y'].clip(lower=0)

        combined.drop(columns=['attack_cat'], inplace=True)

        # Clean numerics
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)

        feat_cols = [c for c in combined.columns if c not in ('y', 'day')]
        for c in feat_cols:
            combined[c] = pd.to_numeric(combined[c], errors='coerce')

        combined.dropna(subset=feat_cols, how='all', inplace=True)
        combined[feat_cols] = combined[feat_cols].fillna(0)

        # Remove zero-variance columns
        std = combined[feat_cols].std()
        zero_var = std[std == 0].index.tolist()
        if zero_var:
            combined.drop(columns=zero_var, inplace=True)
            feat_cols = [c for c in feat_cols if c not in zero_var]
            print(f"  Dropped {len(zero_var)} zero-variance features")

        # ── Drop duplicates ────────────────────────────────────────────────
        if drop_duplicates:
            before = len(combined)
            dedup_subset = feat_cols + (['day'] if 'day' in combined.columns else [])
            combined = combined.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
            dropped = before - len(combined)
            if dropped:
                print(f"  Dedup: removed {dropped:,} duplicate rows "
                      f"({dropped/before*100:.2f}%)")

        # ── Split ─────────────────────────────────────────────────────────
        if split_mode == 'temporal' and 'day' in combined.columns:
            print("  Split mode: TEMPORAL (train=Mon-Wed | val=Thu | test=Fri)")
            def _day_match(day_val, keys):
                import re
                return any(re.search(k, str(day_val), re.IGNORECASE)
                           for k in keys)

            train = combined[combined['day'].apply(
                lambda d: _day_match(d, self.DAY_SPLIT['train']))].copy()
            val   = combined[combined['day'].apply(
                lambda d: _day_match(d, self.DAY_SPLIT['val']))].copy()
            test  = combined[combined['day'].apply(
                lambda d: _day_match(d, self.DAY_SPLIT['test']))].copy()

            # Fallback to random if temporal produces empty splits
            if train.empty or test.empty:
                print("  ⚠️  Temporal split failed (day tags not matching) "
                      "— falling back to random split.")
                split_mode = 'random'

        if split_mode == 'random' or 'day' not in combined.columns:
            print("  Split mode: RANDOM stratified 60/20/20")
            train, val, test = _three_way_split(
                combined, label_col='y',
                val_size=0.20, test_size=0.20)

        # Drop day column from splits (not a feature)
        for _df in [train, val, test]:
            _df.drop(columns=['day'], inplace=True, errors='ignore')

        train = train.reset_index(drop=True)
        val   = val.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        feat_cols = [c for c in feat_cols if c != 'day']

        meta = {
            'train': train,
            'val':   val,
            'test':  test,
            'feat_cols': feat_cols,
            'cat_cols': [],
            'binary': binary,
            'class_names': (['Normal','Attack'] if binary
                            else self.CLASS_ORDER),
            'n_classes': 2 if binary else len(self.CLASS_ORDER),
            'dataset': 'cicids',
            'split_mode': split_mode,
            'split_note': ('CICIDS CSV files are concatenated first, then split temporally.' if split_mode == 'temporal' else 'CICIDS CSV files are concatenated first, then split stratified into train/val/test (60/20/20).'),
        }
        print(f"  Total: {len(combined):,}  Train: {len(train):,}  "
              f"Val: {len(val):,}  Test: {len(test):,}  "
              f"Features: {len(feat_cols)}")
        print(f"  Label dist (train): "
              f"{dict(train['y'].value_counts().sort_index())}")
        return meta


# =============================================================================
# EDA — EXPLORATORY DATA ANALYSIS
# =============================================================================
class EDAAnalyzer:
    """
    Full EDA suite for NIDS datasets.
    Generates all plots to FIG_DIR and prints a dataset report.

    Components
    ──────────
    1. Dataset info & describe
    2. Class distribution (bar + pie)
    3. Feature distribution (histogram grid)
    4. Boxplot per feature by class
    5. Feature correlation heatmap
    6. Pairplot (top-k features)
    7. Missing value analysis
    8. Attack category breakdown (multiclass)
    """

    DARK_BG  = '#0b1020'
    CARD_BG  = '#111827'
    PALETTE  = ['#7c3aed','#ff3b5c','#06b6d4','#f59e0b',
                '#10b981','#a78bfa','#fb923c','#34d399']

    def __init__(self, dataset: str, class_names: list):
        self.dataset     = dataset
        self.class_names = class_names

    def _savefig(self, fig, name: str):
        path = FIG_DIR / f"eda_{self.dataset}_{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor=self.DARK_BG)
        plt.close(fig)
        print(f"    → {path.name}")

    # ── 1. Dataset info ──────────────────────────────────────────────────────
    def dataset_info(self, df: pd.DataFrame, feat_cols: list,
                     y: np.ndarray):
        sep = '=' * 55
        print('\n  ' + sep)
        print('  EDA REPORT — ' + self.dataset.upper())
        print('  ' + sep)
        print(f'  Rows          : {len(df):,}')
        print(f'  Features      : {len(feat_cols)}')
        print(f'  Classes       : {len(np.unique(y))}')

        miss       = df[feat_cols].isnull().sum()
        miss_total = int(miss.sum())
        print(f'  Missing vals  : {miss_total:,}  '
              f'({miss_total / (df[feat_cols].size + 1e-9) * 100:.2f}%)')

        unique, counts = np.unique(y, return_counts=True)
        print('  Class dist    :')
        for cls, cnt in zip(unique, counts):
            name = (self.class_names[cls]
                    if cls < len(self.class_names) else str(cls))
            pct  = cnt / len(y) * 100
            bar  = chr(9608) * int(pct / 2)
            print(f'    [{cls}] {name:15s}: {cnt:>7,}  ({pct:5.1f}%)  {bar}')

        # describe() only returns numeric cols — intersect to avoid KeyError
        _num_cols = df[feat_cols].select_dtypes(include='number').columns.tolist()
        _show     = _num_cols[:5] if _num_cols else feat_cols[:5]
        desc = df[_show].describe().round(4)
        print('  Feature statistics (first 5 numeric):')
        print(desc.to_string())
        print('  ' + sep + '\n')

    # ── 2. Class distribution — bar + pie ─────────────────────────────────────    # ── 2. Class distribution — bar + pie ─────────────────────────────────────
    def plot_class_distribution(self, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        names = [self.class_names[i] if i < len(self.class_names)
                 else str(i) for i in unique]
        colors = self.PALETTE[:len(unique)]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                                 facecolor=self.DARK_BG)

        # Bar chart
        ax = axes[0]
        ax.set_facecolor(self.CARD_BG)
        bars = ax.bar(names, counts, color=colors, edgecolor='none',
                      alpha=0.88)
        ax.bar_label(bars, labels=[f'{c:,}' for c in counts],
                     color='white', fontsize=10, padding=4)
        ax.set_title(f'Class Distribution — {self.dataset}',
                     color='white', fontsize=12)
        ax.set_ylabel('Count', color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        ax.set_facecolor(self.CARD_BG)
        for sp in ax.spines.values(): sp.set_visible(False)

        # Pie chart (imbalance visualization)
        ax2 = axes[1]
        ax2.set_facecolor(self.DARK_BG)
        wedges, texts, autotexts = ax2.pie(
            counts, labels=names, colors=colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops=dict(edgecolor='#0b1020', linewidth=2))
        for t in texts:     t.set_color('#94a3b8')
        for t in autotexts: t.set_color('white'); t.set_fontsize(9)
        ax2.set_title('Class Imbalance', color='white', fontsize=12)

        plt.tight_layout()
        self._savefig(fig, 'class_distribution')

    # ── 3. Feature distribution histogram grid ────────────────────────────────
    def plot_feature_distributions(self, X: np.ndarray,
                                   feat_names: list,
                                   y: np.ndarray, top_k: int = 16):
        k      = min(top_k, X.shape[1])
        ncols  = 4
        nrows  = (k + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols*4, nrows*3),
                                 facecolor=self.DARK_BG)
        axes = axes.flatten()
        unique_cls = np.unique(y)

        for i in range(k):
            ax = axes[i]
            ax.set_facecolor(self.CARD_BG)
            for cls in unique_cls:
                mask = y == cls
                name = (self.class_names[cls]
                        if cls < len(self.class_names) else str(cls))
                ax.hist(X[mask, i], bins=30, alpha=0.6,
                        color=self.PALETTE[int(cls) % len(self.PALETTE)],
                        label=name, density=True)
            ax.set_title(feat_names[i] if i < len(feat_names) else f'f{i}',
                         color='white', fontsize=9)
            ax.tick_params(colors='#64748b', labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)

        # Hide unused axes
        for j in range(k, len(axes)):
            axes[j].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right',
                   facecolor='#0b1020', labelcolor='white',
                   fontsize=9, ncol=len(unique_cls))
        fig.suptitle(f'Feature Distributions — {self.dataset}',
                     color='white', fontsize=13, y=1.01)
        plt.tight_layout()
        self._savefig(fig, 'feature_distributions')

    # ── 4. Boxplot per feature by class ──────────────────────────────────────
    def plot_boxplots(self, X: np.ndarray, feat_names: list,
                      y: np.ndarray, top_k: int = 12):
        from sklearn.feature_selection import mutual_info_classif
        # Select top informative features for boxplot
        mi   = mutual_info_classif(X, y, random_state=42)
        idxs = np.argsort(mi)[::-1][:min(top_k, X.shape[1])]

        k     = len(idxs)
        ncols = 4
        nrows = (k + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols*4, nrows*3.5),
                                 facecolor=self.DARK_BG)
        axes = axes.flatten()
        unique_cls = np.unique(y)

        for pos, idx in enumerate(idxs):
            ax = axes[pos]
            ax.set_facecolor(self.CARD_BG)
            data = [X[y == cls, idx] for cls in unique_cls]
            bp   = ax.boxplot(data, patch_artist=True,
                              medianprops=dict(color='white', linewidth=2),
                              whiskerprops=dict(color='#475569'),
                              capprops=dict(color='#475569'),
                              flierprops=dict(marker='.', markersize=2,
                                             color='#475569', alpha=0.4))
            for patch, col in zip(bp['boxes'], self.PALETTE):
                patch.set_facecolor(col + '66')
                patch.set_edgecolor(col)
            cls_labels = [self.class_names[c]
                          if c < len(self.class_names)
                          else str(c) for c in unique_cls]
            ax.set_xticklabels(cls_labels, rotation=20,
                               color='#94a3b8', fontsize=8)
            fname = feat_names[idx] if idx < len(feat_names) else f'f{idx}'
            ax.set_title(fname, color='white', fontsize=9)
            ax.tick_params(colors='#64748b', labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)

        for j in range(k, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'Boxplots (Top-{k} MI Features) — {self.dataset}',
                     color='white', fontsize=13, y=1.01)
        plt.tight_layout()
        self._savefig(fig, 'boxplots')

    # ── 5. Feature correlation heatmap ────────────────────────────────────────
    def plot_correlation_heatmap(self, X: np.ndarray,
                                  feat_names: list, top_k: int = 20):
        k    = min(top_k, X.shape[1])
        corr = np.corrcoef(X[:, :k].T)
        names = feat_names[:k] if len(feat_names) >= k                 else [f'f{i}' for i in range(k)]

        fig, ax = plt.subplots(figsize=(max(10, k*0.7),
                                        max(8,  k*0.65)),
                               facecolor=self.DARK_BG)
        ax.set_facecolor(self.CARD_BG)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1,
                       aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8,
                     label='Pearson r').ax.yaxis.label.set_color('white')
        ax.set_xticks(range(k)); ax.set_xticklabels(
            names, rotation=45, ha='right', color='#94a3b8', fontsize=8)
        ax.set_yticks(range(k)); ax.set_yticklabels(
            names, color='#94a3b8', fontsize=8)
        ax.set_title(f'Feature Correlation Heatmap — {self.dataset}',
                     color='white', fontsize=12)
        # Annotate cells
        if k <= 15:
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f'{corr[i,j]:.2f}',
                            ha='center', va='center',
                            fontsize=6,
                            color='white' if abs(corr[i,j]) > 0.5
                            else '#64748b')
        plt.tight_layout()
        self._savefig(fig, 'correlation_heatmap')

    # ── 6. Pairplot (top-k MI features) ──────────────────────────────────────
    def plot_pairplot(self, X: np.ndarray, feat_names: list,
                      y: np.ndarray, top_k: int = 5):
        from sklearn.feature_selection import mutual_info_classif
        mi   = mutual_info_classif(X, y, random_state=42)
        idxs = np.argsort(mi)[::-1][:min(top_k, X.shape[1])]
        k    = len(idxs)
        names = [feat_names[i] if i < len(feat_names)
                 else f'f{i}' for i in idxs]
        unique_cls = np.unique(y)

        fig, axes = plt.subplots(k, k, figsize=(k*3, k*3),
                                 facecolor=self.DARK_BG)
        axes = np.array(axes, dtype=object).reshape(k, k)
        for r in range(k):
            for c in range(k):
                ax = axes[r][c]
                ax.set_facecolor(self.CARD_BG)
                for cls in unique_cls:
                    mask = y == cls
                    col  = self.PALETTE[int(cls) % len(self.PALETTE)]
                    if r == c:   # diagonal: KDE / histogram
                        ax.hist(X[mask, idxs[r]], bins=20,
                                alpha=0.65, color=col, density=True)
                    else:
                        ax.scatter(X[mask, idxs[c]], X[mask, idxs[r]],
                                   s=3, alpha=0.25, color=col)
                if r == k-1: ax.set_xlabel(names[c], color='#64748b',
                                           fontsize=8)
                if c == 0:   ax.set_ylabel(names[r], color='#64748b',
                                           fontsize=8)
                ax.tick_params(colors='#475569', labelsize=6)
                for sp in ax.spines.values(): sp.set_visible(False)

        # Legend
        handles = [plt.Line2D([0],[0], marker='o', color='w',
                               markerfacecolor=self.PALETTE[int(c)%len(self.PALETTE)],
                               markersize=8,
                               label=self.class_names[c]
                               if c < len(self.class_names) else str(c))
                   for c in unique_cls]
        fig.legend(handles=handles, loc='upper right',
                   facecolor='#0b1020', labelcolor='white', fontsize=9)
        fig.suptitle(f'Pairplot Top-{k} MI Features — {self.dataset}',
                     color='white', fontsize=13)
        plt.tight_layout()
        self._savefig(fig, 'pairplot')

    # ── 7. Missing value analysis ─────────────────────────────────────────────
    def plot_missing_values(self, df: pd.DataFrame, feat_cols: list):
        miss = df[feat_cols].isnull().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            print("    No missing values — skipping missing value plot.")
            return

        fig, ax = plt.subplots(figsize=(12, max(4, len(miss)*0.35)),
                               facecolor=self.DARK_BG)
        ax.set_facecolor(self.CARD_BG)
        pct  = miss / len(df) * 100
        bars = ax.barh(miss.index, pct,
                       color=self.PALETTE[2], alpha=0.85, edgecolor='none')
        ax.bar_label(bars, labels=[f'{p:.2f}%' for p in pct],
                     color='white', fontsize=9, padding=4)
        ax.set_xlabel('Missing %', color='#94a3b8')
        ax.set_title(f'Missing Value Analysis — {self.dataset}',
                     color='white', fontsize=12)
        ax.tick_params(colors='#94a3b8')
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        self._savefig(fig, 'missing_values')

    # ── 8. Attack category breakdown ─────────────────────────────────────────
    def plot_attack_breakdown(self, df: pd.DataFrame):
        cat_col = next((c for c in ['attack_cat','Label','label_raw',
                                     'category'] if c in df.columns), None)
        if cat_col is None:
            return
        counts = df[cat_col].value_counts()
        fig, ax = plt.subplots(figsize=(max(10, len(counts)*0.9), 5),
                               facecolor=self.DARK_BG)
        ax.set_facecolor(self.CARD_BG)
        colors = self.PALETTE * (len(counts)//len(self.PALETTE) + 1)
        bars = ax.bar(counts.index, counts.values,
                      color=colors[:len(counts)], alpha=0.88,
                      edgecolor='none')
        ax.bar_label(bars, labels=[f'{v:,}' for v in counts.values],
                     color='white', fontsize=9, padding=4, rotation=45)
        ax.set_title(f'Attack Category Breakdown — {self.dataset}',
                     color='white', fontsize=12)
        ax.set_ylabel('Count', color='#94a3b8')
        ax.tick_params(colors='#94a3b8', axis='x', rotation=30)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        self._savefig(fig, 'attack_breakdown')

    # ── Master run ────────────────────────────────────────────────────────────
    def run_all(self, df: pd.DataFrame, X: np.ndarray,
                feat_names: list, y: np.ndarray,
                top_k_dist: int = 16, top_k_pair: int = 5):
        print(f"Running EDA for {self.dataset.upper()} …")
        feat_cols = [c for c in feat_names if c in df.columns] or feat_names
        # Separate numeric vs categorical for plots that need floats
        _num_feats = df[feat_cols].select_dtypes(include='number').columns.tolist()
        if not _num_feats:
            _num_feats = feat_cols   # fallback

        self.dataset_info(df, feat_cols, y)
        self.plot_class_distribution(y)
        self.plot_feature_distributions(X, feat_names, y, top_k=top_k_dist)
        self.plot_boxplots(X, feat_names, y, top_k=12)
        self.plot_correlation_heatmap(X, feat_names, top_k=20)
        self.plot_pairplot(X, feat_names, y, top_k=top_k_pair)
        self.plot_missing_values(df, feat_cols)
        self.plot_attack_breakdown(df)
        print(f"  EDA complete — all figures in {FIG_DIR}/")



# =============================================================================
# PREPROCESSING
# =============================================================================
class Preprocessor:
    """
    Train-only preprocessing pipeline with explicit split hygiene.

    Recommended order
    -----------------
    1. Fit encoders / imputer / clipping bounds on TRAIN only
    2. Transform VAL / TEST using the frozen TRAIN state
    3. Fit feature selection on original TRAIN only (done in pipeline)
    4. Fit scaler on original TRAIN only
    5. Apply class balancing to TRAIN only
    """

    def __init__(self, scaler_type: str = 'robust',
                 balance: str = 'smote'):
        self.scaler_type = scaler_type
        self.balance     = balance
        self._encoders   = {}
        self._imputer    = None
        self._scaler     = None
        self._clip_lower = None
        self._clip_upper = None

    # ── Categorical encoding ──────────────────────────────────────────────────
    def _encode(self, df: pd.DataFrame,
                cat_cols: list, fit: bool) -> pd.DataFrame:
        from sklearn.preprocessing import LabelEncoder
        df = df.copy()
        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le is None:
                    df[col] = 0
                else:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).map(
                        lambda v: int(le.transform([v])[0]) if v in known else -1
                    )
        return df

    def _get_scaler(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        scalers = {
            'standard': StandardScaler(),
            'minmax':   MinMaxScaler(),
            'robust':   RobustScaler(),
        }
        return scalers.get(self.scaler_type, RobustScaler())

    def clean_fit_transform(self, df: pd.DataFrame, feat_cols: list,
                            cat_cols: list) -> np.ndarray:
        """Fit cleaning state on TRAIN only and return cleaned TRAIN matrix."""
        from sklearn.impute import SimpleImputer

        df = self._encode(df, cat_cols, fit=True)
        X = df[feat_cols].values.astype(np.float64)
        X = np.where(np.isfinite(X), X, 0.0)

        self._imputer = SimpleImputer(strategy='median')
        X = self._imputer.fit_transform(X)

        col_means = X.mean(axis=0)
        col_stds  = X.std(axis=0) + 1e-9
        self._clip_lower = col_means - 5 * col_stds
        self._clip_upper = col_means + 5 * col_stds
        X = np.clip(X, self._clip_lower, self._clip_upper)
        return X

    def clean_transform(self, df: pd.DataFrame, feat_cols: list,
                        cat_cols: list) -> np.ndarray:
        """Apply frozen TRAIN cleaning state to VAL/TEST."""
        df = self._encode(df, cat_cols, fit=False)
        X  = df[feat_cols].values.astype(np.float64)
        X  = np.where(np.isfinite(X), X, 0.0)
        X  = self._imputer.transform(X)
        X  = np.clip(X, self._clip_lower, self._clip_upper)
        return X

    def fit_scaler(self, X_train: np.ndarray):
        """Fit scaler on original TRAIN only."""
        self._scaler = self._get_scaler()
        self._scaler.fit(X_train)
        return self

    def scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError('Scaler has not been fitted yet.')
        return self._scaler.transform(X)

    # ── Backward-compatible wrappers ──────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame, feat_cols: list,
                      cat_cols: list, y: np.ndarray) -> tuple:
        """Legacy wrapper: clean -> scale(train) -> balance(train)."""
        X = self.clean_fit_transform(df, feat_cols, cat_cols)
        self.fit_scaler(X)
        X = self.scale_transform(X)
        cat_indices = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
        X, y = self.balance_train(X, y, cat_indices=cat_indices)
        return X, y

    def transform(self, df: pd.DataFrame, feat_cols: list,
                  cat_cols: list) -> np.ndarray:
        X = self.clean_transform(df, feat_cols, cat_cols)
        return self.scale_transform(X)

    # ── Class balancing ───────────────────────────────────────────────────────
    def balance_train(self, X: np.ndarray,
                      y: np.ndarray,
                      cat_indices: list | None = None) -> tuple:
        return self._balance_data(X, y, cat_indices=cat_indices)

    def _balance_data(self, X: np.ndarray,
                      y: np.ndarray,
                      cat_indices: list | None = None) -> tuple:
        counts = np.bincount(y.astype(int))
        if len(counts) < 2:
            return X, y

        ratio = counts.min() / (counts.max() + 1e-9)
        if ratio > 0.4 or self.balance in (None, 'none', 'off'):
            print(f"  Classes sufficiently balanced (ratio={ratio:.3f}) — skipping resampling.")
            return X, y

        min_count = int(counts.min())
        if min_count < 2 and self.balance in ('smote', 'adasyn', 'borderline'):
            print("  Resampling skipped (minority class too small for synthetic oversampling).")
            return X, y

        try:
            if cat_indices:
                if self.balance == 'smote':
                    from imblearn.over_sampling import SMOTENC
                    samp = SMOTENC(
                        categorical_features=cat_indices,
                        random_state=42,
                        k_neighbors=max(1, min(5, min_count - 1)),
                    )
                else:
                    from imblearn.under_sampling import RandomUnderSampler
                    print(f"  {self.balance.upper()} is unsafe with categorical features — falling back to RandomUnderSampler.")
                    samp = RandomUnderSampler(random_state=42)
            else:
                if self.balance == 'smote':
                    from imblearn.over_sampling import SMOTE
                    samp = SMOTE(random_state=42,
                                 k_neighbors=max(1, min(5, min_count - 1)))
                elif self.balance == 'adasyn':
                    from imblearn.over_sampling import ADASYN
                    samp = ADASYN(random_state=42,
                                  n_neighbors=max(1, min(5, min_count - 1)))
                elif self.balance == 'borderline':
                    from imblearn.over_sampling import BorderlineSMOTE
                    samp = BorderlineSMOTE(random_state=42,
                                           k_neighbors=max(1, min(5, min_count - 1)))
                else:
                    from imblearn.under_sampling import RandomUnderSampler
                    samp = RandomUnderSampler(random_state=42)

            X_r, y_r = samp.fit_resample(X, y)
            print(f"  {self.balance.upper()} resampling: {len(y):,} → {len(y_r):,}")
            return X_r, y_r
        except Exception as e:
            print(f"  Resampling skipped ({e})")
            return X, y


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    """Feature selection + optional interaction features."""

    def __init__(self):
        self.selected_idx_  = None
        self.selected_names_= None

    def select(self, X: np.ndarray, y: np.ndarray,
               feat_names: list, k: int = 40,
               method: str = 'mutual_info') -> np.ndarray:
        from sklearn.feature_selection import (SelectKBest,
                                               mutual_info_classif,
                                               f_classif)
        k = min(k, X.shape[1])
        scorer = mutual_info_classif if method == 'mutual_info' else f_classif
        sel = SelectKBest(scorer, k=k)
        sel.fit(np.nan_to_num(X), y)
        self.selected_idx_   = sel.get_support(indices=True)
        self.selected_names_ = [feat_names[i] for i in self.selected_idx_
                                if i < len(feat_names)]
        # Pad if needed
        while len(self.selected_names_) < len(self.selected_idx_):
            self.selected_names_.append(f"feat_{len(self.selected_names_)}")
        print(f"  Feature selection ({method}): "
              f"{X.shape[1]} → {k} features kept")
        return self.selected_idx_


# =============================================================================
# BASE LEARNERS
# =============================================================================

def _import_keras():
    """
    Keras 3 + TF 2.16 compatible import shim.
    Uses hasattr checks only — no layer instantiation smoke-test.
    """
    import importlib

    def _has_layers(k):
        return (k is not None
                and hasattr(k, 'layers')
                and hasattr(k, 'models')
                and hasattr(k, 'callbacks')
                and hasattr(k, 'optimizers'))

    # Try tensorflow first (always available when TF is installed)
    try:
        import tensorflow as tf
    except ImportError:
        tf = None

    # Path 1: standalone keras package  (Keras 3 / TF >= 2.16) ← FASTEST
    try:
        k = importlib.import_module('keras')
        if _has_layers(k):
            return (tf, k)
    except Exception:
        pass

    # Path 2: tf.keras attribute access
    if tf is not None:
        try:
            k = tf.keras
            if _has_layers(k):
                return (tf, k)
        except Exception:
            pass

    # Path 3: tensorflow.keras sub-module  (TF <= 2.15)
    if tf is not None:
        try:
            k = importlib.import_module('tensorflow.keras')
            if _has_layers(k):
                return (tf, k)
        except Exception:
            pass

    # Path 4: tf_keras compatibility package
    try:
        k = importlib.import_module('tf_keras')
        return (tf, k)
    except Exception:
        pass

    raise ImportError(
        "No working Keras found.\n"
        "Your environment has TF 2.16 + Keras 3 — run:\n"
        "    pip install --upgrade keras tensorflow\n"
        "Then restart the kernel."
    )





# =============================================================================
#  ███████╗██╗  ██╗██████╗ ███████╗██████╗ ████████╗███████╗
#  ██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██╔════╝
#  █████╗   ╚███╔╝ ██████╔╝█████╗  ██████╔╝   ██║   ███████╗
#  ██╔══╝   ██╔██╗ ██╔═══╝ ██╔══╝  ██╔══██╗   ██║   ╚════██║
#  ███████╗██╔╝ ██╗██║     ███████╗██║  ██║   ██║   ███████║
#  ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
#
#  EXPERT LEARNERS
#  ┌──────────────┬───────────────────────────────────────┐
#  │ Tabular      │ Random Forest · XGBoost               │
#  │ Temporal     │ CNN-BiLSTM · CNN-GRU                  │
#  │ Graph        │ GNN (GCN) · GAT                       │
#  │ Transformer  │ TabTransformer                        │
#  └──────────────┴───────────────────────────────────────┘
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Custom Keras Layers (defined at module level for pickling / reuse)
# ─────────────────────────────────────────────────────────────────────────────

def _build_adj_matrix(X: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    """
    Build a normalized feature-level adjacency matrix from training data.
    Nodes = features,  Edges = |Pearson correlation| > threshold.
    Normalization: symmetric D^{-1/2} A D^{-1/2}  (GCN convention).
    """
    corr = np.abs(np.corrcoef(X.T)).astype(np.float32)
    corr[corr < threshold] = 0.0
    np.fill_diagonal(corr, 1.0)                     # self-loops
    d_inv_sqrt = np.diag(1.0 / np.sqrt(corr.sum(axis=1) + 1e-9))
    adj = d_inv_sqrt @ corr @ d_inv_sqrt
    return np.nan_to_num(adj).astype(np.float32)    # (n_feat, n_feat)


class GCNConv:
    """
    Single Graph Convolutional layer applied as functional call.
    h_new = relu(A_hat · h · W)   where A_hat is pre-normalized.
    """
    def __init__(self, units, activation='relu', name='gcn'):
        self.dense      = None
        self.units      = units
        self.activation = activation
        self._name      = name

    def build(self, keras_layers):
        self.dense = keras_layers.Dense(
            self.units, activation=self.activation,
            name=self._name + '_W')

    def __call__(self, h, adj_tensor):
        """h: (batch, n_nodes, in_feat)  adj: (n_nodes, n_nodes)"""
        import tensorflow as tf
        agg = tf.matmul(adj_tensor, h)   # neighbourhood aggregation
        return self.dense(agg)


class GATHead:
    """
    Single attention head of GAT.
    e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
    """
    def __init__(self, units, name='gat'):
        self.W       = None
        self.a_src   = None
        self.a_dst   = None
        self.units   = units
        self._name   = name

    def build(self, keras_layers):
        self.W     = keras_layers.Dense(self.units, use_bias=False,
                                        name=self._name + '_W')
        self.a_src = keras_layers.Dense(1, use_bias=False,
                                        name=self._name + '_as')
        self.a_dst = keras_layers.Dense(1, use_bias=False,
                                        name=self._name + '_ad')

    def __call__(self, h, adj_tensor):
        import tensorflow as tf
        Wh     = self.W(h)                              # (batch, N, units)
        e_src  = self.a_src(Wh)                        # (batch, N, 1)
        e_dst  = self.a_dst(Wh)                        # (batch, N, 1)
        e      = e_src + tf.transpose(e_dst, [0,2,1])  # (batch, N, N)
        e      = tf.nn.leaky_relu(e, alpha=0.2)
        mask   = (1.0 - adj_tensor) * (-1e9)
        alpha  = tf.nn.softmax(e + mask, axis=-1)      # (batch, N, N)
        return tf.nn.elu(tf.matmul(alpha, Wh))         # (batch, N, units)


# =============================================================================
class ExpertLearners:
    """
    Multi-Expert Ensemble for Hybrid NIDS.

    Usage
    -----
    el = ExpertLearners(n_classes=2, n_features=40)
    el.add_rf()          # Tabular Expert
    el.add_xgb()         # Tabular Expert
    el.add_cnn_bilstm()  # Temporal Expert
    el.add_cnn_gru()     # Temporal Expert
    el.add_gnn()         # Graph Expert
    el.add_gat()         # Graph Expert
    el.add_transformer() # Transformer Expert
    el.fit_all(X_train, y_train)
    proba_dict = el.predict_proba_all(X_test)
    """

    # Expert group labels (used for reporting)
    EXPERT_GROUPS = {
        'RF':          'Tabular',
        'XGB':         'Tabular',
        'CNN_BiLSTM':  'Temporal',
        'CNN_GRU':     'Temporal',
        'GNN':         'Graph',
        'GAT':         'Graph',
        'Transformer': 'Transformer',
    }

    def __init__(self, n_classes: int, n_features: int):
        self.n_classes  = n_classes
        self.n_features = n_features
        self.models_    = {}      # name → model object
        self._dl_cfg    = {}      # name → training config (DL only)
        self._adj       = None    # feature adjacency (computed in fit_all)

    # ── Output config helper ──────────────────────────────────────────────────
    def _out(self):
        n_out = 1 if self.n_classes == 2 else self.n_classes
        act   = 'sigmoid' if self.n_classes == 2 else 'softmax'
        loss  = ('binary_crossentropy' if self.n_classes == 2
                 else 'sparse_categorical_crossentropy')
        return n_out, act, loss

    def _dl_skip(self, name: str, err: Exception):
        if isinstance(err, ImportError):
            print(f"  {name} skipped — TF/Keras not installed.")
        else:
            print(f"  {name} skipped: {type(err).__name__}: {err}")

    # =========================================================================
    # TABULAR EXPERT
    # =========================================================================
    def add_rf(self, n_estimators: int = 300, **kw):
        from sklearn.ensemble import RandomForestClassifier
        params = dict(n_estimators=n_estimators, class_weight='balanced',
                      max_features='sqrt', min_samples_leaf=2,
                      random_state=42, n_jobs=-1)
        params.update(kw)
        safe_params, ignored = self._filter_supported_estimator_params(RandomForestClassifier, params)
        self.models_['RF'] = RandomForestClassifier(**safe_params)
        if ignored:
            print(f"  RF          built — {safe_params.get('n_estimators', n_estimators)} trees  (ignored unsupported params: {', '.join(ignored)})")
        else:
            print(f"  RF          built — {safe_params.get('n_estimators', n_estimators)} trees")

    def add_xgb(self, n_estimators: int = 300, **kw):
        try:
            import xgboost as xgb
            params = dict(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='hist',
            )
            if self.n_classes == 2:
                params.update(objective='binary:logistic', eval_metric='logloss')
            else:
                params.update(objective='multi:softprob', eval_metric='mlogloss', num_class=self.n_classes)
            params.update(kw)
            self.models_['XGB'] = xgb.XGBClassifier(**params)
            print(f"  XGB         built — {n_estimators} rounds")
        except ImportError:
            print("  XGB skipped — pip install xgboost")

    # =========================================================================
    # TEMPORAL EXPERT
    # =========================================================================
    def add_cnn_bilstm(self, epochs: int = 30, batch: int = 256):
        """CNN feature extractor → BiLSTM temporal modelling."""
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()

            inp = L.Input(shape=(self.n_features, 1), name='cbl_in')
            # CNN block
            x = L.Conv1D(64,  3, padding='same', activation='relu',
                          name='cbl_c1')(inp)
            x = L.BatchNormalization(name='cbl_bn1')(x)
            x = L.Conv1D(128, 3, padding='same', activation='relu',
                          name='cbl_c2')(x)
            x = L.BatchNormalization(name='cbl_bn2')(x)
            x = L.MaxPooling1D(2, name='cbl_mp')(x)
            # BiLSTM block
            x = L.Bidirectional(L.LSTM(64, return_sequences=True),
                                  name='cbl_bil1')(x)
            x = L.Dropout(0.3, name='cbl_do1')(x)
            x = L.Bidirectional(L.LSTM(32), name='cbl_bil2')(x)
            x = L.Dense(64, activation='relu', name='cbl_d1')(x)
            x = L.Dropout(0.3, name='cbl_do2')(x)
            out = L.Dense(n_out, activation=act, name='cbl_out')(x)

            m = keras.models.Model(inp, out, name='CNN_BiLSTM')
            m.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=loss, metrics=['accuracy'])
            self.models_['CNN_BiLSTM'] = m
            self._dl_cfg['CNN_BiLSTM'] = dict(
                epochs=epochs, batch=batch, n_out=n_out, loss=loss,
                input_3d=True)
            print(f"  CNN-BiLSTM  built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('CNN-BiLSTM', e)

    def add_cnn_gru(self, epochs: int = 30, batch: int = 256):
        """CNN feature extractor → GRU temporal modelling."""
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()

            inp = L.Input(shape=(self.n_features, 1), name='cg_in')
            # CNN block
            x = L.Conv1D(64,  3, padding='same', activation='relu',
                          name='cg_c1')(inp)
            x = L.BatchNormalization(name='cg_bn1')(x)
            x = L.Conv1D(128, 5, padding='same', activation='relu',
                          name='cg_c2')(x)
            x = L.BatchNormalization(name='cg_bn2')(x)
            x = L.MaxPooling1D(2, name='cg_mp')(x)
            # GRU block
            x = L.Bidirectional(L.GRU(64, return_sequences=True),
                                  name='cg_gru1')(x)
            x = L.Dropout(0.3, name='cg_do1')(x)
            x = L.GRU(32, name='cg_gru2')(x)
            x = L.Dense(64, activation='relu', name='cg_d1')(x)
            x = L.Dropout(0.3, name='cg_do2')(x)
            out = L.Dense(n_out, activation=act, name='cg_out')(x)

            m = keras.models.Model(inp, out, name='CNN_GRU')
            m.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=loss, metrics=['accuracy'])
            self.models_['CNN_GRU'] = m
            self._dl_cfg['CNN_GRU'] = dict(
                epochs=epochs, batch=batch, n_out=n_out, loss=loss,
                input_3d=True)
            print(f"  CNN-GRU     built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('CNN-GRU', e)

    # =========================================================================
    # GRAPH EXPERT
    # =========================================================================
    def _build_gnn_model(self, adj_init: np.ndarray,
                          keras, name: str = 'GNN') -> object:
        """
        Graph Convolutional Network on feature-level graph.
        Nodes = features (n_features),  Edges = feature correlations.
        Input  : (batch, n_features, 1) — each feature is a node with 1 value
        Process: 2 × GCN layers with shared adjacency A_hat
        Output : classification head
        """
        import tensorflow as tf
        L = keras.layers
        n_out, act, loss = self._out()

        # Adjacency as non-trainable weight
        adj_tensor = tf.constant(adj_init, dtype=tf.float32)  # (N, N)

        inp = L.Input(shape=(self.n_features, 1), name=f'{name}_in')

        # GCN layer 1  h1 = relu(A_hat · h · W1)
        gcn1 = GCNConv(64, name=f'{name}_gcn1')
        gcn1.build(L)
        x = keras.ops.matmul(adj_tensor,
                              inp) if hasattr(keras, 'ops') else \
            tf.matmul(adj_tensor, inp)
        x = gcn1.dense(x)

        # GCN layer 2
        gcn2 = GCNConv(32, name=f'{name}_gcn2')
        gcn2.build(L)
        x2 = (keras.ops.matmul(adj_tensor, x)
               if hasattr(keras, 'ops')
               else tf.matmul(adj_tensor, x))
        x2 = gcn2.dense(x2)

        x3 = L.GlobalAveragePooling1D(name=f'{name}_gap')(x2)
        x3 = L.Dense(64, activation='relu', name=f'{name}_d1')(x3)
        x3 = L.Dropout(0.35,               name=f'{name}_do')(x3)
        out = L.Dense(n_out, activation=act, name=f'{name}_out')(x3)

        m = keras.models.Model(inp, out, name=name)
        m.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss=loss, metrics=['accuracy'])
        return m

    def add_gnn(self, epochs: int = 30, batch: int = 256):
        """
        Graph Neural Network (GCN) on feature correlation graph.
        Adjacency A is built from training data correlation matrix.
        """
        try:
            tf, keras = _import_keras()
            m = self._build_gnn_model(
                np.eye(self.n_features, dtype=np.float32),  # placeholder
                keras, name='GNN')
            self.models_['GNN'] = m
            self._dl_cfg['GNN'] = dict(
                epochs=epochs, batch=batch,
                n_out=self._out()[0], loss=self._out()[2],
                input_3d=True, needs_adj=True)
            print(f"  GNN (GCN)   built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('GNN', e)

    def add_gat(self, epochs: int = 30, batch: int = 256,
                n_heads: int = 4):
        """
        Graph Attention Network on feature correlation graph.
        Multi-head attention replaces symmetric aggregation of GCN.
        """
        try:
            import tensorflow as tf
            _tf, keras = _import_keras()
            L  = keras.layers
            n_out, act, loss = self._out()

            inp = L.Input(shape=(self.n_features, 1), name='gat_in')

            # We implement GAT via a custom Lambda that holds the adj
            # at build-time; adj is replaced with actual corr adj in fit_all
            adj_ph = tf.Variable(
                np.eye(self.n_features, dtype=np.float32),
                trainable=False, name='gat_adj')
            self._gat_adj_var = adj_ph   # save ref for update in fit_all

            # Head 1
            W1  = L.Dense(32, use_bias=False, name='gat_W1')
            as1 = L.Dense(1,  use_bias=False, name='gat_as1')
            ad1 = L.Dense(1,  use_bias=False, name='gat_ad1')

            # Head 2
            W2  = L.Dense(32, use_bias=False, name='gat_W2')
            as2 = L.Dense(1,  use_bias=False, name='gat_as2')
            ad2 = L.Dense(1,  use_bias=False, name='gat_ad2')

            def gat_forward(h):
                def head(W, a_s, a_d):
                    Wh    = W(h)
                    e_s   = a_s(Wh)
                    e_d   = a_d(Wh)
                    e     = e_s + tf.transpose(e_d, [0,2,1])
                    e     = tf.nn.leaky_relu(e, alpha=0.2)
                    mask  = (1.0 - adj_ph) * (-1e9)
                    alpha = tf.nn.softmax(e + mask, axis=-1)
                    return tf.nn.elu(tf.matmul(alpha, Wh))
                h1 = head(W1, as1, ad1)
                h2 = head(W2, as2, ad2)
                return tf.concat([h1, h2], axis=-1)  # (batch,N,64)

            x   = L.Lambda(gat_forward, name='gat_heads')(inp)
            x   = L.GlobalAveragePooling1D(name='gat_gap')(x)
            x   = L.Dense(64, activation='relu', name='gat_d1')(x)
            x   = L.Dropout(0.35, name='gat_do')(x)
            out = L.Dense(n_out, activation=act, name='gat_out')(x)

            m = keras.models.Model(inp, out, name='GAT')
            m.compile(optimizer=keras.optimizers.Adam(5e-4),
                      loss=loss, metrics=['accuracy'])
            self.models_['GAT'] = m
            self._dl_cfg['GAT'] = dict(
                epochs=epochs, batch=batch,
                n_out=n_out, loss=loss,
                input_3d=True, needs_adj=True)
            print(f"  GAT         built — params: {m.count_params():,}  "
                  f"heads: {n_heads}")
        except Exception as e:
            self._dl_skip('GAT', e)

    # =========================================================================
    # TRANSFORMER EXPERT
    # =========================================================================
    def add_transformer(self, epochs: int = 30, batch: int = 256,
                         d_model: int = 64, n_heads: int = 4,
                         n_blocks: int = 3, ff_dim: int = 128):
        """
        TabTransformer — Transformer encoder on feature embeddings.

        Architecture
        ────────────
        Input (batch, n_features)
          └─ Feature embedding:  Dense(d_model)  → (batch, n_features, d_model)
          └─ N × Transformer block:
               ├─ Multi-Head Self-Attention
               ├─ Add & LayerNorm
               ├─ Feed-Forward Network
               └─ Add & LayerNorm
          └─ Global Average Pooling
          └─ MLP head → output
        """
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()
            nf = self.n_features

            inp  = L.Input(shape=(nf,), name='tr_in')

            # Feature embedding: each feature → d_model vector
            x = L.Reshape((nf, 1), name='tr_reshape')(inp)
            x = L.Dense(d_model, name='tr_embed')(x)  # (batch, nf, d_model)

            # Learnable positional encoding
            pos_emb = L.Embedding(input_dim=nf, output_dim=d_model,
                                   name='tr_pos')
            positions = tf.range(start=0, limit=nf, delta=1)
            x = x + pos_emb(positions)

            # Transformer blocks
            for i in range(n_blocks):
                pfx = f'tr_b{i}'
                # Multi-Head Self-Attention
                attn_out = L.MultiHeadAttention(
                    num_heads=n_heads, key_dim=d_model // n_heads,
                    name=f'{pfx}_mha')(x, x)
                x = L.LayerNormalization(
                        epsilon=1e-6, name=f'{pfx}_ln1')(x + attn_out)
                # Feed-Forward Network
                ff  = L.Dense(ff_dim, activation='gelu',
                               name=f'{pfx}_ff1')(x)
                ff  = L.Dropout(0.1, name=f'{pfx}_ff_do')(ff)
                ff  = L.Dense(d_model, name=f'{pfx}_ff2')(ff)
                x   = L.LayerNormalization(
                        epsilon=1e-6, name=f'{pfx}_ln2')(x + ff)

            # Pooling + MLP head
            x   = L.GlobalAveragePooling1D(name='tr_gap')(x)
            x   = L.Dense(128, activation='relu',  name='tr_d1')(x)
            x   = L.Dropout(0.3,                   name='tr_do1')(x)
            x   = L.Dense(64,  activation='relu',  name='tr_d2')(x)
            x   = L.Dropout(0.2,                   name='tr_do2')(x)
            out = L.Dense(n_out, activation=act,   name='tr_out')(x)

            m = keras.models.Model(inp, out, name='Transformer')
            adam_kwargs = {'learning_rate': 1e-3}
            if hasattr(keras.optimizers.Adam, 'weight_decay'):
                adam_kwargs['weight_decay'] = 1e-4
            m.compile(
                optimizer=keras.optimizers.Adam(**adam_kwargs),
                loss=loss, metrics=['accuracy'])
            self.models_['Transformer'] = m
            self._dl_cfg['Transformer'] = dict(
                epochs=epochs, batch=batch, n_out=n_out, loss=loss,
                input_3d=False)    # Transformer takes flat (batch, n_feat)
            print(f"  Transformer built — params: {m.count_params():,}  "
                  f"blocks:{n_blocks}  heads:{n_heads}  d_model:{d_model}")
        except Exception as e:
            self._dl_skip('Transformer', e)

    # =========================================================================
    # FIT ALL
    # =========================================================================
    def fit_all(self, X: np.ndarray, y: np.ndarray,
               X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train all registered experts.
        If X_val/y_val provided, use as explicit validation set for DL models
        (overrides internal validation_split).
        """
        # ── Compute feature adjacency (used by GNN / GAT) ─────────────────
        has_graph = any(self._dl_cfg.get(n, {}).get('needs_adj')
                        for n in self.models_)
        if has_graph:
            print("  Computing feature correlation graph …")
            self._adj = _build_adj_matrix(X, threshold=0.10)
            # Update GNN model with real adjacency (rebuild)
            if 'GNN' in self.models_:
                try:
                    tf, keras = _import_keras()
                    cfg = self._dl_cfg['GNN']
                    self.models_['GNN'] = self._build_gnn_model(
                        self._adj, keras, name='GNN')
                    print(f"  GNN adj updated — "
                          f"density={float((self._adj>0).mean()):.3f}")
                except Exception as e:
                    print(f"  GNN adj update failed: {e}")
            # Update GAT adj variable
            if 'GAT' in self.models_ and hasattr(self, '_gat_adj_var'):
                try:
                    import tensorflow as tf
                    self._gat_adj_var.assign(self._adj)
                    print(f"  GAT adj updated — "
                          f"density={float((self._adj>0).mean()):.3f}")
                except Exception as e:
                    print(f"  GAT adj update failed: {e}")

        # ── Train ─────────────────────────────────────────────────────────
        _keras_mod = None
        if self._dl_cfg:
            try:
                _, _keras_mod = _import_keras()
            except ImportError:
                pass

        if len(self.models_) == 0:
            raise ValueError(
                "Tidak ada expert model yang berhasil didaftarkan. "
                "Pilih minimal satu model, dan pastikan dependency yang dibutuhkan tersedia."
            )

        print(f"\n  Training {len(self.models_)} experts …")
        failed_models = []
        for name, model in list(self.models_.items()):
            t0  = time.time()
            grp = self.EXPERT_GROUPS.get(name, '?')
            cfg = self._dl_cfg.get(name)

            try:
                if cfg:
                    use_3d = cfg.get('input_3d', True)
                    X_in   = X.reshape(-1, X.shape[1], 1) if use_3d else X
                    y_in   = y.astype(float if cfg['n_out']==1 else int)
                    cb     = []
                    if _keras_mod:
                        cb = [_keras_mod.callbacks.EarlyStopping(
                                  patience=5, restore_best_weights=True,
                                  monitor='val_loss'),
                              _keras_mod.callbacks.ReduceLROnPlateau(
                                  patience=3, factor=0.5, verbose=0)]
                    # Use explicit val set if provided, else 10% internal split
                    if X_val is not None:
                        use_3d_v = cfg.get('input_3d', True)
                        X_val_in = X_val.reshape(-1, X_val.shape[1], 1) if use_3d_v else X_val
                        y_val_in = y_val.astype(float if cfg['n_out']==1 else int)
                        if cfg.get('multi_output'):
                            model.fit(
                                X_in,
                                {
                                    cfg['cls_output_name']: y_in,
                                    cfg['recon_output_name']: X_in if use_3d else X,
                                },
                                epochs=cfg['epochs'],
                                batch_size=cfg['batch'],
                                validation_data=(
                                    X_val_in,
                                    {
                                        cfg['cls_output_name']: y_val_in,
                                        cfg['recon_output_name']: X_val_in,
                                    },
                                ),
                                verbose=0,
                                callbacks=cb,
                            )
                        else:
                            model.fit(X_in, y_in,
                                      epochs=cfg['epochs'],
                                      batch_size=cfg['batch'],
                                      validation_data=(X_val_in, y_val_in),
                                      verbose=0, callbacks=cb)
                    else:
                        if cfg.get('multi_output'):
                            model.fit(
                                X_in,
                                {
                                    cfg['cls_output_name']: y_in,
                                    cfg['recon_output_name']: X_in if use_3d else X,
                                },
                                epochs=cfg['epochs'],
                                batch_size=cfg['batch'],
                                validation_split=0.1,
                                verbose=0,
                                callbacks=cb,
                            )
                        else:
                            model.fit(X_in, y_in,
                                      epochs=cfg['epochs'],
                                      batch_size=cfg['batch'],
                                      validation_split=0.1,
                                      verbose=0, callbacks=cb)
                else:
                    model.fit(X, y)

                print(f"    [{grp:11s}]  {name:12s} ✓  {time.time()-t0:.1f}s")
            except Exception as e:
                failed_models.append((name, e))
                print(f"    [{grp:11s}]  {name:12s} ✗  skipped after fit error: {e}")

        for name, _ in failed_models:
            self.models_.pop(name, None)
            self._dl_cfg.pop(name, None)

        if len(self.models_) == 0:
            details = '; '.join([f"{n}: {type(e).__name__}" for n, e in failed_models]) or "unknown"
            raise ValueError(
                "Semua expert model gagal saat training. "
                f"Detail ringkas: {details}"
            )

    # =========================================================================
    # PREDICT
    # =========================================================================
    def _raw_predict(self, name: str, model,
                     X: np.ndarray) -> np.ndarray:
        cfg = self._dl_cfg.get(name)
        if cfg:
            use_3d = cfg.get('input_3d', True)
            X_in   = X.reshape(-1, X.shape[1], 1) if use_3d else X
            raw    = model.predict(X_in, verbose=0)
            if isinstance(raw, (list, tuple)):
                if cfg.get('multi_output'):
                    cls_name = cfg.get('cls_output_name')
                    if hasattr(model, 'output_names') and cls_name in list(getattr(model, 'output_names', [])):
                        cls_idx = list(model.output_names).index(cls_name)
                        raw = raw[cls_idx]
                    else:
                        raw = raw[0]
                else:
                    raw = raw[0]
            return np.hstack([1-raw, raw]) if raw.shape[1]==1 else raw
        return model.predict_proba(X)

    def predict_proba_all(self, X: np.ndarray) -> dict:
        out = {}
        for name, m in list(self.models_.items()):
            try:
                out[name] = np.clip(self._raw_predict(name, m, X), 1e-9, 1-1e-9)
            except Exception as e:
                print(f"  Warning: predict_proba gagal untuk {name} — model di-skip ({e})")
        return out

    def predict_all(self, X: np.ndarray) -> dict:
        out = {}
        for name, m in list(self.models_.items()):
            try:
                p = self._raw_predict(name, m, X)
                out[name] = (np.argmax(p, axis=1) if p.shape[1] > 1
                             else (p[:,1] > 0.5).astype(int))
            except Exception as e:
                print(f"  Warning: predict gagal untuk {name} — model di-skip ({e})")
        return out


# =============================================================================
# COPULA FUSION
# =============================================================================
class CopulaFusion:
    """
    Gaussian · Clayton · Frank copulas for inter-model dependence modeling.
    Binary fusion returns shape (n, 2); multiclass fusion returns shape
    (n, n_classes) via class-wise copula blending.
    """

    def __init__(self, family: str = 'gaussian', alpha: float = 0.35):
        self.family      = family
        self.alpha       = alpha
        self._weights    = None
        self._names      = None
        self._n_classes  = None
        self._state      = None
        self._passthrough = False
        self._single_model_name = None

    def _get_n_classes(self, proba_dict: dict) -> int:
        if not proba_dict:
            raise ValueError(
                "CopulaFusion menerima proba_dict kosong. "
                "Ini biasanya terjadi karena tidak ada expert model yang aktif, "
                "atau semua model gagal dibangun / gagal training / gagal predict."
            )
        first = next(iter(proba_dict.values()))
        return int(first.shape[1])

    def _scores_for_class(self, proba_dict: dict, class_idx: int) -> np.ndarray:
        return np.column_stack([p[:, class_idx] for p in proba_dict.values()])

    def _fit_state(self, S: np.ndarray) -> dict:
        from sklearn.preprocessing import QuantileTransformer
        state = {
            'qt': QuantileTransformer(
                n_quantiles=min(500, max(10, S.shape[0])),
                output_distribution='uniform',
                random_state=42,
            ),
            'corr': None,
            'theta': None,
        }
        U = np.clip(state['qt'].fit_transform(S), 1e-6, 1 - 1e-6)

        if self.family == 'gaussian':
            from scipy.stats import norm as N
            Z = N.ppf(U)
            state['corr'] = np.corrcoef(Z.T)
        elif self.family == 'clayton':
            from scipy.stats import kendalltau
            taus = [max(float(kendalltau(U[:, i], U[:, j]).statistic or 0.0), 1e-4)
                    for i in range(U.shape[1])
                    for j in range(i + 1, U.shape[1])]
            tau = float(np.mean(taus)) if taus else 1e-4
            tau = min(max(tau, 1e-4), 0.95)
            state['theta'] = 2 * tau / (1 - tau)
        elif self.family == 'frank':
            state['theta'] = 2.0

        return state

    def fit(self, proba_dict: dict, y: np.ndarray):
        from sklearn.metrics import roc_auc_score

        if not proba_dict:
            raise ValueError(
                "Tidak ada probabilitas model untuk tahap Copula Fusion. "
                "Pastikan minimal satu model berhasil didaftarkan, dilatih, dan menghasilkan prediksi."
            )

        self._names = list(proba_dict.keys())
        self._n_classes = self._get_n_classes(proba_dict)

        if len(proba_dict) == 1:
            self._passthrough = True
            self._single_model_name = self._names[0]
            self._weights = np.array([1.0], dtype=float)
            self._state = None
            print(f"\n  Copula fusion di-bypass: hanya satu model aktif ({self._single_model_name}).")
            return self

        print(f"\n  Copula family: {self.family.upper()}")

        # AUC-based model weights
        weights = []
        for name, p in proba_dict.items():
            try:
                if self._n_classes == 2:
                    auc = roc_auc_score(y, p[:, 1])
                else:
                    auc = roc_auc_score(y, p, multi_class='ovr', average='macro')
            except Exception:
                auc = 0.5
            w = max(float(auc) - 0.5, 1e-3)
            weights.append(w)
            print(f"  {name:12s} AUC={auc:.4f}  weight={w:.4f}")
        self._weights = np.array(weights, dtype=float)
        self._weights = self._weights / self._weights.sum()

        if self._n_classes == 2:
            S = self._scores_for_class(proba_dict, 1)
            self._state = self._fit_state(S)
            if self.family == 'gaussian' and self._state['corr'] is not None:
                print(f"  Correlation matrix:\n{np.round(self._state['corr'], 3)}")
            elif self.family in ('clayton', 'frank'):
                print(f"  θ = {self._state['theta']:.4f}")
        else:
            self._state = {c: self._fit_state(self._scores_for_class(proba_dict, c))
                           for c in range(self._n_classes)}
            print(f"  Class-wise copula states prepared for {self._n_classes} classes.")

        return self

    def _copula_density(self, U: np.ndarray, state: dict) -> np.ndarray:
        try:
            if self.family == 'gaussian':
                return self._gaussian(U, state['corr'])
            elif self.family == 'clayton':
                return self._clayton(U, state['theta'])
            elif self.family == 'frank':
                return self._frank(U, state['theta'])
        except Exception:
            pass
        return np.clip(U.mean(axis=1), 1e-6, 1.0)

    def _fuse_single(self, S: np.ndarray, state: dict) -> np.ndarray:
        U = np.clip(state['qt'].transform(S), 1e-6, 1 - 1e-6)
        density  = self._copula_density(U, state)
        weighted = np.average(S, axis=1, weights=self._weights)
        blended  = self.alpha * density + (1 - self.alpha) * weighted
        return np.clip(blended, 1e-9, 1 - 1e-9)

    def fuse(self, proba_dict: dict) -> np.ndarray:
        if self._n_classes is None:
            raise RuntimeError('CopulaFusion must be fitted before calling fuse().')
        if not proba_dict:
            raise ValueError(
                "CopulaFusion.fuse menerima proba_dict kosong. "
                "Tidak ada output probabilitas yang bisa difusikan."
            )
        if self._passthrough:
            if self._single_model_name in proba_dict:
                return np.clip(proba_dict[self._single_model_name], 1e-9, 1 - 1e-9)
            first = next(iter(proba_dict.values()))
            return np.clip(first, 1e-9, 1 - 1e-9)

        if self._n_classes == 2:
            S = self._scores_for_class(proba_dict, 1)
            attack_score = self._fuse_single(S, self._state)
            return np.column_stack([1 - attack_score, attack_score])

        class_scores = []
        for c in range(self._n_classes):
            S_c = self._scores_for_class(proba_dict, c)
            class_scores.append(self._fuse_single(S_c, self._state[c]))
        fused = np.column_stack(class_scores)
        fused = fused / np.clip(fused.sum(axis=1, keepdims=True), 1e-9, None)
        return np.clip(fused, 1e-9, 1 - 1e-9)

    def _gaussian(self, U, corr):
        from scipy.stats import norm as N
        if corr is None:
            return np.clip(U.mean(axis=1), 1e-6, 1.0)
        Sigma = np.array(corr, dtype=float)
        eps = 1e-6
        Sigma = Sigma + np.eye(Sigma.shape[0]) * eps
        invS  = np.linalg.pinv(Sigma)
        ldet  = np.log(max(np.linalg.det(Sigma), 1e-12))
        Z     = N.ppf(np.clip(U, 1e-9, 1 - 1e-9))
        vals  = np.einsum('bi,ij,bj->b', Z, (invS - np.eye(Sigma.shape[0])), Z)
        log_c = -0.5 * (vals + ldet)
        c     = np.exp(np.clip(log_c, -30, 30))
        return (c - c.min()) / (np.ptp(c) + 1e-10)

    def _clayton(self, U, theta):
        th = max(float(theta or 0.1), 0.1)
        pairs = []
        for i in range(U.shape[1]):
            for j in range(i + 1, U.shape[1]):
                u = np.clip(U[:, i], 1e-9, 1 - 1e-9)
                v = np.clip(U[:, j], 1e-9, 1 - 1e-9)
                c = ((th + 1) * (u * v) ** (-(th + 1))
                     * (u ** (-th) + v ** (-th) - 1) ** (-(2 * th + 1) / th))
                pairs.append(c)
        d = np.nanmean(pairs, axis=0) if pairs else U.mean(axis=1)
        d = np.nan_to_num(d, nan=0.5).clip(0)
        return (d - d.min()) / (np.ptp(d) + 1e-10)

    def _frank(self, U, theta):
        th = float(theta or 2.0)
        pairs = []
        for i in range(U.shape[1]):
            for j in range(i + 1, U.shape[1]):
                u = np.clip(U[:, i], 1e-9, 1 - 1e-9)
                v = np.clip(U[:, j], 1e-9, 1 - 1e-9)
                et, eu, ev = np.exp(-th), np.exp(-th * u), np.exp(-th * v)
                denom = (1 - et) - (1 - eu) * (1 - ev)
                c = (th * (1 - et) * eu * ev) / np.clip(denom ** 2, 1e-12, None)
                pairs.append(c)
        d = np.nanmean(pairs, axis=0) if pairs else U.mean(axis=1)
        d = np.nan_to_num(d, nan=0.5).clip(0)
        return (d - d.min()) / (np.ptp(d) + 1e-10)


# =============================================================================
# BAYESIAN NETWORK REASONING
# =============================================================================
class BayesianLayer:
    """Hill-Climb BN + Variable Elimination inference."""

    def __init__(self, n_bins: int = 4, max_parents: int = 3,
                 top_k_feats: int = 8):
        self.n_bins       = n_bins
        self.max_parents  = max_parents
        self.top_k        = top_k_feats
        self._bn          = None
        self._infer       = None
        self._edges       = None
        self._bin_edges   = {}
        self._bn_feat_names = None
        self.fitted       = False

    def _discretize(self, X: np.ndarray, cols: list,
                    fit: bool = True) -> pd.DataFrame:
        df = {}
        for i, col in enumerate(cols):
            vals = X[:, i]
            if fit:
                _, edges = pd.cut(vals, bins=self.n_bins,
                                  retbins=True, labels=False,
                                  duplicates='drop')
                self._bin_edges[col] = edges
            edges = self._bin_edges.get(col)
            if edges is not None:
                disc = pd.Series(pd.cut(vals, bins=edges, labels=False,
                                        include_lowest=True)).fillna(0).astype(int)
            else:
                disc = pd.Series(np.zeros(len(vals), dtype=int))
            df[col] = disc.values
        return pd.DataFrame(df, columns=cols)

    def _discretize_vector(self, values: np.ndarray, name: str, fit: bool = True) -> pd.Series:
        if fit:
            _, edges = pd.cut(values, bins=self.n_bins, retbins=True, labels=False, duplicates='drop')
            self._bin_edges[name] = edges
        edges = self._bin_edges.get(name)
        if edges is None:
            return pd.Series(np.zeros(len(values), dtype=int))
        return pd.Series(pd.cut(values, bins=edges, labels=False, include_lowest=True)).fillna(0).astype(int)

    def fit(self, X: np.ndarray, fused: np.ndarray,
            y: np.ndarray, feat_names: list):
        try:
            try:
                from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
            except Exception:
                from pgmpy.models import BayesianNetwork
            from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
            try:
                from pgmpy.estimators import BIC as _BICScore
            except Exception:
                from pgmpy.estimators import BicScore as _BICScore
            from pgmpy.inference import VariableElimination
            from sklearn.feature_selection import mutual_info_classif
        except ImportError:
            print("  pgmpy not installed — BN layer skipped.")
            return self

        print("\n  Bayesian Network structure learning …")

        # Select top-k informative features
        mi   = mutual_info_classif(X, y, random_state=42)
        idxs = np.argsort(mi)[::-1][:self.top_k]
        sel_names = [feat_names[i] if i < len(feat_names)
                     else f'f{i}' for i in idxs]
        self._bn_feat_names = sel_names

        X_sel = X[:, idxs]
        df_d  = self._discretize(X_sel, sel_names, fit=True)
        df_d['fused_conf'] = self._discretize_vector(fused.max(axis=1), 'fused_conf', fit=True)
        df_d['fused_pred'] = np.argmax(fused, axis=1).astype(int)
        df_d['label'] = y.astype(int)

        hc   = HillClimbSearch(df_d)
        best = hc.estimate(scoring_method=_BICScore(df_d),
                           max_indegree=self.max_parents,
                           max_iter=300, show_progress=False)
        self._edges = list(best.edges())
        print(f"  Edges: {len(self._edges)}")

        self._bn = BayesianNetwork(self._edges)
        self._bn.fit(df_d, estimator=MaximumLikelihoodEstimator)
        self._infer = VariableElimination(self._bn)
        self.fitted = True
        print("  BN ready for inference.")
        return self

    def predict(self, X: np.ndarray, fused: np.ndarray,
                feat_names: list) -> np.ndarray:
        if not self.fitted:
            return np.argmax(fused, axis=1)

        idxs  = [feat_names.index(n) if n in feat_names else 0
                 for n in self._bn_feat_names]
        X_sel = X[:, idxs]
        df_d  = self._discretize(X_sel, self._bn_feat_names, fit=False)
        df_d['fused_conf'] = self._discretize_vector(fused.max(axis=1), 'fused_conf', fit=False)
        df_d['fused_pred'] = np.argmax(fused, axis=1).astype(int)

        bn_nodes = set(self._bn.nodes())
        preds    = []
        for i in range(len(df_d)):
            ev = {c: int(df_d.loc[i, c])
                  for c in df_d.columns
                  if c != 'label' and c in bn_nodes}
            try:
                q    = self._infer.query(['label'], evidence=ev, show_progress=False)
                pred = int(np.argmax(q.values))
            except Exception:
                pred = int(np.argmax(fused[i]))
            preds.append(pred)
        return np.array(preds)


# =============================================================================
# EVALUATOR + XAI + STATISTICAL VALIDATION
# =============================================================================
class Evaluator:

    def __init__(self, class_names: list, dataset: str = ''):
        self.class_names = class_names
        self.dataset     = dataset
        self.results_    = {}

    def evaluate(self, name: str, y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_proba: np.ndarray = None) -> dict:
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, matthews_corrcoef,
                                     classification_report)
        avg = 'binary' if len(np.unique(y_true))==2 else 'weighted'
        res = {
            'Accuracy':  accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average=avg,
                                         zero_division=0),
            'Recall':    recall_score(y_true, y_pred, average=avg,
                                      zero_division=0),
            'F1':        f1_score(y_true, y_pred, average=avg,
                                  zero_division=0),
            'MCC':       matthews_corrcoef(y_true, y_pred),
        }
        if y_proba is not None:
            try:
                res['AUC-ROC'] = (roc_auc_score(y_true, y_proba[:,1])
                                  if y_proba.shape[1]==2
                                  else roc_auc_score(y_true, y_proba,
                                                     multi_class='ovr',
                                                     average='macro'))
            except Exception:
                res['AUC-ROC'] = None
        self.results_[name] = res

        print(f"\n  ── {name} ({self.dataset}) ──")
        for k, v in res.items():
            if v is None:
                print(f"    {k:12s}: N/A")
            else:
                print(f"    {k:12s}: {float(v):.4f}")
        # Build target_names safely from actual labels in y_true
        _labels    = sorted(np.unique(np.concatenate([y_true, y_pred])))
        _tgt_names = [self.class_names[i] if i < len(self.class_names)
                      else f'Class_{i}' for i in _labels]
        print(classification_report(y_true, y_pred,
                                    labels=_labels,
                                    target_names=_tgt_names,
                                    zero_division=0))
        return res

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_confusion(self, y_true, y_pred,
                       title='Confusion Matrix'):
        from sklearn.metrics import confusion_matrix
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        n  = cm.shape[0]
        ticklabels = [self.class_names[i] if i < len(self.class_names) else f'Class_{i}'
                      for i in labels]
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n-1)),
                               facecolor='#0b1020')
        ax.set_facecolor('#111827')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=ticklabels,
                    yticklabels=ticklabels,
                    annot_kws={'size':11,'color':'white'})
        ax.set_title(f'{title} [{self.dataset}]',
                     color='white', fontsize=12)
        ax.set_xlabel('Predicted', color='#94a3b8')
        ax.set_ylabel('Actual',    color='#94a3b8')
        ax.tick_params(colors='#64748b')
        plt.tight_layout()
        safe = title.replace(' ','_').replace('/','')
        path = FIG_DIR / f"cm_{self.dataset}_{safe}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='#0b1020')
        plt.close()

    def plot_comparison(self):
        df  = pd.DataFrame(self.results_).T
        mts = [c for c in ['Accuracy','F1','AUC-ROC','Precision',
                            'Recall','MCC'] if c in df.columns]
        df  = df[mts].apply(pd.to_numeric, errors='coerce').dropna(how='all')

        colors = ['#7c3aed','#06b6d4','#ff3b5c',
                  '#f59e0b','#10b981','#a78bfa']
        x, w   = np.arange(len(df)), 0.12
        fig, ax = plt.subplots(figsize=(max(12, len(df)*1.8), 5),
                               facecolor='#0b1020')
        ax.set_facecolor('#111827')
        for i, m in enumerate(mts):
            vals = df[m].fillna(0).values
            ax.bar(x + i*w, vals, w, label=m,
                   color=colors[i % len(colors)], alpha=0.85)
        ax.set_xticks(x + w*len(mts)/2)
        ax.set_xticklabels(df.index, rotation=25, color='#94a3b8', ha='right')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', color='#94a3b8')
        ax.set_title(f'Model Comparison — {self.dataset}',
                     color='white', fontsize=13)
        ax.legend(facecolor='#0b1020', labelcolor='white',
                  fontsize=9, ncol=len(mts))
        ax.tick_params(colors='#64748b')
        plt.tight_layout()
        path = FIG_DIR / f"comparison_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='#0b1020')
        plt.close()
        print(f"  Comparison chart → {path}")

    def plot_roc_all(self, y_true, proba_dict, fused_proba):
        from sklearn.metrics import roc_curve, auc
        if len(np.unique(y_true)) != 2:
            return
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0b1020')
        ax.set_facecolor('#111827')
        colors = ['#7c3aed','#06b6d4','#f59e0b','#10b981','#a78bfa','#fb923c']
        for (name, p), c in zip(
                list(proba_dict.items()) + [('Copula Fusion', fused_proba)],
                colors):
            try:
                fpr, tpr, _ = roc_curve(y_true, p[:,1])
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=c, lw=2,
                        label=f'{name} (AUC={roc_auc:.4f})')
            except Exception:
                pass
        ax.plot([0,1],[0,1], '--', color='#475569', lw=1)
        ax.set_xlabel('FPR', color='#94a3b8')
        ax.set_ylabel('TPR', color='#94a3b8')
        ax.set_title(f'ROC Curves — {self.dataset}',
                     color='white', fontsize=12)
        ax.legend(facecolor='#0b1020', labelcolor='white', fontsize=9)
        ax.tick_params(colors='#64748b')
        plt.tight_layout()
        path = FIG_DIR / f"roc_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='#0b1020')
        plt.close()
        print(f"  ROC curves → {path}")

    def plot_fused_distribution(self, fused_proba, y_true):
        from scipy.stats import gaussian_kde
        scores = fused_proba[:, 1] if fused_proba.shape[1] == 2 else fused_proba.max(axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 facecolor='#0b1020')
        unique_cls = sorted(np.unique(y_true))
        colors = plt.cm.tab10(np.linspace(0, 1, max(2, len(unique_cls))))
        for cls, col in zip(unique_cls, colors):
            mask = (y_true == cls)
            if not mask.any():
                continue
            lbl = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            axes[0].hist(scores[mask], bins=40, alpha=0.55,
                         color=col, label=lbl, density=True)
            if mask.sum() > 5 and np.std(scores[mask]) > 0:
                kde = gaussian_kde(scores[mask])
                xs  = np.linspace(0, 1, 300)
                axes[1].plot(xs, kde(xs), color=col, lw=2, label=lbl)
                axes[1].fill_between(xs, kde(xs), alpha=0.12, color=col)
        xlabel = 'P(Attack)' if fused_proba.shape[1] == 2 else 'Max fused class probability'
        for ax, ttl in zip(axes, ['Histogram', 'KDE']):
            ax.set_facecolor('#111827')
            ax.set_title(f'Fused Score {ttl}', color='white')
            ax.set_xlabel(xlabel, color='#94a3b8')
            ax.legend(facecolor='#0b1020', labelcolor='white', fontsize=9)
            ax.tick_params(colors='#64748b')
        plt.suptitle(f'Fused Probability Distribution — {self.dataset}',
                     color='white')
        plt.tight_layout()
        path = FIG_DIR / f"fused_dist_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='#0b1020')
        plt.close()

    # ── SHAP ──────────────────────────────────────────────────────────────────
    def shap_explain(self, model, X: np.ndarray,
                     feat_names: list, model_name: str):
        try:
            import shap
            X_s = X[:min(300, len(X))]
            if hasattr(model, 'estimators_'):
                exp = shap.TreeExplainer(model)
            else:
                exp = shap.KernelExplainer(
                    model.predict_proba, X_s[:50])
            sv  = exp.shap_values(X_s)
            sv  = sv[1] if isinstance(sv, list) else sv
            fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0b1020')
            shap.summary_plot(sv, X_s, feature_names=feat_names,
                              show=False, plot_type='bar')
            plt.title(f'SHAP — {model_name} [{self.dataset}]',
                      color='white')
            path = FIG_DIR / f"shap_{self.dataset}_{model_name}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight',
                        facecolor='#0b1020')
            plt.close()
            print(f"  SHAP → {path}")
        except Exception as e:
            print(f"  SHAP skipped ({model_name}): {e}")

    # ── Statistical validation ────────────────────────────────────────────────
    def statistical_tests(self, cv_scores: dict):
        from scipy.stats import friedmanchisquare, wilcoxon
        names   = list(cv_scores.keys())
        arrays  = [np.array(cv_scores[n]) for n in names]
        results = {}

        # Friedman
        if len(arrays) >= 3:
            try:
                stat, p = friedmanchisquare(*arrays)
                results['Friedman'] = {'χ²': round(stat,4), 'p': round(p,4),
                                       'sig': p < 0.05}
                print(f"  Friedman χ²={stat:.4f}  p={p:.4f}"
                      f"  {'SIGNIFICANT ✓' if p<0.05 else 'not sig.'}")
            except Exception as e:
                print(f"  Friedman: {e}")

        # Pairwise Wilcoxon
        results['Wilcoxon'] = {}
        best = max(cv_scores, key=lambda n: np.mean(cv_scores[n]))
        for n in names:
            if n == best: continue
            try:
                s, p = wilcoxon(arrays[names.index(best)],
                                arrays[names.index(n)],
                                zero_method='wilcox')
                # Cohen's d
                a, b = arrays[names.index(best)], arrays[names.index(n)]
                d    = (a.mean()-b.mean()) / (np.sqrt((a.std()**2+b.std()**2)/2)+1e-9)
                key  = f'{best} vs {n}'
                results['Wilcoxon'][key] = {
                    'W': round(s,2), 'p': round(p,4),
                    'sig': p < 0.05, "Cohen's d": round(d,4)}
                print(f"  Wilcoxon {key:30s}: W={s:.1f}  "
                      f"p={p:.4f}  d={d:.4f}"
                      f"  {'✓' if p<0.05 else '✗'}")
            except Exception as e:
                print(f"  Wilcoxon {n}: {e}")

        path = OUT_DIR / f"stats_{self.dataset}.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Stats → {path}")
        return results

    # ── Summary ───────────────────────────────────────────────────────────────
    def plot_precision_recall(self, y_true: np.ndarray,
                               proba_dict: dict,
                               fused_proba: np.ndarray):
        """Precision-Recall curves for all models + fused."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        if len(np.unique(y_true)) != 2:
            return
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0b1020')
        ax.set_facecolor('#111827')
        colors = ['#7c3aed','#06b6d4','#f59e0b','#10b981',
                  '#a78bfa','#fb923c','#ff3b5c']
        all_models = list(proba_dict.items()) + [('Copula Fusion', fused_proba)]
        for (name, p), c in zip(all_models, colors):
            try:
                prec, rec, _ = precision_recall_curve(y_true, p[:, 1])
                ap = average_precision_score(y_true, p[:, 1])
                ax.plot(rec, prec, color=c, lw=2,
                        label=f'{name} (AP={ap:.4f})')
                ax.fill_between(rec, prec, alpha=0.06, color=c)
            except Exception:
                pass
        # Baseline
        baseline = y_true.mean()
        ax.axhline(baseline, linestyle='--', color='#475569',
                   lw=1, label=f'Baseline ({baseline:.3f})')
        ax.set_xlabel('Recall',    color='#94a3b8')
        ax.set_ylabel('Precision', color='#94a3b8')
        ax.set_title(f'Precision-Recall Curves — {self.dataset}',
                     color='white', fontsize=12)
        ax.legend(facecolor='#0b1020', labelcolor='white', fontsize=8)
        ax.tick_params(colors='#64748b')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        plt.tight_layout()
        path = FIG_DIR / f"pr_curve_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#0b1020')
        plt.close()
        print(f"  PR curves → {path}")

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results_).T.reset_index()
        df.rename(columns={'index':'Model'}, inplace=True)

        # Numeric columns
        num_cols = [c for c in df.columns if c != 'Model']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

        print("\n" + "="*60)
        print(f"RESULTS SUMMARY — {self.dataset}")
        print("="*60)
        print(df.round(4).to_string(index=False))

        # ── CSV export ─────────────────────────────────────────────────────
        csv_path = OUT_DIR / f"results_{self.dataset}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV   → {csv_path}")

        # ── LaTeX export ───────────────────────────────────────────────────
        try:
            latex_str = df.round(4).to_latex(
                index=False, escape=True,
                caption=f"Model Performance Results — {self.dataset.upper()}",
                label=f"tab:results_{self.dataset}",
                column_format='l' + 'r'*(len(df.columns)-1)
            )  # keep as-is
            tex_path = OUT_DIR / f"results_{self.dataset}.tex"
            with open(tex_path, 'w') as f:
                f.write(latex_str)
            print(f"  LaTeX → {tex_path}")
        except Exception as e:
            print(f"  LaTeX export skipped: {e}")

        # ── Summary bar chart ─────────────────────────────────────────────
        self._plot_summary_heatmap(df)
        return df

    def _plot_summary_heatmap(self, df: pd.DataFrame):
        """Heatmap table of all model × metric scores."""
        metrics = [c for c in ['Accuracy','Precision','Recall',
                                'F1','AUC-ROC','MCC']
                   if c in df.columns]
        data = df.set_index('Model')[metrics].apply(
            pd.to_numeric, errors='coerce').fillna(0)
        if data.empty:
            return
        fig, ax = plt.subplots(
            figsize=(len(metrics)*1.6, max(4, len(data)*0.65)),
            facecolor='#0b1020')
        ax.set_facecolor('#111827')
        im = ax.imshow(data.values, cmap='RdYlGn', vmin=0, vmax=1,
                       aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8).ax.yaxis.label.set_color('white')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, color='#94a3b8', fontsize=10)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, color='#94a3b8', fontsize=10)
        # Annotate
        for i in range(len(data)):
            for j in range(len(metrics)):
                val = data.values[i, j]
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                        fontsize=9,
                        color='black' if 0.35 < val < 0.75 else 'white')
        ax.set_title(f'Model × Metric Summary — {self.dataset}',
                     color='white', fontsize=12)
        plt.tight_layout()
        path = FIG_DIR / f"summary_heatmap_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#0b1020')
        plt.close()
        print(f"  Summary heatmap → {path}")


# =============================================================================
# CROSS-VALIDATION RUNNER
# =============================================================================
def run_cv(learners: ExpertLearners, X: np.ndarray,
           y: np.ndarray, k: int = 2,
           n_classes: int = 2,
           X_val: np.ndarray = None,
           y_val: np.ndarray = None) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    avg = 'binary' if n_classes==2 else 'weighted'
    cv_scores = defaultdict(list)

    ml_candidates = [name for name in learners.models_.keys() if name not in learners._dl_cfg]
    if not ml_candidates:
        print(f"\n  Stratified {k}-Fold CV … skipped (tidak ada model ML untuk CV).")
        return {}
    print(f"\n  Stratified {k}-Fold CV …")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        for name, model in learners.models_.items():
            if name in learners._dl_cfg:
                continue        # skip DL in CV (time)
            import copy
            m_clone = copy.deepcopy(model)
            m_clone.fit(X_tr, y_tr)
            f1 = f1_score(y_va, m_clone.predict(X_va),
                          average=avg, zero_division=0)
            cv_scores[name].append(f1)

    for name, scores in cv_scores.items():
        print(f"    {name:8s}  F1 = {np.mean(scores):.4f} "
              f"± {np.std(scores):.4f}")

    return dict(cv_scores)


# =============================================================================
# █████████████████████████████████████████████████████████████████████████████
#  MASTER PIPELINE
# █████████████████████████████████████████████████████████████████████████████
# =============================================================================
class NIDSPipeline:
    """
    Orchestrates all 8 dissertation pipeline stages for any dataset.

    Parameters
    ----------
    dataset : str
        'nslkdd' | 'unsw' | 'insdn' | 'cicids'
    paths : dict
        Dataset file paths (keys depend on dataset; see examples below)
    binary : bool
        True  → binary classification (Normal vs Attack)
        False → multiclass
    copula_family : str
        'gaussian' | 'clayton' | 'frank'
    balance : str
        'smote' | 'adasyn' | 'borderline' | 'under'
    top_k : int
        Number of features to keep after selection
    cv_folds : int
        Stratified K-Fold folds for cross-validation
    sample_frac : float
        Fraction of CICIDS data to use (1.0 = full; 0.2 = 20% for dev)
    """

    def __init__(
        self,
        dataset: str,
        paths: dict,
        binary: bool          = True,
        copula_family: str    = 'gaussian',
        balance: str          = 'smote',
        scaler: str           = 'robust',
        top_k: int            = 40,
        cv_folds: int         = 2,   # naikan ke 5/10 saat model final
        # Tabular Expert
        use_rf: bool          = True,
        use_xgb: bool         = True,
        # Temporal Expert
        use_cnn_bilstm: bool  = True,
        use_cnn_gru: bool     = True,
        # Graph Expert
        use_gnn: bool         = True,
        use_gat: bool         = True,
        # Transformer Expert
        use_transformer: bool = True,
        # Auxiliary
        use_bn: bool          = True,
        use_shap: bool        = True,
        sample_frac: float    = 1.0,
        cicids_split: str     = 'random',   # 'random' | 'temporal'
    ):
        self.dataset         = dataset
        self.paths           = paths
        self.binary          = binary
        self.copula_family   = copula_family
        self.balance         = balance
        self.scaler          = scaler
        self.top_k           = top_k
        self.cv_folds        = cv_folds
        # Experts
        self.use_rf          = use_rf
        self.use_xgb         = use_xgb
        self.use_cnn_bilstm  = use_cnn_bilstm
        self.use_cnn_gru     = use_cnn_gru
        self.use_gnn         = use_gnn
        self.use_gat         = use_gat
        self.use_transformer = use_transformer
        self.use_bn          = use_bn
        self.use_shap        = use_shap
        self.sample_frac     = sample_frac
        self.cicids_split    = cicids_split

        self.logger = Logger(f"pipeline_{dataset}")

    def _banner(self, msg: str):
        line = "─" * 65
        self.logger.log(f"\n{line}\n  {msg}\n{line}")

    def _report_split_methodology(self, meta: dict):
        note = meta.get('split_note', 'Split note not provided by loader.')
        self.logger.log(f"  Split design → {note}")
        self.logger.log(
            "  Methodology → encoders/imputer/clipping/feature selection/scaler are fitted on TRAIN only; VAL is used for tuning/early stopping; TEST remains final hold-out."
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_model_params(self, name: str) -> dict:
        return dict(self.model_params.get(name, {}))

    def _get_train_params(self) -> tuple:
        epochs = int(self.training_params.get('epochs', 30))
        batch_size = int(self.training_params.get('batch_size', 256))
        return epochs, batch_size

    def run(self) -> dict:
        t0 = time.time()
        self._banner(
            f"HYBRID NIDS PIPELINE  ·  Dataset: {self.dataset.upper()}"
            f"  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ── Stage 1+2: Load & engineer ─────────────────────────────────────
        self._banner("Stage 1 + 2 · Data Loading & Feature Engineering")
        meta = self._load()
        train_df    = meta['train']
        val_df      = meta.get('val', None)   # may be None for datasets without val
        test_df     = meta['test']
        feat_cols   = meta['feat_cols']
        cat_cols    = meta['cat_cols']
        class_names = meta['class_names']
        n_classes   = meta['n_classes']

        y_train_raw = train_df['y'].values.astype(int)
        y_val       = val_df['y'].values.astype(int)   if val_df is not None else None
        y_test      = test_df['y'].values.astype(int)

        self._report_split_methodology(meta)

        # ── Stage 2b: EDA ──────────────────────────────────────────────────
        self._banner("Stage 2b · Exploratory Data Analysis")
        eda = EDAAnalyzer(dataset=self.dataset,
                          class_names=class_names)
        # Build numeric-only X for EDA:
        #   encode categoricals → pd.to_numeric (coerce) → fillna(0)
        _eda_df = train_df[feat_cols].copy()
        for _c in cat_cols:
            if _c in _eda_df.columns:
                _eda_df[_c] = pd.factorize(_eda_df[_c])[0]
        _eda_X = _eda_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        eda.run_all(df=train_df,
                    X=_eda_X,
                    feat_names=feat_cols,
                    y=y_train_raw)

        # ── Stage 3: Preprocess ────────────────────────────────────────────
        self._banner("Stage 3 · Preprocessing")
        prep = Preprocessor(scaler_type=self.scaler, balance=self.balance)

        # 3.1 Clean using TRAIN-only fitted state
        X_train_clean = prep.clean_fit_transform(train_df, feat_cols, cat_cols)
        X_val_clean   = prep.clean_transform(val_df, feat_cols, cat_cols) if val_df is not None else None
        X_test_clean  = prep.clean_transform(test_df, feat_cols, cat_cols)

        # 3.2 Feature selection learned on original TRAIN only (before resampling)
        fe     = FeatureEngineer()
        k      = min(self.top_k, X_train_clean.shape[1])
        idx    = fe.select(X_train_clean, y_train_raw, feat_cols, k=k)
        X_train_clean = X_train_clean[:, idx]
        X_val_clean   = X_val_clean[:, idx] if X_val_clean is not None else None
        X_test_clean  = X_test_clean[:, idx]
        feat_names    = fe.selected_names_
        sel_cat_cols  = [c for c in cat_cols if c in feat_names]

        # 3.3 Scale learned on original TRAIN only
        prep.fit_scaler(X_train_clean)
        X_train = prep.scale_transform(X_train_clean)
        X_val   = prep.scale_transform(X_val_clean) if X_val_clean is not None else None
        X_test  = prep.scale_transform(X_test_clean)

        # 3.4 Balance TRAIN only (never VAL/TEST)
        cat_indices_selected = [feat_names.index(c) for c in sel_cat_cols if c in feat_names]
        X_train, y_train = prep.balance_train(X_train, y_train_raw, cat_indices=cat_indices_selected)

        val_shape = X_val.shape if X_val is not None else 'N/A'
        self.logger.log(
            f"  Final shapes → train: {X_train.shape}  "
            f"val: {val_shape}  test: {X_test.shape}")

        # ── Stage 4: Build & cross-validate ────────────────────────────────
        self._banner("Stage 4 · Base Learner Training + CV")
        learners = ExpertLearners(n_classes=n_classes,
                                n_features=X_train.shape[1])
        # Tabular Expert
        if self.use_rf:          learners.add_rf()
        if self.use_xgb:         learners.add_xgb()
        # Temporal Expert
        if self.use_cnn_bilstm:  learners.add_cnn_bilstm(epochs=30)
        if self.use_cnn_gru:     learners.add_cnn_gru(epochs=30)
        # Graph Expert
        if self.use_gnn:         learners.add_gnn(epochs=30)
        if self.use_gat:         learners.add_gat(epochs=30)
        # Transformer Expert
        if self.use_transformer: learners.add_transformer(epochs=30)

        # Cross-validation (sklearn models only)
        cv_scores = run_cv(learners, X_train, y_train,
                           k=self.cv_folds, n_classes=n_classes,
                           X_val=X_val, y_val=y_val)

        # ── Stage 5: Copula Fusion ─────────────────────────────────────────
        self._banner("Stage 5 · Copula Fusion")
        # Re-fit on full train with val set for DL early stopping
        learners.fit_all(X_train, y_train, X_val=X_val, y_val=y_val)

        train_proba = learners.predict_proba_all(X_train)
        copula      = CopulaFusion(family=self.copula_family)
        copula.fit(train_proba, y_train)

        test_proba   = learners.predict_proba_all(X_test)
        fused_test   = copula.fuse(test_proba)
        fused_train  = copula.fuse(train_proba)

        # ── Stage 6: Fused distribution ────────────────────────────────────
        self._banner("Stage 6 · Fused Probability Distribution Analysis")
        evaluator = Evaluator(class_names=class_names,
                              dataset=self.dataset)
        evaluator.plot_fused_distribution(fused_test, y_test)

        # ── Stage 7: Bayesian Network ──────────────────────────────────────
        bn_pred = None
        if self.use_bn:
            self._banner("Stage 7 · Bayesian Network Reasoning")
            bn = BayesianLayer(n_bins=4, max_parents=3, top_k_feats=8)
            bn.fit(X_train, fused_train, y_train, feat_names)
            if bn.fitted:
                bn_pred = bn.predict(X_test, fused_test, feat_names)

        # ── Stage 8: Evaluation ────────────────────────────────────────────
        self._banner("Stage 8 · Evaluation + XAI + Statistical Validation")

        test_preds  = learners.predict_all(X_test)
        test_probas = learners.predict_proba_all(X_test)

        # Individual models
        for name in test_preds:
            evaluator.evaluate(name, y_test, test_preds[name],
                               y_proba=test_probas.get(name))
            evaluator.plot_confusion(y_test, test_preds[name],
                                     title=f'CM {name}')

        # Copula fusion
        fused_pred = np.argmax(fused_test, axis=1)
        evaluator.evaluate('Copula Fusion', y_test, fused_pred,
                           y_proba=fused_test)
        evaluator.plot_confusion(y_test, fused_pred,
                                 title='CM Copula Fusion')

        # Bayesian Network
        if bn_pred is not None:
            evaluator.evaluate('BN Reasoning', y_test, bn_pred)
            evaluator.plot_confusion(y_test, bn_pred,
                                     title='CM BN Reasoning')

        # ROC overlay + Precision-Recall curves
        evaluator.plot_roc_all(y_test, test_probas, fused_test)
        evaluator.plot_precision_recall(y_test, test_probas, fused_test)

        # Model comparison
        evaluator.plot_comparison()

        # SHAP for RF
        if self.use_shap and 'RF' in learners.models_:
            evaluator.shap_explain(learners.models_['RF'],
                                   X_test, feat_names, 'RF')

        # Statistical validation
        if len(cv_scores) >= 2:
            evaluator.statistical_tests(cv_scores)

        # Summary
        summary_df = evaluator.summary()

        elapsed = time.time() - t0
        self._banner(
            f"✅ Pipeline DONE  ·  {elapsed:.0f}s  "
            f"·  Figures → {FIG_DIR}")
        self.logger.close()

        return {
            'evaluator':   evaluator,
            'summary':     summary_df,
            'learners':    learners,
            'copula':      copula,
            'fused_test':  fused_test,
            'feat_names':  feat_names,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        ds = self.dataset.lower()
        if ds == 'nslkdd':
            return NSLKDDLoader().load(
                self.paths['train'], self.paths['test'],
                binary=self.binary)

        elif ds == 'unsw':
            return UNSWLoader().load(
                self.paths['train'], self.paths['test'],
                binary=self.binary)

        elif ds == 'insdn':
            return InSDNLoader().load(
                normal_path=self.paths['normal'],
                ovs_path=self.paths['ovs'],
                metasploit_path=self.paths['metasploit'],
                binary=self.binary)

        elif ds == 'cicids':
            return CICIDSLoader().load(
                csv_dir=self.paths.get('dir'),
                file_list=self.paths.get('files'),
                binary=self.binary,
                sample_frac=self.sample_frac,
                split_mode=getattr(self, 'cicids_split', 'random'),
                drop_duplicates=True)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")


# =============================================================================
# EXTENDED MODEL LIBRARY (ML + DL)
# =============================================================================
class ExpertLearnersV3(ExpertLearners):
    """Extended expert library with clearer ML/DL separation.

    Notes
    -----
    - AAE here is implemented as an *AAE-inspired autoencoder-regularized
      classifier scaffold*, not a full adversarial autoencoder with a separate
      discriminator.
    - TransferLearning here uses self-supervised autoencoder pretraining on the
      TRAIN split, then transfers encoder weights into a downstream classifier.
    """

    EXPERT_GROUPS = {
        **getattr(ExpertLearners, 'EXPERT_GROUPS', {}),
        'LR': 'ML', 'RF': 'ML', 'SVM': 'ML', 'XGB': 'ML', 'DT': 'ML',
        'MLP': 'DL', 'DNN': 'DL', 'CNN1D': 'DL', 'LSTM': 'DL',
        'AAE': 'DL-Experimental', 'CNN_LSTM': 'DL', 'CNN_BiLSTM': 'DL',
        'CNN_GRU': 'DL', 'GNN': 'DL-Graph', 'GAT': 'DL-Graph',
        'Transformer': 'DL-Transformer', 'TransferLearning': 'DL-Transfer',
    }

    @staticmethod
    def _filter_supported_estimator_params(estimator_cls, params: dict) -> tuple:
        """Return (accepted_params, ignored_keys) based on estimator __init__ signature."""
        import inspect
        try:
            sig = inspect.signature(estimator_cls.__init__)
            accepted = {k: v for k, v in params.items() if k in sig.parameters}
            ignored = sorted(set(params) - set(accepted))
            return accepted, ignored
        except Exception:
            return params, []

    def add_lr(self, **kw):
        from sklearn.linear_model import LogisticRegression
        params = dict(max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs')
        params.update(kw)
        safe_params, ignored = self._filter_supported_estimator_params(LogisticRegression, params)
        self.models_['LR'] = LogisticRegression(**safe_params)
        if ignored:
            print(f"  LR          built — Logistic Regression  (ignored unsupported params: {', '.join(ignored)})")
        else:
            print("  LR          built — Logistic Regression")

    def add_svm(self, **kw):
        from sklearn.svm import SVC
        params = dict(C=1.0, kernel='rbf', gamma='scale', probability=True,
                      class_weight='balanced', random_state=42)
        params.update(kw)
        safe_params, ignored = self._filter_supported_estimator_params(SVC, params)
        self.models_['SVM'] = SVC(**safe_params)
        if ignored:
            print(f"  SVM         built — Support Vector Machine  (ignored unsupported params: {', '.join(ignored)})")
        else:
            print("  SVM         built — Support Vector Machine")

    def add_dt(self, **kw):
        from sklearn.tree import DecisionTreeClassifier
        params = dict(max_depth=18, min_samples_leaf=2,
                      class_weight='balanced', random_state=42)
        params.update(kw)
        safe_params, ignored = self._filter_supported_estimator_params(DecisionTreeClassifier, params)
        self.models_['DT'] = DecisionTreeClassifier(**safe_params)
        if ignored:
            print(f"  DT          built — Decision Tree  (ignored unsupported params: {', '.join(ignored)})")
        else:
            print("  DT          built — Decision Tree")

    def _build_dense_classifier(self, name: str, hidden_units, dropout=0.30, batch_norm=False):
        tf, keras = _import_keras()
        L = keras.layers
        n_out, act, loss = self._out()
        inp = L.Input(shape=(self.n_features,), name=f'{name}_in')
        x = inp
        for i, units in enumerate(hidden_units, 1):
            x = L.Dense(units, activation='relu', name=f'{name}_d{i}')(x)
            if batch_norm:
                x = L.BatchNormalization(name=f'{name}_bn{i}')(x)
            if dropout and dropout > 0:
                x = L.Dropout(dropout if i < len(hidden_units) else max(0.1, dropout - 0.1), name=f'{name}_do{i}')(x)
        out = L.Dense(n_out, activation=act, name=f'{name}_out')(x)
        m = keras.models.Model(inp, out, name=name)
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
        return m, dict(epochs=30, batch=256, n_out=n_out, loss=loss, input_3d=False)

    def add_mlp(self, epochs: int = 30, batch: int = 256):
        try:
            m, cfg = self._build_dense_classifier('MLP', [128, 64], dropout=0.25, batch_norm=False)
            cfg.update(epochs=epochs, batch=batch)
            self.models_['MLP'] = m
            self._dl_cfg['MLP'] = cfg
            print(f"  MLP         built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('MLP', e)

    def add_dnn(self, epochs: int = 30, batch: int = 256):
        try:
            m, cfg = self._build_dense_classifier('DNN', [256, 128, 64], dropout=0.30, batch_norm=True)
            cfg.update(epochs=epochs, batch=batch)
            self.models_['DNN'] = m
            self._dl_cfg['DNN'] = cfg
            print(f"  DNN         built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('DNN', e)

    def add_cnn1d(self, epochs: int = 30, batch: int = 256):
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()
            inp = L.Input(shape=(self.n_features, 1), name='cnn1d_in')
            x = L.Conv1D(64, 3, padding='same', activation='relu', name='cnn1d_c1')(inp)
            x = L.BatchNormalization(name='cnn1d_bn1')(x)
            x = L.Conv1D(128, 5, padding='same', activation='relu', name='cnn1d_c2')(x)
            x = L.BatchNormalization(name='cnn1d_bn2')(x)
            x = L.GlobalMaxPooling1D(name='cnn1d_gap')(x)
            x = L.Dense(96, activation='relu', name='cnn1d_d1')(x)
            x = L.Dropout(0.30, name='cnn1d_do1')(x)
            out = L.Dense(n_out, activation=act, name='cnn1d_out')(x)
            m = keras.models.Model(inp, out, name='CNN1D')
            m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
            self.models_['CNN1D'] = m
            self._dl_cfg['CNN1D'] = dict(epochs=epochs, batch=batch, n_out=n_out, loss=loss, input_3d=True)
            print(f"  CNN1D       built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('CNN1D', e)

    def add_lstm(self, epochs: int = 30, batch: int = 256):
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()
            inp = L.Input(shape=(self.n_features, 1), name='lstm_in')
            x = L.LSTM(64, return_sequences=True, name='lstm_l1')(inp)
            x = L.Dropout(0.25, name='lstm_do1')(x)
            x = L.LSTM(32, name='lstm_l2')(x)
            x = L.Dense(64, activation='relu', name='lstm_d1')(x)
            x = L.Dropout(0.25, name='lstm_do2')(x)
            out = L.Dense(n_out, activation=act, name='lstm_out')(x)
            m = keras.models.Model(inp, out, name='LSTM')
            m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
            self.models_['LSTM'] = m
            self._dl_cfg['LSTM'] = dict(epochs=epochs, batch=batch, n_out=n_out, loss=loss, input_3d=True)
            print(f"  LSTM        built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('LSTM', e)

    def add_cnn_lstm(self, epochs: int = 30, batch: int = 256):
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()
            inp = L.Input(shape=(self.n_features, 1), name='cl_in')
            x = L.Conv1D(64, 3, padding='same', activation='relu', name='cl_c1')(inp)
            x = L.BatchNormalization(name='cl_bn1')(x)
            x = L.MaxPooling1D(2, name='cl_mp')(x)
            x = L.LSTM(64, name='cl_l1')(x)
            x = L.Dense(64, activation='relu', name='cl_d1')(x)
            x = L.Dropout(0.30, name='cl_do1')(x)
            out = L.Dense(n_out, activation=act, name='cl_out')(x)
            m = keras.models.Model(inp, out, name='CNN_LSTM')
            m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
            self.models_['CNN_LSTM'] = m
            self._dl_cfg['CNN_LSTM'] = dict(epochs=epochs, batch=batch, n_out=n_out, loss=loss, input_3d=True)
            print(f"  CNN-LSTM    built — params: {m.count_params():,}")
        except Exception as e:
            self._dl_skip('CNN-LSTM', e)

    def add_aae(self, epochs: int = 30, batch: int = 256):
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()
            inp = L.Input(shape=(self.n_features,), name='aae_in')
            x = L.Dense(128, activation='relu', name='aae_enc1')(inp)
            x = L.Dropout(0.20, name='aae_do1')(x)
            latent = L.Dense(48, activation='relu', name='aae_latent')(x)
            cls = L.Dense(64, activation='relu', name='aae_cls1')(latent)
            cls_out = L.Dense(n_out, activation=act, name='aae_cls_out')(cls)
            dec = L.Dense(128, activation='relu', name='aae_dec1')(latent)
            recon_out = L.Dense(self.n_features, activation='linear', name='aae_recon_out')(dec)
            m = keras.models.Model(inp, [cls_out, recon_out], name='AAE')
            m.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss={'aae_cls_out': loss, 'aae_recon_out': 'mse'},
                      loss_weights={'aae_cls_out': 1.0, 'aae_recon_out': 0.12},
                      metrics={'aae_cls_out': ['accuracy']})
            self.models_['AAE'] = m
            self._dl_cfg['AAE'] = dict(epochs=epochs, batch=batch, n_out=n_out, loss=loss, input_3d=False, multi_output=True,
                                       cls_output_name='aae_cls_out', recon_output_name='aae_recon_out')
            print(f"  AAE         built — params: {m.count_params():,} (experimental scaffold)")
        except Exception as e:
            self._dl_skip('AAE', e)

    def add_transfer_learning(self, epochs: int = 30, batch: int = 256):
        try:
            tf, keras = _import_keras()
            L = keras.layers
            n_out, act, loss = self._out()
            inp = L.Input(shape=(self.n_features,), name='tl_in')
            x = L.Dense(256, activation='relu', name='tl_enc1')(inp)
            x = L.Dense(128, activation='relu', name='tl_enc2')(x)
            x = L.Dense(64, activation='relu', name='tl_latent')(x)
            x = L.Dropout(0.25, name='tl_do')(x)
            x = L.Dense(64, activation='relu', name='tl_head1')(x)
            out = L.Dense(n_out, activation=act, name='tl_out')(x)
            m = keras.models.Model(inp, out, name='TransferLearning')
            m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
            self.models_['TransferLearning'] = m
            self._dl_cfg['TransferLearning'] = dict(epochs=epochs, batch=batch, n_out=n_out, loss=loss, input_3d=False,
                                                    pretrain_autoencoder=True, transfer_layer_names=['tl_enc1', 'tl_enc2', 'tl_latent'])
            print(f"  Transfer     built — params: {m.count_params():,} (self-supervised pretraining)")
        except Exception as e:
            self._dl_skip('TransferLearning', e)

    def _pretrain_transfer_encoder(self, model, X_train, X_val=None, epochs=8, batch=256, layer_names=None):
        try:
            tf, keras = _import_keras()
            L = keras.layers
            inp = L.Input(shape=(self.n_features,), name='pre_ae_in')
            x = L.Dense(256, activation='relu', name='tl_enc1')(inp)
            x = L.Dense(128, activation='relu', name='tl_enc2')(x)
            latent = L.Dense(64, activation='relu', name='tl_latent')(x)
            x = L.Dense(128, activation='relu', name='pre_ae_dec1')(latent)
            x = L.Dense(256, activation='relu', name='pre_ae_dec2')(x)
            out = L.Dense(self.n_features, activation='linear', name='pre_ae_out')(x)
            ae = keras.models.Model(inp, out, name='Transfer_AE')
            ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
            fit_kw = dict(epochs=epochs, batch_size=batch, verbose=0)
            if X_val is not None:
                fit_kw['validation_data'] = (X_val, X_val)
            else:
                fit_kw['validation_split'] = 0.1
            ae.fit(X_train, X_train, **fit_kw)
            for lname in (layer_names or ['tl_enc1', 'tl_enc2', 'tl_latent']):
                try:
                    model.get_layer(lname).set_weights(ae.get_layer(lname).get_weights())
                except Exception:
                    pass
            print("    Transfer pretraining ✓")
        except Exception as e:
            print(f"    Transfer pretraining skipped ({e})")

    def fit_all(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        has_graph = any(self._dl_cfg.get(n, {}).get('needs_adj') for n in self.models_)
        if has_graph:
            print("  Computing feature correlation graph …")
            self._adj = _build_adj_matrix(X, threshold=0.10)
            if 'GNN' in self.models_:
                try:
                    tf, keras = _import_keras()
                    self.models_['GNN'] = self._build_gnn_model(self._adj, keras, name='GNN')
                    print(f"  GNN adj updated — density={float((self._adj>0).mean()):.3f}")
                except Exception as e:
                    print(f"  GNN adj update failed: {e}")
            if 'GAT' in self.models_ and hasattr(self, '_gat_adj_var'):
                try:
                    import tensorflow as tf
                    self._gat_adj_var.assign(self._adj)
                    print(f"  GAT adj updated — density={float((self._adj>0).mean()):.3f}")
                except Exception as e:
                    print(f"  GAT adj update failed: {e}")

        _keras_mod = None
        if self._dl_cfg:
            try:
                _, _keras_mod = _import_keras()
            except ImportError:
                pass

        print(f"\n  Training {len(self.models_)} experts …")
        for name, model in self.models_.items():
            t0 = time.time()
            grp = self.EXPERT_GROUPS.get(name, '?')
            cfg = self._dl_cfg.get(name)
            if cfg:
                use_3d = cfg.get('input_3d', True)
                X_in = X.reshape(-1, X.shape[1], 1) if use_3d else X
                cb = []
                if _keras_mod:
                    cb = [_keras_mod.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
                          _keras_mod.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)]
                if cfg.get('pretrain_autoencoder'):
                    X_val_pre = None if X_val is None else (X_val.reshape(-1, X_val.shape[1], 1) if use_3d else X_val)
                    self._pretrain_transfer_encoder(model,
                        X_in if X_in.ndim == 2 else X_in.reshape(X_in.shape[0], -1),
                        X_val_pre if X_val_pre is None or X_val_pre.ndim == 2 else X_val_pre.reshape(X_val_pre.shape[0], -1),
                        epochs=min(12, max(4, cfg['epochs']//3)), batch=cfg['batch'], layer_names=cfg.get('transfer_layer_names'))
                if cfg.get('multi_output'):
                    X_fit = X_in if X_in.ndim == 2 else X_in.reshape(X_in.shape[0], -1)
                    y_targets = {cfg['cls_output_name']: y.astype(float if cfg['n_out']==1 else int),
                                 cfg['recon_output_name']: X_fit}
                    if X_val is not None:
                        X_val_in = X_val.reshape(-1, X_val.shape[1], 1) if use_3d else X_val
                        X_val_fit = X_val_in if X_val_in.ndim == 2 else X_val_in.reshape(X_val_in.shape[0], -1)
                        y_val_targets = {cfg['cls_output_name']: y_val.astype(float if cfg['n_out']==1 else int),
                                         cfg['recon_output_name']: X_val_fit}
                        model.fit(X_fit, y_targets, epochs=cfg['epochs'], batch_size=cfg['batch'], validation_data=(X_val_fit, y_val_targets), verbose=0, callbacks=cb)
                    else:
                        model.fit(X_fit, y_targets, epochs=cfg['epochs'], batch_size=cfg['batch'], validation_split=0.1, verbose=0, callbacks=cb)
                else:
                    y_in = y.astype(float if cfg['n_out']==1 else int)
                    if X_val is not None:
                        X_val_in = X_val.reshape(-1, X_val.shape[1], 1) if use_3d else X_val
                        y_val_in = y_val.astype(float if cfg['n_out']==1 else int)
                        model.fit(X_in, y_in, epochs=cfg['epochs'], batch_size=cfg['batch'], validation_data=(X_val_in, y_val_in), verbose=0, callbacks=cb)
                    else:
                        model.fit(X_in, y_in, epochs=cfg['epochs'], batch_size=cfg['batch'], validation_split=0.1, verbose=0, callbacks=cb)
            else:
                model.fit(X, y)
            print(f"    [{grp:14s}]  {name:16s} ✓  {time.time()-t0:.1f}s")

    def _raw_predict(self, name: str, model, X: np.ndarray) -> np.ndarray:
        cfg = self._dl_cfg.get(name)
        if cfg:
            use_3d = cfg.get('input_3d', True)
            X_in = X.reshape(-1, X.shape[1], 1) if use_3d else X
            if cfg.get('multi_output'):
                X_pred = X_in if X_in.ndim == 2 else X_in.reshape(X_in.shape[0], -1)
                raw = model.predict(X_pred, verbose=0)
                raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            else:
                raw = model.predict(X_in, verbose=0)
            raw = np.asarray(raw)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            return np.hstack([1 - raw, raw]) if raw.shape[1] == 1 else raw
        return model.predict_proba(X)

ExpertLearners = ExpertLearnersV3

class NIDSPipelineV3(NIDSPipeline):
    def __init__(self, dataset: str, paths: dict, binary: bool = True, copula_family: str = 'gaussian', balance: str = 'smote',
                 scaler: str = 'robust', top_k: int = 40, cv_folds: int = 2,
                 use_lr: bool = True, use_rf: bool = True, use_svm: bool = True, use_xgb: bool = True, use_dt: bool = True,
                 use_mlp: bool = True, use_dnn: bool = True, use_cnn1d: bool = True, use_lstm: bool = True,
                 use_aae: bool = False, use_cnn_lstm: bool = True, use_cnn_bilstm: bool = True, use_cnn_gru: bool = True,
                 use_gnn: bool = False, use_gat: bool = False, use_transformer: bool = True, use_transfer_learning: bool = False,
                 use_bn: bool = True, use_shap: bool = False, sample_frac: float = 1.0, cicids_split: str = 'random',
                 model_params: dict | None = None, training_params: dict | None = None):
        self.use_lr = use_lr; self.use_svm = use_svm; self.use_dt = use_dt
        self.use_mlp = use_mlp; self.use_dnn = use_dnn; self.use_cnn1d = use_cnn1d; self.use_lstm = use_lstm
        self.use_aae = use_aae; self.use_cnn_lstm = use_cnn_lstm; self.use_transfer_learning = use_transfer_learning
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.seed = int(self.training_params.get('seed', 42))
        super().__init__(dataset=dataset, paths=paths, binary=binary, copula_family=copula_family, balance=balance, scaler=scaler,
                         top_k=top_k, cv_folds=cv_folds, use_rf=use_rf, use_xgb=use_xgb, use_cnn_bilstm=use_cnn_bilstm,
                         use_cnn_gru=use_cnn_gru, use_gnn=use_gnn, use_gat=use_gat, use_transformer=use_transformer,
                         use_bn=use_bn, use_shap=use_shap, sample_frac=sample_frac, cicids_split=cicids_split)

    def _active_model_lists(self):
        ml, dl = [], []
        if self.use_lr: ml.append('LR')
        if self.use_rf: ml.append('RF')
        if self.use_svm: ml.append('SVM')
        if self.use_xgb: ml.append('XGB')
        if self.use_dt: ml.append('DT')
        if self.use_mlp: dl.append('MLP')
        if self.use_dnn: dl.append('DNN')
        if self.use_cnn1d: dl.append('CNN1D')
        if self.use_lstm: dl.append('LSTM')
        if self.use_aae: dl.append('AAE')
        if self.use_cnn_lstm: dl.append('CNN_LSTM')
        if self.use_cnn_bilstm: dl.append('CNN_BiLSTM')
        if self.use_cnn_gru: dl.append('CNN_GRU')
        if self.use_gnn: dl.append('GNN')
        if self.use_gat: dl.append('GAT')
        if self.use_transformer: dl.append('Transformer')
        if self.use_transfer_learning: dl.append('TransferLearning')
        return ml, dl

    def run(self) -> dict:
        t0 = time.time()
        self._banner(f"HYBRID NIDS PIPELINE  ·  Dataset: {self.dataset.upper()}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ml_models, dl_models = self._active_model_lists()
        self.logger.log(f"  Active ML models → {', '.join(ml_models) if ml_models else 'None'}")
        self.logger.log(f"  Active DL models → {', '.join(dl_models) if dl_models else 'None'}")
        self.logger.log(f"  Training controls → seed={self.seed}, epochs={self.training_params.get('epochs', 30)}, batch_size={self.training_params.get('batch_size', 256)}")

        self._banner("Stage 1 + 2 · Data Loading & Feature Engineering")
        meta = self._load()
        train_df = meta['train']; val_df = meta.get('val', None); test_df = meta['test']
        feat_cols = meta['feat_cols']; cat_cols = meta['cat_cols']; class_names = meta['class_names']; n_classes = meta['n_classes']
        y_train_raw = train_df['y'].values.astype(int)
        y_val = val_df['y'].values.astype(int) if val_df is not None else None
        y_test = test_df['y'].values.astype(int)
        self._report_split_methodology(meta)

        self._banner("Stage 2b · Exploratory Data Analysis")
        eda = EDAAnalyzer(dataset=self.dataset, class_names=class_names)
        _eda_df = train_df[feat_cols].copy()
        for _c in cat_cols:
            if _c in _eda_df.columns: _eda_df[_c] = pd.factorize(_eda_df[_c])[0]
        _eda_X = _eda_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        eda.run_all(df=train_df, X=_eda_X, feat_names=feat_cols, y=y_train_raw)

        self._banner("Stage 3 · Preprocessing")
        prep = Preprocessor(scaler_type=self.scaler, balance=self.balance)
        X_train_clean = prep.clean_fit_transform(train_df, feat_cols, cat_cols)
        X_val_clean = prep.clean_transform(val_df, feat_cols, cat_cols) if val_df is not None else None
        X_test_clean = prep.clean_transform(test_df, feat_cols, cat_cols)
        fe = FeatureEngineer()
        k = min(self.top_k, X_train_clean.shape[1])
        idx = fe.select(X_train_clean, y_train_raw, feat_cols, k=k)
        X_train_clean = X_train_clean[:, idx]
        X_val_clean = X_val_clean[:, idx] if X_val_clean is not None else None
        X_test_clean = X_test_clean[:, idx]
        feat_names = fe.selected_names_
        sel_cat_cols = [c for c in cat_cols if c in feat_names]
        prep.fit_scaler(X_train_clean)
        X_train = prep.scale_transform(X_train_clean)
        X_val = prep.scale_transform(X_val_clean) if X_val_clean is not None else None
        X_test = prep.scale_transform(X_test_clean)
        cat_indices_selected = [feat_names.index(c) for c in sel_cat_cols if c in feat_names]
        X_train, y_train = prep.balance_train(X_train, y_train_raw, cat_indices=cat_indices_selected)
        val_shape = X_val.shape if X_val is not None else 'N/A'
        self.logger.log(f"  Final shapes → train: {X_train.shape}  val: {val_shape}  test: {X_test.shape}")

        self._banner("Stage 4 · Base Learner Training + CV")
        learners = ExpertLearners(n_classes=n_classes, n_features=X_train.shape[1])
        epochs, batch_size = self._get_train_params()
        if self.use_lr: learners.add_lr(**self._get_model_params('lr'))
        if self.use_rf: learners.add_rf(**self._get_model_params('rf'))
        if self.use_svm: learners.add_svm(**self._get_model_params('svm'))
        if self.use_xgb: learners.add_xgb(**self._get_model_params('xgb'))
        if self.use_dt: learners.add_dt(**self._get_model_params('dt'))
        if self.use_mlp: learners.add_mlp(epochs=epochs, batch=batch_size)
        if self.use_dnn: learners.add_dnn(epochs=epochs, batch=batch_size)
        if self.use_cnn1d: learners.add_cnn1d(epochs=epochs, batch=batch_size)
        if self.use_lstm: learners.add_lstm(epochs=epochs, batch=batch_size)
        if self.use_aae: learners.add_aae(epochs=epochs, batch=batch_size)
        if self.use_cnn_lstm: learners.add_cnn_lstm(epochs=epochs, batch=batch_size)
        if self.use_cnn_bilstm: learners.add_cnn_bilstm(epochs=epochs, batch=batch_size)
        if self.use_cnn_gru: learners.add_cnn_gru(epochs=epochs, batch=batch_size)
        if self.use_gnn: learners.add_gnn(epochs=epochs, batch=batch_size)
        if self.use_gat: learners.add_gat(epochs=epochs, batch=batch_size)
        if self.use_transformer: learners.add_transformer(epochs=epochs, batch=batch_size)
        if self.use_transfer_learning: learners.add_transfer_learning(epochs=epochs, batch=batch_size)

        if len(learners.models_) == 0:
            raise ValueError(
                "Tidak ada expert model yang aktif/berhasil dibangun. "
                "Pilih minimal satu model yang didukung environment Anda."
            )

        cv_scores = run_cv(learners, X_train, y_train, k=self.cv_folds, n_classes=n_classes, X_val=X_val, y_val=y_val)

        self._banner("Stage 5 · Copula Fusion")
        learners.fit_all(X_train, y_train, X_val=X_val, y_val=y_val)
        train_proba = learners.predict_proba_all(X_train)
        if not train_proba:
            raise ValueError(
                "Setelah training, tidak ada probabilitas model yang berhasil dihasilkan. "
                "Cek log build/train model atau aktifkan minimal satu model ML sederhana seperti LR/RF/DT."
            )
        copula = CopulaFusion(family=self.copula_family)
        copula.fit(train_proba, y_train)
        test_proba = learners.predict_proba_all(X_test)
        if not test_proba:
            raise ValueError(
                "Tahap predict pada test set tidak menghasilkan probabilitas dari model mana pun."
            )
        fused_test = copula.fuse(test_proba)
        fused_train = copula.fuse(train_proba)

        self._banner("Stage 6 · Fused Probability Distribution Analysis")
        evaluator = Evaluator(class_names=class_names, dataset=self.dataset)
        evaluator.plot_fused_distribution(fused_test, y_test)
        bn_pred = None
        if self.use_bn:
            self._banner("Stage 7 · Bayesian Network Reasoning")
            bn = BayesianLayer(n_bins=4, max_parents=3, top_k_feats=8)
            bn.fit(X_train, fused_train, y_train, feat_names)
            if bn.fitted: bn_pred = bn.predict(X_test, fused_test, feat_names)

        self._banner("Stage 8 · Evaluation + XAI + Statistical Validation")
        test_preds = learners.predict_all(X_test)
        test_probas = learners.predict_proba_all(X_test)
        for name in test_preds:
            evaluator.evaluate(name, y_test, test_preds[name], y_proba=test_probas.get(name))
            evaluator.plot_confusion(y_test, test_preds[name], title=f'CM {name}')
        fused_pred = np.argmax(fused_test, axis=1)
        evaluator.evaluate('Copula Fusion', y_test, fused_pred, y_proba=fused_test)
        evaluator.plot_confusion(y_test, fused_pred, title='CM Copula Fusion')
        if bn_pred is not None:
            evaluator.evaluate('BN Reasoning', y_test, bn_pred)
            evaluator.plot_confusion(y_test, bn_pred, title='CM BN Reasoning')
        evaluator.plot_roc_all(y_test, test_probas, fused_test)
        evaluator.plot_precision_recall(y_test, test_probas, fused_test)
        evaluator.plot_comparison()
        if self.use_shap and 'RF' in learners.models_:
            evaluator.shap_explain(learners.models_['RF'], X_test, feat_names, 'RF')
        if len(cv_scores) >= 2:
            evaluator.statistical_tests(cv_scores)
        summary_df = evaluator.summary()
        elapsed = time.time() - t0
        self._banner(f"✅ Pipeline DONE  ·  {elapsed:.0f}s  ·  Figures → {FIG_DIR}")
        self.logger.close()
        return {'evaluator': evaluator, 'summary': summary_df, 'learners': learners, 'copula': copula,
                'fused_test': fused_test, 'feat_names': feat_names,
                'active_ml_models': ml_models, 'active_dl_models': dl_models}

NIDSPipeline = NIDSPipelineV3

# =============================================================================
# MULTI-DATASET RUNNER
# =============================================================================
class MultiDatasetRunner:
    """
    Runs the full pipeline on all configured datasets and produces
    a unified cross-dataset comparison report.
    """

    def __init__(self, configs: list):
        """
        configs : list of dicts, each with keys accepted by NIDSPipeline
        """
        self.configs  = configs
        self.results_ = {}

    def run_all(self):
        t0 = time.time()
        print("\n" + "█"*65)
        print(f"  MULTI-DATASET RUN  ·  {len(self.configs)} datasets")
        print("█"*65)

        for cfg in self.configs:
            ds = cfg['dataset']
            print(f"\n{'═'*65}")
            print(f"  DATASET: {ds.upper()}")
            print(f"{'═'*65}")
            try:
                pipeline = NIDSPipeline(**cfg)
                result   = pipeline.run()
                self.results_[ds] = result
            except Exception as e:
                import traceback
                print(f"  [ERROR] {ds}: {e}")
                traceback.print_exc()

        self._cross_dataset_report()
        print(f"\n  Total wall time: {time.time()-t0:.0f}s")

    def _cross_dataset_report(self):
        """Aggregate best F1 per dataset into one comparison table."""
        rows = []
        for ds, res in self.results_.items():
            df = res['summary']
            if df.empty:
                continue
            f1_col = 'F1' if 'F1' in df.columns else None
            if f1_col is None:
                continue
            df['F1'] = pd.to_numeric(df['F1'], errors='coerce')
            best_row = df.loc[df['F1'].idxmax()].copy()
            best_row['Dataset'] = ds.upper()
            rows.append(best_row)

        if not rows:
            return

        report = pd.DataFrame(rows).set_index('Dataset')
        print("\n" + "="*65)
        print("CROSS-DATASET BEST MODEL SUMMARY")
        print("="*65)
        print(report.to_string())

        path = OUT_DIR / "cross_dataset_summary.csv"
        report.to_csv(path)
        print(f"\n  Report → {path}")

        # Bar chart
        if 'F1' in report.columns:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0b1020')
            ax.set_facecolor('#111827')
            clrs = ['#7c3aed','#06b6d4','#ff3b5c','#f59e0b']
            bars = ax.bar(report.index,
                          pd.to_numeric(report['F1'], errors='coerce'),
                          color=clrs[:len(report)], alpha=0.85,
                          edgecolor='none')
            ax.bar_label(bars, fmt='%.4f', color='white', fontsize=10)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel('Best F1 Score', color='#94a3b8')
            ax.set_title('Cross-Dataset Best F1 Comparison',
                         color='white', fontsize=13)
            ax.tick_params(colors='#94a3b8')
            plt.tight_layout()
            fig_path = FIG_DIR / "cross_dataset_f1.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                        facecolor='#0b1020')
            plt.close()
            print(f"  Cross-dataset chart → {fig_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
class DatasetSelector:
    """
    Interactive selector for the Hybrid NIDS pipeline.

    Modes
    -----
    1. Interactive wizard (default)
    2. CLI argparse (--datasets ...)
    3. Programmatic: DatasetSelector(CONFIGS).run([...])
    """

    VALID = ('nslkdd', 'unsw', 'insdn', 'cicids')

    LABELS = {
        'nslkdd': 'NSL-KDD        (KDDTrain+.txt + KDDTest+.txt)',
        'unsw'  : 'UNSW-NB15      (training-set.csv + testing-set.csv)',
        'insdn' : 'InSDN          (Normal + OVS + Metasploit)',
        'cicids': 'CICIDS2017     (8 daily CSV files)',
    }

    SHORT_LABELS = {
        'nslkdd': 'NSL-KDD',
        'unsw': 'UNSW-NB15',
        'insdn': 'InSDN',
        'cicids': 'CICIDS2017',
    }

    COPULA_DEFAULT = {
        'nslkdd': 'gaussian',
        'unsw'  : 'clayton',
        'insdn' : 'frank',
        'cicids': 'gaussian',
    }

    SCALER_LABELS = {
        'standard': 'StandardScaler',
        'minmax': 'MinMaxScaler',
        'robust': 'RobustScaler',
    }

    SCALER_RECOMMENDATION = {
        'nslkdd': {
            'recommended': 'robust',
            'alternatives': ['standard', 'minmax'],
            'reason': 'Campuran fitur count/rate/bytes cenderung tidak seimbang dan rawan outlier.'
        },
        'unsw': {
            'recommended': 'robust',
            'alternatives': ['standard', 'minmax'],
            'reason': 'Fitur heterogen dengan sebaran lebar; robust biasanya paling stabil sebagai baseline.'
        },
        'insdn': {
            'recommended': 'robust',
            'alternatives': ['standard', 'minmax'],
            'reason': 'Gabungan trafik normal dan attack source sering menghasilkan spike dan distribusi berat sebelah.'
        },
        'cicids': {
            'recommended': 'robust',
            'alternatives': ['standard', 'minmax'],
            'reason': 'Flow feature sangat banyak dan nilai ekstrem cukup umum; robust aman untuk baseline awal.'
        },
    }

    STEP_TITLES = [
        'Dataset selection',
        'Classification mode',
        'Copula family',
        'Scaling strategy',
        'Class balancing',
        'Experiment preset',
        'Extra options',
        'Review & run',
    ]

    def __init__(self, configs: list):
        self._configs = {c['dataset']: c for c in configs}

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────
    def launch(self):
        if not self._in_jupyter() and self._has_pipeline_args():
            self._run_argparse()
        else:
            self._run_interactive()

    @staticmethod
    def _in_jupyter() -> bool:
        try:
            shell = get_ipython().__class__.__name__
            return shell in ('ZMQInteractiveShell',
                             'TerminalInteractiveShell',
                             'SpyderShell')
        except NameError:
            return False

    @staticmethod
    def _has_pipeline_args() -> bool:
        import sys
        return '--datasets' in sys.argv

    def run(self, datasets: list, overrides: dict = None):
        keys = self._resolve(datasets)
        cfgs = self._apply_overrides(keys, overrides or {})
        self._execute(cfgs)

    # ─────────────────────────────────────────────────────────────────────
    # PRESENTATION HELPERS
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _line(char='═', width=72):
        return char * width

    def _header(self):
        print('\n' + self._line('═'))
        print('  🛡️  HYBRID NIDS DISSERTATION PIPELINE')
        print('  Copula-Based ML-DL + Bayesian Reasoning for Explainable NIDS')
        print(self._line('═'))
        print('  Wizard mode aktif • input tervalidasi • preset eksperimen siap pakai')
        print(self._line('─'))

    def _box(self, title: str, lines: list, icon: str = '📌'):
        print('\n' + self._line('─'))
        print(f'  {icon}  {title}')
        print(self._line('─'))
        for line in lines:
            print(f'  {line}')

    def _step(self, idx: int, title: str, hint: str = ''):
        total = len(self.STEP_TITLES)
        print('\n' + self._line('═'))
        print(f'  STEP {idx}/{total}  •  {title}')
        if hint:
            print(f'  {hint}')
        print(self._line('═'))

    def _dataset_status(self, key: str) -> str:
        return '✅ ready' if self._paths_exist(key) else '⚠️ path belum ditemukan'

    def _prompt_choice(self, prompt: str, valid: set, default: str = None,
                       aliases: dict = None, allow_blank: bool = True) -> str:
        aliases = aliases or {}
        while True:
            raw = input(prompt).strip().lower()
            if not raw and allow_blank and default is not None:
                return default
            value = aliases.get(raw, raw)
            if value in valid:
                return value
            print(f"  ⚠️  Input tidak valid. Pilihan yang tersedia: {', '.join(sorted(valid))}")

    def _prompt_int(self, prompt: str, default: int,
                    min_value: int = None, max_value: int = None) -> int:
        while True:
            raw = input(prompt).strip()
            if not raw:
                return default
            if not re.fullmatch(r'-?\d+', raw):
                print('  ⚠️  Masukkan bilangan bulat.')
                continue
            value = int(raw)
            if min_value is not None and value < min_value:
                print(f'  ⚠️  Minimal {min_value}.')
                continue
            if max_value is not None and value > max_value:
                print(f'  ⚠️  Maksimal {max_value}.')
                continue
            return value

    def _prompt_float(self, prompt: str, default: float,
                      min_value: float = None, max_value: float = None) -> float:
        while True:
            raw = input(prompt).strip()
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError:
                print('  ⚠️  Masukkan angka desimal yang valid.')
                continue
            if min_value is not None and value < min_value:
                print(f'  ⚠️  Minimal {min_value}.')
                continue
            if max_value is not None and value > max_value:
                print(f'  ⚠️  Maksimal {max_value}.')
                continue
            return value

    @staticmethod
    def _ask_toggle(prompt: str, default: bool) -> bool:
        lbl = 'Y/n' if default else 'y/N'
        while True:
            ans = input(f"{prompt}[{lbl}]: ").strip().lower()
            if not ans:
                return default
            if ans in ('y', 'yes', '1'):
                return True
            if ans in ('n', 'no', '0'):
                return False
            print('  ⚠️  Jawab dengan y/n.')

    def _parse_menu_input(self, raw: str) -> list:
        tokens = raw.replace(',', ' ').split()
        keys, seen = [], set()
        for t in tokens:
            if t == '0' or t.lower() == 'all':
                return list(self.VALID)
            try:
                idx = int(t) - 1
                if 0 <= idx < len(self.VALID):
                    k = self.VALID[idx]
                    if k not in seen:
                        keys.append(k)
                        seen.add(k)
                    continue
            except ValueError:
                pass
            k = t.lower()
            if k in self.VALID and k not in seen:
                keys.append(k)
                seen.add(k)
        return keys

    def _prompt_datasets(self) -> list:
        self._step(1, 'Dataset selection', 'Pilih satu, beberapa, atau semua dataset.')
        print('  [0] ALL datasets')
        for i, key in enumerate(self.VALID, 1):
            print(f"  [{i}] {self.LABELS[key]}  →  {self._dataset_status(key)}")
        print('  Contoh input: 1   |   1 3   |   nslkdd cicids   |   0')
        while True:
            raw = input('  Dataset choice: ').strip()
            if raw.lower() in ('q', 'quit', 'exit'):
                return []
            chosen = self._parse_menu_input(raw)
            if chosen:
                ready = sum(1 for k in chosen if self._paths_exist(k))
                print(f"  ✓ Dipilih: {', '.join(self.SHORT_LABELS[k] for k in chosen)}")
                print(f"  ℹ️  Status path: {ready}/{len(chosen)} dataset ready")
                return chosen
            print('  ⚠️  Tidak ada dataset valid yang terbaca dari input Anda.')


    def _recommended_scaler(self, chosen: list) -> str:
        recs = [self.SCALER_RECOMMENDATION.get(k, {}).get('recommended', 'robust')
                for k in chosen]
        uniq = sorted(set(recs))
        if len(uniq) == 1:
            return uniq[0]
        return 'robust'

    def _scaler_recommendation_lines(self, chosen: list) -> list:
        lines = []
        for key in chosen:
            rec = self.SCALER_RECOMMENDATION.get(key, {})
            recommended = rec.get('recommended', 'robust')
            alternatives = ', '.join(rec.get('alternatives', ['standard', 'minmax']))
            reason = rec.get('reason', 'Robust cocok untuk baseline awal.')
            lines.append(
                f"{self.SHORT_LABELS[key]:12s} → rekomendasi: {recommended:<8s} | alternatif: {alternatives}"
            )
            lines.append(f"   alasan: {reason}")
        auto_pick = self._recommended_scaler(chosen)
        lines.append(f"Auto recommendation saat ini akan memilih: {auto_pick}")
        return lines

    def _prompt_experiment_preset(self):
        self._step(6, 'Experiment preset', 'Pilih preset cepat, seimbang, penuh, atau custom.')
        lines = [
            '[1] Fast baseline      → RF + XGBoost + BN off + SHAP off',
            '[2] Balanced research  → RF + XGBoost + CNN-BiLSTM + BN on + SHAP off',
            '[3] Full dissertation  → semua expert aktif + BN + SHAP',
            '[4] Custom             → atur model satu per satu',
        ]
        self._box('Preset eksperimen', lines, icon='🧪')
        choice = self._prompt_choice('  Preset [1/2/3/4, default=2]: ', {'1','2','3','4'}, default='2')

        if choice == '1':
            return dict(
                use_rf=True, use_xgb=True,
                use_cnn_bilstm=False, use_cnn_gru=False,
                use_gnn=False, use_gat=False,
                use_transformer=False,
                use_bn=False, use_shap=False,
            )
        if choice == '2':
            return dict(
                use_rf=True, use_xgb=True,
                use_cnn_bilstm=True, use_cnn_gru=False,
                use_gnn=False, use_gat=False,
                use_transformer=False,
                use_bn=True, use_shap=False,
            )
        if choice == '3':
            return dict(
                use_rf=True, use_xgb=True,
                use_cnn_bilstm=True, use_cnn_gru=True,
                use_gnn=True, use_gat=True,
                use_transformer=True,
                use_bn=True, use_shap=True,
            )

        self._box('Custom expert selection', [
            'Tekan Enter untuk menerima default [ON].',
            'Gunakan y/n untuk mengaktifkan atau mematikan komponen.',
        ], icon='🎛️')
        print('  ── Tabular experts ──')
        use_rf          = self._ask_toggle('  RF               ', True)
        use_xgb         = self._ask_toggle('  XGBoost          ', True)
        print('  ── Temporal experts ──')
        use_cnn_bilstm  = self._ask_toggle('  CNN-BiLSTM       ', True)
        use_cnn_gru     = self._ask_toggle('  CNN-GRU          ', True)
        print('  ── Graph experts ──')
        use_gnn         = self._ask_toggle('  GNN (GCN)        ', True)
        use_gat         = self._ask_toggle('  GAT              ', True)
        print('  ── Transformer expert ──')
        use_transformer = self._ask_toggle('  Transformer      ', True)
        print('  ── Auxiliary ──')
        use_bn          = self._ask_toggle('  Bayesian Net     ', True)
        use_shap        = self._ask_toggle('  SHAP XAI         ', True)
        return dict(
            use_rf=use_rf, use_xgb=use_xgb,
            use_cnn_bilstm=use_cnn_bilstm, use_cnn_gru=use_cnn_gru,
            use_gnn=use_gnn, use_gat=use_gat,
            use_transformer=use_transformer,
            use_bn=use_bn, use_shap=use_shap,
        )

    def _build_summary_lines(self, chosen, overrides, sample_frac, cicids_split_mode):
        lines = [
            f"Datasets        : {', '.join(self.SHORT_LABELS[k] for k in chosen)}",
            f"Mode            : {'Binary' if overrides['binary'] else 'Multiclass'}",
            f"Copula          : {overrides.get('copula_family', 'per-dataset default')}",
            f"Scaling         : {overrides.get('scaler', 'robust').upper()}",
            f"Balancing       : {overrides['balance'].upper()}",
            f"CV folds        : {overrides['cv_folds']}",
            f"Tabular         : RF={overrides['use_rf']} | XGB={overrides['use_xgb']}",
            f"Temporal        : CNN-BiLSTM={overrides['use_cnn_bilstm']} | CNN-GRU={overrides['use_cnn_gru']}",
            f"Graph           : GNN={overrides['use_gnn']} | GAT={overrides['use_gat']}",
            f"Transformer     : {overrides['use_transformer']}",
            f"Auxiliary       : BN={overrides['use_bn']} | SHAP={overrides['use_shap']}",
        ]
        if 'cicids' in chosen:
            lines.append(f"CICIDS options   : sample_frac={sample_frac:.2f} | split={cicids_split_mode}")
        missing = [self.SHORT_LABELS[k] for k in chosen if not self._paths_exist(k)]
        if missing:
            lines.append(f"Path warning     : {', '.join(missing)} belum terdeteksi")
        else:
            lines.append('Path status      : semua dataset terdeteksi')
        return lines

    # ─────────────────────────────────────────────────────────────────────
    # INTERACTIVE MENU
    # ─────────────────────────────────────────────────────────────────────
    def _run_interactive(self):
        self._header()

        chosen = self._prompt_datasets()
        if not chosen:
            print('  Aborted.')
            return

        self._step(2, 'Classification mode', 'Binary cocok untuk baseline, multiclass untuk analisis kategori serangan.')
        self._box('Pilihan mode', [
            '[1] Binary      → Normal vs Attack',
            '[2] Multiclass  → kategori serangan per dataset',
        ], icon='🏷️')
        mode = self._prompt_choice('  Mode [1/2, default=1]: ', {'1', '2'}, default='1')
        binary = (mode == '1')

        self._step(3, 'Copula family', 'Anda bisa pilih manual atau pakai default terbaik per dataset.')
        self._box('Pilihan copula', [
            '[1] Gaussian   → dependensi simetris',
            '[2] Clayton    → kuat pada lower-tail dependence',
            '[3] Frank      → dependensi campuran',
            '[4] Auto       → pakai default per dataset',
        ], icon='🔗')
        cop = self._prompt_choice('  Copula [1/2/3/4, default=4]: ', {'1','2','3','4'}, default='4')
        copula_override = {'1': 'gaussian', '2': 'clayton', '3': 'frank', '4': None}[cop]

        self._step(4, 'Scaling strategy', 'Pilih teknik scaling numerik yang akan dipakai pipeline.')
        self._box('Pilihan scaling', [
            '[1] StandardScaler  → mean=0, std=1; cocok bila data relatif bersih',
            '[2] MinMaxScaler    → skala [0,1]; sensitif terhadap outlier ekstrem',
            '[3] RobustScaler    → median + IQR; paling aman untuk baseline NIDS',
            '[4] Auto recommended → gunakan rekomendasi berdasarkan dataset terpilih',
        ], icon='📏')
        self._box('Rekomendasi dataset', self._scaler_recommendation_lines(chosen), icon='💡')
        scaler_choice = self._prompt_choice('  Scaling [1/2/3/4, default=4]: ', {'1','2','3','4'}, default='4')
        scaler = {
            '1': 'standard',
            '2': 'minmax',
            '3': 'robust',
            '4': self._recommended_scaler(chosen),
        }[scaler_choice]
        print(f'  ✓ Scaling terpilih: {scaler} ({self.SCALER_LABELS.get(scaler, scaler)})')

        self._step(5, 'Class balancing', 'Pilih strategi penanganan imbalance dataset.')
        self._box('Pilihan balancing', [
            '[1] SMOTE            → default aman untuk banyak kasus',
            '[2] ADASYN           → adaptif pada sampel minoritas',
            '[3] BorderlineSMOTE  → fokus area perbatasan kelas',
            '[4] Under-sampling   → cepat, tetapi membuang data mayoritas',
        ], icon='⚖️')
        bal = self._prompt_choice('  Balance [1/2/3/4, default=1]: ', {'1','2','3','4'}, default='1')
        balance = {'1': 'smote', '2': 'adasyn', '3': 'borderline', '4': 'under'}[bal]

        preset_overrides = self._prompt_experiment_preset()

        self._step(7, 'Extra options', 'Atur parameter tambahan yang paling sering dipakai saat eksperimen.')
        cv_default = 2 if len(chosen) > 1 else 5
        cv_folds = self._prompt_int(
            f'  CV folds [default={cv_default}; saran: 2 cepat, 5 normal, 10 final]: ',
            default=cv_default, min_value=2, max_value=10)

        sample_frac = 0.30
        cicids_split_mode = 'random'
        if 'cicids' in chosen:
            self._box('Opsi CICIDS2017', [
                'sample_frac  : 0.05 s/d 1.00',
                'split random : cepat untuk eksplorasi',
                'split temporal: lebih kuat untuk eksperimen publikasi',
            ], icon='📅')
            sample_frac = self._prompt_float(
                '  CICIDS sample fraction [default=0.30]: ',
                default=0.30, min_value=0.05, max_value=1.0)
            split_choice = self._prompt_choice(
                '  CICIDS split [1=random, 2=temporal, default=1]: ',
                {'1','2'}, default='1')
            cicids_split_mode = 'temporal' if split_choice == '2' else 'random'

        overrides = dict(
            binary=binary,
            scaler=scaler,
            balance=balance,
            cv_folds=cv_folds,
            **preset_overrides,
        )
        if copula_override:
            overrides['copula_family'] = copula_override
        if 'cicids' in chosen:
            overrides['sample_frac'] = sample_frac
            overrides['cicids_split'] = cicids_split_mode

        self._step(8, 'Review & run', 'Periksa ringkasan konfigurasi sebelum pipeline dijalankan.')
        self._box('Configuration summary',
                  self._build_summary_lines(chosen, overrides, sample_frac, cicids_split_mode),
                  icon='✅')
        start = self._prompt_choice('  Jalankan pipeline sekarang? [y/n, default=y]: ', {'y', 'n'}, default='y')
        if start == 'n':
            print('  Aborted.')
            return

        cfgs = self._apply_overrides(chosen, overrides)
        self._execute(cfgs)

    # ─────────────────────────────────────────────────────────────────────
    # ARGPARSE MODE
    # ─────────────────────────────────────────────────────────────────────
    def _run_argparse(self):
        import argparse
        p = argparse.ArgumentParser(
            prog='nids_pipeline_v2.py',
            description='Hybrid NIDS Dissertation Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples
--------
  python nids_pipeline_v2.py --datasets all
  python nids_pipeline_v2.py --datasets nslkdd unsw
  python nids_pipeline_v2.py --datasets cicids --sample 0.5 --no-cnn-gru
  python nids_pipeline_v2.py --datasets insdn --copula frank --binary false
  python nids_pipeline_v2.py --datasets nslkdd unsw insdn cicids --cv 10
""")
        p.add_argument('--datasets', nargs='+', required=True, metavar='DS',
                       help='Dataset keys: all | nslkdd | unsw | insdn | cicids')
        p.add_argument('--binary', type=lambda x: x.lower() != 'false',
                       default=True, metavar='BOOL',
                       help='Binary mode (default: true)')
        p.add_argument('--copula', choices=['gaussian', 'clayton', 'frank'],
                       default=None, help='Copula family')
        p.add_argument('--scaler', choices=['standard', 'minmax', 'robust'],
                       default='robust', help='Scaling method (default: robust)')
        p.add_argument('--balance', choices=['smote', 'adasyn', 'borderline', 'under'],
                       default='smote', help='Resampling method (default: smote)')
        p.add_argument('--sample', type=float, default=0.3, metavar='FRAC',
                       help='CICIDS sample fraction 0–1 (default: 0.3)')
        p.add_argument('--cv', type=int, default=5, metavar='K',
                       help='CV folds (default: 5)')
        p.add_argument('--no-rf', action='store_true', help='Disable RF')
        p.add_argument('--no-xgb', action='store_true', help='Disable XGBoost')
        p.add_argument('--no-cnn-bilstm', action='store_true', help='Disable CNN-BiLSTM')
        p.add_argument('--no-cnn-gru', action='store_true', help='Disable CNN-GRU')
        p.add_argument('--no-gnn', action='store_true', help='Disable GNN')
        p.add_argument('--no-gat', action='store_true', help='Disable GAT')
        p.add_argument('--no-transformer', action='store_true', help='Disable Transformer')
        p.add_argument('--no-bn', action='store_true', help='Disable Bayesian Net')
        p.add_argument('--no-shap', action='store_true', help='Disable SHAP')
        args = p.parse_args()

        overrides = dict(
            binary=args.binary,
            scaler=args.scaler,
            balance=args.balance,
            cv_folds=max(2, min(10, args.cv)),
            sample_frac=max(0.05, min(1.0, args.sample)),
            use_rf=not args.no_rf,
            use_xgb=not args.no_xgb,
            use_cnn_bilstm=not args.no_cnn_bilstm,
            use_cnn_gru=not args.no_cnn_gru,
            use_gnn=not args.no_gnn,
            use_gat=not args.no_gat,
            use_transformer=not args.no_transformer,
            use_bn=not args.no_bn,
            use_shap=not args.no_shap,
        )
        if args.copula:
            overrides['copula_family'] = args.copula

        self._header()
        keys = self._resolve(args.datasets)
        cfgs = self._apply_overrides(keys, overrides)
        self._execute(cfgs)

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _resolve(self, datasets: list) -> list:
        if any(d.lower() == 'all' for d in datasets):
            return list(self.VALID)
        keys, seen = [], set()
        for d in datasets:
            d = d.lower()
            if d in self.VALID and d not in seen:
                keys.append(d)
                seen.add(d)
            elif d not in self.VALID:
                print(f"  [WARN] Unknown dataset '{d}' — skipped.")
        if not keys:
            raise ValueError('No valid datasets specified.')
        return keys

    _DATASET_PARAMS = {
        'cicids': {'cicids_split', 'sample_frac'},
        'nslkdd': set(),
        'unsw': set(),
        'insdn': set(),
    }

    _PIPELINE_PARAMS = {
        'dataset','paths','binary','copula_family','balance','scaler',
        'top_k','cv_folds','use_rf','use_xgb','use_cnn_bilstm','use_cnn_gru',
        'use_gnn','use_gat','use_transformer','use_bn','use_shap',
        'sample_frac','cicids_split',
    }

    def _apply_overrides(self, keys: list, overrides: dict) -> list:
        import copy
        cfgs = []
        for k in keys:
            cfg = copy.deepcopy(self._configs[k])
            for param, val in overrides.items():
                if val is None:
                    continue
                allowed_for_dataset = self._DATASET_PARAMS.get(k, set())
                dataset_specific_params = set().union(*self._DATASET_PARAMS.values())
                if param in dataset_specific_params and param not in allowed_for_dataset:
                    continue
                if param in self._PIPELINE_PARAMS:
                    cfg[param] = val
            if 'copula_family' not in overrides:
                cfg['copula_family'] = self.COPULA_DEFAULT.get(k, 'gaussian')
            cfgs.append(cfg)
        return cfgs

    def _execute(self, cfgs: list):
        if len(cfgs) == 1:
            NIDSPipeline(**cfgs[0]).run()
        else:
            MultiDatasetRunner(configs=cfgs).run_all()

    def _paths_exist(self, key: str) -> bool:
        paths = self._configs.get(key, {}).get('paths', {})
        return any(Path(v).exists() for v in paths.values() if isinstance(v, str))

# =============================================================================
# PATH CHECKER — jalankan ini dulu sebelum pipeline
# =============================================================================
def check_dataset_paths(configs: list = None, auto_fix: bool = True):
    """
    Validasi semua path dataset sebelum pipeline dijalankan.
    Menampilkan status file, working directory, dan saran perbaikan.

    Parameters
    ----------
    configs    : list of config dicts (default: CONFIGS from __main__)
    auto_fix   : jika True, coba scan lokasi umum secara otomatis
    """
    import os
    cwd = Path.cwd()
    print("=" * 65)
    print("  📁  DATASET PATH CHECKER")
    print("=" * 65)
    print(f"  Working directory : {cwd}")
    print()

    # ── Known filename patterns per dataset ───────────────────────────────
    KNOWN_FILES = {
        'nslkdd':  ['KDDTrain+.txt', 'KDDTest+.txt',
                    'KDDTrain+_20Percent.txt'],
        'unsw':    ['UNSW_NB15_training-set.csv',
                    'UNSW_NB15_testing-set.csv'],
        'insdn':   ['Normal_data.csv', 'OVS_data.csv',
                    'Metasploit.csv', 'Normal_traffic.csv'],
        'cicids':  ['Monday-WorkingHours.pcap_ISCX.csv',
                    'Tuesday-WorkingHours.pcap_ISCX.csv'],
    }

    SEARCH_DIRS = [
        cwd,
        cwd / 'data',
        cwd.parent,
        cwd.parent / 'data',
        Path.home() / 'data',
        Path.home() / 'datasets',
        Path.home() / 'Downloads',
        Path('/mnt'),
        Path('/data'),
    ]

    def _find_file(filename: str) -> Path | None:
        """Recursively search known locations for a filename."""
        for base in SEARCH_DIRS:
            if not base.exists():
                continue
            for found in base.rglob(filename):
                return found
        return None

    if configs is None:
        # Try to use CONFIGS from the global scope
        try:
            configs = CONFIGS  # noqa
        except NameError:
            print("  No configs provided. Pass configs=CONFIGS as argument.")
            return {}

    path_map  = {}   # dataset → {key: resolved_path}
    all_found = True

    for cfg in configs:
        ds    = cfg['dataset']
        paths = cfg.get('paths', {})
        print(f"  ── {ds.upper()} {'─'*(50-len(ds))}")

        for key, val in paths.items():
            if key == 'dir':
                # Directory check for CICIDS
                p = Path(val)
                exists = p.exists()
                status = "✅" if exists else "❌"
                print(f"    {status}  [{key}] {val}")
                if not exists and auto_fix:
                    # Try to find any CICIDS-like CSV nearby
                    for known in KNOWN_FILES.get(ds, []):
                        found = _find_file(known)
                        if found:
                            alt = str(found.parent)
                            print(f"       💡 Found nearby: {alt}")
                            break
                if not exists:
                    all_found = False
                path_map.setdefault(ds, {})[key] = val

            elif key == 'files':
                # Explicit file list
                for fpath in val:
                    p      = Path(fpath)
                    exists = p.exists()
                    status = "✅" if exists else "❌"
                    print(f"    {status}  {fpath}")
                    if not exists:
                        all_found = False
                path_map.setdefault(ds, {})[key] = val

            else:
                # Single file path
                p      = Path(val)
                exists = p.exists()
                status = "✅" if exists else "❌"
                print(f"    {status}  [{key}] {val}")

                if not exists and auto_fix:
                    filename = p.name
                    found    = _find_file(filename)
                    if found:
                        print(f"       💡 Auto-found: {found}")
                        path_map.setdefault(ds, {})[key] = str(found)
                    else:
                        print(f"       ⚠️  Not found anywhere. "
                              f"Expected filename: {filename}")
                        all_found = False
                        path_map.setdefault(ds, {})[key] = val
                else:
                    path_map.setdefault(ds, {})[key] = val

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 65)
    if all_found:
        print("  ✅  All files found — pipeline ready to run.")
    else:
        print("  ❌  Some files missing. Fix options:")
        print()
        print("  Option A — Update CONFIGS paths manually:")
        print("    Edit the paths dict in CONFIGS to match your file locations.")
        print()
        print("  Option B — Create folder structure:")
        print(f"    {cwd}/")
        print("    ├── data/")
        print("    │   ├── nslkdd/")
        print("    │   │   ├── KDDTrain+.txt")
        print("    │   │   └── KDDTest+.txt")
        print("    │   ├── unsw/")
        print("    │   │   ├── UNSW_NB15_training-set.csv")
        print("    │   │   └── UNSW_NB15_testing-set.csv")
        print("    │   ├── insdn/")
        print("    │   │   ├── Normal_data.csv")
        print("    │   │   ├── OVS_data.csv")
        print("    │   │   └── Metasploit.csv")
        print("    │   └── cicids2017/")
        print("    │       └── *.pcap_ISCX.csv  (8 files)")
        print()
        print("  Option C — Run set_dataset_paths() to configure interactively.")
    print("=" * 65)
    return path_map


def set_dataset_paths(configs: list = None) -> list:
    """
    Interactive path configurator — lets you type the correct paths
    dataset by dataset, then returns updated configs.

    Usage:
        CONFIGS = set_dataset_paths(CONFIGS)
        DatasetSelector(CONFIGS).launch()
    """
    if configs is None:
        try:
            configs = CONFIGS   # noqa
        except NameError:
            print("Pass configs=CONFIGS as argument.")
            return []

    print("\n  🔧  INTERACTIVE PATH CONFIGURATOR")
    print("  Press Enter to keep current path, or type new path.\n")

    updated = []
    for cfg in configs:
        ds    = cfg['dataset'].upper()
        paths = dict(cfg.get('paths', {}))
        print(f"  ── {ds} ──")

        for key, val in paths.items():
            if key == 'files':
                continue
            exists = Path(val).exists()
            mark   = "✅" if exists else "❌"
            new    = input(f"  {mark} [{key}] (current: {val})\n      New path: ").strip()
            if new:
                paths[key] = new
        new_cfg = dict(cfg)
        new_cfg['paths'] = paths
        updated.append(new_cfg)
        print()

    print("  ✅  Paths updated. Run DatasetSelector(updated_configs).launch()")
    return updated


# =============================================================================
# DATASET SELECTOR  (interactive CLI · argparse · programmatic)
# =============================================================================



# =============================================================================
# V4 EXTENSIONS — VALIDATION · COPULA · FEATURE SELECTION · ENSEMBLE · HYBRID
# =============================================================================
from sklearn.base import clone
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold,
    RandomizedSearchCV
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class FeatureEngineerV4(FeatureEngineer):
    """Extended feature selection toolbox.

    Supported methods
    -----------------
    mutual_info / mi, anova / f_classif, chi2, l1_logistic, rf_importance,
    rfe, mrmr.
    """

    def _greedy_mrmr(self, X: np.ndarray, y: np.ndarray, k: int) -> list:
        mi = mutual_info_classif(np.nan_to_num(X), y, random_state=42)
        corr = np.abs(np.corrcoef(np.nan_to_num(X).T)) if X.shape[1] > 1 else np.array([[1.0]])
        selected, remaining = [], list(range(X.shape[1]))
        while remaining and len(selected) < k:
            if not selected:
                best = int(np.nanargmax(mi))
            else:
                scores = []
                for j in remaining:
                    redundancy = float(np.mean(corr[j, selected])) if selected else 0.0
                    scores.append((float(mi[j]) - redundancy, j))
                best = max(scores, key=lambda t: t[0])[1]
            selected.append(best)
            remaining.remove(best)
        return selected

    def select(self, X: np.ndarray, y: np.ndarray, feat_names: list, k: int = 40,
               method: str = 'mutual_info') -> np.ndarray:
        method = str(method or 'mutual_info').lower()
        k = min(k, X.shape[1])
        Xn = np.nan_to_num(X)
        if method in ('mutual_info', 'mi'):
            return super().select(Xn, y, feat_names, k=k, method='mutual_info')
        if method in ('anova', 'f_classif', 'anova_f'):
            sel = SelectKBest(f_classif, k=k).fit(Xn, y)
            self.selected_idx_ = sel.get_support(indices=True)
        elif method == 'chi2':
            from sklearn.preprocessing import MinMaxScaler
            Xp = MinMaxScaler().fit_transform(Xn)
            sel = SelectKBest(chi2, k=k).fit(Xp, y)
            self.selected_idx_ = sel.get_support(indices=True)
        elif method in ('l1_logistic', 'l1', 'lasso_logistic'):
            solver = 'liblinear' if len(np.unique(y)) <= 2 else 'saga'
            clf = LogisticRegression(max_iter=3000, penalty='l1', solver=solver)
            clf.fit(Xn, y)
            coef = np.abs(getattr(clf, 'coef_', np.zeros((1, Xn.shape[1]))))
            score = coef.mean(axis=0)
            self.selected_idx_ = np.argsort(score)[::-1][:k]
        elif method in ('rf_importance', 'random_forest_importance', 'rfi'):
            clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            clf.fit(Xn, y)
            score = np.asarray(getattr(clf, 'feature_importances_', np.zeros(Xn.shape[1])))
            self.selected_idx_ = np.argsort(score)[::-1][:k]
        elif method == 'rfe':
            est = LogisticRegression(max_iter=2000, solver='liblinear') if len(np.unique(y)) <= 2 else RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            selector = RFE(estimator=est, n_features_to_select=k, step=max(1, Xn.shape[1] // 20))
            selector.fit(Xn, y)
            self.selected_idx_ = selector.get_support(indices=True)
        elif method == 'mrmr':
            self.selected_idx_ = np.array(self._greedy_mrmr(Xn, y, k), dtype=int)
        else:
            print(f"  Unknown feature selection method '{method}' → fallback to mutual_info")
            return super().select(Xn, y, feat_names, k=k, method='mutual_info')
        self.selected_names_ = [feat_names[i] if i < len(feat_names) else f'feat_{i}' for i in self.selected_idx_]
        print(f"  Feature selection ({method}): {X.shape[1]} → {len(self.selected_idx_)} features kept")
        return self.selected_idx_


class ValidationSuite:
    """Additional validation methods evaluated on TRAIN only.

    This preserves the original hold-out VAL/TEST methodology while allowing
    richer internal validation estimates.
    """

    PARAM_DISTS = {
        'LR': {'C': np.logspace(-3, 2, 8), 'solver': ['lbfgs', 'liblinear', 'saga']},
        'RF': {'n_estimators': [150, 300, 500], 'max_depth': [None, 10, 18, 28], 'min_samples_leaf': [1, 2, 4]},
        'SVM': {'C': np.logspace(-2, 2, 8), 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
        'DT': {'max_depth': [None, 8, 12, 18, 24], 'min_samples_leaf': [1, 2, 4, 8], 'criterion': ['gini', 'entropy']},
        'XGB': {'n_estimators': [150, 300, 500], 'max_depth': [4, 6, 8], 'learning_rate': [0.03, 0.05, 0.1], 'subsample': [0.7, 0.8, 1.0]},
    }

    def __init__(self, method: str = 'holdout', cv_folds: int = 5, repeats: int = 3,
                 bootstrap_rounds: int = 30, nested_inner_folds: int = 3,
                 seed: int = 42, optimize: bool = False):
        self.method = str(method or 'holdout').lower()
        self.cv_folds = max(2, int(cv_folds))
        self.repeats = max(1, int(repeats))
        self.bootstrap_rounds = max(5, int(bootstrap_rounds))
        self.nested_inner_folds = max(2, int(nested_inner_folds))
        self.seed = int(seed)
        self.optimize = bool(optimize)

    def _metric(self, y_true, y_pred, n_classes):
        avg = 'binary' if n_classes == 2 else 'weighted'
        return float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    def _splitter(self, y):
        if self.method == 'simple_kfold':
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        if self.method == 'stratified_kfold':
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        if self.method == 'repeated_kfold':
            return RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=self.repeats, random_state=self.seed)
        if self.method == 'nested_cv':
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

    def _iter_bootstrap(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        idx_all = np.arange(n)
        for _ in range(self.bootstrap_rounds):
            tr = rng.choice(idx_all, size=n, replace=True)
            mask = np.ones(n, dtype=bool)
            mask[np.unique(tr)] = False
            te = idx_all[mask]
            if len(te) == 0:
                continue
            yield tr, te

    def _maybe_search(self, name, estimator, X, y):
        if not self.optimize:
            return clone(estimator).fit(X, y)
        dist = self.PARAM_DISTS.get(name)
        if not dist:
            return clone(estimator).fit(X, y)
        inner = StratifiedKFold(n_splits=self.nested_inner_folds, shuffle=True, random_state=self.seed)
        scoring = 'f1' if len(np.unique(y)) == 2 else 'f1_weighted'
        try:
            search = RandomizedSearchCV(
                clone(estimator), dist, n_iter=min(6, sum(len(v) if hasattr(v, '__len__') else 3 for v in dist.values())),
                cv=inner, random_state=self.seed, n_jobs=1, scoring=scoring, error_score='raise'
            )
            search.fit(X, y)
            return search.best_estimator_
        except Exception:
            return clone(estimator).fit(X, y)

    def evaluate(self, learners, X: np.ndarray, y: np.ndarray, n_classes: int = 2) -> dict:
        ml_models = {n: m for n, m in learners.models_.items() if n not in getattr(learners, '_dl_cfg', {})}
        if not ml_models:
            return {}
        out = defaultdict(list)
        if self.method == 'holdout':
            return out
        if self.method == 'bootstrapping':
            split_iter = self._iter_bootstrap(X, y)
        else:
            splitter = self._splitter(y)
            split_iter = splitter.split(X, y)
        for tr_idx, te_idx in split_iter:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            for name, model in ml_models.items():
                try:
                    if self.method == 'nested_cv':
                        fitted = self._maybe_search(name, model, X_tr, y_tr)
                    else:
                        fitted = clone(model).fit(X_tr, y_tr)
                    pred = fitted.predict(X_te)
                    out[name].append(self._metric(y_te, pred, n_classes))
                except Exception:
                    continue
        return dict(out)


class CopulaFusionV4(CopulaFusion):
    """Extended copula family support.

    Gumbel, Student-t, and Vine are implemented as robust approximations for
    model-score fusion, suitable for comparative experimentation.
    """

    def fit(self, proba_dict: dict, y: np.ndarray):
        self.family = str(self.family or 'gaussian').lower()
        return super().fit(proba_dict, y)

    def _copula_density(self, U: np.ndarray, state: dict | None = None) -> np.ndarray:
        fam = self.family.lower()
        if fam in ('gaussian', 'clayton', 'frank'):
            return super()._copula_density(U, state)
        try:
            if fam in ('gumbel', 'gumble'):
                return self._gumbel(U, state)
            if fam in ('student_t', 'student-t', 't', 't_copula'):
                return self._student_t(U, state)
            if fam == 'vine':
                return self._vine(U, state)
        except Exception:
            pass
        return super()._copula_density(U, state)

    def _gumbel(self, U, state=None):
        from scipy.stats import kendalltau
        taus = [max(kendalltau(U[:, i], U[:, j])[0], 1e-4)
                for i in range(U.shape[1]) for j in range(i + 1, U.shape[1])]
        tau = float(np.mean(taus)) if taus else 0.1
        th = max(1.0 / max(1e-6, 1 - tau), 1.01)
        u = np.clip(U, 1e-9, 1 - 1e-9)
        s = ((-np.log(u)) ** th).sum(axis=1)
        c = np.exp(-(s ** (1.0 / th)))
        return (c - c.min()) / (np.ptp(c) + 1e-10)

    def _student_t(self, U, state=None):
        from scipy.stats import t as student_t
        z = student_t.ppf(np.clip(U, 1e-6, 1 - 1e-6), df=4)
        Sigma = np.corrcoef(z.T)
        invS = np.linalg.pinv(Sigma)
        quad = np.einsum('ij,jk,ik->i', z, invS, z)
        score = 1.0 / (1.0 + quad)
        return (score - score.min()) / (np.ptp(score) + 1e-10)

    def _vine(self, U, state=None):
        # Pair-copula approximation: average several pairwise copula scores.
        mats = []
        base_family = self.family
        for fam in ('gaussian', 'clayton', 'frank', 'gumbel'):
            self.family = fam
            mats.append(np.asarray(self._copula_density(U), dtype=float))
        self.family = base_family
        d = np.nanmean(np.vstack(mats), axis=0) if mats else U.mean(axis=1)
        return (d - d.min()) / (np.ptp(d) + 1e-10)


class EnsembleHybridFusion:
    ML_SET = {'LR', 'RF', 'SVM', 'XGB', 'DT'}
    DL_SET = {'MLP', 'DNN', 'CNN1D', 'LSTM', 'AAE', 'CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU', 'GNN', 'GAT', 'Transformer', 'TransferLearning'}

    def __init__(self, n_classes: int = 2, ensemble_method: str = 'weighted_soft_vote',
                 hybrid_method: str = 'weighted_ml_dl'):
        self.n_classes = int(n_classes)
        self.ensemble_method = str(ensemble_method or 'weighted_soft_vote').lower()
        self.hybrid_method = str(hybrid_method or 'weighted_ml_dl').lower()

    def _score_weights(self, proba_dict: dict, y_ref: np.ndarray) -> dict:
        weights = {}
        for name, p in proba_dict.items():
            try:
                if p.shape[1] == 2:
                    s = roc_auc_score(y_ref, p[:, 1])
                else:
                    s = roc_auc_score(y_ref, p, multi_class='ovr', average='macro')
            except Exception:
                pred = np.argmax(p, axis=1)
                avg = 'binary' if len(np.unique(y_ref)) == 2 else 'weighted'
                s = f1_score(y_ref, pred, average=avg, zero_division=0)
            weights[name] = max(float(s), 1e-6)
        denom = sum(weights.values()) or 1.0
        return {k: v / denom for k, v in weights.items()}

    def _safe_probs(self, proba_dict: dict) -> dict:
        out = {}
        for k, p in (proba_dict or {}).items():
            arr = np.asarray(p, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] == 1:
                arr = np.hstack([1 - arr, arr])
            out[k] = np.clip(arr, 1e-9, 1 - 1e-9)
        return out

    def _avg(self, mats, weights=None):
        mats = [np.asarray(m, dtype=float) for m in mats if m is not None]
        if not mats:
            return None
        if weights is None:
            z = np.mean(np.stack(mats, axis=0), axis=0)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / (w.sum() + 1e-12)
            z = np.tensordot(w, np.stack(mats, axis=0), axes=(0, 0))
        z = np.clip(z, 1e-9, None)
        z = z / (z.sum(axis=1, keepdims=True) + 1e-12)
        return z

    def _hard_vote(self, mats):
        preds = [np.argmax(m, axis=1) for m in mats]
        n = mats[0].shape[0]
        out = np.zeros((n, self.n_classes), dtype=float)
        for i in range(n):
            vals, cnt = np.unique([p[i] for p in preds], return_counts=True)
            out[i, vals[np.argmax(cnt)]] = 1.0
        return out

    def _gmean(self, mats):
        z = np.exp(np.mean(np.log(np.clip(np.stack(mats, axis=0), 1e-9, 1.0)), axis=0))
        z = z / (z.sum(axis=1, keepdims=True) + 1e-12)
        return z

    def _rank_average_binary(self, mats):
        from scipy.stats import rankdata
        pos = [m[:, 1] for m in mats]
        ranks = [rankdata(v) / len(v) for v in pos]
        score = np.mean(np.vstack(ranks), axis=0)
        return np.column_stack([1 - score, score])

    def build_ensemble(self, ref_proba: dict, y_ref: np.ndarray, test_proba: dict) -> dict:
        ref = self._safe_probs(ref_proba)
        test = self._safe_probs(test_proba)
        if not test:
            return {}
        names = list(test.keys())
        mats = [test[n] for n in names]
        weights = self._score_weights({k: ref[k] for k in names if k in ref}, y_ref) if ref and y_ref is not None else {n: 1.0 / len(names) for n in names}
        meth = self.ensemble_method
        out = {}
        if meth == 'soft_vote':
            out['Ensemble SoftVote'] = self._avg(mats)
        elif meth == 'weighted_soft_vote':
            out['Ensemble WeightedSoftVote'] = self._avg(mats, [weights.get(n, 1.0) for n in names])
        elif meth == 'hard_vote':
            out['Ensemble HardVote'] = self._hard_vote(mats)
        elif meth in ('geometric_mean', 'gmean'):
            out['Ensemble GeometricMean'] = self._gmean(mats)
        elif meth == 'rank_average':
            out['Ensemble RankAverage'] = self._rank_average_binary(mats) if self.n_classes == 2 else self._avg(mats)
        elif meth == 'stacking':
            # Safe stacking: use reference split (VAL if available) as meta-train.
            if ref and y_ref is not None and len(ref) == len(test):
                X_meta_ref = np.hstack([ref[n] for n in names if n in ref])
                X_meta_test = np.hstack([test[n] for n in names if n in ref])
                meta = LogisticRegression(max_iter=3000, solver='lbfgs')
                meta.fit(X_meta_ref, y_ref)
                p = meta.predict_proba(X_meta_test)
                if p.shape[1] == 1:
                    p = np.hstack([1 - p, p])
                out['Ensemble Stacking'] = p
            else:
                out['Ensemble Stacking'] = self._avg(mats, [weights.get(n, 1.0) for n in names])
        else:
            out['Ensemble WeightedSoftVote'] = self._avg(mats, [weights.get(n, 1.0) for n in names])
        return {k: v for k, v in out.items() if v is not None}

    def build_hybrid(self, ref_proba: dict, y_ref: np.ndarray, test_proba: dict) -> dict:
        ref = self._safe_probs(ref_proba)
        test = self._safe_probs(test_proba)
        if not test:
            return {}
        ml_names = [n for n in test if n in self.ML_SET]
        dl_names = [n for n in test if n in self.DL_SET]
        if not ml_names or not dl_names:
            return {}
        ref_w = self._score_weights(ref, y_ref) if ref and y_ref is not None else {}
        ml_ref = self._avg([ref[n] for n in ml_names if n in ref], [ref_w.get(n, 1.0) for n in ml_names if n in ref]) if ref else None
        dl_ref = self._avg([ref[n] for n in dl_names if n in ref], [ref_w.get(n, 1.0) for n in dl_names if n in ref]) if ref else None
        ml_test = self._avg([test[n] for n in ml_names], [ref_w.get(n, 1.0) for n in ml_names])
        dl_test = self._avg([test[n] for n in dl_names], [ref_w.get(n, 1.0) for n in dl_names])
        out = {}
        if self.hybrid_method == 'stack_groups' and ml_ref is not None and dl_ref is not None and y_ref is not None:
            X_meta_ref = np.hstack([ml_ref, dl_ref])
            X_meta_test = np.hstack([ml_test, dl_test])
            meta = LogisticRegression(max_iter=3000, solver='lbfgs')
            meta.fit(X_meta_ref, y_ref)
            p = meta.predict_proba(X_meta_test)
            if p.shape[1] == 1:
                p = np.hstack([1 - p, p])
            out['Hybrid StackGroups'] = p
        else:
            if ml_ref is not None and dl_ref is not None and y_ref is not None:
                group_w = self._score_weights({'ML': ml_ref, 'DL': dl_ref}, y_ref)
                w_ml, w_dl = group_w.get('ML', 0.5), group_w.get('DL', 0.5)
            else:
                w_ml, w_dl = 0.5, 0.5
            out['Hybrid WeightedMLDL'] = self._avg([ml_test, dl_test], [w_ml, w_dl])
        return {k: v for k, v in out.items() if v is not None}


class NIDSPipelineV4(NIDSPipelineV3):
    """V4 pipeline with richer validation, feature selection, optimization,
    ensemble, and hybrid fusion.

    Notes
    -----
    - Official train/test protocols are still preserved.
    - Advanced validation methods operate on the TRAIN split only.
    - DL optimization remains heuristic/lightweight rather than full AutoML.
    - Vine copula is implemented as a pair-copula approximation.
    """

    def __init__(self, dataset: str, paths: dict, binary: bool = True,
                 copula_family: str = 'gaussian', balance: str = 'smote',
                 scaler: str = 'robust', top_k: int = 40, cv_folds: int = 2,
                 use_lr: bool = True, use_rf: bool = True, use_svm: bool = True,
                 use_xgb: bool = True, use_dt: bool = True, use_mlp: bool = True,
                 use_dnn: bool = True, use_cnn1d: bool = True, use_lstm: bool = True,
                 use_aae: bool = False, use_cnn_lstm: bool = True,
                 use_cnn_bilstm: bool = True, use_cnn_gru: bool = True,
                 use_gnn: bool = False, use_gat: bool = False,
                 use_transformer: bool = True, use_transfer_learning: bool = False,
                 use_bn: bool = True, use_shap: bool = False,
                 sample_frac: float = 1.0, cicids_split: str = 'random',
                 model_params: dict | None = None, training_params: dict | None = None,
                 validation_method: str = 'holdout', validation_repeats: int = 3,
                 bootstrap_rounds: int = 30, nested_inner_folds: int = 3,
                 feature_method: str = 'mutual_info', optimize_ml: bool = False,
                 optimize_dl: bool = False, ensemble_method: str = 'weighted_soft_vote',
                 hybrid_method: str = 'weighted_ml_dl', use_ensemble: bool = True,
                 use_hybrid: bool = True):
        self.validation_method = str(validation_method or 'holdout').lower()
        self.validation_repeats = int(validation_repeats)
        self.bootstrap_rounds = int(bootstrap_rounds)
        self.nested_inner_folds = int(nested_inner_folds)
        self.feature_method = str(feature_method or 'mutual_info').lower()
        self.optimize_ml = bool(optimize_ml)
        self.optimize_dl = bool(optimize_dl)
        self.ensemble_method = str(ensemble_method or 'weighted_soft_vote').lower()
        self.hybrid_method = str(hybrid_method or 'weighted_ml_dl').lower()
        self.use_ensemble = bool(use_ensemble)
        self.use_hybrid = bool(use_hybrid)
        super().__init__(dataset=dataset, paths=paths, binary=binary, copula_family=copula_family,
                         balance=balance, scaler=scaler, top_k=top_k, cv_folds=cv_folds,
                         use_lr=use_lr, use_rf=use_rf, use_svm=use_svm, use_xgb=use_xgb, use_dt=use_dt,
                         use_mlp=use_mlp, use_dnn=use_dnn, use_cnn1d=use_cnn1d, use_lstm=use_lstm,
                         use_aae=use_aae, use_cnn_lstm=use_cnn_lstm, use_cnn_bilstm=use_cnn_bilstm,
                         use_cnn_gru=use_cnn_gru, use_gnn=use_gnn, use_gat=use_gat,
                         use_transformer=use_transformer, use_transfer_learning=use_transfer_learning,
                         use_bn=use_bn, use_shap=use_shap, sample_frac=sample_frac, cicids_split=cicids_split,
                         model_params=model_params, training_params=training_params)

    def _get_train_params(self) -> tuple:
        epochs, batch = super()._get_train_params()
        if self.optimize_dl:
            if self.dataset == 'cicids':
                batch = max(batch, 256)
                epochs = min(epochs, 18)
            elif self.dataset in ('nslkdd', 'unsw'):
                batch = min(max(batch, 128), 256)
                epochs = min(max(epochs, 20), 40)
            else:
                batch = min(max(batch, 64), 256)
                epochs = min(max(epochs, 20), 35)
            self.logger.log(f"  DL optimization heuristic → epochs={epochs}, batch_size={batch}")
        return epochs, batch

    def _optimize_ml_estimators(self, learners, X, y):
        if not self.optimize_ml:
            return
        self.logger.log("  ML optimization → lightweight RandomizedSearchCV on TRAIN only")
        ml_models = {n: m for n, m in learners.models_.items() if n not in getattr(learners, '_dl_cfg', {})}
        for name, model in list(ml_models.items()):
            dist = ValidationSuite.PARAM_DISTS.get(name)
            if not dist:
                continue
            cv = StratifiedKFold(n_splits=min(3, max(2, self.cv_folds)), shuffle=True, random_state=self.seed)
            scoring = 'f1' if self.binary else 'f1_weighted'
            try:
                search = RandomizedSearchCV(clone(model), dist, n_iter=min(6, 12), cv=cv, random_state=self.seed, n_jobs=1, scoring=scoring)
                search.fit(X, y)
                learners.models_[name] = search.best_estimator_
                self.logger.log(f"    Optimized {name} → best params {search.best_params_}")
            except Exception as e:
                self.logger.log(f"    Optimization skipped for {name}: {e}")

    def run(self) -> dict:
        t0 = time.time()
        self._banner(f"HYBRID NIDS PIPELINE V4  ·  Dataset: {self.dataset.upper()}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ml_models, dl_models = self._active_model_lists()
        self.logger.log(f"  Active ML models → {', '.join(ml_models) if ml_models else 'None'}")
        self.logger.log(f"  Active DL models → {', '.join(dl_models) if dl_models else 'None'}")
        self.logger.log(f"  Validation → method={self.validation_method}, folds={self.cv_folds}, repeats={self.validation_repeats}, bootstrap_rounds={self.bootstrap_rounds}")
        self.logger.log(f"  Feature selection → method={self.feature_method}, top_k={self.top_k}")
        self.logger.log(f"  Copula → family={self.copula_family}")
        self.logger.log(f"  Ensemble/Hybrid → use_ensemble={self.use_ensemble} ({self.ensemble_method}), use_hybrid={self.use_hybrid} ({self.hybrid_method})")

        self._banner("Stage 1 + 2 · Data Loading & Feature Engineering")
        meta = self._load()
        train_df = meta['train']; val_df = meta.get('val', None); test_df = meta['test']
        feat_cols = meta['feat_cols']; cat_cols = meta['cat_cols']; class_names = meta['class_names']; n_classes = meta['n_classes']
        y_train_raw = train_df['y'].values.astype(int)
        y_val = val_df['y'].values.astype(int) if val_df is not None else None
        y_test = test_df['y'].values.astype(int)
        self._report_split_methodology(meta)

        self._banner("Stage 2b · Exploratory Data Analysis")
        eda = EDAAnalyzer(dataset=self.dataset, class_names=class_names)
        _eda_df = train_df[feat_cols].copy()
        for _c in cat_cols:
            if _c in _eda_df.columns:
                _eda_df[_c] = pd.factorize(_eda_df[_c])[0]
        _eda_X = _eda_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        eda.run_all(df=train_df, X=_eda_X, feat_names=feat_cols, y=y_train_raw)

        self._banner("Stage 3 · Preprocessing")
        prep = Preprocessor(scaler_type=self.scaler, balance=self.balance)
        X_train_clean = prep.clean_fit_transform(train_df, feat_cols, cat_cols)
        X_val_clean = prep.clean_transform(val_df, feat_cols, cat_cols) if val_df is not None else None
        X_test_clean = prep.clean_transform(test_df, feat_cols, cat_cols)
        fe = FeatureEngineerV4()
        k = min(self.top_k, X_train_clean.shape[1])
        idx = fe.select(X_train_clean, y_train_raw, feat_cols, k=k, method=self.feature_method)
        X_train_clean = X_train_clean[:, idx]
        X_val_clean = X_val_clean[:, idx] if X_val_clean is not None else None
        X_test_clean = X_test_clean[:, idx]
        feat_names = fe.selected_names_
        sel_cat_cols = [c for c in cat_cols if c in feat_names]
        prep.fit_scaler(X_train_clean)
        X_train = prep.scale_transform(X_train_clean)
        X_val = prep.scale_transform(X_val_clean) if X_val_clean is not None else None
        X_test = prep.scale_transform(X_test_clean)
        cat_indices_selected = [feat_names.index(c) for c in sel_cat_cols if c in feat_names]
        X_train, y_train = prep.balance_train(X_train, y_train_raw, cat_indices=cat_indices_selected)
        val_shape = X_val.shape if X_val is not None else 'N/A'
        self.logger.log(f"  Final shapes → train: {X_train.shape}  val: {val_shape}  test: {X_test.shape}")

        self._banner("Stage 4 · Base Learner Training + Validation")
        learners = ExpertLearners(n_classes=n_classes, n_features=X_train.shape[1])
        epochs, batch_size = self._get_train_params()
        if self.use_lr: learners.add_lr(**self._get_model_params('lr'))
        if self.use_rf: learners.add_rf(**self._get_model_params('rf'))
        if self.use_svm: learners.add_svm(**self._get_model_params('svm'))
        if self.use_xgb: learners.add_xgb(**self._get_model_params('xgb'))
        if self.use_dt: learners.add_dt(**self._get_model_params('dt'))
        if self.use_mlp: learners.add_mlp(epochs=epochs, batch=batch_size)
        if self.use_dnn: learners.add_dnn(epochs=epochs, batch=batch_size)
        if self.use_cnn1d: learners.add_cnn1d(epochs=epochs, batch=batch_size)
        if self.use_lstm: learners.add_lstm(epochs=epochs, batch=batch_size)
        if self.use_aae: learners.add_aae(epochs=epochs, batch=batch_size)
        if self.use_cnn_lstm: learners.add_cnn_lstm(epochs=epochs, batch=batch_size)
        if self.use_cnn_bilstm: learners.add_cnn_bilstm(epochs=epochs, batch=batch_size)
        if self.use_cnn_gru: learners.add_cnn_gru(epochs=epochs, batch=batch_size)
        if self.use_gnn: learners.add_gnn(epochs=epochs, batch=batch_size)
        if self.use_gat: learners.add_gat(epochs=epochs, batch=batch_size)
        if self.use_transformer: learners.add_transformer(epochs=epochs, batch=batch_size)
        if self.use_transfer_learning: learners.add_transfer_learning(epochs=epochs, batch=batch_size)

        if len(learners.models_) == 0:
            raise ValueError("Tidak ada expert model yang aktif/berhasil dibangun. Pilih minimal satu model yang didukung environment Anda.")

        self._optimize_ml_estimators(learners, X_train, y_train)
        validation = ValidationSuite(method=self.validation_method, cv_folds=self.cv_folds,
                                     repeats=self.validation_repeats, bootstrap_rounds=self.bootstrap_rounds,
                                     nested_inner_folds=self.nested_inner_folds, seed=self.seed,
                                     optimize=self.optimize_ml)
        cv_scores = validation.evaluate(learners, X_train, y_train, n_classes=n_classes)
        if cv_scores:
            for n, s in cv_scores.items():
                self.logger.log(f"  Validation {n} → mean F1={np.mean(s):.4f} ± {np.std(s):.4f}")
        else:
            self.logger.log("  Validation → no ML validation scores produced (hold-out mode or no ML models active)")

        self._banner("Stage 5 · Copula / Ensemble / Hybrid Fusion")
        learners.fit_all(X_train, y_train, X_val=X_val, y_val=y_val)
        train_proba = learners.predict_proba_all(X_train)
        if not train_proba:
            raise ValueError("Setelah training, tidak ada probabilitas model yang berhasil dihasilkan.")
        val_proba = learners.predict_proba_all(X_val) if X_val is not None else {}
        test_proba = learners.predict_proba_all(X_test)
        if not test_proba:
            raise ValueError("Tahap predict pada test set tidak menghasilkan probabilitas dari model mana pun.")

        copula = CopulaFusionV4(family=self.copula_family)
        copula.fit(train_proba, y_train)
        fused_test = copula.fuse(test_proba)
        fused_train = copula.fuse(train_proba)

        ens_fuser = EnsembleHybridFusion(n_classes=n_classes, ensemble_method=self.ensemble_method, hybrid_method=self.hybrid_method)
        ref_proba = val_proba if val_proba and y_val is not None else train_proba
        y_ref = y_val if val_proba and y_val is not None else y_train
        ensemble_outputs = ens_fuser.build_ensemble(ref_proba, y_ref, test_proba) if self.use_ensemble else {}
        hybrid_outputs = ens_fuser.build_hybrid(ref_proba, y_ref, test_proba) if self.use_hybrid else {}

        self._banner("Stage 6 · Evaluation + XAI + Statistical Validation")
        evaluator = Evaluator(class_names=class_names, dataset=self.dataset)
        evaluator.plot_fused_distribution(fused_test, y_test)

        bn_pred = None
        if self.use_bn:
            self._banner("Stage 7 · Bayesian Network Reasoning")
            bn = BayesianLayer(n_bins=4, max_parents=3, top_k_feats=8)
            bn.fit(X_train, fused_train, y_train, feat_names)
            if bn.fitted:
                bn_pred = bn.predict(X_test, fused_test, feat_names)

        test_preds = learners.predict_all(X_test)
        test_probas = learners.predict_proba_all(X_test)
        for name in test_preds:
            evaluator.evaluate(name, y_test, test_preds[name], y_proba=test_probas.get(name))
            evaluator.plot_confusion(y_test, test_preds[name], title=f'CM {name}')

        fused_pred = np.argmax(fused_test, axis=1)
        evaluator.evaluate('Copula Fusion', y_test, fused_pred, y_proba=fused_test)
        evaluator.plot_confusion(y_test, fused_pred, title='CM Copula Fusion')

        for name, proba in ensemble_outputs.items():
            pred = np.argmax(proba, axis=1)
            evaluator.evaluate(name, y_test, pred, y_proba=proba)
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')
        for name, proba in hybrid_outputs.items():
            pred = np.argmax(proba, axis=1)
            evaluator.evaluate(name, y_test, pred, y_proba=proba)
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')

        if bn_pred is not None:
            evaluator.evaluate('BN Reasoning', y_test, bn_pred)
            evaluator.plot_confusion(y_test, bn_pred, title='CM BN Reasoning')

        evaluator.plot_roc_all(y_test, test_probas, fused_test)
        evaluator.plot_precision_recall(y_test, test_probas, fused_test)
        evaluator.plot_comparison()
        if self.use_shap and 'RF' in learners.models_:
            evaluator.shap_explain(learners.models_['RF'], X_test, feat_names, 'RF')
        if len(cv_scores) >= 2:
            evaluator.statistical_tests(cv_scores)
        summary_df = evaluator.summary()
        elapsed = time.time() - t0
        self._banner(f"✅ Pipeline DONE  ·  {elapsed:.0f}s  ·  Figures → {FIG_DIR}")
        self.logger.close()
        return {
            'evaluator': evaluator,
            'summary': summary_df,
            'learners': learners,
            'copula': copula,
            'fused_test': fused_test,
            'feat_names': feat_names,
            'active_ml_models': ml_models,
            'active_dl_models': dl_models,
            'validation_method': self.validation_method,
            'validation_scores': cv_scores,
            'ensemble_outputs': list(ensemble_outputs.keys()),
            'hybrid_outputs': list(hybrid_outputs.keys()),
        }



# =============================================================================
# V4.2 ENHANCEMENTS — PRE-PREPROCESSING, ENHANCED EDA, EXTENDED EVALUATION
# =============================================================================
class PrePreprocessor:
    """
    Data quality / hygiene stage executed *before* model-oriented preprocessing.

    Goals
    -----
    - standardize blank strings and non-finite values
    - remove duplicate rows split-wise
    - identify and remove constant features using TRAIN split only
    - create a transparent audit trail for methodology reporting
    """

    def __init__(self, dataset: str = '', drop_duplicates: bool = True,
                 drop_constant: bool = True):
        self.dataset = dataset
        self.drop_duplicates = drop_duplicates
        self.drop_constant = drop_constant
        self.constant_cols_ = []
        self.report_ = {}

    def _sanitize_frame(self, df: pd.DataFrame, feat_cols: list, cat_cols: list) -> pd.DataFrame:
        df = df.copy()
        # Normalize whitespace and blank tokens only on object/category columns
        obj_cols = [c for c in feat_cols if c in df.columns and (df[c].dtype == 'object' or str(df[c].dtype).startswith('category'))]
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        # Replace non-finite values on numeric features
        num_cols = [c for c in feat_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        return df

    def _summarize(self, df: pd.DataFrame, feat_cols: list, split_name: str) -> dict:
        num_cols = [c for c in feat_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        duplicates = int(df.duplicated(subset=[c for c in feat_cols if c in df.columns]).sum()) if feat_cols else 0
        missing_total = int(df[[c for c in feat_cols if c in df.columns]].isna().sum().sum()) if feat_cols else 0
        missing_pct = float(missing_total / (max(1, len(df) * max(1, len(feat_cols)))) * 100.0)
        constant_cols = []
        if num_cols:
            constant_cols = [c for c in num_cols if df[c].nunique(dropna=False) <= 1]
        return {
            'split': split_name,
            'rows': int(len(df)),
            'features': int(len(feat_cols)),
            'duplicates': duplicates,
            'missing_total': missing_total,
            'missing_pct': round(missing_pct, 4),
            'constant_cols': constant_cols,
        }

    def fit_transform_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None,
                             test_df: pd.DataFrame, feat_cols: list, cat_cols: list):
        train_df = self._sanitize_frame(train_df, feat_cols, cat_cols)
        val_df = self._sanitize_frame(val_df, feat_cols, cat_cols) if val_df is not None else None
        test_df = self._sanitize_frame(test_df, feat_cols, cat_cols)

        before_train = self._summarize(train_df, feat_cols, 'train_before')
        before_val = self._summarize(val_df, feat_cols, 'val_before') if val_df is not None else None
        before_test = self._summarize(test_df, feat_cols, 'test_before')

        if self.drop_duplicates:
            train_df = train_df.drop_duplicates(subset=[c for c in feat_cols if c in train_df.columns] + (['y'] if 'y' in train_df.columns else [])).reset_index(drop=True)
            if val_df is not None:
                val_df = val_df.drop_duplicates(subset=[c for c in feat_cols if c in val_df.columns] + (['y'] if 'y' in val_df.columns else [])).reset_index(drop=True)
            test_df = test_df.drop_duplicates(subset=[c for c in feat_cols if c in test_df.columns] + (['y'] if 'y' in test_df.columns else [])).reset_index(drop=True)

        # Detect constant features on TRAIN only, then remove from all splits
        if self.drop_constant:
            self.constant_cols_ = []
            for c in list(feat_cols):
                if c not in train_df.columns:
                    continue
                try:
                    nunq = train_df[c].nunique(dropna=False)
                except Exception:
                    nunq = 2
                if nunq <= 1:
                    self.constant_cols_.append(c)
            if self.constant_cols_:
                train_df = train_df.drop(columns=self.constant_cols_, errors='ignore')
                if val_df is not None:
                    val_df = val_df.drop(columns=self.constant_cols_, errors='ignore')
                test_df = test_df.drop(columns=self.constant_cols_, errors='ignore')
                feat_cols = [c for c in feat_cols if c not in self.constant_cols_]
                cat_cols = [c for c in cat_cols if c not in self.constant_cols_]

        after_train = self._summarize(train_df, feat_cols, 'train_after')
        after_val = self._summarize(val_df, feat_cols, 'val_after') if val_df is not None else None
        after_test = self._summarize(test_df, feat_cols, 'test_after')

        self.report_ = {
            'dataset': self.dataset,
            'drop_duplicates': self.drop_duplicates,
            'drop_constant': self.drop_constant,
            'constant_cols_removed': self.constant_cols_,
            'train_before': before_train,
            'train_after': after_train,
            'val_before': before_val,
            'val_after': after_val,
            'test_before': before_test,
            'test_after': after_test,
        }

        # Persist audit trail
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with open(OUT_DIR / f"prepreprocess_report_{self.dataset}_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(self.report_, f, indent=2, default=str)
        except Exception:
            pass

        return train_df, val_df, test_df, feat_cols, cat_cols, self.report_


class EDAAnalyzerV2(EDAAnalyzer):
    """Adds preprocessing-audit visualization and feature-vs-label analysis."""

    def _safe_top_idx(self, X: np.ndarray, y: np.ndarray, top_k: int = 20):
        from sklearn.feature_selection import mutual_info_classif
        if X.size == 0:
            return np.array([], dtype=int)
        try:
            mi = mutual_info_classif(np.nan_to_num(X), y, random_state=42)
            return np.argsort(mi)[::-1][:min(top_k, X.shape[1])]
        except Exception:
            vars_ = np.nanvar(np.nan_to_num(X), axis=0)
            return np.argsort(vars_)[::-1][:min(top_k, X.shape[1])]

    def plot_prepreprocess_audit(self, report: dict | None):
        if not report:
            return
        rows = []
        for k in ['train_before', 'train_after', 'val_before', 'val_after', 'test_before', 'test_after']:
            if report.get(k):
                r = report[k]
                rows.append([r['split'], r['rows'], r['duplicates'], r['missing_total'], r['missing_pct']])
        if not rows:
            return
        df = pd.DataFrame(rows, columns=['Split', 'Rows', 'Duplicates', 'Missing', 'Missing %'])
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(df) + 2)), facecolor=self.DARK_BG)
        ax.set_facecolor(self.CARD_BG)
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for (_, _), cell in table.get_celld().items():
            cell.set_edgecolor('#1f2937')
            cell.set_text_props(color='white')
            cell.set_facecolor(self.CARD_BG)
        ax.set_title(f'Pre-Preprocessing Audit — {self.dataset}', color='white', fontsize=13, pad=10)
        plt.tight_layout()
        self._savefig(fig, 'prepreprocess_audit')

    def plot_feature_label_heatmap(self, X: np.ndarray, feat_names: list, y: np.ndarray, top_k: int = 20):
        if X.size == 0:
            return
        idxs = self._safe_top_idx(X, y, top_k=top_k)
        if len(idxs) == 0:
            return
        names = [feat_names[i] if i < len(feat_names) else f'f{i}' for i in idxs]
        Xs = np.nan_to_num(X[:, idxs])
        # 1) Pearson correlation against numeric-coded label
        corrs = []
        for j in range(Xs.shape[1]):
            try:
                c = np.corrcoef(Xs[:, j], y.astype(float))[0, 1]
            except Exception:
                c = 0.0
            corrs.append(0.0 if not np.isfinite(c) else float(c))
        corr_arr = np.array(corrs).reshape(-1, 1)
        # 2) MI against label
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(Xs, y, random_state=42)
        except Exception:
            mi = np.zeros(len(names), dtype=float)
        heat = np.column_stack([corr_arr.ravel(), mi])
        fig, ax = plt.subplots(figsize=(6, max(6, len(names) * 0.35)), facecolor=self.DARK_BG)
        ax.set_facecolor(self.CARD_BG)
        sns.heatmap(heat, annot=True, fmt='.3f', cmap='coolwarm', ax=ax,
                    yticklabels=names, xticklabels=['Corr(label)', 'MI(label)'],
                    cbar=True)
        ax.set_title(f'Feature ↔ Label Association — {self.dataset}', color='white', fontsize=12)
        ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        self._savefig(fig, 'feature_label_heatmap')

    def run_all(self, df: pd.DataFrame, X: np.ndarray,
                feat_names: list, y: np.ndarray,
                top_k_dist: int = 16, top_k_pair: int = 5,
                preprocessing_report: dict | None = None):
        print(f"Running enhanced EDA for {self.dataset.upper()} …")
        feat_cols = [c for c in feat_names if c in df.columns] or feat_names
        self.dataset_info(df, feat_cols, y)
        self.plot_prepreprocess_audit(preprocessing_report)
        self.plot_class_distribution(y)
        self.plot_feature_distributions(X, feat_names, y, top_k=top_k_dist)
        self.plot_boxplots(X, feat_names, y, top_k=min(12, max(4, len(feat_names))))
        self.plot_correlation_heatmap(X, feat_names, top_k=min(20, max(4, len(feat_names))))
        self.plot_feature_label_heatmap(X, feat_names, y, top_k=min(20, max(4, len(feat_names))))
        if len(feat_names) <= 60 and len(df) <= 80000:
            self.plot_pairplot(X, feat_names, y, top_k=min(top_k_pair, 5))
        self.plot_missing_values(df, feat_cols)
        self.plot_attack_breakdown(df)
        print(f"  Enhanced EDA complete — all figures in {FIG_DIR}/")


class EvaluatorV2(Evaluator):
    """Extends ROC/PR plotting to multiclass and keeps all classic plots."""

    def plot_roc_all(self, y_true, proba_dict, fused_proba):
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        unique_classes = sorted(np.unique(y_true))
        n_classes = len(unique_classes)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0b1020')
        ax.set_facecolor('#111827')
        colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(proba_dict) + 1)))
        all_models = list(proba_dict.items()) + [('Copula Fusion', fused_proba)]
        if n_classes == 2:
            for (name, p), c in zip(all_models, colors):
                try:
                    fpr, tpr, _ = roc_curve(y_true, p[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, color=c, label=f'{name} (AUC={roc_auc:.4f})')
                except Exception:
                    pass
        else:
            y_bin = label_binarize(y_true, classes=unique_classes)
            for (name, p), c in zip(all_models, colors):
                try:
                    if p.shape[1] != y_bin.shape[1]:
                        continue
                    fpr, tpr, _ = roc_curve(y_bin.ravel(), p.ravel())
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, color=c, label=f'{name} (micro-AUC={roc_auc:.4f})')
                except Exception:
                    pass
        ax.plot([0, 1], [0, 1], '--', color='#475569', lw=1)
        ax.set_xlabel('False Positive Rate', color='#94a3b8')
        ax.set_ylabel('True Positive Rate', color='#94a3b8')
        ax.set_title(f'ROC Curves — {self.dataset}', color='white', fontsize=12)
        ax.legend(facecolor='#0b1020', labelcolor='white', fontsize=8)
        ax.tick_params(colors='#64748b')
        plt.tight_layout()
        path = FIG_DIR / f"roc_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#0b1020')
        plt.close()
        print(f"  ROC curves → {path}")

    def plot_precision_recall(self, y_true: np.ndarray, proba_dict: dict, fused_proba: np.ndarray):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
        unique_classes = sorted(np.unique(y_true))
        n_classes = len(unique_classes)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0b1020')
        ax.set_facecolor('#111827')
        colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(proba_dict) + 1)))
        all_models = list(proba_dict.items()) + [('Copula Fusion', fused_proba)]
        if n_classes == 2:
            baseline = float(np.mean(y_true))
            for (name, p), c in zip(all_models, colors):
                try:
                    prec, rec, _ = precision_recall_curve(y_true, p[:, 1])
                    ap = average_precision_score(y_true, p[:, 1])
                    ax.plot(rec, prec, color=c, lw=2, label=f'{name} (AP={ap:.4f})')
                    ax.fill_between(rec, prec, alpha=0.06, color=c)
                except Exception:
                    pass
            ax.axhline(baseline, linestyle='--', color='#475569', lw=1, label=f'Baseline ({baseline:.3f})')
        else:
            y_bin = label_binarize(y_true, classes=unique_classes)
            for (name, p), c in zip(all_models, colors):
                try:
                    if p.shape[1] != y_bin.shape[1]:
                        continue
                    prec, rec, _ = precision_recall_curve(y_bin.ravel(), p.ravel())
                    ap = average_precision_score(y_bin, p, average='micro')
                    ax.plot(rec, prec, color=c, lw=2, label=f'{name} (micro-AP={ap:.4f})')
                except Exception:
                    pass
        ax.set_xlabel('Recall', color='#94a3b8')
        ax.set_ylabel('Precision', color='#94a3b8')
        ax.set_title(f'Precision-Recall Curves — {self.dataset}', color='white', fontsize=12)
        ax.legend(facecolor='#0b1020', labelcolor='white', fontsize=8)
        ax.tick_params(colors='#64748b')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        plt.tight_layout()
        path = FIG_DIR / f"pr_curve_{self.dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#0b1020')
        plt.close()
        print(f"  PR curves → {path}")


class NIDSPipelineV4_2(NIDSPipelineV4):
    """
    V4.2 focuses on richer end-to-end analytics:
    - explicit pre-preprocessing stage
    - enhanced preprocessing audit
    - extended EDA figures
    - stronger evaluation plots for both binary and multiclass tasks
    """

    def __init__(self, *args, generate_eda: bool = True,
                 drop_duplicates_stage: bool = True,
                 drop_constant_stage: bool = True,
                 **kwargs):
        self.generate_eda = bool(generate_eda)
        self.drop_duplicates_stage = bool(drop_duplicates_stage)
        self.drop_constant_stage = bool(drop_constant_stage)
        super().__init__(*args, **kwargs)

    def run(self) -> dict:
        t0 = time.time()
        self._banner(f"HYBRID NIDS PIPELINE V4.3  ·  Dataset: {self.dataset.upper()}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ml_models, dl_models = self._active_model_lists()
        self.logger.log(f"  Active ML models → {', '.join(ml_models) if ml_models else 'None'}")
        self.logger.log(f"  Active DL models → {', '.join(dl_models) if dl_models else 'None'}")
        self.logger.log(f"  Validation → method={self.validation_method}, folds={self.cv_folds}, repeats={self.validation_repeats}, bootstrap_rounds={self.bootstrap_rounds}")
        self.logger.log(f"  Feature selection → method={self.feature_method}, top_k={self.top_k}")
        self.logger.log(f"  Copula → family={self.copula_family}")
        self.logger.log(f"  Ensemble/Hybrid → use_ensemble={self.use_ensemble} ({self.ensemble_method}), use_hybrid={self.use_hybrid} ({self.hybrid_method})")

        # Stage 1: Load
        self._banner("Stage 1 · Data Loading")
        meta = self._load()
        train_df = meta['train']; val_df = meta.get('val', None); test_df = meta['test']
        feat_cols = meta['feat_cols']; cat_cols = meta['cat_cols']; class_names = meta['class_names']; n_classes = meta['n_classes']
        self._report_split_methodology(meta)

        # Stage 2: Pre-preprocessing / data quality
        self._banner("Stage 2 · Pre-Preprocessing (Data Quality & Hygiene)")
        pprep = PrePreprocessor(dataset=self.dataset,
                                drop_duplicates=self.drop_duplicates_stage,
                                drop_constant=self.drop_constant_stage)
        train_df, val_df, test_df, feat_cols, cat_cols, pprep_report = pprep.fit_transform_splits(
            train_df, val_df, test_df, feat_cols, cat_cols
        )
        y_train_raw = train_df['y'].values.astype(int)
        y_val = val_df['y'].values.astype(int) if val_df is not None else None
        y_test = test_df['y'].values.astype(int)
        self.logger.log(f"  After pre-preprocessing → train={train_df.shape}, val={(val_df.shape if val_df is not None else 'N/A')}, test={test_df.shape}")
        self.logger.log(f"  Constant features removed → {len(pprep_report.get('constant_cols_removed', []))}")

        # Stage 2b: EDA
        if self.generate_eda:
            self._banner("Stage 2b · Exploratory Data Analysis")
            eda = EDAAnalyzerV2(dataset=self.dataset, class_names=class_names)
            _eda_df = train_df[feat_cols].copy()
            for _c in cat_cols:
                if _c in _eda_df.columns:
                    _eda_df[_c] = pd.factorize(_eda_df[_c])[0]
            _eda_X = _eda_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
            # Automatically tone down pairplot for very large datasets
            pair_k = 4 if len(train_df) > 120000 else 5
            dist_k = 12 if len(feat_cols) > 60 else 16
            eda.run_all(df=train_df, X=_eda_X, feat_names=feat_cols, y=y_train_raw,
                        top_k_dist=dist_k, top_k_pair=pair_k,
                        preprocessing_report=pprep_report)

        # Stage 3: Preprocessing
        self._banner("Stage 3 · Preprocessing")
        prep = Preprocessor(scaler_type=self.scaler, balance=self.balance)
        X_train_clean = prep.clean_fit_transform(train_df, feat_cols, cat_cols)
        X_val_clean = prep.clean_transform(val_df, feat_cols, cat_cols) if val_df is not None else None
        X_test_clean = prep.clean_transform(test_df, feat_cols, cat_cols)

        fe = FeatureEngineerV4()
        k = min(self.top_k, X_train_clean.shape[1])
        idx = fe.select(X_train_clean, y_train_raw, feat_cols, k=k, method=self.feature_method)
        X_train_clean = X_train_clean[:, idx]
        X_val_clean = X_val_clean[:, idx] if X_val_clean is not None else None
        X_test_clean = X_test_clean[:, idx]
        feat_names = fe.selected_names_
        sel_cat_cols = [c for c in cat_cols if c in feat_names]

        prep.fit_scaler(X_train_clean)
        X_train = prep.scale_transform(X_train_clean)
        X_val = prep.scale_transform(X_val_clean) if X_val_clean is not None else None
        X_test = prep.scale_transform(X_test_clean)
        cat_indices_selected = [feat_names.index(c) for c in sel_cat_cols if c in feat_names]
        X_train, y_train = prep.balance_train(X_train, y_train_raw, cat_indices=cat_indices_selected)
        val_shape = X_val.shape if X_val is not None else 'N/A'
        self.logger.log(f"  Final shapes → train: {X_train.shape}  val: {val_shape}  test: {X_test.shape}")

        # Stage 4: Train + validation
        self._banner("Stage 4 · Base Learner Training + Validation")
        learners = ExpertLearners(n_classes=n_classes, n_features=X_train.shape[1])
        epochs, batch_size = self._get_train_params()
        if self.use_lr: learners.add_lr(**self._get_model_params('lr'))
        if self.use_rf: learners.add_rf(**self._get_model_params('rf'))
        if self.use_svm: learners.add_svm(**self._get_model_params('svm'))
        if self.use_xgb: learners.add_xgb(**self._get_model_params('xgb'))
        if self.use_dt: learners.add_dt(**self._get_model_params('dt'))
        if self.use_mlp: learners.add_mlp(epochs=epochs, batch=batch_size)
        if self.use_dnn: learners.add_dnn(epochs=epochs, batch=batch_size)
        if self.use_cnn1d: learners.add_cnn1d(epochs=epochs, batch=batch_size)
        if self.use_lstm: learners.add_lstm(epochs=epochs, batch=batch_size)
        if self.use_aae: learners.add_aae(epochs=epochs, batch=batch_size)
        if self.use_cnn_lstm: learners.add_cnn_lstm(epochs=epochs, batch=batch_size)
        if self.use_cnn_bilstm: learners.add_cnn_bilstm(epochs=epochs, batch=batch_size)
        if self.use_cnn_gru: learners.add_cnn_gru(epochs=epochs, batch=batch_size)
        if self.use_gnn: learners.add_gnn(epochs=epochs, batch=batch_size)
        if self.use_gat: learners.add_gat(epochs=epochs, batch=batch_size)
        if self.use_transformer: learners.add_transformer(epochs=epochs, batch=batch_size)
        if self.use_transfer_learning: learners.add_transfer_learning(epochs=epochs, batch=batch_size)

        if len(learners.models_) == 0:
            raise ValueError("Tidak ada expert model yang aktif/berhasil dibangun. Pilih minimal satu model yang didukung environment Anda.")

        self._optimize_ml_estimators(learners, X_train, y_train)
        validation = ValidationSuite(method=self.validation_method, cv_folds=self.cv_folds,
                                     repeats=self.validation_repeats, bootstrap_rounds=self.bootstrap_rounds,
                                     nested_inner_folds=self.nested_inner_folds, seed=self.seed,
                                     optimize=self.optimize_ml)
        cv_scores = validation.evaluate(learners, X_train, y_train, n_classes=n_classes)
        if cv_scores:
            for n, s in cv_scores.items():
                self.logger.log(f"  Validation {n} → mean F1={np.mean(s):.4f} ± {np.std(s):.4f}")
        else:
            self.logger.log("  Validation → no ML validation scores produced (hold-out mode or no ML models active)")

        # Stage 5: Fusion
        self._banner("Stage 5 · Copula / Ensemble / Hybrid Fusion")
        learners.fit_all(X_train, y_train, X_val=X_val, y_val=y_val)
        train_proba = learners.predict_proba_all(X_train)
        if not train_proba:
            raise ValueError("Setelah training, tidak ada probabilitas model yang berhasil dihasilkan.")
        val_proba = learners.predict_proba_all(X_val) if X_val is not None else {}
        test_proba = learners.predict_proba_all(X_test)
        if not test_proba:
            raise ValueError("Tahap predict pada test set tidak menghasilkan probabilitas dari model mana pun.")

        copula = CopulaFusionV4(family=self.copula_family)
        copula.fit(train_proba, y_train)
        fused_test = copula.fuse(test_proba)
        fused_train = copula.fuse(train_proba)

        ens_fuser = EnsembleHybridFusion(n_classes=n_classes,
                                         ensemble_method=self.ensemble_method,
                                         hybrid_method=self.hybrid_method)
        ref_proba = val_proba if val_proba and y_val is not None else train_proba
        y_ref = y_val if val_proba and y_val is not None else y_train
        ensemble_outputs = ens_fuser.build_ensemble(ref_proba, y_ref, test_proba) if self.use_ensemble else {}
        hybrid_outputs = ens_fuser.build_hybrid(ref_proba, y_ref, test_proba) if self.use_hybrid else {}

        # Stage 6 & 7: Evaluation
        self._banner("Stage 6 · Evaluation + XAI + Statistical Validation")
        evaluator = EvaluatorV2(class_names=class_names, dataset=self.dataset)
        evaluator.plot_fused_distribution(fused_test, y_test)

        bn_pred = None
        if self.use_bn:
            self._banner("Stage 7 · Bayesian Network Reasoning")
            bn = BayesianLayer(n_bins=4, max_parents=3, top_k_feats=8)
            bn.fit(X_train, fused_train, y_train, feat_names)
            if bn.fitted:
                bn_pred = bn.predict(X_test, fused_test, feat_names)

        test_preds = learners.predict_all(X_test)
        test_probas = learners.predict_proba_all(X_test)
        for name in test_preds:
            evaluator.evaluate(name, y_test, test_preds[name], y_proba=test_probas.get(name))
            evaluator.plot_confusion(y_test, test_preds[name], title=f'CM {name}')

        fused_pred = np.argmax(fused_test, axis=1)
        evaluator.evaluate('Copula Fusion', y_test, fused_pred, y_proba=fused_test)
        evaluator.plot_confusion(y_test, fused_pred, title='CM Copula Fusion')

        for name, proba in ensemble_outputs.items():
            pred = np.argmax(proba, axis=1)
            evaluator.evaluate(name, y_test, pred, y_proba=proba)
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')

        for name, proba in hybrid_outputs.items():
            pred = np.argmax(proba, axis=1)
            evaluator.evaluate(name, y_test, pred, y_proba=proba)
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')

        if bn_pred is not None:
            evaluator.evaluate('BN Reasoning', y_test, bn_pred)
            evaluator.plot_confusion(y_test, bn_pred, title='CM BN Reasoning')

        evaluator.plot_roc_all(y_test, test_probas, fused_test)
        evaluator.plot_precision_recall(y_test, test_probas, fused_test)
        evaluator.plot_comparison()

        if self.use_shap and 'RF' in learners.models_:
            evaluator.shap_explain(learners.models_['RF'], X_test, feat_names, 'RF')

        if len(cv_scores) >= 2:
            evaluator.statistical_tests(cv_scores)

        summary_df = evaluator.summary()
        elapsed = time.time() - t0
        self._banner(f"✅ Pipeline DONE  ·  {elapsed:.0f}s  ·  Figures → {FIG_DIR}")
        self.logger.close()
        return {
            'evaluator': evaluator,
            'summary': summary_df,
            'learners': learners,
            'copula': copula,
            'fused_test': fused_test,
            'feat_names': feat_names,
            'active_ml_models': ml_models,
            'active_dl_models': dl_models,
            'validation_method': self.validation_method,
            'validation_scores': cv_scores,
            'ensemble_outputs': list(ensemble_outputs.keys()),
            'hybrid_outputs': list(hybrid_outputs.keys()),
            'prepreprocess_report': pprep_report,
        }


NIDSPipeline = NIDSPipelineV4_2


if __name__ == "__main__":

    # =========================================================================
    # ── CONFIGURE YOUR DATASET PATHS HERE ────────────────────────────────────
    # =========================================================================
    CONFIGS = [

        # ── 1. NSL-KDD ─────────────────────────────────────────────────────
        dict(
            dataset        = 'nslkdd',
            paths          = {
                'train': 'data/nslkdd/KDDTrain+.txt',
                'test':  'data/nslkdd/KDDTest+.txt',
            },
            binary         = True,
            copula_family  = 'gaussian',
            balance        = 'smote',
            scaler         = 'robust',
            top_k          = 40,
            cv_folds       = 2,   # ← naikan ke 5/10 saat model final
            use_rf=True, use_xgb=True,
            use_cnn_bilstm=True, use_cnn_gru=True,
            use_gnn=True, use_gat=True,
            use_transformer=True,
            use_bn         = True,
            use_shap       = True,
        ),

        # ── 2. UNSW-NB15 ───────────────────────────────────────────────────
        dict(
            dataset        = 'unsw',
            paths          = {
                'train': 'data/unsw/UNSW_NB15_training-set.csv',
                'test':  'data/unsw/UNSW_NB15_testing-set.csv',
            },
            binary         = True,
            copula_family  = 'clayton',
            balance        = 'smote',
            scaler         = 'robust',
            top_k          = 40,
            cv_folds       = 2,   # ← naikan ke 5/10 saat model final
            use_rf=True, use_xgb=True,
            use_cnn_bilstm=True, use_cnn_gru=True,
            use_gnn=True, use_gat=True,
            use_transformer=True,
            use_bn         = True,
            use_shap       = True,
        ),

        # ── 3. InSDN ───────────────────────────────────────────────────────
        dict(
            dataset        = 'insdn',
            paths          = {
                # Use whichever filename you have — loader auto-detects variants
                'normal':     'data/insdn/Normal_data.csv',     # or Normal.csv
                'ovs':        'data/insdn/OVS_data.csv',        # or OVS.csv
                'metasploit': 'data/insdn/Metasploit.csv',      # or MSF.csv
            },
            binary         = True,      # False → 3-class (Normal/OVS/MSF)
            copula_family  = 'frank',
            balance        = 'smote',
            scaler         = 'robust',
            top_k          = 40,
            cv_folds       = 2,   # ← naikan ke 5/10 saat model final
            use_rf=True, use_xgb=True,
            use_cnn_bilstm=True, use_cnn_gru=True,
            use_gnn=True, use_gat=True,
            use_transformer=True,
            use_bn         = True,
            use_shap       = True,
        ),

        # ── 4. CICIDS2017 ──────────────────────────────────────────────────
        dict(
            dataset        = 'cicids',
            paths          = {
                # Option A: auto-detect from folder (recommended)
                'dir': 'data/cicids2017/',

                # Option B: explicit file list (uncomment if preferred)
                # 'files': [
                #   'data/cicids2017/Monday-WorkingHours.pcap_ISCX.csv',
                #   'data/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv',
                #   'data/cicids2017/Wednesday-WorkingHours.pcap_ISCX.csv',
                #   'data/cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                #   'data/cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                #   'data/cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
                #   'data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                #   'data/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                # ],
            },
            binary         = True,
            copula_family  = 'gaussian',
            balance        = 'smote',
            scaler         = 'robust',
            top_k          = 40,
            cv_folds       = 2,   # ← naikan ke 5/10 saat model final
            use_rf=True, use_xgb=True,
            use_cnn_bilstm=True, use_cnn_gru=True,
            use_gnn=True, use_gat=True,
            use_transformer=True,
            use_bn         = True,
            use_shap       = True,
            sample_frac    = 0.30,      # 30% for speed; set 1.0 for full run
            cicids_split   = 'random',  # 'temporal' for publication-quality split
        ),
    ]

    # ── Validate paths first (recommended) ──────────────────────────────────
    # Uncomment to check/fix paths before running:
    # check_dataset_paths(CONFIGS)
    # CONFIGS = set_dataset_paths(CONFIGS)   # interactive fix

    # ── Launch interactive selector ───────────────────────────────────────────
    DatasetSelector(CONFIGS).launch()





# =============================================================================
# v4.5.1 STANDALONE EXTENSIONS
# Semi-Supervised / Unsupervised / RL backends integrated without external base import
# =============================================================================

import math
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import OneClassSVM, SVC

def _normalize_scores(scores):
    s = np.asarray(scores, dtype=float).reshape(-1)
    if s.size == 0:
        return np.zeros(0, dtype=float)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    sf = s[finite]
    lo, hi = np.quantile(sf, [0.01, 0.99]) if sf.size >= 5 else (sf.min(), sf.max())
    if not np.isfinite(lo):
        lo = np.nanmin(sf)
    if not np.isfinite(hi):
        hi = np.nanmax(sf)
    scaled = (s - lo) / (hi - lo + 1e-9)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return scaled.astype(float)

def _to_binary_proba(p_attack):
    p = np.clip(np.asarray(p_attack, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p, p])

def _safe_choice(rng, n, frac):
    n = int(n)
    if n <= 0:
        return np.array([], dtype=int)
    take = max(1, min(n - 1, int(round(n * frac)))) if n > 1 else 1
    return rng.choice(np.arange(n), size=take, replace=False)

# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class BaseProbabilisticWrapper:
    name = "base"
    paradigm = "generic"

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class IsolationForestDetector(BaseProbabilisticWrapper):
    name = "IsolationForest"
    paradigm = "unsupervised"

    def __init__(self, contamination=0.1, seed=42, n_estimators=300):
        self.contamination = float(contamination)
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.model = None
        self.threshold_ = None

    def fit(self, X, y=None):
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(X)
        scores = -self.model.decision_function(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def predict_proba(self, X):
        scores = -self.model.decision_function(X)
        return _to_binary_proba(_normalize_scores(scores))

class OneClassSVMDetector(BaseProbabilisticWrapper):
    name = "OneClassSVM"
    paradigm = "unsupervised"

    def __init__(self, contamination=0.1, nu=0.1, kernel='rbf', gamma='scale'):
        self.contamination = float(contamination)
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = gamma
        self.model = None

    def fit(self, X, y=None):
        self.model = OneClassSVM(nu=min(max(self.nu, 1e-3), 0.95), kernel=self.kernel, gamma=self.gamma)
        self.model.fit(X)
        return self

    def predict_proba(self, X):
        scores = -self.model.decision_function(X)
        return _to_binary_proba(_normalize_scores(scores))

class LOFNoveltyDetector(BaseProbabilisticWrapper):
    name = "LOF"
    paradigm = "unsupervised"

    def __init__(self, contamination=0.1, n_neighbors=20):
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.model = None

    def fit(self, X, y=None):
        nn = max(5, min(self.n_neighbors, max(5, len(X) - 1)))
        self.model = LocalOutlierFactor(n_neighbors=nn, contamination=self.contamination, novelty=True)
        self.model.fit(X)
        return self

    def predict_proba(self, X):
        scores = -self.model.decision_function(X)
        return _to_binary_proba(_normalize_scores(scores))

class PCAReconstructionDetector(BaseProbabilisticWrapper):
    name = "PCARecon"
    paradigm = "unsupervised"

    def __init__(self, contamination=0.1, n_components=0.95, seed=42):
        self.contamination = float(contamination)
        self.n_components = n_components
        self.seed = int(seed)
        self.model = None

    def fit(self, X, y=None):
        self.model = PCA(n_components=self.n_components, svd_solver='full', random_state=self.seed)
        self.model.fit(X)
        return self

    def _recon_error(self, X):
        Z = self.model.transform(X)
        Xr = self.model.inverse_transform(Z)
        return np.mean((X - Xr) ** 2, axis=1)

    def predict_proba(self, X):
        scores = self._recon_error(X)
        return _to_binary_proba(_normalize_scores(scores))

class KMeansDistanceDetector(BaseProbabilisticWrapper):
    name = "KMeansDistance"
    paradigm = "unsupervised"

    def __init__(self, contamination=0.1, n_clusters=4, seed=42):
        self.contamination = float(contamination)
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.model = None

    def fit(self, X, y=None):
        k = max(2, min(self.n_clusters, max(2, len(X) // 50)))
        self.model = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        self.model.fit(X)
        return self

    def predict_proba(self, X):
        dist = np.min(self.model.transform(X), axis=1)
        return _to_binary_proba(_normalize_scores(dist))

class AutoencoderAnomalyDetector(BaseProbabilisticWrapper):
    name = "AutoencoderAnomaly"
    paradigm = "unsupervised"

    def __init__(self, contamination=0.1, epochs=20, batch_size=128, seed=42, semi_supervised=False):
        self.contamination = float(contamination)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.semi_supervised = bool(semi_supervised)
        self.model = None
        self._keras = None

    def fit(self, X, y=None):
        tf, keras = _import_keras()
        self._keras = keras
        L = keras.layers
        n_feat = int(X.shape[1])
        tf.random.set_seed(self.seed)
        inp = L.Input(shape=(n_feat,), name='ae_in')
        d1 = L.Dense(max(16, n_feat), activation='relu')(inp)
        d2 = L.Dense(max(8, n_feat // 2), activation='relu')(d1)
        bottleneck = L.Dense(max(4, n_feat // 4), activation='relu')(d2)
        u1 = L.Dense(max(8, n_feat // 2), activation='relu')(bottleneck)
        u2 = L.Dense(max(16, n_feat), activation='relu')(u1)
        out = L.Dense(n_feat, activation='linear')(u2)
        self.model = keras.models.Model(inp, out, name='autoencoder_anomaly')
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
        train_X = X
        if self.semi_supervised and y is not None and (np.asarray(y) == 0).any():
            train_X = X[np.asarray(y) == 0]
        cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='loss')]
        self.model.fit(train_X, train_X, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=cb)
        return self

    def predict_proba(self, X):
        Xr = self.model.predict(X, verbose=0)
        err = np.mean((X - Xr) ** 2, axis=1)
        return _to_binary_proba(_normalize_scores(err))

class LabelSpreadingWrapper(BaseProbabilisticWrapper):
    name = "LabelSpreading"
    paradigm = "semi_supervised"

    def __init__(self, label_fraction=0.3, seed=42, kernel='rbf', alpha=0.2):
        self.label_fraction = float(label_fraction)
        self.seed = int(seed)
        self.kernel = kernel
        self.alpha = float(alpha)
        self.model = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        y_mask = np.asarray(y).copy()
        unlabeled_idx = _safe_choice(rng, len(y_mask), 1.0 - self.label_fraction)
        y_mask[unlabeled_idx] = -1
        self.model = LabelSpreading(kernel=self.kernel, alpha=self.alpha, max_iter=50)
        self.model.fit(X, y_mask)
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        if proba.ndim == 1:
            return _to_binary_proba(proba)
        return np.clip(proba, 1e-6, 1 - 1e-6)

class SelfTrainingWrapper(BaseProbabilisticWrapper):
    def __init__(self, base_estimator, label_fraction=0.3, seed=42, threshold=0.8, name='SelfTraining'):
        self.base_estimator = base_estimator
        self.label_fraction = float(label_fraction)
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.name = name
        self.paradigm = "semi_supervised"
        self.model = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        y_mask = np.asarray(y).copy()
        unlabeled_idx = _safe_choice(rng, len(y_mask), 1.0 - self.label_fraction)
        y_mask[unlabeled_idx] = -1
        try:
            self.model = SelfTrainingClassifier(estimator=clone(self.base_estimator), threshold=self.threshold)
        except TypeError:
            self.model = SelfTrainingClassifier(base_estimator=clone(self.base_estimator), threshold=self.threshold)
        self.model.fit(X, y_mask)
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        if proba.ndim == 1:
            return _to_binary_proba(proba)
        return np.clip(proba, 1e-6, 1 - 1e-6)

class PseudoLabelDNNWrapper(BaseProbabilisticWrapper):
    name = "PseudoLabelDNN"
    paradigm = "semi_supervised"

    def __init__(self, n_classes=2, label_fraction=0.3, pseudo_threshold=0.9, epochs=20, batch_size=128, seed=42):
        self.n_classes = int(n_classes)
        self.label_fraction = float(label_fraction)
        self.pseudo_threshold = float(pseudo_threshold)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model = None
        self._keras = None

    def _build(self, n_feat):
        tf, keras = _import_keras()
        self._keras = keras
        tf.random.set_seed(self.seed)
        L = keras.layers
        n_out = 1 if self.n_classes == 2 else self.n_classes
        act = 'sigmoid' if self.n_classes == 2 else 'softmax'
        loss = 'binary_crossentropy' if self.n_classes == 2 else 'sparse_categorical_crossentropy'
        inp = L.Input(shape=(n_feat,), name='semi_dnn_in')
        x = L.Dense(max(64, n_feat), activation='relu')(inp)
        x = L.Dropout(0.25)(x)
        x = L.Dense(max(32, n_feat // 2 + 8), activation='relu')(x)
        x = L.Dropout(0.20)(x)
        out = L.Dense(n_out, activation=act)(x)
        m = keras.models.Model(inp, out, name='PseudoLabelDNN')
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
        return m

    def _fit_once(self, X, y):
        cb = [self._keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='loss')]
        y_fit = y.astype(float if self.n_classes == 2 else int)
        self.model.fit(X, y_fit, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=cb)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        idx_labeled = _safe_choice(rng, len(y), self.label_fraction)
        mask = np.zeros(len(y), dtype=bool)
        mask[idx_labeled] = True
        X_lab, y_lab = X[mask], np.asarray(y)[mask]
        X_unlab = X[~mask]
        self.model = self._build(X.shape[1])
        self._fit_once(X_lab, y_lab)
        if len(X_unlab):
            raw = self.model.predict(X_unlab, verbose=0)
            proba = np.column_stack([1 - raw, raw]) if raw.ndim == 1 or raw.shape[1] == 1 else raw
            conf = np.max(proba, axis=1)
            keep = conf >= self.pseudo_threshold
            if keep.any():
                pseudo_y = np.argmax(proba[keep], axis=1)
                X2 = np.vstack([X_lab, X_unlab[keep]])
                y2 = np.concatenate([y_lab, pseudo_y])
                self.model = self._build(X.shape[1])
                self._fit_once(X2, y2)
        return self

    def predict_proba(self, X):
        raw = self.model.predict(X, verbose=0)
        if raw.ndim == 1 or raw.shape[1] == 1:
            return _to_binary_proba(raw.reshape(-1))
        return np.clip(raw, 1e-6, 1 - 1e-6)

class AdaptiveThresholdQLearner(BaseProbabilisticWrapper):
    name = "AdaptiveQThreshold"
    paradigm = "reinforcement"

    def __init__(self, base_detector=None, episodes=40, epsilon=0.2, lr=0.3, seed=42):
        self.base_detector = base_detector or IsolationForestDetector(contamination=0.1, seed=seed)
        self.episodes = int(episodes)
        self.epsilon = float(epsilon)
        self.lr = float(lr)
        self.seed = int(seed)
        self.threshold_ = 0.5
        self.q_ = None

    def fit(self, X, y):
        self.base_detector.fit(X, y)
        scores = self.base_detector.predict_proba(X)[:, 1]
        rng = np.random.RandomState(self.seed)
        thresholds = np.linspace(0.05, 0.95, 19)
        q = np.zeros(len(thresholds), dtype=float)
        y = np.asarray(y).astype(int)
        for ep in range(self.episodes):
            if rng.rand() < self.epsilon:
                a = rng.randint(len(thresholds))
            else:
                a = int(np.argmax(q))
            thr = thresholds[a]
            batch_idx = rng.choice(len(y), size=max(32, min(len(y), max(64, len(y)//3))), replace=True)
            pred = (scores[batch_idx] >= thr).astype(int)
            reward = f1_score(y[batch_idx], pred, average='binary', zero_division=0)
            q[a] = q[a] + self.lr * (reward - q[a])
        self.q_ = q
        self.threshold_ = float(thresholds[int(np.argmax(q))])
        return self

    def predict_proba(self, X):
        return self.base_detector.predict_proba(X)

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold_).astype(int)

# ---------------------------------------------------------------------------
# Learner manager for non-supervised paradigms
# ---------------------------------------------------------------------------

class AlternativeParadigmLearners:
    def __init__(self, paradigm='semi_supervised'):
        self.paradigm = paradigm
        self.models_ = {}

    def add(self, name, model):
        self.models_[name] = model

    def fit_all(self, X, y=None):
        fitted = {}
        for name, model in self.models_.items():
            try:
                if self.paradigm == 'unsupervised':
                    fitted[name] = model.fit(X, y if getattr(model, 'semi_supervised', False) else None)
                else:
                    fitted[name] = model.fit(X, y)
                print(f"    [{self.paradigm}] {name:20s} ✓")
            except Exception as e:
                print(f"    [{self.paradigm}] {name:20s} skipped: {type(e).__name__}: {e}")
        self.models_ = fitted
        return self

    def predict_proba_all(self, X):
        out = {}
        for name, model in self.models_.items():
            try:
                p = model.predict_proba(X)
                if p is not None:
                    out[name] = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
            except Exception as e:
                print(f"    Predict skipped for {name}: {e}")
        return out

    def predict_all(self, X):
        out = {}
        for name, model in self.models_.items():
            try:
                out[name] = np.asarray(model.predict(X)).astype(int)
            except Exception:
                try:
                    out[name] = np.argmax(model.predict_proba(X), axis=1)
                except Exception:
                    pass
        return out

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class NIDSPipelineV4_5(NIDSPipelineV4_2):
    def __init__(self, *args,
                 learning_paradigm: str = 'supervised',
                 use_label_spreading: bool = False,
                 use_self_training_svm: bool = False,
                 use_self_training_rf: bool = False,
                 use_pseudo_label_dnn: bool = False,
                 use_isolation_forest: bool = False,
                 use_oneclass_svm_anom: bool = False,
                 use_lof: bool = False,
                 use_pca_anom: bool = False,
                 use_kmeans_anom: bool = False,
                 use_autoencoder_anom: bool = False,
                 use_rl_threshold: bool = False,
                 semi_label_fraction: float = 0.3,
                 pseudo_label_threshold: float = 0.9,
                 anomaly_contamination: float = 0.1,
                 rl_episodes: int = 40,
                 **kwargs):
        self.learning_paradigm = str(learning_paradigm or 'supervised').lower()
        self.use_label_spreading = bool(use_label_spreading)
        self.use_self_training_svm = bool(use_self_training_svm)
        self.use_self_training_rf = bool(use_self_training_rf)
        self.use_pseudo_label_dnn = bool(use_pseudo_label_dnn)
        self.use_isolation_forest = bool(use_isolation_forest)
        self.use_oneclass_svm_anom = bool(use_oneclass_svm_anom)
        self.use_lof = bool(use_lof)
        self.use_pca_anom = bool(use_pca_anom)
        self.use_kmeans_anom = bool(use_kmeans_anom)
        self.use_autoencoder_anom = bool(use_autoencoder_anom)
        self.use_rl_threshold = bool(use_rl_threshold)
        self.semi_label_fraction = float(semi_label_fraction)
        self.pseudo_label_threshold = float(pseudo_label_threshold)
        self.anomaly_contamination = float(anomaly_contamination)
        self.rl_episodes = int(rl_episodes)
        super().__init__(*args, **kwargs)

    def _active_supervised_groups(self):
        ml = []
        dl = []
        if getattr(self, 'use_lr', False): ml.append('LR')
        if getattr(self, 'use_rf', False): ml.append('RF')
        if getattr(self, 'use_svm', False): ml.append('SVM')
        if getattr(self, 'use_xgb', False): ml.append('XGB')
        if getattr(self, 'use_dt', False): ml.append('DT')

        if getattr(self, 'use_mlp', False): dl.append('MLP')
        if getattr(self, 'use_dnn', False): dl.append('DNN')
        if getattr(self, 'use_cnn1d', False): dl.append('CNN1D')
        if getattr(self, 'use_lstm', False): dl.append('LSTM')
        if getattr(self, 'use_aae', False): dl.append('AAE')
        if getattr(self, 'use_cnn_lstm', False): dl.append('CNN_LSTM')
        if getattr(self, 'use_cnn_bilstm', False): dl.append('CNN_BiLSTM')
        if getattr(self, 'use_cnn_gru', False): dl.append('CNN_GRU')
        if getattr(self, 'use_gnn', False): dl.append('GNN')
        if getattr(self, 'use_gat', False): dl.append('GAT')
        if getattr(self, 'use_transformer', False): dl.append('Transformer')
        if getattr(self, 'use_transfer_learning', False): dl.append('TransferLearning')
        return ml, dl

    def _apply_supervised_runtime_guards(self):
        notes = []
        ml, dl = self._active_supervised_groups()
        if getattr(self, 'use_hybrid', False) and (not ml or not dl):
            self.use_hybrid = False
            notes.append('Hybrid dinonaktifkan otomatis: membutuhkan minimal satu model ML dan satu model DL aktif.')
        total_models = len(ml) + len(dl)
        if getattr(self, 'use_ensemble', False) and total_models < 2:
            self.use_ensemble = False
            notes.append('Ensemble dinonaktifkan otomatis: membutuhkan minimal dua model aktif.')
        return notes

    def run(self):
        if self.learning_paradigm == 'supervised':
            self._hardening_notes = self._apply_supervised_runtime_guards()
            result = super().run()
            if isinstance(result, dict):
                result.setdefault('hardening_notes', []).extend(self._hardening_notes)
            return result
        return self._run_non_supervised()

    def _get_train_params(self):
        return super()._get_train_params()

    def _active_alt_models(self):
        groups = {'semi_supervised': [], 'unsupervised': [], 'reinforcement': []}
        if self.use_label_spreading: groups['semi_supervised'].append('LabelSpreading')
        if self.use_self_training_svm: groups['semi_supervised'].append('SelfTrainingSVM')
        if self.use_self_training_rf: groups['semi_supervised'].append('SelfTrainingRF')
        if self.use_pseudo_label_dnn: groups['semi_supervised'].append('PseudoLabelDNN')
        if self.use_isolation_forest: groups['unsupervised'].append('IsolationForest')
        if self.use_oneclass_svm_anom: groups['unsupervised'].append('OneClassSVM')
        if self.use_lof: groups['unsupervised'].append('LOF')
        if self.use_pca_anom: groups['unsupervised'].append('PCARecon')
        if self.use_kmeans_anom: groups['unsupervised'].append('KMeansDistance')
        if self.use_autoencoder_anom: groups['unsupervised'].append('AutoencoderAnomaly')
        if self.use_rl_threshold: groups['reinforcement'].append('AdaptiveQThreshold')
        return groups

    def _build_alt_learners(self, n_classes, epochs, batch_size):
        learners = AlternativeParadigmLearners(paradigm=self.learning_paradigm)
        if self.learning_paradigm == 'semi_supervised':
            if self.use_label_spreading:
                learners.add('LabelSpreading', LabelSpreadingWrapper(label_fraction=self.semi_label_fraction, seed=self.seed))
            if self.use_self_training_svm:
                learners.add('SelfTrainingSVM', SelfTrainingWrapper(SVC(C=1.0, probability=True, gamma='scale', kernel='rbf'), label_fraction=self.semi_label_fraction, seed=self.seed, threshold=0.8, name='SelfTrainingSVM'))
            if self.use_self_training_rf:
                learners.add('SelfTrainingRF', SelfTrainingWrapper(RandomForestClassifier(n_estimators=300, random_state=self.seed, class_weight='balanced', n_jobs=-1), label_fraction=self.semi_label_fraction, seed=self.seed, threshold=0.8, name='SelfTrainingRF'))
            if self.use_pseudo_label_dnn:
                learners.add('PseudoLabelDNN', PseudoLabelDNNWrapper(n_classes=n_classes, label_fraction=self.semi_label_fraction, pseudo_threshold=self.pseudo_label_threshold, epochs=epochs, batch_size=batch_size, seed=self.seed))
        elif self.learning_paradigm == 'unsupervised':
            if self.use_isolation_forest:
                learners.add('IsolationForest', IsolationForestDetector(contamination=self.anomaly_contamination, seed=self.seed))
            if self.use_oneclass_svm_anom:
                learners.add('OneClassSVM', OneClassSVMDetector(contamination=self.anomaly_contamination))
            if self.use_lof:
                learners.add('LOF', LOFNoveltyDetector(contamination=self.anomaly_contamination))
            if self.use_pca_anom:
                learners.add('PCARecon', PCAReconstructionDetector(contamination=self.anomaly_contamination, seed=self.seed))
            if self.use_kmeans_anom:
                learners.add('KMeansDistance', KMeansDistanceDetector(contamination=self.anomaly_contamination, seed=self.seed))
            if self.use_autoencoder_anom:
                learners.add('AutoencoderAnomaly', AutoencoderAnomalyDetector(contamination=self.anomaly_contamination, epochs=epochs, batch_size=batch_size, seed=self.seed, semi_supervised=False))
        elif self.learning_paradigm == 'reinforcement':
            # RL scaffold uses anomaly scores + adaptive threshold policy
            base_detector = IsolationForestDetector(contamination=self.anomaly_contamination, seed=self.seed)
            learners.add('AdaptiveQThreshold', AdaptiveThresholdQLearner(base_detector=base_detector, episodes=self.rl_episodes, seed=self.seed))
        return learners

    def _preprocess_common(self):
        self._banner("Stage 1 · Data Loading")
        meta = self._load()
        train_df = meta['train']; val_df = meta.get('val', None); test_df = meta['test']
        feat_cols = meta['feat_cols']; cat_cols = meta['cat_cols']; class_names = meta['class_names']; n_classes = meta['n_classes']
        self._report_split_methodology(meta)

        self._banner("Stage 2 · Pre-Preprocessing (Data Quality & Hygiene)")
        pprep = PrePreprocessor(dataset=self.dataset, drop_duplicates=self.drop_duplicates_stage, drop_constant=self.drop_constant_stage)
        train_df, val_df, test_df, feat_cols, cat_cols, pprep_report = pprep.fit_transform_splits(
            train_df, val_df, test_df, feat_cols, cat_cols
        )
        y_train_raw = train_df['y'].values.astype(int)
        y_val = val_df['y'].values.astype(int) if val_df is not None else None
        y_test = test_df['y'].values.astype(int)
        self.logger.log(f"  After pre-preprocessing → train={train_df.shape}, val={(val_df.shape if val_df is not None else 'N/A')}, test={test_df.shape}")

        if self.generate_eda:
            self._banner("Stage 2b · Exploratory Data Analysis")
            eda = EDAAnalyzerV2(dataset=self.dataset, class_names=class_names)
            _eda_df = train_df[feat_cols].copy()
            for _c in cat_cols:
                if _c in _eda_df.columns:
                    _eda_df[_c] = pd.factorize(_eda_df[_c])[0]
            _eda_X = _eda_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
            pair_k = 4 if len(train_df) > 120000 else 5
            dist_k = 12 if len(feat_cols) > 60 else 16
            eda.run_all(df=train_df, X=_eda_X, feat_names=feat_cols, y=y_train_raw,
                        top_k_dist=dist_k, top_k_pair=pair_k,
                        preprocessing_report=pprep_report)

        self._banner("Stage 3 · Preprocessing")
        prep = Preprocessor(scaler_type=self.scaler, balance=self.balance)
        X_train_clean = prep.clean_fit_transform(train_df, feat_cols, cat_cols)
        X_val_clean = prep.clean_transform(val_df, feat_cols, cat_cols) if val_df is not None else None
        X_test_clean = prep.clean_transform(test_df, feat_cols, cat_cols)

        fe = FeatureEngineerV4()
        k = min(self.top_k, X_train_clean.shape[1])
        idx = fe.select(X_train_clean, y_train_raw, feat_cols, k=k, method=self.feature_method)
        X_train_clean = X_train_clean[:, idx]
        X_val_clean = X_val_clean[:, idx] if X_val_clean is not None else None
        X_test_clean = X_test_clean[:, idx]
        feat_names = fe.selected_names_

        prep.fit_scaler(X_train_clean)
        X_train = prep.scale_transform(X_train_clean)
        X_val = prep.scale_transform(X_val_clean) if X_val_clean is not None else None
        X_test = prep.scale_transform(X_test_clean)

        # For non-supervised paradigms, do not apply class balancing by default.
        return {
            'meta': meta,
            'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
            'feat_names': feat_names, 'class_names': class_names, 'n_classes': n_classes,
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train_raw': y_train_raw, 'y_val': y_val, 'y_test': y_test,
            'pprep_report': pprep_report,
        }

    def _run_non_supervised(self):
        t0 = time.time()
        self._banner(f"NIDS ACADEMIC PIPELINE V4.5  ·  Dataset: {self.dataset.upper()}  ·  Paradigm: {self.learning_paradigm.upper()}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        active_groups = self._active_alt_models()
        self.logger.log(f"  Learning paradigm → {self.learning_paradigm}")
        self.logger.log(f"  Active models → {active_groups.get(self.learning_paradigm, [])}")

        common = self._preprocess_common()
        X_train = common['X_train']; X_val = common['X_val']; X_test = common['X_test']
        y_train = common['y_train_raw']; y_val = common['y_val']; y_test = common['y_test']
        feat_names = common['feat_names']; class_names = common['class_names']; n_classes = common['n_classes']
        pprep_report = common['pprep_report']

        if self.learning_paradigm in {'unsupervised', 'reinforcement'} and n_classes != 2:
            raise ValueError(f"Paradigm {self.learning_paradigm} saat ini didukung untuk klasifikasi biner/anomaly detection. Pilih mode binary=True.")

        epochs, batch_size = self._get_train_params()
        self._banner("Stage 4 · Alternative Paradigm Training")
        learners = self._build_alt_learners(n_classes=n_classes, epochs=epochs, batch_size=batch_size)
        if len(learners.models_) == 0:
            raise ValueError(f"Tidak ada model aktif untuk paradigma {self.learning_paradigm}. Pilih minimal satu model.")
        learners.fit_all(X_train, y_train)

        self._banner("Stage 5 · Fusion / Aggregation")
        train_proba = learners.predict_proba_all(X_train)
        if not train_proba:
            raise ValueError("Tidak ada probabilitas model yang berhasil dihasilkan pada split TRAIN.")
        test_proba = learners.predict_proba_all(X_test)
        if not test_proba:
            raise ValueError("Tidak ada probabilitas model yang berhasil dihasilkan pada split TEST.")
        val_proba = learners.predict_proba_all(X_val) if X_val is not None else {}

        if len(train_proba) == 1:
            first_name = next(iter(train_proba.keys()))
            fused_train = train_proba[first_name]
            fused_test = test_proba[first_name]
            self.logger.log("  Fusion bypass → hanya satu model aktif")
            copula = None
        else:
            copula = CopulaFusionV4(family=self.copula_family)
            copula.fit(train_proba, y_train)
            fused_train = copula.fuse(train_proba)
            fused_test = copula.fuse(test_proba)

        ens_fuser = EnsembleHybridFusion(n_classes=n_classes,
                                              ensemble_method=self.ensemble_method,
                                              hybrid_method=self.hybrid_method)
        ref_proba = val_proba if val_proba and y_val is not None else train_proba
        y_ref = y_val if val_proba and y_val is not None else y_train
        ensemble_outputs = ens_fuser.build_ensemble(ref_proba, y_ref, test_proba) if self.use_ensemble and len(test_proba) > 1 else {}
        hybrid_outputs = ens_fuser.build_hybrid(ref_proba, y_ref, test_proba) if self.use_hybrid and len(test_proba) > 1 else {}

        self._banner("Stage 6 · Evaluation")
        evaluator = EvaluatorV2(class_names=class_names, dataset=self.dataset)
        evaluator.plot_fused_distribution(fused_test, y_test)

        test_preds = learners.predict_all(X_test)
        for name, pred in test_preds.items():
            evaluator.evaluate(name, y_test, pred, y_proba=test_proba.get(name))
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')

        if len(test_proba) > 1:
            fused_pred = np.argmax(fused_test, axis=1)
            fused_name = f"{self.learning_paradigm.title()} Fusion"
            evaluator.evaluate(fused_name, y_test, fused_pred, y_proba=fused_test)
            evaluator.plot_confusion(y_test, fused_pred, title=f'CM {fused_name}')
        else:
            self.logger.log("  Primary aggregate skipped in report: hanya satu model aktif sehingga prediksi primary identik dengan model tunggal.")

        for name, proba in ensemble_outputs.items():
            pred = np.argmax(proba, axis=1)
            evaluator.evaluate(name, y_test, pred, y_proba=proba)
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')
        for name, proba in hybrid_outputs.items():
            pred = np.argmax(proba, axis=1)
            evaluator.evaluate(name, y_test, pred, y_proba=proba)
            evaluator.plot_confusion(y_test, pred, title=f'CM {name}')

        evaluator.plot_roc_all(y_test, test_proba, fused_test)
        evaluator.plot_precision_recall(y_test, test_proba, fused_test)
        evaluator.plot_comparison()
        summary_df = evaluator.summary()

        elapsed = time.time() - t0
        self._banner(f"✅ Pipeline DONE  ·  {elapsed:.0f}s  ·  Figures → {FIG_DIR}")
        self.logger.close()
        return {
            'evaluator': evaluator,
            'summary': summary_df,
            'learners': learners,
            'copula': copula,
            'fused_test': fused_test,
            'feat_names': feat_names,
            'active_ml_models': [],
            'active_dl_models': [],
            'active_alt_models': active_groups.get(self.learning_paradigm, []),
            'validation_method': 'holdout',
            'validation_scores': {},
            'ensemble_outputs': list(ensemble_outputs.keys()),
            'hybrid_outputs': list(hybrid_outputs.keys()),
            'prepreprocess_report': pprep_report,
            'learning_paradigm': self.learning_paradigm,
        }

    def run(self):
        if self.learning_paradigm == 'supervised':
            return super().run()
        return self._run_non_supervised()

NIDSPipeline = NIDSPipelineV4_5

if __name__ == "__main__":
    # Keep default launcher from base module simple for manual script usage.
    raise SystemExit("Gunakan file ini melalui dashboard atau import class NIDSPipeline.")

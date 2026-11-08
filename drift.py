"""Data drift: PSI (Population Stability Index). <0.1 OK, 0.1-0.25 WARN, >0.25 ALERT"""
import numpy as np
import pandas as pd


def _props(counts):
    counts = np.asarray(counts, float)
    return (counts / max(counts.sum(), 1)).tolist()


def psi(expected, actual, eps=1e-4):
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(df, num_cols, cat_cols, bins=10):
    """Training data er distribution save kore rakhi (pore compare korbo)."""
    ref = {"numeric": {}, "categorical": {}}
    # 0/1 ba kom-value column (jemon is_foreign) ke category hishebe dhori
    low = [c for c in num_cols if df[c].nunique() <= 5]
    num_cols = [c for c in num_cols if c not in low]
    cat_cols = list(cat_cols) + low
    for c in num_cols:
        s = df[c].dropna()
        edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)[1:-1]))
        idx = np.searchsorted(edges, s, side="right")
        counts = np.bincount(idx, minlength=len(edges) + 1)
        ref["numeric"][c] = {"edges": edges.tolist(), "expected": _props(counts)}
    for c in cat_cols:
        ref["categorical"][c] = df[c].astype(str).value_counts(normalize=True).to_dict()
    return ref


def compare(ref, df):
    """Notun data vs training data: protita column er PSI."""
    out = {}
    for c, info in ref["numeric"].items():
        s = df[c].dropna()
        idx = np.searchsorted(info["edges"], s, side="right")
        counts = np.bincount(idx, minlength=len(info["edges"]) + 1)
        out[c] = psi(info["expected"], _props(counts))
    for c, exp in ref["categorical"].items():
        cats = list(exp)
        act = df[c].astype(str).value_counts(normalize=True)
        e = [exp[k] for k in cats] + [0.0]
        a = [float(act.get(k, 0.0)) for k in cats]
        a.append(max(0.0, 1 - sum(a)))  # notun/unseen category
        out[c] = psi(e, a)
    return out


def status(value):
    return "OK" if value < 0.1 else "WARN" if value < 0.25 else "ALERT"

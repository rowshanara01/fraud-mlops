"""Usage: python monitor.py data/new_batch_drifted.csv   -> drift report (ALERT hole exit code 1)"""
import sys

import pandas as pd

import drift
from registry import load_latest

path = sys.argv[1] if len(sys.argv) > 1 else "data/new_batch_ok.csv"
version, bundle, metrics, reference = load_latest()
new = pd.read_csv(path)

report = drift.compare(reference, new)
print(f"model {version} | batch: {path} ({len(new)} rows)\n")
worst = "OK"
for col, v in sorted(report.items(), key=lambda kv: -kv[1]):
    s = drift.status(v)
    worst = s if s == "ALERT" or (s == "WARN" and worst == "OK") else worst
    print(f"{col:18s} PSI={v:6.3f}  {s}")

# Prediction drift: model koto % transaction ke fraud bolche vs training time
proba = bundle["model"].predict_proba(new[bundle["columns"]])[:, 1]
flag = float((proba >= bundle["threshold"]).mean())
print(f"\nflag rate now={flag:.3%} vs reference={metrics['reference_flag_rate']:.3%}")
print("\nOVERALL:", worst, "-> retrain kora dorkar" if worst == "ALERT" else "")
sys.exit(1 if worst == "ALERT" else 0)

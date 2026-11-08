"""Chhoto model registry: model/registry/<version>/ + model/latest.txt"""
import json
from pathlib import Path

import joblib

ROOT = Path("model")


def load_latest():
    version = (ROOT / "latest.txt").read_text().strip()
    d = ROOT / "registry" / version
    bundle = joblib.load(d / "model.joblib")
    metrics = json.loads((d / "metrics.json").read_text())
    reference = json.loads((d / "reference.json").read_text())
    return version, bundle, metrics, reference

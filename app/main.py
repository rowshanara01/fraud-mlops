"""Fraud API: validation, versioned model, latency + prediction logging (monitoring er jonno)"""
import json
import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from registry import load_latest

VERSION, BUNDLE, METRICS, _ = load_latest()
MODEL, THRESHOLD, COLUMNS = BUNDLE["model"], BUNDLE["threshold"], BUNDLE["columns"]
LOG = Path("logs/predictions.jsonl")
LOG.parent.mkdir(exist_ok=True)

app = FastAPI(title="Fraud Detection API", version=VERSION)


class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    category: Literal["grocery", "online", "travel", "electronics", "restaurant"]
    distance_km: Optional[float] = Field(default=None, ge=0)
    is_foreign: Literal[0, 1]
    txn_count_24h: int = Field(ge=0, le=1000)
    new_device: Literal[0, 1]
    account_age_days: int = Field(ge=0)


def score(txns: list[Transaction]):
    start = time.perf_counter()
    df = pd.DataFrame([t.model_dump() for t in txns])[COLUMNS]
    probs = MODEL.predict_proba(df)[:, 1]
    ms = (time.perf_counter() - start) * 1000
    results = [{"fraud_probability": round(float(p), 4), "is_fraud": bool(p >= THRESHOLD)} for p in probs]
    with open(LOG, "a") as f:  # prediction log -> pore drift/performance monitor kora jay
        for t, r in zip(txns, results):
            f.write(json.dumps({"ts": time.time(), "version": VERSION, **t.model_dump(), **r}) + "\n")
    return results, round(ms, 2)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": VERSION}


@app.get("/model-info")
def model_info():
    return {"version": VERSION, **METRICS}


@app.post("/predict")
def predict(txn: Transaction):
    results, ms = score([txn])
    return {**results[0], "model_version": VERSION, "latency_ms": ms}


@app.post("/predict-batch")
def predict_batch(txns: list[Transaction]):
    results, ms = score(txns)
    return {"count": len(results), "model_version": VERSION, "latency_ms": ms, "results": results}

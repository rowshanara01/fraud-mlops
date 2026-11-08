"""Training: compare -> imbalanced-aware threshold -> evaluate -> versioned save (+ drift reference + run log)"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import drift

SEED = 42
MIN_PRECISION = 0.50  # business rule: alert gulor komport 50% sotti fraud hote hobe

df = pd.read_csv("data/transactions.csv")
X, y = df.drop(columns="is_fraud"), df["is_fraud"]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"fraud rate: {y.mean():.3%}  (imbalanced!)")

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
candidates = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "hist_gb": HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.05,
                                              max_depth=3, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# Imbalanced data te PR-AUC (average_precision) accuracy er cheye onek bhalo metric
scores = {}
for name, est in candidates.items():
    s = cross_val_score(Pipeline([("pre", pre), ("clf", est)]), X_train, y_train, cv=cv,
                        scoring="average_precision")
    scores[name] = float(s.mean())
    print(f"{name:8s} CV PR-AUC = {s.mean():.3f} +/- {s.std():.3f}")
best = max(scores, key=scores.get)
model = Pipeline([("pre", pre), ("clf", candidates[best])])
print("best:", best)

# Threshold: precision >= MIN_PRECISION rekhe jotota beshi recall (fraud dhora) possible
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
ok = np.where(prec[:-1] >= MIN_PRECISION)[0]
threshold = float(thr[ok[np.argmax(rec[:-1][ok])]]) if len(ok) else 0.5
print(f"threshold = {threshold:.3f}")

model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = proba >= threshold
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics = {
    "best_model": best, "cv_pr_auc": scores, "threshold": threshold,
    "test_pr_auc": float(average_precision_score(y_test, proba)),
    "test_precision": float(tp / max(tp + fp, 1)), "test_recall": float(tp / max(tp + fn, 1)),
    "test_flag_rate": float(pred.mean()), "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "reference_flag_rate": float(pred.mean()),
}
print(json.dumps(metrics, indent=2))

# ---- Versioned save (registry) ----
version = datetime.now().strftime("%Y%m%d-%H%M%S")
d = Path("model/registry") / version
d.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "threshold": threshold, "columns": X.columns.tolist()}, d / "model.joblib")
(d / "metrics.json").write_text(json.dumps(metrics, indent=2))
(d / "reference.json").write_text(json.dumps(drift.build_reference(X_train, num_cols, cat_cols)))
Path("model/latest.txt").write_text(version)
with open("model/runs.jsonl", "a") as f:  # chhoto experiment log
    f.write(json.dumps({"version": version, **{k: metrics[k] for k in ("best_model", "test_pr_auc", "test_precision", "test_recall")}}) + "\n")
print("saved version:", version)

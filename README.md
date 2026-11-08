#### Fraud Detection (Production-style ML)

## Chalanor niyom ##
pip install -r requirements.txt
python generate_data.py                     # train data + ok batch + drifted batch
python train.py                             # compare -> threshold -> versioned save
pytest -q
uvicorn app.main:app --reload               # localhost:8000/docs
python monitor.py data/new_batch_ok.csv        # drift OK
python monitor.py data/new_batch_drifted.csv   # drift ALERT (exit code 1)

## Company te ja ja use hoy (ei project e ache)
- Imbalanced data (fraud ~3.6%): PR-AUC, class_weight, precision-constrained threshold
- Model registry/versioning: model/registry/<version>/ (model + metrics + drift reference)
- Experiment log: model/runs.jsonl
- API: validation, latency, prediction logging (logs/predictions.jsonl), /model-info
- Drift monitoring: PSI per feature + prediction flag rate; ALERT hole exit code 1 (cron/CI e kaje lage)
- Tests (API + drift), Dockerfile, GitHub Actions CI (.github/workflows/ci.yml)

## Challenge
1. monitor.py ke cron/Airflow diye roj chalao, ALERT hole Slack/email pathao
2. logs/predictions.jsonl theke monitor chalao (real traffic simulation)
3. Auto-retrain: ALERT hole train.py chalano, notun version e rollback support
4. SHAP diye per-transaction explanation
5. Cost-based threshold (FN/FP cost) add koro

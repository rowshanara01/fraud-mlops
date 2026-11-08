from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
FRAUDY = {"amount": 2500, "hour": 3, "category": "electronics", "distance_km": 400,
          "is_foreign": 1, "txn_count_24h": 9, "new_device": 1, "account_age_days": 10}
NORMAL = {"amount": 30, "hour": 14, "category": "grocery", "distance_km": 3,
          "is_foreign": 0, "txn_count_24h": 1, "new_device": 0, "account_age_days": 1500}


def test_health_has_version():
    assert "model_version" in client.get("/health").json()


def test_fraudy_scores_higher_and_is_flagged():
    f = client.post("/predict", json=FRAUDY).json()
    n = client.post("/predict", json=NORMAL).json()
    assert f["fraud_probability"] > n["fraud_probability"]
    assert f["is_fraud"] is True and n["is_fraud"] is False


def test_missing_distance_ok():
    body = {k: v for k, v in NORMAL.items() if k != "distance_km"}
    assert client.post("/predict", json=body).status_code == 200


def test_bad_input_rejected():
    assert client.post("/predict", json={**NORMAL, "hour": 99}).status_code == 422
    assert client.post("/predict", json={**NORMAL, "category": "gold"}).status_code == 422


def test_batch():
    r = client.post("/predict-batch", json=[FRAUDY, NORMAL]).json()
    assert r["count"] == 2

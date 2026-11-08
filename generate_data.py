"""Synthetic transaction data (fraud ~3.6%). Sathe 'drift' kora notun batch-o banay (monitoring dekhar jonno)."""
import numpy as np
import pandas as pd


def make(n, seed, drift=False, with_label=True):
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.8, 1.0, n)
    if drift:
        amount *= 1.8  # manush hothat beshi khoroch korche
    hour = rng.integers(0, 24, n)
    category = rng.choice(["grocery", "online", "travel", "electronics", "restaurant"], n,
                          p=[0.3, 0.3, 0.1, 0.1, 0.2])
    distance = rng.exponential(15, n)
    is_foreign = (rng.random(n) < (0.20 if drift else 0.05)).astype(int)
    txn24 = rng.poisson(2, n)
    new_device = (rng.random(n) < 0.10).astype(int)
    age = rng.integers(1, 3000, n)

    lin = (0.9 * np.log1p(amount) + 1.8 * is_foreign + 1.5 * new_device
           + 0.35 * txn24 + 0.01 * distance + 1.3 * (hour < 5)
           + 0.9 * (category == "electronics") + 0.7 * (category == "online")
           - 0.0004 * age + rng.normal(0, 0.4, n))
    z = 2.5 * (lin - np.quantile(lin, 0.975))  # top ~2.5% sondehojonok = fraud
    fraud = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "category": category,
        "distance_km": distance.round(2), "is_foreign": is_foreign,
        "txn_count_24h": txn24, "new_device": new_device, "account_age_days": age,
    })
    df.loc[rng.choice(n, int(n * 0.03), replace=False), "distance_km"] = np.nan
    if with_label:
        df["is_fraud"] = fraud
    return df


if __name__ == "__main__":
    train = make(20000, seed=1)
    train.to_csv("data/transactions.csv", index=False)
    print("train rows:", len(train), "| fraud rate:", round(train.is_fraud.mean(), 4))
    # production e label pore ashe, tai notun batch e label nai
    make(3000, seed=2, with_label=False).to_csv("data/new_batch_ok.csv", index=False)
    make(3000, seed=3, drift=True, with_label=False).to_csv("data/new_batch_drifted.csv", index=False)
    print("saved new_batch_ok.csv + new_batch_drifted.csv")

import pandas as pd
import drift


def _ref():
    df = pd.read_csv("data/transactions.csv").drop(columns="is_fraud")
    num = df.select_dtypes("number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    return drift.build_reference(df, num, cat)


def test_same_distribution_no_alert():
    rep = drift.compare(_ref(), pd.read_csv("data/new_batch_ok.csv"))
    assert max(rep.values()) < 0.1


def test_drifted_batch_alerts():
    rep = drift.compare(_ref(), pd.read_csv("data/new_batch_drifted.csv"))
    assert rep["amount"] > 0.25 and rep["is_foreign"] > 0.1

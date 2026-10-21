import numpy as np
import pandas as pd

from src.ml_signal import attach_ml_up_proba, training_history_start


def test_attach_ml_up_proba_adds_column():
    idx = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    df = pd.DataFrame({"Close": close, "Volatility": np.full(len(idx), 0.25)})
    out = attach_ml_up_proba(df, eval_start="2025-03-01", min_train_rows=30)
    assert "ML_Up_Proba" in out.columns
    assert len(out) == len(df)
    assert out["ML_Up_Proba"].between(0.0, 1.0).all()


def test_training_history_starts_before_eval_window():
    start = training_history_start("2024-06-01")
    assert start == (pd.Timestamp("2024-06-01") - pd.Timedelta(days=504)).date().isoformat()
    assert start < "2024-06-01"

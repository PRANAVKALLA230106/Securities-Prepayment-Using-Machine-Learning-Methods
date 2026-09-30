"""End-to-end smoke test on synthetic data in the exact Freddie Mac file format."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepay import pipeline  # noqa: E402
from prepay.features import build_table  # noqa: E402
from prepay import freddie  # noqa: E402
from scripts.make_synthetic_data import make  # noqa: E402


def _cfg(tmp: Path) -> dict:
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    cfg["paths"] = {k: str(tmp / k) for k in ["raw_dir", "macro_dir", "processed_dir", "results_dir"]}
    cfg["svm"]["train_rows"] = 1500
    cfg["neural_net"]["epochs"] = 2
    cfg["gradient_boosting"]["max_iter"] = 30
    return cfg


def test_labels_are_next_month_and_leak_free(tmp_path):
    make(tmp_path, years=[2012], loans_per_year=200, seed=1)
    for src, dst in [("raw", "raw_dir"), ("macro", "macro_dir")]:
        (tmp_path / src).rename(tmp_path / dst)
    orig = freddie.select_loans(freddie.load_origination(tmp_path / "raw_dir", 2012), 360, None, 0)
    perf = freddie.load_performance(tmp_path / "raw_dir", 2012, set(orig["loan_id"]))
    df = build_table(orig, perf, None)

    usable = df[df["y_next"].notna()]
    # every positive row is followed by that loan's payoff row
    payoff_rows = df[df["y_same"] == 1]
    assert usable["y_next"].sum() > 0
    assert usable["y_next"].sum() <= len(payoff_rows)
    # payoff rows (balance already 0) are never used as feature rows in the fixed task
    assert df.loc[df["y_same"] == 1, "y_next"].isna().all()
    # masked zero balances on active rows are not treated as "paid off"
    assert (usable["upb_ratio"] > 0).all()


def test_full_pipeline_runs(tmp_path):
    cfg = _cfg(tmp_path)
    make(tmp_path / "gen", years=[2008, 2016], loans_per_year=400, seed=3)
    (tmp_path / "gen" / "raw").rename(tmp_path / "raw_dir")
    (tmp_path / "gen" / "macro").rename(tmp_path / "macro_dir")

    data_path = pipeline.prepare(cfg, tmp_path)
    res = pipeline.run(cfg, tmp_path, data_path)

    assert set(res["key"]) == set(cfg["models"])
    assert res["roc_auc"].between(0, 1).all()
    out = tmp_path / "results_dir"
    assert (out / "RESULTS.md").exists()
    assert (out / "figures" / "leakage_demo.png").exists()
    assert len(pd.read_csv(out / "metrics.csv")) == len(cfg["models"])

    # demo loads the saved models and scores held-out rows without training
    import demo

    shown = demo.run_demo(cfg, tmp_path, n=10, seed=0)
    assert len(shown) == 10 and shown["y"].sum() == 5
    assert all(shown[f"p_{m}"].between(0, 1).all() for m in cfg["models"])

    # loan-level split: no loan in two sets
    df = pd.read_parquet(data_path)
    df = df[df["y_next"].notna()].reset_index(drop=True)
    from prepay.features import split

    tr, va, te = split(df, cfg["split"], cfg["seed"])
    sets = [set(df.loc[i, "loan_id"]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])

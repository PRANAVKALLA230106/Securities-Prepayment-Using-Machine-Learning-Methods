"""Live demo: score held-out loan-months with the saved models. No training.

    python demo.py                  # 10 held-out test loan-months (half of them prepay)
    python demo.py --n 6 --seed 7   # a different draw
    python demo.py --synthetic      # use the models from `run_all.py --synthetic`

Needs a finished `python run_all.py` first: it saves every model, the fitted
preprocessor and a few test-set loan-months to models/. None of these rows
were used for training, threshold selection or calibration.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepay import evaluate as ev  # noqa: E402

SHORT = {
    "baseline": "Base",
    "logistic_base": "LR",
    "logistic_l1": "LR-L1",
    "logistic_l2": "LR-L2",
    "svm_poly": "SVM-p",
    "svm_rbf": "SVM-r",
    "svm_sigmoid": "SVM-s",
    "gda_linear": "LDA",
    "gda_quadratic": "QDA",
    "neural_net": "NN",
    "gradient_boosting": "GBM",
}

KEY_FEATURES = {
    "credit_score": ("FICO", "{:.0f}"),
    "orig_interest_rate": ("rate", "{:.2f}"),
    "mortgage_rate": ("mkt", "{:.2f}"),
    "rate_incentive": ("incent", "{:+.2f}"),
    "loan_age": ("age", "{:.0f}"),
    "upb_ratio": ("upb%", "{:.2f}"),
    "mtm_ltv": ("mtmLTV", "{:.0f}"),
    "burnout_months": ("burn", "{:.0f}"),
}


def _fmt(v, f):
    return "-" if pd.isna(v) else f.format(v)


def run_demo(cfg: dict, root: Path, n: int = 10, seed: int = 42) -> pd.DataFrame:
    model_dir = root / cfg["paths"].get("models_dir", "models")
    res_dir = root / cfg["paths"]["results_dir"]
    if not (model_dir / "preprocessor.joblib").exists():
        raise SystemExit(f"No saved models in {model_dir}. Run `python run_all.py` first.")

    prep = joblib.load(model_dir / "preprocessor.joblib")
    sample = pd.read_parquet(model_dir / "demo_sample.parquet")
    names = [m for m in cfg["models"] if (model_dir / f"{m}.joblib").exists()]
    models = {m: joblib.load(model_dir / f"{m}.joblib") for m in names}

    # half prepay, half not (prepayment is ~1-2% of rows, so a random draw would show none)
    rng = np.random.default_rng(seed)
    pos, neg = sample.index[sample["y_next"] == 1], sample.index[sample["y_next"] == 0]
    k = min(n // 2, len(pos))
    pick = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, min(n - k, len(neg)), replace=False)])
    rows = sample.loc[rng.permutation(pick)].reset_index(drop=True)
    X = prep.transform(rows)
    y = rows["y_next"].astype(int).to_numpy()

    print(f"\n{len(rows)} held-out TEST loan-months (not seen in training). Task: will the loan pay off NEXT month?\n")
    head = f"{'#':>2}  {'loan_id':<13}{'month':>7}  " + "".join(f"{lab:>8}" for lab, _ in KEY_FEATURES.values())
    print(head)
    print("-" * len(head))
    for i, r in rows.iterrows():
        feats = "".join(f"{_fmt(r.get(c), f):>8}" for c, (_, f) in KEY_FEATURES.items())
        print(f"{i + 1:>2}  {str(r['loan_id']):<13}{int(r['period']):>7}  {feats}")
    print(
        "\nFICO = credit score, rate = loan's rate, mkt = 30y market rate, incent = rate - mkt (refinance incentive),"
        "\nage = months since origination, upb% = balance left / original, mtmLTV = mark-to-market LTV,"
        "\nburn = months the loan already had a refinance incentive and did not prepay"
    )

    probs, preds = {}, {}
    for m, model in models.items():
        s = model.score(X)
        probs[m] = ev.calibrated_proba(model.calibrator, s)
        preds[m] = (s >= model.threshold).astype(int)

    print("\nPrepay probability per model (calibrated on the validation set); * = model predicts PREPAY\n")
    head = f"{'#':>2}  {'true':<9}" + "".join(f"{SHORT.get(m, m):>8}" for m in names)
    print(head)
    print("-" * len(head))
    for i in range(len(rows)):
        cells = "".join(f"{100 * probs[m][i]:>6.1f}%{'*' if preds[m][i] else ' '}" for m in names)
        print(f"{i + 1:>2}  {'PREPAY' if y[i] else 'no':<9}{cells}")
    print(f"{'':>2}  {'correct':<9}" + "".join(f"{f'{int((preds[m] == y).sum())}/{len(y)}':>8}" for m in names))

    metrics_path = res_dir / "metrics.csv"
    if metrics_path.exists():
        t = pd.read_csv(metrics_path).sort_values("pr_auc", ascending=False)
        cols = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
        print(f"\nFull test-set comparison ({metrics_path.relative_to(root)}, sorted by PR-AUC):\n")
        print(f"{'model':<24}" + "".join(f"{c:>11}" for c in ["accuracy", "precision", "recall", "F1", "PR-AUC", "ROC-AUC"]))
        for _, r in t.iterrows():
            print(f"{r['model']:<24}{100 * r['accuracy']:>10.2f}%" + "".join(f"{r[c]:>11.3f}" for c in cols[1:]))
        print(
            "\nNote: the demo rows are half prepayments; in the real test set only "
            f"{100 * (t.iloc[0]['tp'] + t.iloc[0]['fn']) / (t.iloc[0][['tn', 'fp', 'fn', 'tp']].sum()):.2f}% of "
            "loan-months prepay, which is why accuracy alone is misleading."
        )

    return pd.DataFrame({"y": y, **{f"p_{m}": probs[m] for m in names}, **{f"pred_{m}": preds[m] for m in names}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=10, help="number of loan-months to show")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true", help="use the models from `run_all.py --synthetic`")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.synthetic:
        cfg["paths"]["models_dir"], cfg["paths"]["results_dir"] = "models_synthetic", "results_synthetic"
    t = time.time()
    run_demo(cfg, ROOT, n=args.n, seed=args.seed)
    print(f"\n(demo ran in {time.time() - t:.1f}s, no training)")


if __name__ == "__main__":
    main()

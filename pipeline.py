"""End-to-end pipeline: prepare data -> train models -> write results."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import freddie, macro
from .features import Preprocessor, build_table, feature_columns, split
from .models import build

PRETTY = {
    "baseline": "Baseline (always 'no')",
    "logistic_base": "Logistic",
    "logistic_l1": "Logistic L1",
    "logistic_l2": "Logistic L2",
    "svm_poly": "SVM (poly)",
    "svm_rbf": "SVM (RBF)",
    "svm_sigmoid": "SVM (sigmoid)",
    "gda_linear": "GDA (linear)",
    "gda_quadratic": "GDA (quadratic)",
    "neural_net": "Neural network",
    "gradient_boosting": "Gradient boosting",
}


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------- data
def prepare(cfg: dict, root: Path, rebuild: bool = False) -> Path:
    paths = {k: root / v for k, v in cfg["paths"].items()}
    out = paths["processed_dir"] / "loan_months.parquet"
    meta_path = paths["processed_dir"] / "meta.json"

    years = cfg["data"]["years"] or freddie.find_years(paths["raw_dir"])
    if not years:
        raise SystemExit(
            f"No Freddie Mac sample files found in {paths['raw_dir']}.\n"
            "Download sample_YYYY.zip files (see README) and put them there."
        )
    meta = {"years": years, "data": cfg["data"], "macro": cfg["macro"], "seed": cfg["seed"]}
    if out.exists() and meta_path.exists() and not rebuild:
        if json.loads(meta_path.read_text()) == json.loads(json.dumps(meta)):
            print(f"[data] using cached {_rel(out, root)} (run with --rebuild to redo)")
            return out

    print(f"[data] building dataset from vintages {years}")
    mac = macro.load_macro(cfg["macro"], paths["macro_dir"]) if cfg["macro"]["enabled"] else None
    tables = []
    for year in years:
        t = time.time()
        orig = freddie.load_origination(paths["raw_dir"], year)
        sel = freddie.select_loans(orig, cfg["data"]["loan_term_months"], cfg["data"]["loans_per_year"], cfg["seed"] + year)
        perf = freddie.load_performance(paths["raw_dir"], year, set(sel["loan_id"]))
        tab = build_table(sel, perf, mac)
        tables.append(tab)
        print(
            f"  {year}: {len(sel):,} loans, {len(tab):,} loan-months, "
            f"{int(tab['y_same'].sum()):,} payoffs ({time.time() - t:.0f}s)"
        )
    df = pd.concat(tables, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object and c != "loan_id":
            df[c] = df[c].astype("category")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[data] saved {len(df):,} rows to {_rel(out, root)}")
    return out


# ----------------------------------------------------------------------------- models
def _fit_eval(name, cfg, X, y, idx, seed):
    tr, va, te = idx
    model = build(name, cfg, seed)
    t = time.time()
    model.fit(X[tr], y[tr], X[va], y[va])
    fit_s = time.time() - t
    s_va, s_te = model.score(X[va]), model.score(X[te])
    thr = ev.pick_threshold(y[va], s_va)
    m = ev.metrics(y[te], s_te, thr)
    m.update(threshold=thr, fit_seconds=round(fit_s, 1))
    return model, m, s_te


def run(cfg: dict, root: Path, data_path: Path, only: list[str] | None = None) -> pd.DataFrame:
    seed = cfg["seed"]
    res_dir = root / cfg["paths"]["results_dir"]
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(data_path)
    df = full[full["y_next"].notna()].reset_index(drop=True)
    use_macro = cfg["macro"]["enabled"]
    num, cat = feature_columns(use_macro)
    tr, va, te = split(df, cfg["split"], seed)
    prep = Preprocessor(num, cat).fit(df.loc[tr])
    X = prep.transform(df)
    y = df["y_next"].to_numpy(int)
    idx = (tr.to_numpy(), va.to_numpy(), te.to_numpy())

    summary = {
        "vintages": sorted(int(v) for v in df["vintage"].unique()),
        "loans": int(df["loan_id"].nunique()),
        "loan_months": len(df),
        "features": X.shape[1],
        "prepay_rate": float(y.mean()),
        "split": cfg["split"]["method"],
        "train/val/test rows": [len(tr), len(va), len(te)],
        "test_prepay_rate": float(y[idx[2]].mean()),
    }
    print("[model] dataset:", json.dumps(summary))

    names = only or cfg["models"]
    rows, scores, fitted = [], {}, {}
    for name in names:
        print(f"[model] {PRETTY.get(name, name)} ...")
        model, m, s_te = _fit_eval(name, cfg, X, y, idx, seed)
        fitted[name], scores[name] = model, s_te
        m = {"model": PRETTY.get(name, name), "key": name, **m}
        rows.append(m)
        print(
            f"      test: acc {m['accuracy']:.4f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
            f"PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  ({m['fit_seconds']}s)"
        )
        ev.plot_confusion(m, PRETTY.get(name, name), fig_dir / f"confusion_{name}.png")
        pd.DataFrame(rows).to_csv(res_dir / "metrics.csv", index=False)  # save progress

    yte = y[idx[2]]
    ev.plot_metric_bars(rows, "pr_auc", "PR-AUC", fig_dir / "pr_auc.png", ref=yte.mean(), ref_label="random")
    ev.plot_metric_bars(rows, "roc_auc", "ROC-AUC", fig_dir / "roc_auc.png", ref=0.5, ref_label="random")
    # best model of each family (logistic, svm, gda, ...), then the top 4 of those
    best_per_family = {}
    for r in sorted(rows, key=lambda r: -r["pr_auc"]):
        fam = r["key"].split("_")[0]
        if r["key"] != "baseline" and fam not in best_per_family:
            best_per_family[fam] = r["key"]
    top = list(best_per_family.values())[:4]
    ev.plot_pr_curves({PRETTY[k]: (yte, scores[k]) for k in top}, yte, fig_dir / "pr_curves.png")
    ev.plot_pca(X[idx[0]], y[idx[0]], cfg["evaluation"]["pca_points"], seed, fig_dir / "pca_3d.png")
    if "logistic_l1" in fitted:
        ev.plot_top_features(prep.columns, fitted["logistic_l1"].coef_, fig_dir / "top_features.png")

    leak = None
    if cfg["leakage_demo"]["enabled"]:
        leak = leakage_demo(cfg, full, num, cat, rows, X, y, idx, fig_dir)

    write_report(cfg, root, res_dir, rows, summary, leak)
    return pd.DataFrame(rows)


def leakage_demo(cfg, full, num, cat, rows, X, y, idx, fig_dir):
    """Paper setup: same-month label, payoff rows kept, raw balance, random row split."""
    print("[leakage demo] re-running L2 logistic regression with the paper's setup ...")
    seed = cfg["seed"]
    d = full  # modified in place; not needed afterwards
    d["upb_ratio"] = d["upb_ratio_raw"]  # raw balance: 0 in the payoff month
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(d))
    n_tr, n_va = int(0.6 * len(d)), int(0.2 * len(d))
    tr, va, te = perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]
    num_p = [c for c in num if c != "upb_masked"]
    prep = Preprocessor(num_p, cat).fit(d.iloc[tr])
    Xp = prep.transform(d)
    yp = d["y_same"].to_numpy(int)
    _, paper, _ = _fit_eval("logistic_l2", cfg, Xp, yp, (tr, va, te), seed)
    fixed = next((r for r in rows if r["key"] == "logistic_l2"), None)
    if fixed is None:
        _, fixed, _ = _fit_eval("logistic_l2", cfg, X, y, idx, seed)
    print(
        f"      paper setup: acc {paper['accuracy']:.4f}  recall {paper['recall']:.3f}  PR-AUC {paper['pr_auc']:.3f}\n"
        f"      fixed setup: acc {fixed['accuracy']:.4f}  recall {fixed['recall']:.3f}  PR-AUC {fixed['pr_auc']:.3f}"
    )
    ev.plot_leakage(paper, fixed, fig_dir / "leakage_demo.png")
    return {"paper": paper, "fixed": fixed}


# ----------------------------------------------------------------------------- report
def _pct(x):
    return f"{100 * x:.2f}%"


def write_report(cfg, root, res_dir, rows, summary, leak):
    lines = [
        "# Results",
        "",
        f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python run_all.py`._",
        "",
        "## Data",
        "",
        f"- Vintages: {', '.join(map(str, summary['vintages']))}",
        f"- Loans: {summary['loans']:,}  |  loan-months: {summary['loan_months']:,}  |  features: {summary['features']}",
        f"- Monthly prepayment rate: {_pct(summary['prepay_rate'])} "
        f"(so always answering 'no prepayment' is already {_pct(1 - summary['test_prepay_rate'])} accurate on test)",
        f"- Split: **{summary['split']}**, train / val / test rows = "
        + " / ".join(f"{n:,}" for n in summary["train/val/test rows"]),
        "- Decision threshold for each model: the one with the best F1 on the validation set.",
        "",
        "## Test-set performance",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | True no-prepay | True prepay | Fit (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["pr_auc"]):
        lines.append(
            f"| {r['model']} | {_pct(r['accuracy'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"**{r['pr_auc']:.3f}** | {r['roc_auc']:.3f} | {_pct(r['true_no_prepay_rate'])} | "
            f"{_pct(r['true_prepay_rate'])} | {r['fit_seconds']} |"
        )
    lines += [
        "",
        "Sorted by PR-AUC (area under the precision-recall curve), the most informative single "
        "number when the positive class is rare. \"True no-prepay\" and \"True prepay\" are the "
        "paper's Table 1 columns (specificity and recall).",
        "",
        "![PR-AUC](figures/pr_auc.png)",
        "",
        "![PR curves](figures/pr_curves.png)",
        "",
    ]
    if leak:
        p, f = leak["paper"], leak["fixed"]
        lines += [
            "## Leakage check: paper setup vs fixed setup",
            "",
            "Same model (L2 logistic regression), same loans. The paper setup labels each row "
            "\"prepaid this month\", keeps the payoff row (where the balance is already 0) and "
            "splits rows at random.",
            "",
            "| Setup | Accuracy | Precision | Recall | PR-AUC |",
            "|---|---|---|---|---|",
            f"| Paper (leaky) | {_pct(p['accuracy'])} | {p['precision']:.3f} | {p['recall']:.3f} | {p['pr_auc']:.3f} |",
            f"| Fixed | {_pct(f['accuracy'])} | {f['precision']:.3f} | {f['recall']:.3f} | {f['pr_auc']:.3f} |",
            "",
            "![Leakage demo](figures/leakage_demo.png)",
            "",
        ]
    lines += [
        "## Other figures",
        "",
        "- `figures/confusion_<model>.png` - confusion matrix per model (test set)",
        "- `figures/pca_3d.png` - 3-component PCA of the training features",
        "- `figures/top_features.png` - largest L1 logistic regression coefficients",
        "- `figures/roc_auc.png` - ROC-AUC per model",
        "",
        "![PCA](figures/pca_3d.png)",
        "",
        "![Top features](figures/top_features.png)",
        "",
    ]
    (res_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] results written to {_rel(res_dir, root)}/RESULTS.md")

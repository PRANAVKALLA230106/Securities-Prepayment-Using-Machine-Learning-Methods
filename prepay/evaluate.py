"""Metrics, threshold selection and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# Reference categorical palette (fixed order) + text/surface tokens
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
SEQ_BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


def pick_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Threshold that maximizes F1 on the validation set."""
    if np.all(s_val == s_val[0]):
        return float(s_val[0]) + 1e-9  # constant scorer (baseline) -> predicts all 0
    p, r, t = precision_recall_curve(y_val, s_val)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.nanargmax(f1)])


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    const = np.all(s == s[0])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": 0.5 if const else roc_auc_score(y, s),
        "pr_auc": float(y.mean()) if const else average_precision_score(y, s),
        # the paper's table columns
        "true_no_prepay_rate": tn / (tn + fp) if tn + fp else 0.0,
        "true_prepay_rate": rec,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(m: dict, title: str, path: Path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
    ax.imshow(cm, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(cm.max(), 2)))
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        light = v > np.sqrt(max(cm.max(), 1))
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if light else INK, fontsize=11)
    ax.set_xticks([0, 1], ["No prepay", "Prepay"])
    ax.set_yticks([0, 1], ["No prepay", "Prepay"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save(fig, path)


def plot_metric_bars(rows: list[dict], key: str, label: str, path: Path, ref: float | None = None, ref_label=""):
    rows = sorted(rows, key=lambda r: r[key])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.2))
    names = [r["model"] for r in rows]
    vals = [r[key] for r in rows]
    ax.barh(names, vals, color=BLUE, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.3f}", va="center", color=INK2, fontsize=9)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.text(ref, len(rows) - 0.4, f" {ref_label}", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel(label)
    ax.set_xlim(0, max(max(vals) * 1.18, 0.05))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{label} on the test set")
    _save(fig, path)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray, path: Path):
    """PR curves for up to 4 models (fixed color order)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for color, (name, (yt, s)) in zip([BLUE, ORANGE, AQUA, YELLOW], curves.items()):
        p, r, _ = precision_recall_curve(yt, s)
        ax.plot(r, p, color=color, lw=2, label=f"{name} (AP {average_precision_score(yt, s):.3f})")
    ax.axhline(y.mean(), color=INK2, lw=1, ls="--")
    ax.text(0.99, y.mean(), "random guessing", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("Recall (share of prepayments caught)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Precision-recall curves, top models")
    _save(fig, path)


def plot_pca(X: np.ndarray, y: np.ndarray, n: int, seed: int, path: Path):
    rng = np.random.default_rng(seed)
    # balanced sample so the rare class is visible
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(n // 2, len(pos), len(neg))
    idx = np.concatenate([rng.choice(neg, k, replace=False), rng.choice(pos, k, replace=False)])
    Z = PCA(n_components=3, random_state=seed).fit(X).transform(X[idx])
    yy = y[idx]
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor(SURFACE)
    ax.set_box_aspect(None, zoom=0.85)
    for cls, color, name in [(0, BLUE, "No prepayment next month"), (1, ORANGE, "Prepays next month")]:
        z = Z[yy == cls]
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=8, alpha=0.55, color=color, label=name, edgecolors="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(loc="upper left")
    ax.set_title("PCA, 3 components (balanced sample)")
    _save(fig, path)


def plot_top_features(names: list[str], coef: np.ndarray, path: Path, k: int = 15):
    order = np.argsort(np.abs(coef))[::-1][:k][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.36 * k + 1.2))
    vals = coef[order]
    colors = [ORANGE if v > 0 else BLUE for v in vals]
    ax.barh([names[i] for i in order], vals, color=colors, height=0.6)
    ax.axvline(0, color=INK2, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Coefficient on standardized feature")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=ORANGE, label="raises prepayment odds"), Patch(color=BLUE, label="lowers prepayment odds")],
        loc="lower right",
    )
    ax.set_title("Top features, L1 logistic regression")
    _save(fig, path)


def plot_leakage(paper: dict, fixed: dict, path: Path):
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("pr_auc", "PR-AUC")]
    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    for off, color, (lab, m) in [(-w / 2, ORANGE, ("Paper setup (leaky)", paper)), (w / 2, BLUE, ("Fixed setup", fixed))]:
        vals = [m[k] for k, _ in keys]
        ax.bar(x + off, vals, width=w - 0.02, color=color, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [k for _, k in keys])
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2)
    ax.set_title("L2 logistic regression: paper setup vs fixed setup")
    _save(fig, path)

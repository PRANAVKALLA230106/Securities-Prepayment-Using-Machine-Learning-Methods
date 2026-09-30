"""All classifiers used in the project.

Every model exposes ``fit(X_train, y_train, X_val, y_val)`` and
``score(X) -> np.ndarray`` where a higher score means "more likely to prepay".
Scores do not need to be probabilities; thresholds are picked on validation.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

_SK_NEW = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def _logreg(kind: str, C: float, seed: int) -> LogisticRegression:
    """Logistic regression that works on old and new scikit-learn APIs."""
    if kind == "none":
        return LogisticRegression(C=np.inf, max_iter=2000) if _SK_NEW else LogisticRegression(penalty=None, max_iter=2000)
    if kind == "l1":
        if _SK_NEW:
            return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", tol=1e-3, max_iter=500, random_state=seed)
        return LogisticRegression(penalty="l1", C=C, solver="saga", tol=1e-3, max_iter=500, random_state=seed)
    if _SK_NEW:
        return LogisticRegression(C=C, l1_ratio=0.0, max_iter=2000)
    return LogisticRegression(penalty="l2", C=C, max_iter=2000)


def _subsample(X, y, n, seed):
    if n is None or len(y) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


class Baseline:
    """Always predicts 'no prepayment'."""

    def fit(self, X, y, Xv, yv):
        return self

    def score(self, X):
        return np.zeros(len(X))


class Logistic:
    def __init__(self, kind, cfg, seed):
        self.kind, self.cfg, self.seed = kind, cfg, seed

    def fit(self, X, y, Xv, yv):
        X, y = _subsample(X, y, self.cfg["max_train_rows"], self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.model = _logreg(self.kind, self.cfg["C"], self.seed).fit(X, y)
        self.coef_ = self.model.coef_.ravel()
        return self

    def score(self, X):
        return self.model.decision_function(X)


class SVM:
    def __init__(self, kernel, cfg, seed):
        self.kernel, self.cfg, self.seed = kernel, cfg, seed

    def fit(self, X, y, Xv, yv):
        X, y = _subsample(X, y, self.cfg["train_rows"], self.seed)
        self.model = SVC(kernel=self.kernel, degree=self.cfg["poly_degree"], gamma="scale", cache_size=1000)
        self.model.fit(X, y)
        return self

    def score(self, X):
        # predict in blocks to keep memory flat
        return np.concatenate([self.model.decision_function(X[i : i + 50_000]) for i in range(0, len(X), 50_000)])


class GDA:
    def __init__(self, kind, cfg):
        self.kind, self.cfg = kind, cfg

    def fit(self, X, y, Xv, yv):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "linear":
                self.model = LinearDiscriminantAnalysis().fit(X, y)
                return self
            best = None
            for r in self.cfg["qda_reg_params"]:
                m = QuadraticDiscriminantAnalysis(reg_param=r).fit(X, y)
                ap = average_precision_score(yv, self._score(m, Xv))
                print(f"      QDA reg_param={r}: val PR-AUC={ap:.4f}")
                if best is None or ap > best[0]:
                    best = (ap, r, m)
            self.reg_param, self.model = best[1], best[2]
        return self

    @staticmethod
    def _score(m, X):
        # log-odds are numerically safer than probabilities (QDA saturates at 0/1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lp = m.predict_log_proba(X)
        return lp[:, 1] - lp[:, 0]

    def score(self, X):
        return self._score(self.model, X)


class GradientBoosting:
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed

    def fit(self, X, y, Xv, yv):
        self.model = HistGradientBoostingClassifier(
            max_iter=self.cfg["max_iter"],
            learning_rate=self.cfg["learning_rate"],
            early_stopping=True,
            random_state=self.seed,
        ).fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


def build(name: str, cfg: dict, seed: int):
    if name == "baseline":
        return Baseline()
    if name.startswith("logistic_"):
        return Logistic({"base": "none", "l1": "l1", "l2": "l2"}[name.split("_")[1]], cfg["logistic"], seed)
    if name.startswith("svm_"):
        return SVM(name.split("_")[1], cfg["svm"], seed)
    if name.startswith("gda_"):
        return GDA(name.split("_")[1], cfg["gda"])
    if name == "neural_net":
        from .nn import NeuralNet  # torch is only imported if needed

        return NeuralNet(cfg["neural_net"], seed)
    if name == "gradient_boosting":
        return GradientBoosting(cfg["gradient_boosting"], seed)
    raise ValueError(f"Unknown model {name!r}")


def timed_fit(model, *args):
    t = time.time()
    model.fit(*args)
    return time.time() - t

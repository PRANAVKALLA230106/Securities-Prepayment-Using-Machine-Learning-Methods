"""Turning raw loan data into a leak-free modeling table.

The prediction task
-------------------
For every loan that is still active at the end of month ``t`` we predict
whether it pays off in full (Freddie Mac zero-balance code ``01``) during month
``t+1``. Every feature is known at the end of month ``t``.

Why this fixes the paper's leakage
----------------------------------
In the paper each row's label was "prepaid in *this* month" and features came
from that same row, including the current balance - which is already ~0 in the
payoff month. The model could simply read the answer off ``current_UPB``.
Here the payoff row itself is never used as a feature row, so balance-type
features cannot give the answer away.

The paper-style table is still built (``paper_style=True``) for the leakage
demonstration in the results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_SENTINELS = {
    "credit_score": 9999,
    "mi_pct": 999,
    "num_units": 99,
    "orig_cltv": 999,
    "orig_dti": 999,
    "orig_ltv": 999,
    "num_borrowers": 99,
}

CATEGORICAL = [
    "first_time_homebuyer",
    "occupancy_status",
    "channel",
    "ppm_flag",
    "property_type",
    "loan_purpose",
    "super_conforming",
    "relief_refi",
    "property_state",
]

ORIG_NUMERIC = [
    "credit_score",
    "mi_pct",
    "num_units",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_ltv",
    "orig_interest_rate",
    "num_borrowers",
]

DYNAMIC_NUMERIC = [
    "loan_age",
    "remaining_months",
    "upb_ratio",
    "upb_masked",
    "delinquency",
    "modified",
    "payment_deferral",
    "current_interest_rate",
    "month_sin",
    "month_cos",
]

MACRO_NUMERIC = [
    "mortgage_rate",
    "mortgage_rate_chg3",
    "rate_incentive",
    "burnout_months",
    "hpi_yoy",
    "hpi_since_orig",
    "mtm_ltv",
    "unemployment",
    "unemployment_chg12",
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_origination(orig: pd.DataFrame) -> pd.DataFrame:
    o = orig.copy()
    for c in ORIG_NUMERIC + ["first_payment_date"]:
        o[c] = _num(o[c])
    for c, sentinel in NUMERIC_SENTINELS.items():
        o.loc[o[c] >= sentinel, c] = np.nan
    for c in CATEGORICAL:
        o[c] = o[c].fillna("NA").astype(str).str.strip().replace({"": "NA", "9": "NA", "99": "NA"})
    return o


def clean_performance(perf: pd.DataFrame) -> pd.DataFrame:
    p = perf.copy()
    p["period"] = _num(p["period"]).astype("Int64")
    for c in ["current_upb", "loan_age", "remaining_months", "current_interest_rate"]:
        p[c] = _num(p[c])
    # delinquency: 0,1,2,... months; "RA" (REO acquisition) and other codes -> 12
    d = _num(p["delinquency"])
    p["delinquency"] = d.mask(d.isna() & p["delinquency"].notna(), 12).fillna(0).clip(upper=12)
    p["modified"] = p["mod_flag"].isin(["Y", "P"]).astype(np.int8)
    p["payment_deferral"] = p["payment_deferral"].isin(["Y", "P"]).astype(np.int8)
    zb = p["zb_code"].astype(str).str.strip().str.zfill(2)
    p["zb_code"] = zb.where(p["zb_code"].notna())
    p = p.dropna(subset=["period"]).sort_values(["loan_id", "period"]).reset_index(drop=True)
    return p


def build_table(orig: pd.DataFrame, perf: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    """One row per loan-month with both labels and all raw features.

    Columns ``y_next`` (fixed task; NaN where not usable) and ``y_same`` (paper task).
    """
    o = clean_origination(orig)
    p = clean_performance(perf)
    df = p.merge(o, on="loan_id", how="inner", validate="many_to_one")

    g = df.groupby("loan_id", sort=False)
    df["next_zb"] = g["zb_code"].shift(-1)
    df["has_next"] = g["period"].shift(-1).notna()
    active = df["zb_code"].isna()

    # --- labels ---------------------------------------------------------------
    payoff_next = df["next_zb"].eq("01") & (df["remaining_months"].fillna(99) > 1)  # exclude maturity
    df["y_next"] = np.where(active & df["has_next"], payoff_next.astype(float), np.nan)
    df["y_same"] = df["zb_code"].eq("01").astype(float)

    # --- dynamic features -----------------------------------------------------
    # Freddie Mac shows current UPB as 0 in a loan's first months (and in the
    # payoff month). On an *active* row a zero is a mask, not a real balance.
    df["upb_ratio_raw"] = df["current_upb"] / df["orig_upb"]
    masked = active & (df["current_upb"].fillna(0) <= 0)
    df["upb_masked"] = masked.astype(np.int8)
    df["upb_ratio"] = df["upb_ratio_raw"].where(~masked, 1.0).clip(0, 1.2)
    df["current_interest_rate"] = df["current_interest_rate"].fillna(df["orig_interest_rate"])
    month = (df["period"] % 100).astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- macro features -------------------------------------------------------
    if macro is not None:
        per = df["period"].astype("int64")
        for c in ["mortgage_rate", "mortgage_rate_chg3", "hpi_yoy", "unemployment", "unemployment_chg12"]:
            df[c] = per.map(macro[c])
        hpi_now = per.map(macro["hpi"])
        hpi_orig = df["first_payment_date"].map(macro["hpi"])
        df["hpi_since_orig"] = hpi_now / hpi_orig - 1
        df["mtm_ltv"] = df["orig_ltv"] * df["upb_ratio"] / (hpi_now / hpi_orig)
        df["rate_incentive"] = df["current_interest_rate"] - df["mortgage_rate"]
        # burnout: how many past months the loan had a >= 0.5pp refi incentive
        inc = (df["rate_incentive"] >= 0.5).astype(np.int16)
        df["burnout_months"] = inc.groupby(df["loan_id"]).cumsum() - inc
    else:
        for c in MACRO_NUMERIC:
            df[c] = np.nan

    drop = ["next_zb", "has_next", "mod_flag", "zb_code", "amortization_type", "io_indicator", "orig_loan_term"]
    return df.drop(columns=[c for c in drop if c in df])


def feature_columns(use_macro: bool) -> tuple[list[str], list[str]]:
    num = ORIG_NUMERIC + DYNAMIC_NUMERIC + (MACRO_NUMERIC if use_macro else [])
    return num, CATEGORICAL


class Preprocessor:
    """Median-impute + missing flags + one-hot + standardize, fitted on TRAIN only."""

    def __init__(self, numeric: list[str], categorical: list[str], min_cat_count: int = 50):
        self.numeric, self.categorical, self.min_cat_count = numeric, categorical, min_cat_count

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        num = df[self.numeric]
        self.medians = num.median()
        self.flag_cols = [c for c in self.numeric if num[c].isna().any()]
        self.levels = {}
        for c in self.categorical:
            vc = df[c].value_counts()
            self.levels[c] = sorted(vc[vc >= self.min_cat_count].index.tolist())
        X = self._raw(df)
        # drop constant columns (useless, and break QDA / scaling)
        std = X.std(axis=0)
        self.keep = std > 0
        X = X.loc[:, self.keep]
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        self.columns = X.columns.tolist()
        return self

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [df[self.numeric].fillna(self.medians)]
        if self.flag_cols:
            parts.append(df[self.flag_cols].isna().astype(np.float32).add_suffix("_missing"))
        for c in self.categorical:
            for lvl in self.levels[c]:
                parts.append((df[c] == lvl).astype(np.float32).rename(f"{c}={lvl}"))
        X = pd.concat(parts, axis=1).astype(np.float32)
        return X

    def transform(self, df: pd.DataFrame, chunk: int = 200_000) -> np.ndarray:
        out = np.empty((len(df), len(self.columns)), dtype=np.float32)
        mean, std = self.mean.to_numpy(np.float32), self.std.to_numpy(np.float32)
        for i in range(0, len(df), chunk):
            X = self._raw(df.iloc[i : i + chunk]).loc[:, self.keep].to_numpy(np.float32)
            out[i : i + chunk] = (X - mean) / std
        return out


def split(df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Return row indices for train / val / test."""
    if cfg["method"] == "loan":
        loans = np.sort(np.asarray(df["loan_id"].unique(), dtype=object))
        loans = loans[np.random.default_rng(seed).permutation(len(loans))]
        n_tr = int(len(loans) * cfg["train_frac"])
        n_va = int(len(loans) * cfg["val_frac"])
        tr, va = set(loans[:n_tr]), set(loans[n_tr : n_tr + n_va])
        lid = df["loan_id"]
        return df.index[lid.isin(tr)], df.index[lid.isin(va)], df.index[~lid.isin(tr | va)]
    if cfg["method"] == "time":
        periods = np.sort(df["period"].unique())
        a = periods[int(len(periods) * cfg["time_train_frac"])]
        b = periods[int(len(periods) * (cfg["time_train_frac"] + cfg["time_val_frac"]))]
        per = df["period"]
        return df.index[per < a], df.index[(per >= a) & (per < b)], df.index[per >= b]
    raise ValueError(f"unknown split method {cfg['method']!r}")

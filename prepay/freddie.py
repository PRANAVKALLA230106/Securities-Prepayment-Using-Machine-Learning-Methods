"""Reading the Freddie Mac Single-Family Loan-Level *Sample* dataset.

Each vintage comes as ``sample_YYYY.zip`` containing two pipe-delimited files
without headers:

* ``sample_orig_YYYY.txt`` - one row per loan, static origination data
* ``sample_svcg_YYYY.txt`` - one row per loan per month, performance data

Column positions follow the Freddie Mac "Single-Family Loan-Level Dataset
General User Guide". Only the columns we use are read, which keeps memory low.
Files may also be placed unzipped in the raw folder.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# 0-based column positions -> our column names
ORIG_COLS = {
    0: "credit_score",
    1: "first_payment_date",
    2: "first_time_homebuyer",
    5: "mi_pct",
    6: "num_units",
    7: "occupancy_status",
    8: "orig_cltv",
    9: "orig_dti",
    10: "orig_upb",
    11: "orig_ltv",
    12: "orig_interest_rate",
    13: "channel",
    14: "ppm_flag",
    15: "amortization_type",
    16: "property_state",
    17: "property_type",
    19: "loan_id",
    20: "loan_purpose",
    21: "orig_loan_term",
    22: "num_borrowers",
    25: "super_conforming",
    28: "relief_refi",
    30: "io_indicator",
}

SVCG_COLS = {
    0: "loan_id",
    1: "period",
    2: "current_upb",
    3: "delinquency",
    4: "loan_age",
    5: "remaining_months",
    7: "mod_flag",
    8: "zb_code",
    10: "current_interest_rate",
    24: "payment_deferral",
}

_YEAR_RE = re.compile(r"sample_(?:orig_|svcg_)?(\d{4})")


def find_years(raw_dir: Path) -> list[int]:
    """All vintage years available in ``raw_dir`` (zipped or unzipped)."""
    years = set()
    for p in Path(raw_dir).rglob("sample_*"):
        m = _YEAR_RE.search(p.name)
        if m and p.suffix.lower() in (".zip", ".txt"):
            years.add(int(m.group(1)))
    return sorted(years)


# Freddie Mac has used both "svcg" and "perf" for the performance file's name
# across different downloads of the sample dataset; accept either.
_KIND_ALIASES = {"svcg": ["svcg", "perf"], "orig": ["orig"]}


def _open_member(raw_dir: Path, year: int, kind: str):
    """Return a binary file handle for sample_{kind}_{year}.txt (kind = orig|svcg)."""
    raw_dir = Path(raw_dir)
    names = [f"sample_{alias}_{year}.txt" for alias in _KIND_ALIASES[kind]]
    for name in names:
        loose = list(raw_dir.rglob(name))
        if loose:
            return open(loose[0], "rb")
    for z in raw_dir.rglob(f"sample_{year}.zip"):
        zf = zipfile.ZipFile(z)
        for member in zf.namelist():
            if any(member.endswith(name) for name in names):
                return zf.open(member)
    raise FileNotFoundError(
        f"Could not find any of {names} (or sample_{year}.zip) under {raw_dir}"
    )


def _n_columns(raw_dir: Path, year: int, kind: str) -> int:
    with _open_member(raw_dir, year, kind) as fh:
        first = io.TextIOWrapper(fh, encoding="latin-1").readline()
    return first.rstrip("\r\n").count("|") + 1


def _layout(kind: str, colmap: dict, ncol: int) -> dict:
    """Adjust column positions to the file's actual layout.

    The User Guide layout has 32 origination columns with separate seller (23)
    and servicer (24) name columns. The sample files we downloaded have 31: a
    single name column at 23, so every later column sits one position earlier.
    """
    if kind == "orig" and ncol == 31:
        return {(i - 1 if i >= 24 else i): n for i, n in colmap.items()}
    return colmap


def _read(raw_dir, year, kind, colmap, chunksize=None):
    ncol = _n_columns(raw_dir, year, kind)
    use = {i: n for i, n in _layout(kind, colmap, ncol).items() if i < ncol}
    fh = _open_member(raw_dir, year, kind)
    reader = pd.read_csv(
        io.TextIOWrapper(fh, encoding="latin-1"),
        sep="|",
        header=None,
        usecols=sorted(use),
        dtype=str,
        chunksize=chunksize,
        na_values=[""],
        keep_default_na=False,
    )
    rename = lambda df: df.rename(columns=use)  # noqa: E731
    if chunksize is None:
        out = rename(reader)
        fh.close()
        return out
    return (rename(c) for c in reader)


def load_origination(raw_dir: Path, year: int) -> pd.DataFrame:
    df = _read(raw_dir, year, "orig", ORIG_COLS)
    for c in ORIG_COLS.values():
        if c not in df:
            df[c] = np.nan
    df["vintage"] = year
    return df


def load_performance(raw_dir: Path, year: int, loan_ids: set[str]) -> pd.DataFrame:
    """Monthly rows for the given loans, read in chunks to limit memory."""
    parts = []
    for chunk in _read(raw_dir, year, "svcg", SVCG_COLS, chunksize=1_000_000):
        chunk = chunk[chunk["loan_id"].isin(loan_ids)]
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(SVCG_COLS.values()))
    for c in SVCG_COLS.values():
        if c not in df:
            df[c] = np.nan
    return df


def select_loans(orig: pd.DataFrame, loan_term: int, n: int | None, seed: int) -> pd.DataFrame:
    """Keep fixed-rate, fully amortizing loans of the given term and sample ``n``."""
    term = pd.to_numeric(orig["orig_loan_term"], errors="coerce")
    keep = term.eq(loan_term)
    keep &= orig["amortization_type"].fillna("FRM").str.upper().eq("FRM")
    keep &= orig["io_indicator"].fillna("N").str.upper().ne("Y")
    orig = orig[keep]
    if n is not None and len(orig) > n:
        orig = orig.sample(n=n, random_state=seed)
    return orig.reset_index(drop=True)
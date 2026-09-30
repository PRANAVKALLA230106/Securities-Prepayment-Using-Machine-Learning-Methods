"""Macroeconomic indicators from FRED (St. Louis Fed).

* Mortgage rate  - Freddie Mac PMMS 30-year fixed (weekly, averaged to monthly)
* House prices   - FHFA purchase-only HPI, US, monthly (hosted on FRED)
* Unemployment   - BLS unemployment rate, monthly (hosted on FRED)

Each series is cached as ``<macro_dir>/<SERIES_ID>.csv``. If the download
fails (no internet, firewall), download the CSV manually from
https://fred.stlouisfed.org/series/<SERIES_ID> ("Download" -> CSV) and save it
under that name.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _download(series_id: str, dest: Path) -> None:
    url = FRED_URL.format(sid=series_id)
    req = urllib.request.Request(url, headers={"User-Agent": "mbs-prepayment-ml"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def load_series(series_id: str, macro_dir: Path) -> pd.Series:
    """Monthly series indexed by YYYYMM integer."""
    path = Path(macro_dir) / f"{series_id}.csv"
    if not path.exists():
        print(f"  downloading {series_id} from FRED ...")
        try:
            _download(series_id, path)
        except Exception as e:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Could not download {series_id} ({e}).\n"
                f"Download it manually from https://fred.stlouisfed.org/series/{series_id} "
                f"(Download -> CSV) and save it as {path}, or set macro.enabled: false in config.yaml."
            ) from e
    df = pd.read_csv(path)
    # FRED CSVs have a date column (DATE or observation_date) and a value column.
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
    ).dropna()
    s = s.groupby(s.index.year * 100 + s.index.month).mean()
    s.index.name = "period"
    return s


def load_macro(cfg: dict, macro_dir: Path) -> pd.DataFrame:
    """DataFrame indexed by YYYYMM with mortgage_rate, hpi, unemployment + derived columns."""
    m = pd.DataFrame(
        {
            "mortgage_rate": load_series(cfg["mortgage_rate"], macro_dir),
            "hpi": load_series(cfg["hpi"], macro_dir),
            "unemployment": load_series(cfg["unemployment"], macro_dir),
        }
    ).sort_index()
    # make sure every month exists so shifts are month-accurate
    idx = pd.period_range(
        pd.Period(f"{m.index.min() // 100}-{m.index.min() % 100:02d}", "M"),
        pd.Period(f"{m.index.max() // 100}-{m.index.max() % 100:02d}", "M"),
        freq="M",
    )
    m = m.reindex([p.year * 100 + p.month for p in idx]).ffill()
    m["mortgage_rate_chg3"] = m["mortgage_rate"] - m["mortgage_rate"].shift(3)
    m["hpi_yoy"] = m["hpi"] / m["hpi"].shift(12) - 1
    m["unemployment_chg12"] = m["unemployment"] - m["unemployment"].shift(12)
    m.index.name = "period"
    return m

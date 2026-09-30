"""Generate FAKE data in the exact Freddie Mac sample-file format.

Used only to test that the pipeline runs end to end (``python run_all.py --synthetic``
and ``pytest``). The numbers mean nothing - never report results from it.

Writes:
    <out>/raw/sample_YYYY.zip   (sample_orig_YYYY.txt + sample_svcg_YYYY.txt, 32 pipe-delimited columns each)
    <out>/macro/MORTGAGE30US.csv, HPIPONM226S.csv, UNRATE.csv   (FRED CSV format)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "WA", "AZ"]
CUTOFF = 202512


def _months(start: int, end: int) -> list[int]:
    out, y, m = [], start // 100, start % 100
    while y * 100 + m <= end:
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _macro() -> pd.DataFrame:
    per = _months(199101, CUTOFF)
    t = np.array([(p // 100) + (p % 100 - 1) / 12 for p in per])
    rate = np.interp(t, [1991, 1999, 2001, 2003, 2007, 2012, 2016, 2019, 2021, 2023, 2026], [9.5, 7.0, 7.0, 5.5, 6.3, 3.6, 3.7, 4.0, 2.9, 6.8, 6.3])
    rng = np.random.default_rng(0)
    rate = rate + rng.normal(0, 0.12, len(t))
    hpi = 100 * np.exp(np.interp(t, [1991, 2006, 2011, 2019, 2022, 2026], [0, 0.75, 0.45, 0.95, 1.35, 1.45]))
    unemp = np.interp(t, [1991, 2000, 2007, 2010, 2019, 2020.3, 2021, 2026], [6.8, 4.0, 4.6, 9.8, 3.6, 13.0, 5.5, 4.2])
    return pd.DataFrame({"period": per, "rate": rate, "hpi": hpi, "unemp": unemp}).set_index("period")


def _write_fred(macro: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dates = [f"{p // 100}-{p % 100:02d}-01" for p in macro.index]
    for sid, col in [("MORTGAGE30US", "rate"), ("HPIPONM226S", "hpi"), ("UNRATE", "unemp")]:
        pd.DataFrame({"observation_date": dates, sid: macro[col].round(2)}).to_csv(out / f"{sid}.csv", index=False)


def _loan_rows(year: int, n: int, macro: pd.DataFrame, rng: np.random.Generator):
    orig_rows, perf_rows = [], []
    for i in range(n):
        lid = f"F{year % 100:02d}Q{rng.integers(1, 5)}{i:07d}"
        om = year * 100 + int(rng.integers(1, 13))
        months = _months(om, CUTOFF)
        if len(months) < 3:
            continue
        fpd = months[1]
        rate0 = round(float(macro.loc[om, "rate"] + rng.normal(0.25, 0.35)), 3)
        upb0 = int(rng.integers(80, 700)) * 1000
        fico = int(np.clip(rng.normal(735, 50), 580, 830)) if rng.random() > 0.01 else 9999
        ltv = int(np.clip(rng.normal(76, 14), 20, 97))
        dti = int(np.clip(rng.normal(35, 9), 5, 50)) if rng.random() > 0.03 else 999
        state = STATES[rng.integers(len(STATES))]
        occ = rng.choice(["P", "I", "S"], p=[0.88, 0.08, 0.04])
        purpose = rng.choice(["P", "C", "N"], p=[0.45, 0.25, 0.30])
        nb = int(rng.choice([1, 2]))
        k = (fpd // 100) * 12 + fpd % 100 - 1 + 359
        maturity = (k // 12) * 100 + k % 12 + 1
        orig = [""] * 32
        orig[0], orig[1], orig[2], orig[3] = fico, fpd, rng.choice(["Y", "N"]), maturity
        orig[4], orig[5], orig[6], orig[7] = int(rng.integers(10000, 49999)), (25 if ltv > 80 else 0), 1, occ
        orig[8], orig[9], orig[10], orig[11], orig[12] = ltv, dti, upb0, ltv, rate0
        orig[13], orig[14], orig[15], orig[16] = rng.choice(["R", "B", "C"]), "N", "FRM", state
        orig[17], orig[18], orig[19], orig[20], orig[21] = rng.choice(["SF", "PU", "CO"]), f"{rng.integers(100, 999)}00", lid, purpose, 360
        orig[22], orig[23], orig[24] = nb, "Other sellers", "Other servicers"
        orig[28], orig[29], orig[30] = "N", 9, "N"
        orig_rows.append(orig)

        r = rate0 / 1200
        pmt = upb0 * r / (1 - (1 + r) ** -360)
        bal, delinq, burn = float(upb0), 0, 0
        for age, per in enumerate(months[1:], start=0):
            mkt = macro.loc[per, "rate"]
            inc = rate0 - mkt
            season = 0.25 * np.sin(2 * np.pi * ((per % 100) - 3) / 12)
            logit = -5.6 + 1.6 * np.clip(inc, -2, 2.5) + season + 0.004 * (fico - 735) + 0.5 * min(age, 30) / 30 - 0.01 * burn
            logit += 0.6 * (purpose == "C") - 0.5 * (occ == "I")
            burn += inc > 0.5
            zb, zbd = "", ""
            if age >= 1 and rng.random() < 1 / (1 + np.exp(-logit)):
                zb, zbd, bal_out = "01", per, 0.0
            else:
                delinq = min(delinq + 1, 12) if rng.random() < (0.004 if delinq == 0 else 0.5) else 0
                if delinq >= 6 and rng.random() < 0.3:
                    zb, zbd, bal_out = "09", per, 0.0
                else:
                    bal = max(bal * (1 + r) - pmt, 0.0)
                    bal_out = 0.0 if age < 6 else round(bal, 2)  # Freddie masks UPB early in the loan
            row = [""] * 32
            row[0], row[1], row[2], row[3] = lid, per, bal_out, delinq
            row[4], row[5], row[7], row[8], row[9] = age, 360 - age, "N" if age % 7 else "", zb, zbd
            row[10], row[11], row[24] = rate0, 0, "N"
            perf_rows.append(row)
            if zb:
                break
    return orig_rows, perf_rows


def _to_txt(rows) -> bytes:
    return ("\n".join("|".join(str(v) for v in r) for r in rows) + "\n").encode("latin-1")


def make(out: Path, years=(2005, 2012, 2019), loans_per_year=1500, seed=42) -> None:
    out = Path(out)
    macro = _macro()
    _write_fred(macro, out / "macro")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for y in years:
        o, p = _loan_rows(y, loans_per_year, macro, rng)
        with zipfile.ZipFile(out / "raw" / f"sample_{y}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"sample_orig_{y}.txt", _to_txt(o))
            z.writestr(f"sample_svcg_{y}.txt", _to_txt(p))
    print(f"[synthetic] wrote fake data for {list(years)} to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--years", nargs="+", type=int, default=[2005, 2012, 2019])
    ap.add_argument("--loans", type=int, default=1500)
    a = ap.parse_args()
    make(Path(a.out), a.years, a.loans)

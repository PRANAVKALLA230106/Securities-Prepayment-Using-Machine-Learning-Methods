"""Run the whole project: build the dataset, train every model, write results/.

    python run_all.py                       # everything, using config.yaml
    python run_all.py --models logistic_l2 gda_quadratic   # just some models
    python run_all.py --rebuild             # rebuild the dataset from the raw files
    python run_all.py --synthetic           # smoke test on generated fake data
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepay import pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", help="subset of models to run (names as in config.yaml)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the processed dataset")
    ap.add_argument("--prepare-only", action="store_true", help="only build the dataset")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="generate fake data in the Freddie Mac format and run on it (writes to data/synthetic/ and results_synthetic/)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.synthetic:
        from scripts.make_synthetic_data import make

        cfg = copy.deepcopy(cfg)
        base = "data/synthetic"
        cfg["paths"] = {
            "raw_dir": f"{base}/raw",
            "macro_dir": f"{base}/macro",
            "processed_dir": f"{base}/processed",
            "results_dir": "results_synthetic",
        }
        cfg["data"]["years"] = None
        cfg["svm"]["train_rows"] = min(cfg["svm"]["train_rows"], 4000)
        cfg["neural_net"]["epochs"] = min(cfg["neural_net"]["epochs"], 8)
        make(ROOT / base, years=[2005, 2012, 2019], loans_per_year=1500, seed=cfg["seed"])

    t = time.time()
    data_path = pipeline.prepare(cfg, ROOT, rebuild=args.rebuild or args.synthetic)
    if not args.prepare_only:
        pipeline.run(cfg, ROOT, data_path, only=args.models)
    print(f"Finished in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()

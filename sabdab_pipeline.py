#!/usr/bin/env python3
"""
SAbDab Full Pipeline - orchestrates all 6 steps.
See individual step modules for details.

Usage:
    python sabdab_pipeline.py --step all
    python sabdab_pipeline.py --step 1
    python sabdab_pipeline.py --step 2
    python sabdab_pipeline.py --step 3 --n_jobs 8
    python sabdab_pipeline.py --step 4
    python sabdab_pipeline.py --step 5
    python sabdab_pipeline.py --step 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_config import *
from pipeline_steps import step1, step2, step3, step4, step5, step6, load_ppi_list


def main():
    parser = argparse.ArgumentParser(description="SAbDab PIsToN-MINT pipeline")
    parser.add_argument("--step", default="all",
                        help="Step to run: 1|2|3|4|5|6|all")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Parallel jobs for step 3")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-PPI timeout (seconds) for step 3")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    steps = (
        ["1", "2", "3", "4", "5", "6"]
        if args.step == "all"
        else [args.step]
    )

    ppi_list, labels = None, None

    for s in steps:
        if s == "1":
            ppi_list, labels = step1()
        else:
            if ppi_list is None:
                ppi_list, labels = load_ppi_list()
            if s == "2":
                step2(ppi_list)
            elif s == "3":
                step3(ppi_list, n_jobs=args.n_jobs, timeout=args.timeout)
            elif s == "4":
                step4(ppi_list, labels,
                      train_ratio=args.train_ratio,
                      val_ratio=args.val_ratio,
                      seed=args.seed)
            elif s == "5":
                step5()
            elif s == "6":
                step6()
            else:
                print(f"Unknown step: {s}")
                sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
CLI entry point for pre-extracting PIsToN + MINT embeddings.

This is Phase 1 of the training pipeline. Run this once to cache
embeddings, then train the MLP classifier quickly via run_training.py.

Usage:
    python run_extract_embeddings.py \
        --ppi_list data/all_ppis.txt \
        --labels data/labels.csv \
        --pdb_dir ./pdbs \
        --out_dir ./output \
        --piston_ckpt path/to/PIsToN.pth \
        --mint_ckpt path/to/mint.ckpt
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract PIsToN + MINT embeddings for fusion training"
    )

    parser.add_argument("--ppi_list", type=str, required=True,
                        help="Text file with all PPI IDs (one per line)")
    parser.add_argument("--labels", type=str, default=None,
                        help="CSV file with columns: PPI,label (optional, not used for extraction)")
    parser.add_argument("--pdb_dir", type=str, default="./pdbs",
                        help="Directory with PDB files")
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory")

    parser.add_argument("--piston_ckpt", type=str, default=None)
    parser.add_argument("--mint_ckpt", type=str, required=True,
                        help="Path to MINT model .ckpt")
    parser.add_argument("--mint_config", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from config.default_config import get_default_config, ensure_dirs

    config = get_default_config(
        pdb_dir=args.pdb_dir,
        out_dir=args.out_dir,
        piston_ckpt=args.piston_ckpt,
        mint_ckpt=args.mint_ckpt,
        mint_json_cfg=args.mint_config,
    )
    ensure_dirs(config)

    ppi_list = [l.strip() for l in open(args.ppi_list) if l.strip()]
    print(f"Extracting embeddings for {len(ppi_list)} PPIs...")

    # Load labels if provided (not strictly needed for extraction)
    labels = {}
    if args.labels:
        with open(args.labels) as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])

    from training.train import extract_and_cache_embeddings

    extract_and_cache_embeddings(ppi_list, labels, config, device=device)


if __name__ == "__main__":
    main()

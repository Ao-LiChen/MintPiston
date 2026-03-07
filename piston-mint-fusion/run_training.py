#!/usr/bin/env python
"""
CLI entry point for training the PIsToN + MINT fusion MLP classifier.

Usage:
    python run_training.py \
        --train_list data/train.txt \
        --val_list data/val.txt \
        --labels data/labels.csv \
        --out_dir ./output \
        --lr 1e-3 --batch_size 8 --max_epochs 100
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
        description="Train the PIsToN + MINT fusion MLP classifier"
    )

    parser.add_argument("--train_list", type=str, required=True,
                        help="Text file with training PPI IDs (one per line)")
    parser.add_argument("--val_list", type=str, required=True,
                        help="Text file with validation PPI IDs (one per line)")
    parser.add_argument("--labels", type=str, required=True,
                        help="CSV file with columns: PPI,label")
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory (must contain embeddings/ from Phase 1)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from config.default_config import get_default_config, ensure_dirs

    config = get_default_config(
        pdb_dir=".",
        out_dir=args.out_dir,
    )

    # Override fusion config with CLI args
    config["fusion"]["lr"] = args.lr
    config["fusion"]["batch_size"] = args.batch_size
    config["fusion"]["max_epochs"] = args.max_epochs
    config["fusion"]["patience"] = args.patience
    config["fusion"]["hidden_dim"] = args.hidden_dim
    config["fusion"]["dropout"] = args.dropout
    config["fusion"]["seed"] = args.seed

    ensure_dirs(config)

    # Load PPI lists
    train_ppis = [l.strip() for l in open(args.train_list) if l.strip()]
    val_ppis = [l.strip() for l in open(args.val_list) if l.strip()]

    # Load labels
    labels = {}
    with open(args.labels) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])

    print(f"Training: {len(train_ppis)} PPIs")
    print(f"Validation: {len(val_ppis)} PPIs")
    print(f"Labels: {len(labels)} entries ({sum(labels.values())} positive)")

    from training.train import train_fusion

    best_path = train_fusion(
        config, train_ppis, val_ppis, labels, labels, device=device
    )
    print(f"\nDone. Best model: {best_path}")


if __name__ == "__main__":
    main()

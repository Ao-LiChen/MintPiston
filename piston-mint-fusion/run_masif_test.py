#!/usr/bin/env python
"""
End-to-end script for running the PIsToN+MINT fusion pipeline on the
masif_test dataset (Zenodo pre-processed data).

Runs Phase 1 (embedding extraction) and Phase 2 (MLP training) in sequence.

Usage:
    python run_masif_test.py \
        --data_root "C:/Users/liche/Downloads/masif_test.tar/masif_test/masif_test_copy" \
        --mint_ckpt saved_models/mint.ckpt \
        --out_dir ./output \
        --device cuda

Before running:
    1. Download masif_test data from Zenodo (already done)
    2. Download MINT checkpoint:
         wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt -O saved_models/mint.ckpt
    3. Generate data splits:
         python prepare_masif_data.py --data_root <path_to_masif_test_copy> --out_dir data
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Run PIsToN+MINT fusion on masif_test data"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to masif_test_copy directory"
    )
    parser.add_argument("--mint_ckpt", type=str, required=True,
                        help="Path to MINT .ckpt file")
    parser.add_argument("--piston_ckpt", type=str, default=None,
                        help="Path to PIsToN .pth (auto-detected if omitted)")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--device", type=str, default=None)

    # Optional: skip phases
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip Phase 1; use existing embeddings")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip Phase 2; only extract embeddings")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Resolve paths ---
    data_root = args.data_root
    grid_dir = os.path.join(data_root, "prepare_energies_16R", "07-grid")
    pdb_dir = os.path.join(data_root, "PDB")

    # Verify data files exist
    data_dir = os.path.join(_project_root, "data")
    for f in ["all_ppis.txt", "train.txt", "val.txt", "labels.csv"]:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run prepare_masif_data.py first.")
            sys.exit(1)

    # --- Build config ---
    from config.default_config import get_default_config, ensure_dirs

    config = get_default_config(
        pdb_dir=pdb_dir,
        out_dir=args.out_dir,
        piston_ckpt=args.piston_ckpt,
        mint_ckpt=args.mint_ckpt,
    )

    # Override grid dir to point to masif_test pre-processed data
    config["dirs"]["grid"] = grid_dir
    config["dirs"]["pdb_dir"] = pdb_dir

    # masif_test is 2-chain (A, Z)
    config["fusion"]["n_chains"] = 2
    config["fusion"]["mint_dim"] = 2560  # 2 * 1280

    # Training hyperparams
    config["fusion"]["lr"] = args.lr
    config["fusion"]["batch_size"] = args.batch_size
    config["fusion"]["max_epochs"] = args.max_epochs
    config["fusion"]["patience"] = args.patience
    config["fusion"]["seed"] = args.seed

    ensure_dirs(config)

    # --- Load data ---
    all_ppis = [l.strip() for l in open(os.path.join(data_dir, "all_ppis.txt")) if l.strip()]
    train_ppis = [l.strip() for l in open(os.path.join(data_dir, "train.txt")) if l.strip()]
    val_ppis = [l.strip() for l in open(os.path.join(data_dir, "val.txt")) if l.strip()]

    labels = {}
    with open(os.path.join(data_dir, "labels.csv")) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])

    print(f"All PPIs: {len(all_ppis)}")
    print(f"Train: {len(train_ppis)} | Val: {len(val_ppis)}")
    print(f"Labels: {sum(labels.values())} pos / {len(labels) - sum(labels.values())} neg")

    # =============================================
    # Phase 1: Extract embeddings
    # =============================================
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("Phase 1: Extracting PIsToN + MINT embeddings")
        print("=" * 60)

        from training.train import extract_and_cache_embeddings
        extract_and_cache_embeddings(all_ppis, labels, config, device=device)
    else:
        print("\nSkipping Phase 1 (--skip_extract)")

    # =============================================
    # Phase 2: Train MLP classifier
    # =============================================
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Phase 2: Training fusion MLP classifier")
        print("=" * 60)

        from training.train import train_fusion
        best_path = train_fusion(
            config, train_ppis, val_ppis, labels, labels, device=device
        )
        print(f"\nDone. Best model saved to: {best_path}")
    else:
        print("\nSkipping Phase 2 (--skip_train)")


if __name__ == "__main__":
    main()

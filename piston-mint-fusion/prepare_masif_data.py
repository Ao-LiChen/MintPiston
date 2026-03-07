#!/usr/bin/env python
"""
Prepare data files for the masif_test dataset.

Reads the pre-processed masif_test data and generates:
  - data/all_ppis.txt       (all PPI identifiers)
  - data/train.txt          (training split, 64%)
  - data/val.txt            (validation split, 16%)
  - data/test.txt           (test split, 20%)
  - data/labels.csv         (PPI -> binary label)

Split is done at the complex level (all models from the same PDB
go to the same split) to prevent data leakage.

Labels come from capri_quality_masif_test.csv:
  - CAPRI_quality 1,2,3 -> label 1 (acceptable or better)
  - CAPRI_quality 4     -> label 0 (incorrect)

Usage:
    python prepare_masif_data.py \
        --data_root "C:/Users/liche/Downloads/masif_test.tar/masif_test/masif_test_copy" \
        --out_dir data
"""

import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Prepare masif_test data for fusion training")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to masif_test_copy directory"
    )
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of complexes for test set")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of remaining complexes for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_native", action="store_true", default=False,
        help="Also include native complexes (from prepare_energies_16R_native)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # --- Read CAPRI quality labels ---
    quality_csv = os.path.join(args.data_root, "capri_quality_masif_test.csv")
    labels = {}
    with open(quality_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ppi = row["PPI"]
            quality = int(row["CAPRI_quality"])
            # CAPRI: 1=high, 2=medium, 3=acceptable, 4=incorrect
            labels[ppi] = 1 if quality <= 3 else 0

    print(f"Loaded {len(labels)} labels from CAPRI quality CSV")
    n_pos = sum(v for v in labels.values())
    n_neg = len(labels) - n_pos
    print(f"  Positive (acceptable+): {n_pos}")
    print(f"  Negative (incorrect):   {n_neg}")

    # --- Verify grid files exist for each PPI ---
    docking_grid_dir = os.path.join(
        args.data_root, "prepare_energies_16R", "07-grid"
    )
    native_grid_dir = os.path.join(
        args.data_root, "prepare_energies_16R_native", "07-grid"
    )

    all_ppis = []
    missing = []

    for ppi in labels:
        grid_path = os.path.join(docking_grid_dir, f"{ppi}.npy")
        if os.path.exists(grid_path):
            all_ppis.append(ppi)
        else:
            missing.append(ppi)

    if missing:
        print(f"  Warning: {len(missing)} PPIs in CSV have no grid file (skipped)")

    # Optionally include native complexes
    if args.include_native:
        native_grid_files = [
            f for f in os.listdir(native_grid_dir)
            if f.endswith(".npy") and not f.endswith("_resnames.npy")
        ]
        for gf in native_grid_files:
            ppi = gf.replace(".npy", "")
            if ppi not in labels:
                labels[ppi] = 1  # native = positive
                all_ppis.append(ppi)
        print(f"  Added {len(native_grid_files)} native complexes")

    print(f"Total PPIs with grid files: {len(all_ppis)}")

    # --- Split into train/val/test ---
    # Group by PDB complex (prefix before -model-) to avoid data leakage
    complex_groups = {}
    for ppi in all_ppis:
        # e.g. "1A79-model-2-pos_A_Z" -> base "1A79"
        base = ppi.split("-model-")[0] if "-model-" in ppi else ppi.split("_")[0]
        if base not in complex_groups:
            complex_groups[base] = []
        complex_groups[base].append(ppi)

    bases = sorted(complex_groups.keys())
    random.shuffle(bases)

    n_test = max(1, int(len(bases) * args.test_ratio))
    n_val = max(1, int(len(bases) * args.val_ratio))

    test_bases = set(bases[:n_test])
    val_bases = set(bases[n_test:n_test + n_val])
    train_bases = set(bases[n_test + n_val:])

    train_ppis = [ppi for b in train_bases for ppi in complex_groups[b]]
    val_ppis = [ppi for b in val_bases for ppi in complex_groups[b]]
    test_ppis = [ppi for b in test_bases for ppi in complex_groups[b]]

    random.shuffle(train_ppis)
    random.shuffle(val_ppis)
    random.shuffle(test_ppis)

    print(f"Train: {len(train_ppis)} PPIs from {len(train_bases)} complexes")
    print(f"Val:   {len(val_ppis)} PPIs from {len(val_bases)} complexes")
    print(f"Test:  {len(test_ppis)} PPIs from {len(test_bases)} complexes")

    # --- Write output files ---
    def write_list(filename, ppis):
        path = os.path.join(args.out_dir, filename)
        with open(path, "w") as f:
            for ppi in ppis:
                f.write(ppi + "\n")
        print(f"Wrote {path}")

    write_list("all_ppis.txt", sorted(all_ppis))
    write_list("train.txt", train_ppis)
    write_list("val.txt", val_ppis)
    write_list("test.txt", test_ppis)

    # labels.csv
    labels_path = os.path.join(args.out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PPI", "label"])
        for ppi in sorted(all_ppis):
            writer.writerow([ppi, labels[ppi]])
    print(f"Wrote {labels_path}")

    # Print label distribution for each split
    def dist_str(ppis):
        pos = sum(labels[p] for p in ppis)
        neg = len(ppis) - pos
        return f"{pos} pos / {neg} neg"

    print(f"\nTrain: {dist_str(train_ppis)}")
    print(f"Val:   {dist_str(val_ppis)}")
    print(f"Test:  {dist_str(test_ppis)}")


if __name__ == "__main__":
    main()

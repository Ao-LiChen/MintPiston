#!/usr/bin/env python3
"""
Update ppi_list.txt, labels.csv, and train/val/test splits to include
HDOCK-generated decoy PPIs (d1) that exist in RAW_PDB_DIR.

Decoys are assigned to the same split as their corresponding native PPI
to avoid data leakage (same PDB structure in train and test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_config import (
    log, OUT_DIR, GRID_DIR, RAW_PDB_DIR,
    PPI_LIST_FILE, LABELS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE,
)


def load_list(path):
    if not path.exists():
        return []
    return [l.strip() for l in open(path) if l.strip()]


def load_labels(path):
    labels = {}
    if not path.exists():
        return labels
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


def get_native_chain_combos():
    """Map PDB base ID -> list of chain combinations from native grid files."""
    combos = {}
    for f in GRID_DIR.iterdir():
        name = f.stem
        if "nat" not in name or "resnames" in name or not f.suffix == ".npy":
            continue
        parts = name.split("_")
        if len(parts) >= 3:
            base = parts[0].replace("nat", "")
            chains = "_".join(parts[1:])
            combos.setdefault(base, set()).add(chains)
    return combos


def main():
    log("Updating pipeline metadata with HDOCK decoys...")

    existing_labels = load_labels(LABELS_FILE)
    existing_ppis = set(load_list(PPI_LIST_FILE))
    train_ppis = set(load_list(TRAIN_FILE))
    val_ppis = set(load_list(VAL_FILE))
    test_ppis = set(load_list(TEST_FILE))

    native_to_split = {}
    for ppi in train_ppis:
        native_to_split[ppi] = "train"
    for ppi in val_ppis:
        native_to_split[ppi] = "val"
    for ppi in test_ppis:
        native_to_split[ppi] = "test"

    nat_combos = get_native_chain_combos()

    d1_pdbs = sorted(
        f.stem for f in RAW_PDB_DIR.iterdir()
        if f.name.endswith("d1.pdb")
    )

    new_ppis = []
    new_labels = {}
    new_train, new_val, new_test = [], [], []
    skipped_no_native = 0

    for d1_base in d1_pdbs:
        pdb_base = d1_base.replace("d1", "")
        if pdb_base not in nat_combos:
            skipped_no_native += 1
            continue

        for chains in sorted(nat_combos[pdb_base]):
            decoy_ppi = f"{d1_base}_{chains}"
            if decoy_ppi in existing_labels:
                continue

            new_ppis.append(decoy_ppi)
            new_labels[decoy_ppi] = 0

            native_ppi = f"{pdb_base}nat_{chains}"
            split = native_to_split.get(native_ppi)
            if split == "train":
                new_train.append(decoy_ppi)
            elif split == "val":
                new_val.append(decoy_ppi)
            elif split == "test":
                new_test.append(decoy_ppi)
            else:
                new_train.append(decoy_ppi)

    if not new_ppis:
        log("No new decoy PPIs to add.")
        return

    log(f"Adding {len(new_ppis)} new decoy PPIs")
    log(f"  -> train: {len(new_train)}, val: {len(new_val)}, test: {len(new_test)}")
    log(f"  Skipped (no native chain combos): {skipped_no_native}")

    with open(PPI_LIST_FILE, "a") as f:
        for ppi in new_ppis:
            f.write(ppi + "\n")

    with open(LABELS_FILE, "a") as f:
        for ppi, label in new_labels.items():
            f.write(f"{ppi},{label}\n")

    with open(TRAIN_FILE, "a") as f:
        for ppi in new_train:
            f.write(ppi + "\n")
    with open(VAL_FILE, "a") as f:
        for ppi in new_val:
            f.write(ppi + "\n")
    with open(TEST_FILE, "a") as f:
        for ppi in new_test:
            f.write(ppi + "\n")

    final_labels = load_labels(LABELS_FILE)
    final_ppis = load_list(PPI_LIST_FILE)
    final_train = load_list(TRAIN_FILE)
    final_val = load_list(VAL_FILE)
    final_test = load_list(TEST_FILE)

    nat_count = sum(1 for v in final_labels.values() if v == 1)
    dec_count = sum(1 for v in final_labels.values() if v == 0)

    log(f"\nFinal counts:")
    log(f"  ppi_list.txt: {len(final_ppis)}")
    log(f"  labels.csv:   {len(final_labels)} (native={nat_count}, decoy={dec_count})")
    log(f"  train.txt:    {len(final_train)}")
    log(f"  val.txt:      {len(final_val)}")
    log(f"  test.txt:     {len(final_test)}")
    log(f"  Class ratio:  {dec_count/(nat_count+dec_count)*100:.1f}% decoy")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract PIsToN + MINT embeddings for decoy PPIs that have grid files
but are missing embeddings.

Runs the fusion pipeline's extract_and_cache_embeddings on decoy PPIs.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "piston-mint-fusion"))

from pipeline_config import (
    log, OUT_DIR, GRID_DIR, EMBED_DIR, RAW_PDB_DIR,
    PPI_LIST_FILE, LABELS_FILE, MINT_CKPT,
)

import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    ppi_list = [l.strip() for l in open(PPI_LIST_FILE) if l.strip()]
    labels = {}
    with open(LABELS_FILE) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])

    ppis_need_embed = []
    for ppi in ppi_list:
        grid_path = GRID_DIR / f"{ppi}.npy"
        piston_out = EMBED_DIR / f"{ppi}_piston.npy"
        mint_out = EMBED_DIR / f"{ppi}_mint.npy"

        if grid_path.exists() and not (piston_out.exists() and mint_out.exists()):
            ppis_need_embed.append(ppi)

    log(f"PPIs needing embedding extraction: {len(ppis_need_embed)}")
    if not ppis_need_embed:
        log("Nothing to do.")
        return

    decoy_count = sum(1 for p in ppis_need_embed if "nat" not in p)
    native_count = len(ppis_need_embed) - decoy_count
    log(f"  Native: {native_count}, Decoy: {decoy_count}")

    mint_ckpt = str(MINT_CKPT)
    if not os.path.exists(mint_ckpt):
        alt = str(Path(__file__).parent / "piston-mint-fusion" / "saved_models" / "mint.ckpt")
        if os.path.exists(alt):
            mint_ckpt = alt
        else:
            alt2 = str(Path(__file__).parent / "mint.ckpt")
            if os.path.exists(alt2):
                mint_ckpt = alt2
    log(f"MINT checkpoint: {mint_ckpt}")

    from config.default_config import get_default_config, ensure_dirs

    config = get_default_config(
        pdb_dir=str(RAW_PDB_DIR),
        out_dir=str(OUT_DIR),
        mint_ckpt=mint_ckpt,
    )

    config["dirs"]["grid"] = str(GRID_DIR)
    config["dirs"]["embeddings"] = str(EMBED_DIR)
    config["dirs"]["pdb_dir"] = str(RAW_PDB_DIR)
    ensure_dirs(config)

    from training.train import extract_and_cache_embeddings

    extract_and_cache_embeddings(ppis_need_embed, labels, config, device=device)
    log("Done.")


if __name__ == "__main__":
    main()

"""Shared paths and constants for the SAbDab pipeline."""
from pathlib import Path
from datetime import datetime

BASE_DIR    = Path("/home/chenaoli/piston")
MINT_DIR    = BASE_DIR / "MintPiston"
SABDAB_TSV  = BASE_DIR / "sabdab_summary_all.tsv"
PDB_DIR     = BASE_DIR / "sabdab_processed" / "PDB"     # downloaded native PDBs
DECOY_DIR   = BASE_DIR / "sabdab_processed" / "decoys"  # pre-generated decoys
SIF_FILE    = MINT_DIR / "downloads" / "piston.sif"
PISTON_ROOT = MINT_DIR / "piston-main"
FUSION_ROOT = MINT_DIR / "piston-mint-fusion"
MINT_ROOT   = MINT_DIR / "mint-main"
MINT_CKPT   = BASE_DIR / "mint.ckpt"

OUT_DIR      = MINT_DIR / "pipeline_out"
DATA_PREP    = OUT_DIR / "data_preparation"
RAW_PDB_DIR  = DATA_PREP / "00-raw_pdbs"
GRID_DIR     = OUT_DIR / "grid"
EMBED_DIR    = OUT_DIR / "embeddings"
LOG_DIR      = OUT_DIR / "logs"
SAVED_MODELS = OUT_DIR / "saved_models"

PPI_LIST_FILE = OUT_DIR / "ppi_list.txt"
LABELS_FILE   = OUT_DIR / "labels.csv"
TRAIN_FILE    = OUT_DIR / "train.txt"
VAL_FILE      = OUT_DIR / "val.txt"
TEST_FILE     = OUT_DIR / "test.txt"
FAILED_FILE   = OUT_DIR / "failed_ppis.txt"
CONFIG_FILE   = OUT_DIR / "piston_config.py"

N_DECOYS = 1


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

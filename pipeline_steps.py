"""Step implementations for the SAbDab pipeline."""
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from pipeline_config import (
    log, SABDAB_TSV, PDB_DIR, DECOY_DIR, SIF_FILE, PISTON_ROOT,
    FUSION_ROOT, MINT_ROOT, MINT_CKPT,
    OUT_DIR, DATA_PREP, RAW_PDB_DIR, GRID_DIR, EMBED_DIR, LOG_DIR, SAVED_MODELS,
    PPI_LIST_FILE, LABELS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE,
    FAILED_FILE, CONFIG_FILE, N_DECOYS,
)
from pipeline_utils import ppi_encode, ppi_decode_pid, source_pdb_for, get_base_pdb


# ============================================================
# STEP 1 - Build PPI list from TSV
# ============================================================

def step1():
    """
    Parse sabdab_summary_all.tsv.
    For each row with a protein antigen and a downloaded native PDB:
      - add native entry  (label=1)
      - add decoys 1..5 that exist in DECOY_DIR (label=0)
    Returns (ppi_list, labels_dict).
    """
    log("[Step 1] Reading SAbDab TSV ...")
    downloaded = {p.stem.lower() for p in PDB_DIR.glob("*.pdb")}
    log(f"  Downloaded PDBs available: {len(downloaded)}")

    ppi_list, labels, seen = [], {}, set()
    skip_ag = skip_pdb = 0

    with open(SABDAB_TSV, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pdb     = row["pdb"].strip().lower()
            hchain  = row["Hchain"].strip()
            lchain  = row["Lchain"].strip()
            ag_raw  = row["antigen_chain"].strip()
            ag_type = row["antigen_type"].strip().lower()

            # Only protein antigens with valid chain info
            if not ag_raw or ag_raw in ("NA", "nan") or "protein" not in ag_type:
                skip_ag += 1
                continue

            # First non-empty antigen chain
            ag_chains = [
                c.strip() for c in ag_raw.split("|")
                if c.strip() not in ("", "NA", "nan")
            ]
            if not ag_chains:
                skip_ag += 1
                continue
            ch2 = ag_chains[0]

            # Antibody side = H+L or just H (nanobody)
            ch1_parts = [c for c in [hchain, lchain] if c and c not in ("NA", "nan")]
            if not ch1_parts:
                skip_ag += 1
                continue
            ch1 = "".join(ch1_parts)

            # Skip if native PDB was not downloaded
            if pdb not in downloaded:
                skip_pdb += 1
                continue

            # Native (label=1)
            nat_id = ppi_encode(pdb, "nat", ch1, ch2)
            if nat_id not in seen:
                ppi_list.append(nat_id)
                labels[nat_id] = 1
                seen.add(nat_id)

            # Decoys 1..5 — only if file already exists in DECOY_DIR (label=0)
            for n in range(1, N_DECOYS + 1):
                kind = f"d{n}"
                if source_pdb_for(pdb, kind).exists():
                    dec_id = ppi_encode(pdb, kind, ch1, ch2)
                    if dec_id not in seen:
                        ppi_list.append(dec_id)
                        labels[dec_id] = 0
                        seen.add(dec_id)

    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = sum(1 for v in labels.values() if v == 0)
    log(f"  Skipped (no protein antigen): {skip_ag}")
    log(f"  Skipped (PDB not downloaded): {skip_pdb}")
    log(f"  Total PPIs: {len(ppi_list)}  (native={n_pos}, decoy={n_neg})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PPI_LIST_FILE, "w") as f:
        f.write("\n".join(ppi_list) + "\n")
    with open(LABELS_FILE, "w") as f:
        f.write("PPI,label\n")
        for p in ppi_list:
            f.write(f"{p},{labels[p]}\n")
    log(f"  -> {PPI_LIST_FILE}")
    log(f"  -> {LABELS_FILE}")
    return ppi_list, labels


def load_ppi_list():
    """Load PPI list and labels from disk (written by step1)."""
    ppi_list = [l.strip() for l in open(PPI_LIST_FILE) if l.strip()]
    labels = {}
    with open(LABELS_FILE) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return ppi_list, labels


# ============================================================
# STEP 2 - Setup directories + place PDB files
# ============================================================

def step2(ppi_list):
    """Create directory tree, write piston_config.py, copy PDB files."""
    log("[Step 2] Setting up dirs and placing PDB files ...")

    for sub in [
        "00-raw_pdbs", "01-protonated_pdb", "02-refined_pdb",
        "03-cropped_pdbs", "04-chains_pdbs", "05-surface_ply", "06-patches"
    ]:
        (DATA_PREP / sub).mkdir(parents=True, exist_ok=True)
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS.mkdir(parents=True, exist_ok=True)

    # Write piston_config.py
    cfg = (
        "import os\n"
        "config = {}\n"
        "config['dirs'] = {}\n"
        f"config['dirs']['data_prepare']   = '{DATA_PREP}/'\n"
        f"config['dirs']['raw_pdb']        = '{DATA_PREP}/00-raw_pdbs/'\n"
        f"config['dirs']['protonated_pdb'] = '{DATA_PREP}/01-protonated_pdb/'\n"
        f"config['dirs']['refined']        = '{DATA_PREP}/02-refined_pdb/'\n"
        f"config['dirs']['cropped_pdb']    = '{DATA_PREP}/03-cropped_pdbs/'\n"
        f"config['dirs']['chains_pdb']     = '{DATA_PREP}/04-chains_pdbs/'\n"
        f"config['dirs']['surface_ply']    = '{DATA_PREP}/05-surface_ply/'\n"
        f"config['dirs']['patches']        = '{DATA_PREP}/06-patches/'\n"
        f"config['dirs']['grid']           = '{GRID_DIR}/'\n"
        f"config['dirs']['tmp']            = '{OUT_DIR}/tmp/'\n"
        "config['ppi_const'] = {}\n"
        "config['ppi_const']['contact_d']       = 5\n"
        "config['ppi_const']['surf_contact_r']  = 1\n"
        "config['ppi_const']['patch_r']         = 16\n"
        "config['ppi_const']['crop_r']          = 17\n"
        "config['ppi_const']['points_in_patch'] = 400\n"
        "config['mesh'] = {}\n"
        "config['mesh']['mesh_res'] = 1.0\n"
        "os.environ['TMP']    = config['dirs']['tmp']\n"
        "os.environ['TMPDIR'] = config['dirs']['tmp']\n"
        "os.environ['TEMP']   = config['dirs']['tmp']\n"
    )
    with open(CONFIG_FILE, "w") as f:
        f.write(cfg)
    log(f"  Config -> {CONFIG_FILE}")

    # Copy PDB files into raw_pdb dir as {pid}.pdb
    copied = already = missing = 0
    for ppi in tqdm(ppi_list, desc="  Copying PDBs"):
        pid = ppi.split("_")[0]          # e.g. '8vtdnat'
        dst = RAW_PDB_DIR / f"{pid}.pdb"
        if dst.exists():
            already += 1
            continue
        try:
            pdb, kind = ppi_decode_pid(pid)
        except ValueError as e:
            log(f"  WARN: {e}")
            missing += 1
            continue
        src = source_pdb_for(pdb, kind)
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    log(f"  Copied: {copied}  Already present: {already}  Missing source: {missing}")


# ============================================================
# STEP 3 - Run piston.sif (multithreaded)
# ============================================================

def _run_one_piston(ppi, timeout):
    """Run piston.sif for a single PPI. Returns (ppi, ok, err_msg)."""
    if (GRID_DIR / f"{ppi}.npy").exists():
        return ppi, True, "already done"
    cmd = [
        "apptainer", "exec",
        str(SIF_FILE),
        "python3", f"{PISTON_ROOT}/piston.py",
        "--config", str(CONFIG_FILE),
        "prepare", "--ppi", ppi, "--no_download",
    ]
    try:
        env = os.environ.copy()
        env["APPTAINER_BIND"] = f"{PISTON_ROOT}:{PISTON_ROOT},{OUT_DIR}:{OUT_DIR}"
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        (LOG_DIR / f"{ppi}.log").write_text(
            f"CMD: {' '.join(cmd)}\nRC: {r.returncode}\n"
            f"=STDOUT=\n{r.stdout}\n=STDERR=\n{r.stderr}"
        )
        if r.returncode == 0:
            return ppi, True, None
        return ppi, False, f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        return ppi, False, "timeout"
    except Exception as e:
        return ppi, False, str(e)


def step3(ppi_list, n_jobs=4, timeout=600):
    """Run piston.sif for all PPIs in parallel. Skip already-done ones."""
    log(f"[Step 3] Running piston.sif  n_jobs={n_jobs}  timeout={timeout}s")
    if not SIF_FILE.exists():
        log(f"  ERROR: SIF not found: {SIF_FILE}")
        return

    todo = [p for p in ppi_list if not (GRID_DIR / f"{p}.npy").exists()]
    log(f"  Already done: {len(ppi_list) - len(todo)}  To process: {len(todo)}")
    if not todo:
        log("  Nothing to do.")
        return

    success_count = 0
    failed = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_run_one_piston, p, timeout): p for p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  piston"):
            ppi, ok, err = fut.result()
            if ok:
                success_count += 1
            else:
                failed.append((ppi, err or ""))

    total_done = (len(ppi_list) - len(todo)) + success_count
    log(f"  New successes: {success_count}  Failed: {len(failed)}  Total done: {total_done}/{len(ppi_list)}")
    if failed:
        with open(FAILED_FILE, "w") as f:
            for ppi, err in failed:
                f.write(f"{ppi}\t{err}\n")
        log(f"  Failed list -> {FAILED_FILE}")
    log(f"  Grid files: {len(list(GRID_DIR.glob('*.npy')))}")


# ============================================================
# STEP 4 - Split train / val / test
# ============================================================

def step4(ppi_list, labels, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Group PPIs by base PDB id, then split groups into train/val/test.
    This prevents leakage: native+decoys of the same PDB stay together.
    Splits ALL PPIs in the list (does not require grid files).
    PPIs without grid files will be skipped at embedding/training time.
    """
    log("[Step 4] Splitting train/val/test ...")

    if not ppi_list:
        log("  No PPIs found — run step 1 first.")
        return

    # Optionally report grid coverage
    with_grid = [p for p in ppi_list if (GRID_DIR / f"{p}.npy").exists()]
    log(f"  Total PPIs: {len(ppi_list)}  (with grid: {len(with_grid)})")

    # Group by base PDB
    groups = defaultdict(list)
    for p in ppi_list:
        groups[get_base_pdb(p)].append(p)

    pdb_ids = sorted(groups.keys())
    random.seed(seed)
    random.shuffle(pdb_ids)

    n = len(pdb_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_pdbs = pdb_ids[:n_train]
    val_pdbs   = pdb_ids[n_train:n_train + n_val]
    test_pdbs  = pdb_ids[n_train + n_val:]

    train_ppis = [p for pdb in train_pdbs for p in groups[pdb]]
    val_ppis   = [p for pdb in val_pdbs   for p in groups[pdb]]
    test_ppis  = [p for pdb in test_pdbs  for p in groups[pdb]]

    for path, subset in [(TRAIN_FILE, train_ppis), (VAL_FILE, val_ppis), (TEST_FILE, test_ppis)]:
        with open(path, "w") as f:
            f.write("\n".join(subset) + "\n")

    def stats(subset):
        pos = sum(1 for p in subset if labels.get(p, 0) == 1)
        return f"{len(subset)} PPIs ({pos} native / {len(subset)-pos} decoy)"

    log(f"  Train: {stats(train_ppis)}  -> {TRAIN_FILE}")
    log(f"  Val:   {stats(val_ppis)}   -> {VAL_FILE}")
    log(f"  Test:  {stats(test_ppis)}  -> {TEST_FILE}")
    return train_ppis, val_ppis, test_ppis


# ============================================================
# STEP 5 - Extract PIsToN + MINT embeddings
# ============================================================

def step5():
    """
    Use the piston-mint-fusion pipeline to extract embeddings for
    all PPIs with completed grid files.
    Reads train/val/test splits and the labels CSV.
    """
    log("[Step 5] Extracting PIsToN + MINT embeddings ...")

    if not MINT_CKPT.exists():
        log(f"  ERROR: MINT checkpoint not found: {MINT_CKPT}")
        log("  Please place mint.ckpt at the expected path and retry.")
        return

    # Collect all PPIs across splits that have grid files
    all_ppis = []
    for split_file in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if split_file.exists():
            all_ppis += [l.strip() for l in open(split_file) if l.strip()]

    if not all_ppis:
        log("  No PPIs found — run steps 1/3/4 first.")
        return

    log(f"  Total PPIs to embed: {len(all_ppis)}")

    # Add fusion root to path and call the extraction script
    extract_script = FUSION_ROOT / "run_extract_embeddings.py"
    cmd = [
        sys.executable, str(extract_script),
        "--ppi_list",  str(PPI_LIST_FILE),
        "--labels",    str(LABELS_FILE),
        "--pdb_dir",   str(RAW_PDB_DIR),
        "--out_dir",   str(OUT_DIR),
        "--mint_ckpt", str(MINT_CKPT),
    ]
    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(FUSION_ROOT))
    if result.returncode != 0:
        log(f"  ERROR: embedding extraction failed (exit {result.returncode})")
    else:
        n_emb = len(list(EMBED_DIR.glob("*_piston.npy")))
        log(f"  Done. Embeddings saved: {n_emb} PPIs -> {EMBED_DIR}")


# ============================================================
# STEP 6 - Train fusion model + benchmark
# ============================================================

def step6():
    """
    Train the fusion MLP on cached embeddings and evaluate on the test set.
    """
    log("[Step 6] Training fusion model ...")

    for f in [TRAIN_FILE, VAL_FILE, TEST_FILE, LABELS_FILE]:
        if not f.exists():
            log(f"  ERROR: missing {f} — run steps 1/4 first.")
            return

    n_train_emb = len(list(EMBED_DIR.glob("*_piston.npy")))
    if n_train_emb == 0:
        log("  ERROR: no embeddings found — run step 5 first.")
        return
    log(f"  Found {n_train_emb} PIsToN embeddings in {EMBED_DIR}")

    train_script = FUSION_ROOT / "run_training.py"
    cmd = [
        sys.executable, str(train_script),
        "--train_list", str(TRAIN_FILE),
        "--val_list",   str(VAL_FILE),
        "--labels",     str(LABELS_FILE),
        "--out_dir",    str(OUT_DIR),
        "--max_epochs", "100",
        "--patience",   "15",
        "--hidden_dim", "512",
        "--batch_size", "32",
        "--lr",         "1e-3",
    ]
    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(FUSION_ROOT))
    if result.returncode != 0:
        log(f"  ERROR: training failed (exit {result.returncode})")
        return

    # Evaluate on test set
    log("  Evaluating on test set ...")
    _evaluate_test_set()


def _evaluate_test_set():
    """Load best model and evaluate on test split."""
    try:
        import torch
        import numpy as np
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        fusion_sys = str(FUSION_ROOT)
        if fusion_sys not in sys.path:
            sys.path.insert(0, fusion_sys)

        from training.evaluate import evaluate
        from training.dataset import CachedEmbeddingDataset
        from models.fusion_model import FusionClassifier
        from torch.utils.data import DataLoader
        import torch.nn as nn

        # Load test PPIs and labels
        test_ppis = [l.strip() for l in open(TEST_FILE) if l.strip()]
        labels = {}
        with open(LABELS_FILE) as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])

        dataset = CachedEmbeddingDataset(test_ppis, labels, str(EMBED_DIR))
        loader  = DataLoader(dataset, batch_size=32, shuffle=False)

        if len(dataset) == 0:
            log("  No test embeddings found, skipping evaluation.")
            return

        # Auto-detect dims from first sample
        sample = dataset[0]
        piston_dim = sample["piston_emb"].shape[0]
        mint_dim   = sample["mint_emb"].shape[0]
        input_dim  = piston_dim + mint_dim

        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = FusionClassifier(input_dim=input_dim, hidden_dim=512, dropout=0.2)

        # Load best checkpoint
        ckpt_path = SAVED_MODELS / "fusion_classifier_best.pth"
        if not ckpt_path.exists():
            ckpt_path = FUSION_ROOT / "saved_models" / "fusion_classifier_best.pth"
        if not ckpt_path.exists():
            log(f"  WARNING: no best model checkpoint found, skipping eval.")
            return

        classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
        classifier.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        metrics = evaluate(classifier, loader, device, loss_fn)

        log("  === Test Set Results ===")
        log(f"  AUC:       {metrics['auc']:.4f}")
        log(f"  Accuracy:  {metrics['accuracy']:.4f}")
        log(f"  F1:        {metrics['f1']:.4f}")
        log(f"  Precision: {metrics['precision']:.4f}")
        log(f"  Recall:    {metrics['recall']:.4f}")
        log(f"  Loss:      {metrics['loss']:.4f}")

        # Save results JSON
        results_path = OUT_DIR / "benchmark_results.json"
        import json
        with open(results_path, "w") as f:
            json.dump({
                "test_ppis": len(test_ppis),
                "test_evaluated": len(dataset),
                **{k: float(v) for k, v in metrics.items()}
            }, f, indent=2)
        log(f"  Results saved -> {results_path}")

    except Exception as e:
        import traceback
        log(f"  ERROR during evaluation: {e}")
        traceback.print_exc()

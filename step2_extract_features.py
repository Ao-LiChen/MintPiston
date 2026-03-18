#!/usr/bin/env python3
"""
Step 2: Extract features from generated decoy PDBs using Piston.
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time

sys.path.insert(0, '/home/chenaoli/piston/MintPiston')

from pipeline_config import (
    log, OUT_DIR, GRID_DIR, DATA_PREP, RAW_PDB_DIR,
    PDB_DIR, N_DECOYS, PISTON_ROOT, SIF_FILE
)

def get_decoy_pdbs():
    """Get list of decoy PDBs that need feature extraction."""
    decoy_pdbs = list(RAW_PDB_DIR.glob("*d1.pdb"))
    decoys_to_process = []
    
    for pdb_file in decoy_pdbs:
        pdb_id = pdb_file.stem  # e.g., "1a2yd1"
        
        # Find corresponding native to get chain info
        native_pdb = RAW_PDB_DIR / f"{pdb_id.replace('d1', 'nat')}.pdb"
        if not native_pdb.exists():
            continue
        
        # Find native grid files to infer chain combinations
        native_id = pdb_id.replace('d1', 'nat')
        native_grids = list(GRID_DIR.glob(f"{native_id}_*.npy"))
        
        for grid_file in native_grids:
            if "resnames" in grid_file.stem:
                continue
            
            # Extract chain info from native grid filename
            # e.g., "1a2ynat_BA_C.npy" -> "BA_C"
            parts = grid_file.stem.split('_')
            if len(parts) >= 3:
                ch1 = parts[1]
                ch2 = parts[2]
                decoy_ppi = f"{pdb_id}_{ch1}_{ch2}"
                
                # Check if grid already exists
                decoy_grid = GRID_DIR / f"{decoy_ppi}.npy"
                if not decoy_grid.exists():
                    decoys_to_process.append(decoy_ppi)
    
    return sorted(set(decoys_to_process))

def extract_features(decoy_ppi, timeout=1800):
    """
    Extract features for a decoy PPI using Piston.
    
    Args:
        decoy_ppi: e.g., "1a2yd1_BA_C"
        timeout: timeout in seconds
    
    Returns:
        (success, message)
    """
    try:
        # Check if grid already exists
        decoy_grid = GRID_DIR / f"{decoy_ppi}.npy"
        if decoy_grid.exists():
            return True, f"Already exists: {decoy_ppi}"
        
        # Check if decoy PDB exists
        pdb_id = decoy_ppi.split('_')[0]
        decoy_pdb = RAW_PDB_DIR / f"{pdb_id}.pdb"
        if not decoy_pdb.exists():
            return False, f"Decoy PDB not found: {decoy_pdb}"
        
        # Run Piston to extract features (no --no_download so protonation runs)
        piston_cmd = [
            "timeout", str(timeout),
            "apptainer", "exec",
            "--bind", f"{PISTON_ROOT}:{PISTON_ROOT}",
            "--bind", f"{OUT_DIR}:{OUT_DIR}",
            str(SIF_FILE),
            "python3", f"{PISTON_ROOT}/piston.py",
            "--config", f"{OUT_DIR}/piston_config.py",
            "prepare",
            "--ppi", decoy_ppi,
        ]
        
        result = subprocess.run(piston_cmd, capture_output=True, text=True, timeout=timeout+10)
        
        if result.returncode == 0 and decoy_grid.exists():
            return True, f"Generated: {decoy_ppi}"
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return False, f"Piston failed for {decoy_ppi}: {error_msg}"
    
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {decoy_ppi}"
    except Exception as e:
        return False, f"Error: {decoy_ppi} - {str(e)[:200]}"

def main():
    log("[Step 2: Feature Extraction] Starting...")
    
    # Get decoy PPIs that need feature extraction
    decoys = get_decoy_pdbs()
    log(f"Found {len(decoys)} decoy PPIs to process")
    
    if len(decoys) == 0:
        log("No decoys to process. Run step1_generate_decoy_pdbs.py first.")
        return
    
    total_tasks = len(decoys)
    completed = 0
    failed = 0
    
    log(f"Extracting features for {total_tasks} decoys...")
    log("Using 8 parallel workers with 1800s timeout per task")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        
        # Submit all tasks
        for decoy_ppi in decoys:
            future = executor.submit(extract_features, decoy_ppi, timeout=1800)
            futures[future] = decoy_ppi
        
        # Process results
        for i, future in enumerate(as_completed(futures), 1):
            decoy_ppi = futures[future]
            try:
                success, msg = future.result()
                if success:
                    completed += 1
                    if i % 10 == 0 or completed <= 5:
                        log(f"[{i}/{total_tasks}] Success: {msg}")
                else:
                    failed += 1
                    log(f"[{i}/{total_tasks}] Failed: {msg}")
            except Exception as e:
                failed += 1
                log(f"[{i}/{total_tasks}] Exception: {decoy_ppi} - {str(e)[:100]}")
    
    log(f"\n[Step 2: Feature Extraction] Complete!")
    log(f"  Completed: {completed}")
    log(f"  Failed: {failed}")
    log(f"  Total: {completed + failed}")

if __name__ == "__main__":
    main()

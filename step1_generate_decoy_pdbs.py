#!/usr/bin/env python3
"""
Step 1: Generate HDOCK decoy PDBs for all native PPIs.
Only generates PDB files, does not extract features.
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time
import shutil

sys.path.insert(0, '/home/chenaoli/piston/MintPiston')

from pipeline_config import (
    log, OUT_DIR, GRID_DIR, DATA_PREP, RAW_PDB_DIR,
    PDB_DIR, N_DECOYS, PISTON_ROOT, SIF_FILE
)

def get_native_ppis_with_grid():
    """Get list of native PPIs that have grid files."""
    grid_files = list(GRID_DIR.glob("*nat*.npy"))
    natives = []
    for gf in grid_files:
        ppi_id = gf.stem
        if "resnames" not in ppi_id and "nat" in ppi_id:
            natives.append(ppi_id)
    return sorted(set(natives))

def extract_pdb_chain(pdb_full_file, pdb_chain_file, ch):
    """Extract specific chains from PDB file."""
    with open(pdb_full_file, 'r') as f:
        with open(pdb_chain_file, 'w') as out:
            for line in f.readlines():
                if (line[0:4]=='ATOM' or line[0:6]=='HETATM') and line[21] in ch:
                    out.write(line)

def extract_model(pdb_file, out_pdb, model_num):
    """Extract a specific model from multi-model PDB file (ATOM lines only)."""
    to_write = False
    with open(pdb_file, 'r') as f:
        with open(out_pdb, 'w') as out:
            for line in f.readlines():
                if line[:6]=='ENDMDL':
                    to_write=False
                if to_write:
                    # Only write ATOM and HETATM lines
                    if line[:4]=='ATOM' or line[:6]=='HETATM':
                        out.write(line)
                if line[:5]=='MODEL':
                    if line.split(' ')[-1].strip('\n') == str(model_num):
                        to_write = True

def generate_decoy_pdb(native_ppi, timeout=600):
    """
    Generate one decoy PDB for a native PPI using HDOCK.
    Only generates PDB, does not extract features.
    
    Args:
        native_ppi: e.g., "1a2ynat_HL_C"
        timeout: timeout in seconds for HDOCK
    
    Returns:
        (success, message)
    """
    try:
        # Parse PPI ID
        parts = native_ppi.split('_')
        if len(parts) != 3:
            return False, f"Invalid PPI format: {native_ppi}"
        
        pdb_id = parts[0].replace('nat', '')
        ch1 = parts[1]
        ch2 = parts[2]
        
        # Check if decoy PDB already exists
        decoy_pdb = RAW_PDB_DIR / f"{pdb_id}d1.pdb"
        if decoy_pdb.exists():
            return True, f"Already exists: {pdb_id}d1.pdb"
        
        # Check if native PDB exists
        native_pdb = RAW_PDB_DIR / f"{pdb_id}nat.pdb"
        if not native_pdb.exists():
            return False, f"Native PDB not found: {native_pdb}"
        
        # Create working directory
        work_dir = DATA_PREP / 'docked' / native_ppi
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract chains
        ch1_file = work_dir / f"{pdb_id}_{ch1}.pdb"
        ch2_file = work_dir / f"{pdb_id}_{ch2}.pdb"
        
        extract_pdb_chain(str(native_pdb), str(ch1_file), ch1)
        extract_pdb_chain(str(native_pdb), str(ch2_file), ch2)
        
        if not ch1_file.exists() or not ch2_file.exists():
            return False, f"Failed to extract chains for {native_ppi}"
        
        # Step 2: Run HDOCK in container
        out_prefix = f"{pdb_id}_{ch1}_{ch2}"
        out_file = work_dir / f"{out_prefix}.out"
        
        hdock_cmd = [
            "apptainer", "exec",
            "--bind", f"{DATA_PREP}:{DATA_PREP}",
            "--pwd", str(work_dir),
            str(SIF_FILE),
            "hdock",
            f"{pdb_id}_{ch1}.pdb", f"{pdb_id}_{ch2}.pdb",
            "-out", f"{out_prefix}.out"
        ]
        
        result = subprocess.run(hdock_cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0 or not out_file.exists():
            return False, f"HDOCK failed for {native_ppi}: {result.stderr[:200]}"
        
        # Step 3: Generate PDB models using createpl
        docked_pdb = work_dir / f"{out_prefix}.pdb"
        
        createpl_cmd = [
            "apptainer", "exec",
            "--bind", f"{DATA_PREP}:{DATA_PREP}",
            "--pwd", str(work_dir),
            str(SIF_FILE),
            "createpl",
            f"{out_prefix}.out", f"{out_prefix}.pdb",
            "-complex", "-nmax", "100"
        ]
        
        result = subprocess.run(createpl_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0 or not docked_pdb.exists():
            return False, f"createpl failed for {native_ppi}: {result.stderr[:200]}"
        
        # Step 4: Extract the 1st model as decoy
        extract_model(str(docked_pdb), str(decoy_pdb), 1)
        
        if not decoy_pdb.exists():
            return False, f"Failed to extract decoy: {decoy_pdb}"
        
        # Clean up intermediate files to save space
        shutil.rmtree(work_dir, ignore_errors=True)
        
        return True, f"Generated: {pdb_id}d1.pdb"
    
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {native_ppi}"
    except Exception as e:
        return False, f"Error: {native_ppi} - {str(e)[:200]}"

def main():
    log("[Step 1: HDOCK Decoy PDB Generation] Starting...")
    
    # Get native PPIs with grid files
    natives = get_native_ppis_with_grid()
    log(f"Found {len(natives)} native PPIs with grid files")
    
    if len(natives) == 0:
        log("ERROR: No native PPIs found!")
        return
    
    # Run full dataset
    # Randomly shuffle to distribute workload
    import random
    random.seed(42)
    random.shuffle(natives)
    # natives = natives[:10]  # Uncomment for testing
    
    total_tasks = len(natives)
    completed = 0
    failed = 0
    
    log(f"Generating {total_tasks} decoy PDBs...")
    log("Using 8 parallel workers with 600s timeout per task")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        
        # Submit all tasks
        for native_ppi in natives:
            future = executor.submit(generate_decoy_pdb, native_ppi, timeout=600)
            futures[future] = native_ppi
        
        # Process results
        for i, future in enumerate(as_completed(futures), 1):
            native_ppi = futures[future]
            try:
                success, msg = future.result()
                if success:
                    completed += 1
                    log(f"[{i}/{total_tasks}] Success: {msg}")
                else:
                    failed += 1
                    log(f"[{i}/{total_tasks}] Failed: {msg}")
            except Exception as e:
                failed += 1
                log(f"[{i}/{total_tasks}] Exception: {native_ppi} - {str(e)[:100]}")
    
    log(f"\n[Step 1: HDOCK Decoy PDB Generation] Complete!")
    log(f"  Completed: {completed}")
    log(f"  Failed: {failed}")
    log(f"  Total: {completed + failed}")
    log(f"\nNext step: Run step2_extract_features.py to extract features")

if __name__ == "__main__":
    main()

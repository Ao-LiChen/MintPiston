# PIsToN-MINT Fusion

A framework that combines **PIsToN** (structure-based surface analysis) and **MINT** (sequence-based protein language model) for protein-protein interface quality prediction, evaluated on the MaSIF-test benchmark.

## Results on MaSIF-test

Benchmark on 262 test samples from the [MaSIF-test dataset](https://github.com/LPDI-EPFL/masif), comparing three classifiers trained on top of frozen backbone embeddings:

| Model       |   AUC  |   Acc  |    F1  |  Prec  |   Rec  |   MCC  |
|-------------|--------|--------|--------|--------|--------|--------|
| PIsToN-only | 0.9565 | 0.8817 | 0.8755 | 0.9237 | 0.8321 | 0.7671 |
| MINT-only   | 0.8209 | 0.7023 | 0.7383 | 0.6587 | 0.8397 | 0.4208 |
| **Fusion**  | **0.9573** | **0.8931** | **0.8939** | 0.8872 | **0.9008** | **0.7864** |

The fusion model achieves the best AUC (0.9573) and F1 (0.8939), combining PIsToN's precision with MINT's recall.

To reproduce:

```bash
python run_benchmark.py \
    --data_root /path/to/masif_test_copy \
    --mint_ckpt saved_models/mint.ckpt \
    --out_dir ./output \
    --device cuda
```

---

## Overview

This project fuses two complementary approaches:

- **PIsToN** extracts biophysical features from protein surfaces via MaSIF-based triangulation, producing 2D interface maps analyzed by a Vision Transformer → **16-dim structural embedding**
- **MINT** uses an ESM-2 language model with multimer inter-chain attention to capture sequence context at the residue level → **1280-dim embedding per chain**

The key bridge: `patch_residue_mapper.py` identifies which residues fall inside PIsToN's surface patch and tells MINT to pool only over those interface residues—both modalities focus on the same atomic region.

```
PDB File
  ├── [PIsToN preprocessing]   surface triangulation → 2D grid (32×32×13) + 13 FireDock energy terms
  │         └── [PIsToN ViT] (frozen)   →   16-dim structural embedding
  │
  └── [Sequence extraction]   per-chain amino acid sequences
            ├── [Patch residue mapper]   resnames.npy → interface residue indices
            └── [MINT ESM-2] (frozen)   layer-33 → mean-pool over patch residues → N×1280-dim

  [16 + N×1280] → [MLP classifier] → P(native interface)

  2-chain dimer:    16 + 2×1280 = 2576-dim
  3-chain Ab-Ag:    16 + 3×1280 = 3856-dim
```

---

## Project Structure

```
MinP/
  piston-main/           # PIsToN source code and pretrained weights
  mint-main/             # MINT source code
  piston-mint-fusion/    # This project
    config/
      default_config.py        # Unified configuration
    data/
      labels.csv               # PPI → binary label (0/1)
      all_ppis.txt / train.txt / val.txt / test.txt
    data_prepare/
      sequence_extractor.py    # PDB → per-chain sequences + residue maps
    models/
      piston_embedder.py       # PIsToN backbone wrapper (16-dim CLS output)
      mint_embedder.py         # MINT backbone wrapper (N×1280-dim output)
      patch_residue_mapper.py  # resnames.npy → MINT token indices
      fusion_model.py          # MLP classifier head
    training/
      dataset.py               # Dataset classes (raw and cached-embedding modes)
      train.py                 # Training loop with early stopping
      evaluate.py              # AUC, accuracy, F1, MCC, etc.
    inference/
      pipeline.py              # End-to-end prediction pipeline
    saved_models/
      PIsToN-only_best.pth     # Trained PIsToN-only MLP head
      MINT-only_best.pth       # Trained MINT-only MLP head
      Fusion_best.pth          # Trained fusion MLP head
    run_benchmark.py           # Reproduce MaSIF-test benchmark (3-way comparison)
    run_training.py            # Train on custom data
    run_extract_embeddings.py  # Pre-extract backbone embeddings
    run_inference.py           # Predict on new PDB files
    prepare_masif_data.py      # Prepare MaSIF-test labels and splits
    environment.yml
    setup.py
```

---

## Installation

### Requirements

- Linux or Windows with WSL2 (recommended); Windows native supported
- CUDA GPU with ≥8 GB VRAM
- Conda

### 1. Set up directory layout

```
MinP/
  piston-main/
  mint-main/
  piston-mint-fusion/
```

### 2. Create environment

```bash
cd piston-mint-fusion
conda env create -f environment.yml
conda activate piston-mint
pip install -e .
```

### 3. Model checkpoints

```
piston-main/saved_models/PIsToN_multiAttn_contrast.pth   # included in piston-main
mint-main/data/esm2_t33_650M_UR50D.json                  # included in mint-main
piston-mint-fusion/saved_models/mint.ckpt                # download separately (see Data & Weights)
```

### 4. PIsToN preprocessing tools (only needed for new data)

PIsToN preprocessing requires external tools (MaSIF, MSMS, APBS, DSSP, FireDock, PDB2PQR). The easiest approach is the pre-built **Singularity container**:

```bash
wget https://users.cs.fiu.edu/~vsteb002/piston_sif/piston.sif
```

If you only want to run on the MaSIF-test dataset, the preprocessed data is available directly (see below) and this step is not needed.

---

## Data & Weights

The preprocessed MaSIF-test data and MINT checkpoint are distributed together as a single archive on Zenodo:

> **DOI**: *(to be added after upload)*

The archive contains:
```
masif_test_copy/
  PDB/                             1,376 PDB files (receptor + ligand chain splits)
  prepare_energies_16R/07-grid/    2,712 .npy interface maps + 1,361 .ref energy files
  capri_quality_masif_test.csv     CAPRI quality labels

mint.ckpt                          MINT ESM-2 650M checkpoint (PyTorch Lightning)
```

After downloading, place `mint.ckpt` in `piston-mint-fusion/saved_models/`.

---

## Running the Benchmark

```bash
python run_benchmark.py \
    --data_root /path/to/masif_test_copy \
    --mint_ckpt saved_models/mint.ckpt \
    --out_dir ./output \
    --device cuda
```

This runs three phases automatically:
1. Extract PIsToN (16-dim) and MINT (2560-dim) embeddings for all 1,356 PPIs
2. Train three MLP classifiers (PIsToN-only, MINT-only, Fusion) with early stopping
3. Evaluate on 262 test samples and print the comparison table

Results are saved to `./output/benchmark_results.csv` and `./output/benchmark_results.json`.

---

## Inference on New Data

### Single complex

```bash
python run_inference.py \
    --pdb /path/to/complex.pdb \
    --side1 A \
    --side2 B \
    --mint_ckpt saved_models/mint.ckpt \
    --fusion_ckpt saved_models/Fusion_best.pth \
    --out_dir ./results
```

### Antibody-antigen (3-chain)

```bash
python run_inference.py \
    --pdb /path/to/ab_ag.pdb \
    --side1 HL \
    --side2 C \
    --mint_ckpt saved_models/mint.ckpt \
    --fusion_ckpt saved_models/Fusion_best.pth \
    --out_dir ./results
```

Output: `Native probability: 0.87` (higher = more native-like interface).

---

## Training on Custom Data

**Phase 1**: Preprocess your PDB files with PIsToN (`piston.py prepare`) to generate the grid, resnames, and `.ref` energy files. Then prepare labels and splits.

**Phase 2**: Pre-extract backbone embeddings (runs each frozen backbone once):

```bash
python run_extract_embeddings.py \
    --ppi_list data/all_ppis.txt \
    --pdb_dir /path/to/pdbs \
    --grid_dir /path/to/07-grid \
    --out_dir ./output \
    --mint_ckpt saved_models/mint.ckpt
```

**Phase 3**: Train the MLP classifier head:

```bash
python run_training.py \
    --train_list data/train.txt \
    --val_list data/val.txt \
    --labels data/labels.csv \
    --out_dir ./output \
    --lr 1e-3 \
    --batch_size 8 \
    --max_epochs 100
```

---

## Input Data Format

| File | Shape / Format | Description |
|------|---------------|-------------|
| `{ppi}.npy` | (32, 32, 13) float32 | 2D interface map, 13 biophysical channels |
| `{ppi}_resnames.npy` | object array | Surface vertex → `"chain:resid:resname-atomid:atomname"` |
| `refined-out-{ppi}.ref` | text | 13 FireDock energy terms |
| `{PDB_ID}.pdb` | standard PDB | Full complex structure for sequence extraction |

PPI identifier format: `{PDB_ID}_{side1_chains}_{side2_chains}`
- `1A0G_A_B` — simple dimer
- `1A14_HL_N` — antibody (H+L chains) vs antigen (N)
- `1A79-model-2-pos_A_Z` — MaSIF-test docking model

---

## References

- **PIsToN**: Stebliankin et al. "Evaluating protein binding interfaces with transformer networks." *Nature Machine Intelligence* (2023). https://doi.org/10.1038/s42256-023-00715-4
- **MINT**: Ullanat et al. "Learning the language of protein-protein interactions." *Nature Communications* (2026).
- **MaSIF**: Gainza et al. "Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning." *Nature Methods* (2020).
- **ESM-2**: Lin et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* (2023).

## License

- PIsToN: CC BY-NC-ND 4.0 (non-commercial use only)
- MINT: MIT License
- This fusion code: CC BY-NC-ND 4.0 (follows the most restrictive component)

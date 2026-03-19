# MintPiston

A framework combining **PIsToN** (structure-based surface analysis) and **MINT** (sequence-based protein language model) for protein-protein interface quality prediction.

## Benchmark Results

### MaSIF-test (262 test samples)

Benchmark on the [MaSIF-test dataset](https://github.com/LPDI-EPFL/masif), comparing three MLP classifiers trained on frozen backbone embeddings:

| Model       |   AUC  |   Acc  |    F1  |  Prec  |   Rec  |   MCC  |
|-------------|--------|--------|--------|--------|--------|--------|
| PIsToN-only | 0.9565 | 0.8817 | 0.8755 | 0.9237 | 0.8321 | 0.7671 |
| MINT-only   | 0.8209 | 0.7023 | 0.7383 | 0.6587 | 0.8397 | 0.4208 |
| **Fusion**  | **0.9573** | **0.8931** | **0.8939** | 0.8872 | **0.9008** | **0.7864** |

### SAbDab antibody-antigen (1,676 test samples)

Benchmark on SAbDab-derived antibody-antigen interfaces with HDOCK-generated decoys, class-weighted loss, and zero-padding for variable MINT dimensions (2560 -> 3840):

| Model       |   AUC  |   Acc  |    F1  |  Prec  |   Rec  |   MCC  |   N  | Pos  | Neg |
|-------------|--------|--------|--------|--------|--------|--------|------|------|-----|
| PIsToN-only | 0.6672 | 0.6510 | 0.7570 | 0.8660 | 0.6723 | 0.1897 | 1676 | 1355 | 321 |
| MINT-only   | 0.7628 | 0.5811 | 0.6692 | **0.9257** | 0.5240 | 0.2736 | 1676 | 1355 | 321 |
| **Fusion**  | **0.8127** | **0.7440** | **0.8267** | 0.9134 | **0.7550** | **0.3784** | 1676 | 1355 | 321 |

---

## Overview

This project fuses two complementary approaches:

- **PIsToN** extracts biophysical features from protein surfaces via MaSIF-based triangulation, producing 2D interface maps analyzed by a Vision Transformer -> **16-dim structural embedding**
- **MINT** uses an ESM-2 language model with multimer inter-chain attention to capture sequence context at the residue level -> **1280-dim embedding per chain**

`patch_residue_mapper.py` identifies which residues fall inside PIsToN's surface patch and tells MINT to pool only over those interface residues -- both modalities focus on the same atomic region.

```
PDB File
  +-- [PIsToN preprocessing]   surface triangulation -> 2D grid (32x32x13) + 13 FireDock energy terms
  |         +-- [PIsToN ViT] (frozen)   ->   16-dim structural embedding
  |
  +-- [Sequence extraction]   per-chain amino acid sequences
            +-- [Patch residue mapper]   resnames.npy -> interface residue indices
            +-- [MINT ESM-2] (frozen)   layer-33 -> mean-pool over patch residues -> Nx1280-dim

  [16 + Nx1280] -> [MLP classifier] -> P(native interface)

  2-chain dimer:    16 + 2x1280 = 2576-dim
  3-chain Ab-Ag:    16 + 3x1280 = 3856-dim
```

---

## Repository Layout

```
MintPiston/
  piston-mint-fusion/            Core fusion framework
    models/                      PIsToN embedder, MINT embedder, fusion MLP
    training/                    Dataset, trainer, evaluator
    inference/                   Single-complex inference pipeline
    benchmarks/sabdab/           SAbDab benchmark script
    run_benchmark.py             MaSIF-test benchmark
    run_masif_test.py            MaSIF-test evaluation
    run_inference.py             Inference CLI
    run_training.py              Training CLI
    run_extract_embeddings.py    Embedding extraction CLI
  piston-main/                   PIsToN model
  mint-main/                     MINT model
  *.py, *.sh                     SAbDab pipeline scripts (see below)
```

---

## Installation

### 1. Create environment

```bash
cd piston-mint-fusion
conda env create -f environment.yml
conda activate piston-mint
pip install -e .
```

### 2. Model checkpoints

```
piston-main/saved_models/PIsToN_multiAttn_contrast.pth   # included in repo
mint-main/data/esm2_t33_650M_UR50D.json                  # included in repo
piston-mint-fusion/saved_models/mint.ckpt                 # download (see Data & Weights)
```

### 3. PIsToN preprocessing tools (only needed for new data)

PIsToN preprocessing requires external tools (MaSIF, MSMS, APBS, DSSP, FireDock, PDB2PQR). The easiest approach is the pre-built Singularity container:

```bash
wget https://users.cs.fiu.edu/~vsteb002/piston_sif/piston.sif
```

If you only want to reproduce the benchmarks, the preprocessed data is available directly (see below) and this step is not needed.

---

## Data & Weights

| Dataset | Contents | Link |
|---------|----------|------|
| MaSIF-test data + MINT checkpoint | 1,376 PDB files, 2,712 grid maps, CAPRI labels, `mint.ckpt` | [Google Drive](https://drive.google.com/file/d/11MONX_6Y_O6Oyqf-jOD0orDofJ4RNfqP/view?usp=sharing) |
| SAbDab pipeline_out | 8,452 PDB files (native+decoy), PIsToN grids, embeddings, labels, splits | [Google Drive](https://drive.google.com/file/d/1BwTtCkWKMTvXvNtcsaKoJx_dFEidfGYW/view?usp=sharing) |

### MaSIF-test data

After downloading, unzip so that `masif_test_copy/` sits next to `piston-mint-fusion/`. Place `mint.ckpt` in `piston-mint-fusion/saved_models/`.

```
masif_test_copy/
  PDB/                             1,376 PDB files
  prepare_energies_16R/07-grid/    2,712 .npy grid maps + 1,361 .ref energy files
  capri_quality_masif_test.csv     CAPRI quality labels
```

### SAbDab data

After downloading `pipeline_out_sabdab.zip`, unzip so that `pipeline_out/` sits next to `piston-mint-fusion/`:

```
pipeline_out/
  labels.csv                       16,710 PPI labels (native=11,041, decoy=5,669)
  ppi_list.txt                     16,710 PPI identifiers
  train.txt / val.txt / test.txt   train/val/test splits (no PDB-level leakage)
  benchmark_v2_results.json        SAbDab benchmark results
  data_preparation/00-raw_pdbs/    8,452 native + HDOCK decoy PDB files (7.6 GB)
  grid/                            PIsToN grid .npy + _resnames.npy files (12,586 PPIs, 1.7 GB)
  embeddings/                      pre-extracted {ppi}_piston.npy + {ppi}_mint.npy (11,342 PPIs, 212 MB)
```

---

## Reproducing the Benchmarks

### MaSIF-test benchmark

```bash
cd piston-mint-fusion
python run_benchmark.py \
    --data_root /path/to/masif_test_copy \
    --mint_ckpt saved_models/mint.ckpt \
    --out_dir ./output \
    --device cuda
```

This extracts PIsToN (16-dim) and MINT (2560-dim) embeddings for 1,356 PPIs, trains three MLP classifiers (PIsToN-only, MINT-only, Fusion) with early stopping, and evaluates on 262 test samples.

### SAbDab benchmark

```bash
cd MintPiston
python run_benchmark_v2.py --device cuda --max_epochs 100 --patience 15
```

This trains and evaluates three MLP classifiers (PIsToN-only, MINT-only, Fusion) on pre-extracted embeddings from `pipeline_out/`. Uses class-weighted loss, zero-padding for variable MINT dimensions, and proper train/val/test splits with no PDB-level leakage.

---

## Inference on New Data

### Simple dimer

```bash
cd piston-mint-fusion
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
cd piston-mint-fusion
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
cd piston-mint-fusion
python run_extract_embeddings.py \
    --ppi_list data/all_ppis.txt \
    --pdb_dir /path/to/pdbs \
    --out_dir ./output \
    --mint_ckpt saved_models/mint.ckpt
```

**Phase 3**: Train the MLP classifier head:

```bash
cd piston-mint-fusion
python run_training.py \
    --train_list data/train.txt \
    --val_list data/val.txt \
    --labels data/labels.csv \
    --out_dir ./output \
    --lr 1e-3 --batch_size 8 --max_epochs 100
```

---

## SAbDab Scripts

Scripts at the repository root for SAbDab benchmark and data management:

| File | Purpose |
|------|---------|
| `run_benchmark_v2.py` | SAbDab benchmark: trains and evaluates PIsToN-only, MINT-only, and Fusion classifiers |
| `extract_decoy_embeddings.py` | Extract MINT+PIsToN embeddings for PPIs with grid files |
| `pipeline_config.py` | Shared paths and constants |
| `pipeline_utils.py` | PPI ID encoding/decoding helpers |

---

## Input Data Format

| File | Shape / Format | Description |
|------|---------------|-------------|
| `{ppi}.npy` | (32, 32, 13) float32 | 2D interface map, 13 biophysical channels |
| `{ppi}_resnames.npy` | object array | Surface vertex -> `"chain:resid:resname-atomid:atomname"` |
| `refined-out-{ppi}.ref` | text | 13 FireDock energy terms |
| `{PDB_ID}.pdb` | standard PDB | Full complex structure for sequence extraction |

PPI identifier format: `{PDB_ID}_{side1_chains}_{side2_chains}`
- `1A0G_A_B` -- simple dimer
- `1A14_HL_N` -- antibody (H+L chains) vs antigen (N)
- `1A79-model-2-pos_A_Z` -- MaSIF-test docking model

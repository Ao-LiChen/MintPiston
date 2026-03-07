"""
Dataset classes for the PIsToN + MINT fusion model.

Provides two dataset modes:
1. FusionDataset: loads raw data (grid, energies, PDB) and computes everything
   on-the-fly. Suitable for inference or small-scale experiments.
2. CachedEmbeddingDataset: loads pre-extracted PIsToN + MINT embeddings from
   disk. Suitable for efficient MLP training (recommended).

Supports N-chain complexes (e.g. antibody H+L + antigen = 3 chains).
PIsToN PPI format: PID_side1_side2 where each side can be multi-letter
(e.g. '1AHW_HL_C' means chains H,L vs chain C).
"""

import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data_prepare.sequence_extractor import parse_ppi_identifier


# Standardization constants from PIsToN training set
# (copied from piston-main/utils/dataset.py lines 248-264)
FEATURE_MEAN = np.array([
    0.06383528408485302, 0.043833505848899605, -0.08456032982438057,
    0.007828608135306595, -0.06060602411612203, 0.06383528408485302,
    0.043833505848899605, -0.08456032982438057, 0.007828608135306595,
    -0.06060602411612203, 11.390402735801011, 0.1496338245579665,
    0.1496338245579665,
])

FEATURE_STD = np.array([
    0.4507792893174703, 0.14148081793902434, 0.16581325050002976,
    0.28599861830017204, 0.6102229371168204, 0.4507792893174703,
    0.14148081793902434, 0.16581325050002976, 0.28599861830017204,
    0.6102229371168204, 7.265311558033949, 0.18003612950610695,
    0.18003612950610695,
])

ENERGY_MEAN = np.array([
    -193.1392953586498, -101.97838818565408, 264.2099535864983,
    -17.27086075949363, 16.329959915611877, -102.78101054852341,
    36.531006329113836, -27.1124789029536, 16.632626582278455,
    -8.784924050632918, -6.206329113924051, -1.8290084388185655,
    -11.827215189873417,
])

ENERGY_STD = np.array([
    309.23521244706757, 66.75799437657368, 9792.783784373369,
    25.384427268309658, 7.929941961525389, 94.05055841984323,
    47.22518557457095, 24.392679889433445, 17.57399925906454,
    7.041949880295568, 6.99554122803362, 2.557571754303165,
    13.666329541281653,
])


def _learn_background_mask(grid_shape):
    """Circular mask: zero outside patch radius."""
    h, w = grid_shape[0], grid_shape[1]
    radius = h / 2.0
    mask = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            x = c - radius
            y = radius - r
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[r][c] = 1
    return mask


def _read_energies(energies_dir, ppi):
    """Read FireDock energy terms from .ref file."""
    energies_path = os.path.join(energies_dir, f"refined-out-{ppi}.ref")
    to_read = False
    all_energies = None
    if os.path.exists(energies_path):
        with open(energies_path, "r") as f:
            for line in f.readlines():
                if to_read:
                    all_energies = line.split("|")
                    all_energies = [x.strip(" ") for x in all_energies]
                    all_energies = all_energies[5:18]
                    all_energies = [float(x) for x in all_energies]
                    all_energies = np.array(all_energies)
                    break
                if "Sol # |" in line:
                    to_read = True
    if all_energies is None:
        all_energies = np.zeros(13)
    all_energies = np.nan_to_num(all_energies)
    return all_energies


def load_and_scale_grid(grid_dir, ppi):
    """
    Load a PIsToN interface map and apply standard scaling + masking.

    Returns:
        grid: (13, H, W) float32 tensor
        energies: (13,) float32 tensor
    """
    grid_path = os.path.join(grid_dir, f"{ppi}.npy")
    grid = np.load(grid_path, allow_pickle=True)  # (H, W, 13)
    mask = _learn_background_mask(grid.shape)
    grid = np.swapaxes(grid, -1, 0).astype(np.float32)  # -> (13, H, W)

    # Standard scale features
    for i in range(grid.shape[0]):
        grid[i] = (grid[i] - FEATURE_MEAN[i]) / FEATURE_STD[i]
    # Mask out values outside the circular patch
    grid = (grid * mask[np.newaxis, :, :]).astype(np.float32)

    # Energies
    energies = _read_energies(grid_dir, ppi).astype(np.float32)
    for i in range(len(energies)):
        energies[i] = (energies[i] - ENERGY_MEAN[i]) / ENERGY_STD[i]

    return grid, energies


def tokenize_chains(alphabet, sequences, chain_order):
    """
    Tokenize N chain sequences in MINT format.

    Each chain is encoded as: <cls> + residues + <eos>
    All chains are concatenated, and each chain gets a unique chain_id.

    Args:
        alphabet: ESM-1b Alphabet instance
        sequences: dict[chain_letter -> sequence_str]
        chain_order: list of individual chain letters, e.g. ['H', 'L', 'C']

    Returns:
        tokens: (T,) long tensor
        chain_ids: (T,) long tensor with values 0, 1, 2, ...
    """
    all_enc = []
    all_chain_ids = []

    for idx, chain_letter in enumerate(chain_order):
        seq = sequences[chain_letter].replace("J", "L")
        enc = alphabet.encode("<cls>" + seq + "<eos>")
        all_enc.extend(enc)
        all_chain_ids.extend([idx] * len(enc))

    tokens = torch.tensor(all_enc, dtype=torch.long)
    chain_ids = torch.tensor(all_chain_ids, dtype=torch.long)
    return tokens, chain_ids


class FusionDataset(Dataset):
    """
    Full dataset that loads raw grid/energies/PDB data and prepares
    all inputs for the FusionModel on the fly.

    Supports N-chain complexes via PIsToN's PPI format (PID_side1_side2).
    """

    def __init__(self, ppi_list, labels, config, pdb_dir=None):
        """
        Args:
            ppi_list: list of PPI identifiers (PID_side1_side2, e.g. '1AHW_HL_C')
            labels: dict mapping ppi -> label (1 or 0)
            config: unified config dict
            pdb_dir: directory with PDB files (defaults to config['dirs']['pdb_dir'])
        """
        self.ppi_list = ppi_list
        self.labels = labels
        self.config = config
        self.grid_dir = config["dirs"]["grid"]
        self.pdb_dir = pdb_dir or config["dirs"]["pdb_dir"]

        # Set up MINT tokenizer
        mint_root = config["mint"]["root"]
        if mint_root not in sys.path:
            sys.path.insert(0, mint_root)
        from mint.data import Alphabet

        self.alphabet = Alphabet.from_architecture("ESM-1b")

        # Filter to only PPIs with available grid files
        self.valid_ppis = [
            ppi for ppi in ppi_list
            if os.path.exists(os.path.join(self.grid_dir, f"{ppi}.npy"))
        ]
        if len(self.valid_ppis) < len(ppi_list):
            print(
                f"FusionDataset: {len(ppi_list) - len(self.valid_ppis)} PPIs "
                f"skipped (no grid file). Using {len(self.valid_ppis)} PPIs."
            )

    def __len__(self):
        return len(self.valid_ppis)

    def __getitem__(self, idx):
        from data_prepare.sequence_extractor import extract_sequences_from_pdb
        from models.patch_residue_mapper import (
            parse_resnames,
            get_unique_patch_residues,
            map_patch_residues_to_mint_tokens,
        )

        ppi = self.valid_ppis[idx]
        pid, side1, side2, all_chains = parse_ppi_identifier(ppi)
        label = float(self.labels.get(ppi, 0))

        # 1. Load PIsToN grid + energies
        grid, energies = load_and_scale_grid(self.grid_dir, ppi)

        # 2. Load resnames and get patch residues (per individual PDB chain)
        resnames_path = os.path.join(self.grid_dir, f"{ppi}_resnames.npy")
        parsed = parse_resnames(resnames_path)
        patch_residues = get_unique_patch_residues(parsed)

        # 3. Extract sequences from PDB (per individual chain)
        pdb_path = os.path.join(self.pdb_dir, f"{pid}.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(self.pdb_dir, f"pdb{pid.lower()}.ent")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(self.pdb_dir, f"{pid}.ent")

        seq_info = extract_sequences_from_pdb(pdb_path, all_chains)

        # 4. Tokenize sequences for MINT (each chain gets its own chain_id)
        sequences = {ch: seq_info[ch]["sequence"] for ch in all_chains}
        tokens, chain_ids = tokenize_chains(self.alphabet, sequences, all_chains)

        # 5. Map patch residues to MINT token indices
        patch_token_indices, _ = map_patch_residues_to_mint_tokens(
            patch_residues, seq_info, all_chains
        )

        return {
            "grid": torch.from_numpy(grid),
            "energies": torch.from_numpy(energies),
            "tokens": tokens,
            "chain_ids": chain_ids,
            "patch_token_indices": patch_token_indices,
            "label": torch.tensor(label, dtype=torch.float32),
            "ppi": ppi,
        }


def fusion_collate_fn(batch):
    """
    Custom collate function to handle variable-length token sequences.
    Pads tokens and chain_ids to the max length in the batch.
    """
    grids = torch.stack([b["grid"] for b in batch])
    energies = torch.stack([b["energies"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    ppis = [b["ppi"] for b in batch]

    # Pad tokens and chain_ids
    max_len = max(b["tokens"].shape[0] for b in batch)
    # Use padding_idx = 1 (from ESM-1b alphabet)
    padded_tokens = torch.ones(len(batch), max_len, dtype=torch.long)
    padded_chain_ids = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["tokens"].shape[0]
        padded_tokens[i, :L] = b["tokens"]
        padded_chain_ids[i, :L] = b["chain_ids"]

    # Patch token indices: keep as list of dicts (not easily tensorized)
    patch_token_indices = [b["patch_token_indices"] for b in batch]

    return {
        "grid": grids,
        "energies": energies,
        "tokens": padded_tokens,
        "chain_ids": padded_chain_ids,
        "patch_token_indices": patch_token_indices,
        "label": labels,
        "ppi": ppis,
    }


class CachedEmbeddingDataset(Dataset):
    """
    Dataset for efficient MLP training using pre-extracted embeddings.

    Expects a directory with:
        {ppi}_piston.npy  -- (16,) float32
        {ppi}_mint.npy    -- (n_chains * 1280,) float32
    """

    def __init__(self, ppi_list, labels, embeddings_dir):
        self.labels = labels
        self.embeddings_dir = embeddings_dir

        self.valid_ppis = [
            ppi for ppi in ppi_list
            if (
                os.path.exists(os.path.join(embeddings_dir, f"{ppi}_piston.npy"))
                and os.path.exists(os.path.join(embeddings_dir, f"{ppi}_mint.npy"))
            )
        ]
        print(f"CachedEmbeddingDataset: {len(self.valid_ppis)} PPIs available.")

    def __len__(self):
        return len(self.valid_ppis)

    def __getitem__(self, idx):
        ppi = self.valid_ppis[idx]
        piston_emb = np.load(
            os.path.join(self.embeddings_dir, f"{ppi}_piston.npy")
        )
        mint_emb = np.load(
            os.path.join(self.embeddings_dir, f"{ppi}_mint.npy")
        )
        label = float(self.labels.get(ppi, 0))
        return {
            "piston_emb": torch.from_numpy(piston_emb).float(),
            "mint_emb": torch.from_numpy(mint_emb).float(),
            "label": torch.tensor(label, dtype=torch.float32),
            "ppi": ppi,
        }

"""
End-to-end inference pipeline for the PIsToN + MINT fusion model.

Given a PDB file and chain identifiers, runs:
1. PIsToN preprocessing (surface triangulation, patch extraction, grid conversion)
2. Sequence extraction and patch-residue mapping
3. Both backbone models (frozen) to extract embeddings
4. MLP classifier to produce a native/decoy probability

Supports N-chain complexes (e.g. antibody H+L + antigen = 3 chains).
PIsToN notation: side1 and side2 are multi-letter strings where each
character is a PDB chain letter (e.g. side1='HL', side2='C').
"""

import os
import sys

import numpy as np
import torch

from config.default_config import get_default_config, load_mint_cfg, ensure_dirs
from data_prepare.sequence_extractor import (
    extract_sequences_from_pdb,
    parse_ppi_identifier,
)
from models.patch_residue_mapper import (
    parse_resnames,
    get_unique_patch_residues,
    map_patch_residues_to_mint_tokens,
)
from models.piston_embedder import PIsToNEmbedder
from models.mint_embedder import MINTPatchEmbedder
from models.fusion_model import FusionClassifier
from training.dataset import load_and_scale_grid, tokenize_chains


class FusionPipeline:
    """
    End-to-end inference pipeline for protein complex interface quality prediction.

    Supports 2-chain dimers and N-chain complexes (e.g. antibody-antigen).

    Usage:
        pipeline = FusionPipeline(config, fusion_ckpt_path, device='cuda')
        # Antibody-antigen: side1='HL' (heavy+light), side2='A' (antigen)
        result = pipeline.predict("path/to/complex.pdb", side1="HL", side2="A")
        print(f"Native probability: {result['probability']:.3f}")
    """

    def __init__(self, config, fusion_ckpt_path=None, device="cuda"):
        """
        Args:
            config: unified config dict from get_default_config()
            fusion_ckpt_path: path to trained fusion classifier .pth file.
                If None, uses default location.
            device: torch device string
        """
        self.config = config
        self.device = torch.device(device)

        # Load PIsToN backbone
        print("Loading PIsToN backbone...")
        self.piston = PIsToNEmbedder.from_config(config, device=device)

        # Load MINT backbone
        print("Loading MINT backbone...")
        self.mint = MINTPatchEmbedder.from_config(config, device=device)

        # Load fusion classifier
        # Note: input_dim depends on n_chains.  We load with a default and
        # potentially resize at predict time if the checkpoint dictates it.
        fusion_cfg = config["fusion"]
        input_dim = fusion_cfg["piston_dim"] + fusion_cfg["mint_dim"]
        self.classifier = FusionClassifier(
            input_dim=input_dim,
            hidden_dim=fusion_cfg["hidden_dim"],
            dropout=fusion_cfg["dropout"],
        ).to(self.device)

        if fusion_ckpt_path is None:
            fusion_ckpt_path = os.path.join(
                config["dirs"]["saved_models"], "fusion_classifier_best.pth"
            )
        if os.path.exists(fusion_ckpt_path):
            state_dict = torch.load(fusion_ckpt_path, map_location=self.device)
            # Auto-detect input dim from checkpoint
            ckpt_input_dim = state_dict["net.0.weight"].shape[1]
            if ckpt_input_dim != input_dim:
                print(f"Adjusting classifier input_dim from {input_dim} to {ckpt_input_dim}")
                self.classifier = FusionClassifier(
                    input_dim=ckpt_input_dim,
                    hidden_dim=fusion_cfg["hidden_dim"],
                    dropout=fusion_cfg["dropout"],
                ).to(self.device)
            self.classifier.load_state_dict(state_dict)
            print(f"Fusion classifier loaded from {fusion_ckpt_path}")
        else:
            print(
                f"WARNING: No fusion classifier found at {fusion_ckpt_path}. "
                "Using random weights."
            )
        self.classifier.eval()

        # Set up MINT tokenizer
        mint_root = config["mint"]["root"]
        if mint_root not in sys.path:
            sys.path.insert(0, mint_root)
        from mint.data import Alphabet

        self.alphabet = Alphabet.from_architecture("ESM-1b")

    def preprocess(self, pdb_path, side1, side2):
        """
        Run PIsToN preprocessing on a PDB file.

        Args:
            pdb_path: path to the PDB file
            side1: chain letters for side 1, e.g. 'HL' or 'A'
            side2: chain letters for side 2, e.g. 'C' or 'B'

        Returns:
            ppi: PPI identifier string (PID_side1_side2)
        """
        from data_prepare.prepare import run_piston_preprocessing

        pid = os.path.splitext(os.path.basename(pdb_path))[0]
        ppi = f"{pid}_{side1}_{side2}"

        # Ensure PDB is in the raw_pdb directory
        raw_pdb_dir = self.config["dirs"]["raw_pdb"]
        os.makedirs(raw_pdb_dir, exist_ok=True)
        target_pdb = os.path.join(raw_pdb_dir, os.path.basename(pdb_path))
        if not os.path.exists(target_pdb):
            import shutil
            shutil.copy2(pdb_path, target_pdb)

        ensure_dirs(self.config)
        run_piston_preprocessing([ppi], self.config)
        return ppi

    def predict(self, pdb_path, side1, side2, skip_preprocessing=False):
        """
        Run the full prediction pipeline on a protein complex.

        Args:
            pdb_path: path to the PDB file
            side1: chain letters for side 1, e.g. 'HL' for antibody H+L chains
            side2: chain letters for side 2, e.g. 'C' for antigen
            skip_preprocessing: if True, assume PIsToN grid files already exist

        Returns:
            dict with keys:
                ppi: str
                probability: float (0-1)
                logit: float
                piston_embedding: numpy array (16,)
                mint_embedding: numpy array (n_chains * 1280,)
        """
        pid = os.path.splitext(os.path.basename(pdb_path))[0]
        ppi = f"{pid}_{side1}_{side2}"
        all_chains = list(side1) + list(side2)

        # Step 1: Preprocessing (if needed)
        if not skip_preprocessing:
            ppi = self.preprocess(pdb_path, side1, side2)

        grid_dir = self.config["dirs"]["grid"]

        # Step 2: Load PIsToN grid + energies
        grid, energies = load_and_scale_grid(grid_dir, ppi)
        grid_t = torch.from_numpy(grid).unsqueeze(0).to(self.device)
        energies_t = torch.from_numpy(energies).unsqueeze(0).float().to(self.device)

        # Step 3: Extract PIsToN embedding
        piston_emb = self.piston(grid_t, energies_t)  # (1, 16)

        # Step 4: Parse patch residues
        resnames_path = os.path.join(grid_dir, f"{ppi}_resnames.npy")
        parsed = parse_resnames(resnames_path)
        patch_residues = get_unique_patch_residues(parsed)

        # Step 5: Extract sequences from PDB (per individual chain)
        seq_info = extract_sequences_from_pdb(pdb_path, all_chains)

        # Step 6: Tokenize for MINT (each chain gets its own chain_id)
        sequences = {ch: seq_info[ch]["sequence"] for ch in all_chains}
        tokens, chain_ids_t = tokenize_chains(self.alphabet, sequences, all_chains)
        tokens = tokens.unsqueeze(0).to(self.device)
        chain_ids_t = chain_ids_t.unsqueeze(0).to(self.device)

        # Step 7: Map patch residues to MINT token indices
        patch_token_indices, _ = map_patch_residues_to_mint_tokens(
            patch_residues, seq_info, all_chains
        )

        # Step 8: Extract MINT embedding
        mint_emb = self.mint(tokens, chain_ids_t, patch_token_indices)

        # Step 9: Run fusion classifier
        combined = torch.cat([piston_emb, mint_emb], dim=1)
        with torch.no_grad():
            logit = self.classifier(combined).squeeze()
            probability = torch.sigmoid(logit).item()

        return {
            "ppi": ppi,
            "probability": probability,
            "logit": logit.item(),
            "piston_embedding": piston_emb.cpu().numpy()[0],
            "mint_embedding": mint_emb.cpu().numpy()[0],
        }

    def predict_batch(self, ppi_list, pdb_dir=None, skip_preprocessing=True):
        """
        Batch prediction on pre-processed PPIs.

        Args:
            ppi_list: list of PPI identifiers (PID_side1_side2)
            pdb_dir: directory containing PDB files
            skip_preprocessing: if True, assume grid files already exist

        Returns:
            list of result dicts
        """
        results = []
        grid_dir = self.config["dirs"]["grid"]
        if pdb_dir is None:
            pdb_dir = self.config["dirs"]["pdb_dir"]

        for ppi in ppi_list:
            pid, side1, side2, _ = parse_ppi_identifier(ppi)
            pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(pdb_dir, f"pdb{pid.lower()}.ent")

            try:
                result = self.predict(
                    pdb_path, side1, side2,
                    skip_preprocessing=skip_preprocessing,
                )
                results.append(result)
            except Exception as e:
                print(f"Error predicting {ppi}: {e}")
                results.append({"ppi": ppi, "probability": None, "error": str(e)})

        return results

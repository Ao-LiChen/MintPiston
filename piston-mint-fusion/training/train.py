"""
Training script for the PIsToN + MINT fusion MLP head.

Supports two modes:
1. extract_and_cache_embeddings(): Run both frozen backbones once and
   save embeddings to disk.
2. train_fusion(): Train the MLP head on cached embeddings (fast) or
   end-to-end through both backbones (slow but simpler).

Supports N-chain complexes (e.g. antibody H+L + antigen).
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_prepare.sequence_extractor import parse_ppi_identifier
from training.dataset import (
    FusionDataset,
    CachedEmbeddingDataset,
    fusion_collate_fn,
    load_and_scale_grid,
    tokenize_chains,
)
from training.evaluate import evaluate
from models.fusion_model import FusionModel, FusionClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_and_cache_embeddings(ppi_list, labels, config, device="cuda"):
    """
    Phase 1: Pre-extract PIsToN and MINT embeddings for all PPIs.

    Saves per-PPI embedding files:
        {embeddings_dir}/{ppi}_piston.npy  (16,)
        {embeddings_dir}/{ppi}_mint.npy    (n_chains * 1280,)

    This avoids running the large MINT model during every training epoch.
    """
    from models.piston_embedder import PIsToNEmbedder
    from models.mint_embedder import MINTPatchEmbedder
    from data_prepare.sequence_extractor import extract_sequences_from_pdb
    from models.patch_residue_mapper import (
        parse_resnames,
        get_unique_patch_residues,
        map_patch_residues_to_mint_tokens,
    )

    embeddings_dir = config["dirs"]["embeddings"]
    os.makedirs(embeddings_dir, exist_ok=True)
    grid_dir = config["dirs"]["grid"]
    pdb_dir = config["dirs"]["pdb_dir"]

    # Load PIsToN model
    print("Loading PIsToN model...")
    piston = PIsToNEmbedder.from_config(config, device=device)

    # Load MINT model
    print("Loading MINT model...")
    mint = MINTPatchEmbedder.from_config(config, device=device)

    # Set up MINT tokenizer
    mint_root = config["mint"]["root"]
    if mint_root not in sys.path:
        sys.path.insert(0, mint_root)
    from mint.data import Alphabet

    alphabet = Alphabet.from_architecture("ESM-1b")

    success_count = 0
    skip_count = 0

    for ppi in tqdm(ppi_list, desc="Extracting embeddings"):
        piston_out = os.path.join(embeddings_dir, f"{ppi}_piston.npy")
        mint_out = os.path.join(embeddings_dir, f"{ppi}_mint.npy")

        # Skip if already extracted
        if os.path.exists(piston_out) and os.path.exists(mint_out):
            success_count += 1
            continue

        grid_path = os.path.join(grid_dir, f"{ppi}.npy")
        if not os.path.exists(grid_path):
            skip_count += 1
            continue

        try:
            pid, side1, side2, all_chains = parse_ppi_identifier(ppi)

            # --- PIsToN embedding ---
            grid, energies = load_and_scale_grid(grid_dir, ppi)
            grid_t = torch.from_numpy(grid).unsqueeze(0).float().to(device)
            energies_t = torch.from_numpy(energies).unsqueeze(0).float().to(device)
            piston_emb = piston(grid_t, energies_t).cpu().numpy()[0]

            # --- MINT embedding ---
            # Parse patch residues
            resnames_path = os.path.join(grid_dir, f"{ppi}_resnames.npy")
            parsed = parse_resnames(resnames_path)
            patch_residues = get_unique_patch_residues(parsed)

            # Extract sequences (per individual PDB chain)
            pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(pdb_dir, f"pdb{pid.lower()}.ent")
            seq_info = extract_sequences_from_pdb(pdb_path, all_chains)

            # Tokenize (each chain gets its own chain_id)
            sequences = {ch: seq_info[ch]["sequence"] for ch in all_chains}
            tokens, chain_ids_t = tokenize_chains(alphabet, sequences, all_chains)
            tokens = tokens.unsqueeze(0).to(device)
            chain_ids_t = chain_ids_t.unsqueeze(0).to(device)

            # Map patch residues to token indices
            patch_token_indices, _ = map_patch_residues_to_mint_tokens(
                patch_residues, seq_info, all_chains
            )
            mint_emb = mint(tokens, chain_ids_t, patch_token_indices).cpu().numpy()[0]

            # Save
            np.save(piston_out, piston_emb)
            np.save(mint_out, mint_emb)
            success_count += 1

        except Exception as e:
            print(f"  Error processing {ppi}: {e}")
            skip_count += 1
            continue

    print(
        f"Embedding extraction complete. "
        f"Success: {success_count}, Skipped: {skip_count}"
    )


def train_fusion(config, train_ppis, val_ppis, train_labels, val_labels, device="cuda"):
    """
    Phase 2: Train the MLP classifier head on cached embeddings.

    Args:
        config: unified config dict
        train_ppis: list of training PPI identifiers
        val_ppis: list of validation PPI identifiers
        train_labels: dict mapping ppi -> label (0 or 1)
        val_labels: dict mapping ppi -> label (0 or 1)
        device: torch device
    """
    fusion_cfg = config["fusion"]
    set_seed(fusion_cfg.get("seed", 42))

    embeddings_dir = config["dirs"]["embeddings"]
    save_dir = config["dirs"]["saved_models"]
    os.makedirs(save_dir, exist_ok=True)

    # Build datasets
    train_dataset = CachedEmbeddingDataset(train_ppis, train_labels, embeddings_dir)
    val_dataset = CachedEmbeddingDataset(val_ppis, val_labels, embeddings_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=fusion_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=fusion_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Auto-detect input dim from first sample
    sample = train_dataset[0]
    actual_mint_dim = sample["mint_emb"].shape[0]
    actual_piston_dim = sample["piston_emb"].shape[0]
    input_dim = actual_piston_dim + actual_mint_dim
    print(f"Auto-detected dims: PIsToN={actual_piston_dim}, MINT={actual_mint_dim}, total={input_dim}")

    # Build classifier
    classifier = FusionClassifier(
        input_dim=input_dim,
        hidden_dim=fusion_cfg["hidden_dim"],
        dropout=fusion_cfg["dropout"],
    ).to(device)

    optimizer = AdamW(
        classifier.parameters(),
        lr=fusion_cfg["lr"],
        weight_decay=fusion_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    patience = fusion_cfg["patience"]
    best_model_path = os.path.join(save_dir, "fusion_classifier_best.pth")

    for epoch in range(fusion_cfg["max_epochs"]):
        # --- Training ---
        classifier.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            piston_emb = batch["piston_emb"].to(device)
            mint_emb = batch["mint_emb"].to(device)
            labels = batch["label"].to(device)

            combined = torch.cat([piston_emb, mint_emb], dim=1)
            logits = classifier(combined).squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.shape[0]
            n_train += labels.shape[0]

        avg_train_loss = train_loss / max(n_train, 1)

        # --- Validation ---
        val_metrics = evaluate(classifier, val_loader, device, loss_fn)
        val_auc = val_metrics["auc"]
        val_loss = val_metrics["loss"]

        scheduler.step(val_auc)

        print(
            f"Epoch {epoch + 1}/{fusion_cfg['max_epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(classifier.state_dict(), best_model_path)
            print(f"  -> New best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training complete. Best validation AUC: {best_auc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    return best_model_path

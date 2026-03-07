"""
FusionModel: combines PIsToN structural embeddings with MINT sequence
embeddings through a trainable MLP classifier head for interface quality
prediction (native vs decoy).
"""

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """
    MLP classifier head operating on concatenated PIsToN + MINT embeddings.
    """

    def __init__(self, input_dim=2576, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    """
    Full fusion model that combines:
    - PIsToN structural embedding (16-dim, frozen)
    - MINT sequence embedding (2560-dim, frozen)
    - Trainable MLP classifier head (2576 -> 1)

    For training efficiency, this also supports a "cached" mode where
    pre-extracted embeddings are concatenated directly (bypassing the
    backbone forward passes).
    """

    def __init__(self, piston_embedder, mint_embedder, fusion_config):
        """
        Args:
            piston_embedder: PIsToNEmbedder instance (frozen)
            mint_embedder: MINTPatchEmbedder instance (frozen)
            fusion_config: dict with keys hidden_dim, dropout, piston_dim, mint_dim
        """
        super().__init__()
        self.piston_embedder = piston_embedder
        self.mint_embedder = mint_embedder

        piston_dim = fusion_config.get("piston_dim", 16)
        mint_dim = fusion_config.get("mint_dim", 2560)
        hidden_dim = fusion_config.get("hidden_dim", 512)
        dropout = fusion_config.get("dropout", 0.2)

        input_dim = piston_dim + mint_dim
        self.classifier = FusionClassifier(input_dim, hidden_dim, dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        grid,
        energies,
        tokens,
        chain_ids,
        patch_token_indices,
        labels=None,
    ):
        """
        Full forward pass through both backbones and the MLP head.

        Args:
            grid: (B, 13, H, W) PIsToN interface map
            energies: (B, 13) FireDock energies
            tokens: (B, T) MINT token tensor
            chain_ids: (B, T) chain ID tensor
            patch_token_indices: dict[chain_id -> list[token_idx]]
            labels: (B,) float tensor, 1=native, 0=decoy (optional)

        Returns:
            logits: (B, 1)
            loss: scalar (only if labels provided)
        """
        piston_emb = self.piston_embedder(grid, energies)  # (B, 16)
        mint_emb = self.mint_embedder(
            tokens, chain_ids, patch_token_indices
        )  # (B, 2560)

        combined = torch.cat([piston_emb, mint_emb], dim=1)  # (B, 2576)
        logits = self.classifier(combined)  # (B, 1)

        if labels is not None:
            loss = self.loss_fn(logits.squeeze(-1), labels.float())
            return logits, loss
        return logits

    def forward_from_embeddings(self, piston_emb, mint_emb, labels=None):
        """
        Forward pass using pre-extracted (cached) embeddings.
        This is much faster for training since it avoids running the
        large backbone models.

        Args:
            piston_emb: (B, 16) pre-extracted PIsToN embeddings
            mint_emb: (B, 2560) pre-extracted MINT embeddings
            labels: (B,) float tensor (optional)

        Returns:
            logits: (B, 1)
            loss: scalar (only if labels provided)
        """
        combined = torch.cat([piston_emb, mint_emb], dim=1)
        logits = self.classifier(combined)

        if labels is not None:
            loss = self.loss_fn(logits.squeeze(-1), labels.float())
            return logits, loss
        return logits

"""
Wrapper around MINT's ESM2 model that extracts per-residue embeddings
for the subset of residues covered by PIsToN's interface patch.

Supports N-chain complexes (e.g. antibody H + L + antigen = 3 chains).
Each individual PDB chain gets its own chain_id, enabling proper
inter-chain attention in MINT's multimer attention mechanism.

Token layout for a 3-chain complex (e.g. H, L, C):
    [<cls> H1..Hn <eos> <cls> L1..Lm <eos> <cls> C1..Ck <eos> <pad>...]
    chain_ids:
    [  0    0...0    0     1    1...1    1     2    2...2    2    ...]

Output: mean-pooled patch-residue embeddings per chain, concatenated.
For 3 chains: (B, 3*1280) = (B, 3840).
"""

import sys
from collections import OrderedDict

import torch
import torch.nn as nn


class MINTPatchEmbedder(nn.Module):
    """
    Loads a pre-trained MINT ESM2 multimer model and extracts mean-pooled
    embeddings from patch-relevant residue positions only.

    Handles any number of chains (2, 3, or more).
    """

    def __init__(self, mint_root, cfg, checkpoint_path, device="cuda"):
        """
        Args:
            mint_root: path to mint-main directory (for imports)
            cfg: argparse.Namespace with ESM2 config fields
                 (encoder_layers, encoder_embed_dim, encoder_attention_heads, token_dropout)
            checkpoint_path: path to MINT .ckpt file
            device: torch device
        """
        super().__init__()

        # Make MINT importable
        if mint_root not in sys.path:
            sys.path.insert(0, mint_root)

        from mint.model.esm2 import ESM2
        from mint.data import Alphabet

        self.alphabet = Alphabet.from_architecture("ESM-1b")

        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=True,
        )

        # Load checkpoint -- MINT saves with 'model.' prefix in state_dict keys
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        new_checkpoint = OrderedDict(
            (key.replace("model.", ""), value)
            for key, value in checkpoint["state_dict"].items()
        )
        self.model.load_state_dict(new_checkpoint)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(device)

        self.embed_dim = cfg.encoder_embed_dim
        self.repr_layer = 33

    def forward(self, tokens, chain_ids, patch_token_indices):
        """
        Extract mean-pooled per-chain embeddings for patch residues only.

        Args:
            tokens: (B, T) tokenized input for all chains concatenated
            chain_ids: (B, T) integer chain ID per token (0, 1, 2, ...)
            patch_token_indices: dict[chain_letter -> list[int]] mapping
                individual PDB chain letters (e.g. 'H', 'L', 'C') to token
                indices into the T dimension.
                The order of keys determines the output concatenation order.

        Returns:
            embedding: (B, n_chains * embed_dim)
                For 2 chains: (B, 2560)
                For 3 chains: (B, 3840)
        """
        with torch.no_grad():
            result = self.model(tokens, chain_ids, repr_layers=[self.repr_layer])
            chain_out = result["representations"][self.repr_layer]  # (B, T, 1280)

        B = chain_out.shape[0]
        device = chain_out.device
        chain_keys = list(patch_token_indices.keys())
        chain_embeddings = []

        for chain_idx, chain_key in enumerate(chain_keys):
            indices = patch_token_indices[chain_key]
            if len(indices) > 0:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                # Select patch residue tokens: (B, n_patch_residues, 1280)
                selected = chain_out[:, idx_tensor, :]
                # Mean pool over the selected residues
                chain_emb = selected.mean(dim=1)  # (B, 1280)
            else:
                # Fallback: if no patch residues found for this chain, average
                # all non-special tokens for that chain
                chain_mask = (chain_ids == chain_idx)
                # Exclude special tokens
                special_mask = (
                    tokens.eq(self.model.cls_idx)
                    | tokens.eq(self.model.eos_idx)
                    | tokens.eq(self.model.padding_idx)
                )
                valid_mask = chain_mask & ~special_mask  # (B, T)
                valid_mask_exp = valid_mask.unsqueeze(-1).expand_as(chain_out)
                masked_out = chain_out * valid_mask_exp.float()
                counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
                chain_emb = masked_out.sum(dim=1) / counts  # (B, 1280)

            chain_embeddings.append(chain_emb)

        # Concatenate all chains: (B, n_chains * 1280)
        return torch.cat(chain_embeddings, dim=-1)

    @staticmethod
    def from_config(config, device="cuda"):
        """Convenience constructor from the unified config dict."""
        from config.default_config import load_mint_cfg

        cfg = load_mint_cfg(config["mint"]["config_json"])
        return MINTPatchEmbedder(
            mint_root=config["mint"]["root"],
            cfg=cfg,
            checkpoint_path=config["mint"]["checkpoint_path"],
            device=device,
        )

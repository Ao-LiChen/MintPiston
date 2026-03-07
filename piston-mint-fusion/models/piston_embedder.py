"""
Wrapper around PIsToN_multiAttn that extracts the 16-dim L2-normalized
CLS embedding from the feature transformer -- the representation used
before prototypical classification.
"""

import sys
import torch
import torch.nn as nn
import numpy as np


class PIsToNEmbedder(nn.Module):
    """
    Loads a pre-trained PIsToN_multiAttn model and exposes a forward()
    method that returns the intermediate 16-dim CLS embedding instead of
    the prototype-distance scores.
    """

    def __init__(self, piston_root, model_path, params, img_size=32, device="cuda"):
        """
        Args:
            piston_root: path to piston-main directory (for imports)
            model_path: path to PIsToN_multiAttn_contrast.pth
            params: dict with keys dim_head, hidden_size, dropout, attn_dropout,
                    n_heads, patch_size, transformer_depth
            img_size: spatial dimension of the interface map (default 32)
            device: torch device
        """
        super().__init__()

        # Make PIsToN importable
        if piston_root not in sys.path:
            sys.path.insert(0, piston_root)

        from networks.PIsToN_multiAttn import PIsToN_multiAttn
        from networks.ViT_pytorch import get_ml_config

        model_config = get_ml_config(params)
        self.model = PIsToN_multiAttn(
            model_config, img_size=img_size, num_classes=2
        ).float()

        # Load pre-trained weights
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(device)

    def forward(self, grid, energies):
        """
        Extract the 16-dim L2-normalized CLS embedding.

        This replicates PIsToN_multiAttn.forward() lines 117-137 but stops
        before the prototype distance computation.

        Args:
            grid: (B, 13, H, W) standardized interface map tensor
            energies: (B, 13) standardized FireDock energy tensor

        Returns:
            embedding: (B, 16) L2-normalized embedding
        """
        with torch.no_grad():
            all_x = []
            for i, feature in enumerate(self.model.index_dict.keys()):
                img_tmp = grid[:, self.model.index_dict[feature][0], :, :]
                energy_tmp = energies[:, self.model.index_dict[feature][1]]
                x, _attn = self.model.spatial_transformers_list[i](
                    img_tmp, energy_tmp
                )
                all_x.append(x)

            x = torch.stack(all_x, dim=1)  # (B, 5, hidden_size)

            B = x.shape[0]
            cls_tokens = self.model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 6, hidden_size)

            x, _feature_attn = self.model.feature_transformer(x)
            x = x[:, 0]  # CLS token output
            x = nn.functional.normalize(x)  # L2 normalization

        return x  # (B, 16)

    @staticmethod
    def from_config(config, device="cuda"):
        """Convenience constructor from the unified config dict."""
        return PIsToNEmbedder(
            piston_root=config["piston"]["root"],
            model_path=config["piston"]["model_path"],
            params=config["piston"]["params"],
            img_size=config["piston"]["img_size"],
            device=device,
        )

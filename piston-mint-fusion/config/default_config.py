"""
Unified configuration for the PIsToN + MINT fusion model.

Consolidates paths, model hyperparameters, and training settings
for both backbone models and the fusion MLP head.
"""

import os
import json
import argparse


def get_default_config(
    pdb_dir,
    out_dir,
    piston_ckpt=None,
    mint_ckpt=None,
    mint_json_cfg=None,
):
    """
    Build the unified configuration dictionary.

    Args:
        pdb_dir: directory containing input PDB files
        out_dir: root output directory for intermediate and final results
        piston_ckpt: path to PIsToN pre-trained model (.pth)
        mint_ckpt: path to MINT pre-trained checkpoint (.ckpt)
        mint_json_cfg: path to MINT ESM2 config JSON
    Returns:
        config dict
    """

    # Resolve default paths relative to this file
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _piston_root = os.path.join(os.path.dirname(_project_root), "piston-main")
    _mint_root = os.path.join(os.path.dirname(_project_root), "mint-main")

    if piston_ckpt is None:
        piston_ckpt = os.path.join(
            _piston_root, "saved_models", "PIsToN_multiAttn_contrast.pth"
        )
    if mint_json_cfg is None:
        mint_json_cfg = os.path.join(_mint_root, "data", "esm2_t33_650M_UR50D.json")

    config = {}

    # ---- PIsToN backbone ----
    config["piston"] = {
        "root": _piston_root,
        "model_path": piston_ckpt,
        "params": {
            "dim_head": 16,
            "hidden_size": 16,
            "dropout": 0,
            "attn_dropout": 0,
            "n_heads": 8,
            "patch_size": 4,
            "transformer_depth": 8,
        },
        "patch_r": 16,
        "img_size": 32,  # 2 * patch_r
        "embedding_dim": 16,
    }

    # ---- MINT backbone ----
    config["mint"] = {
        "root": _mint_root,
        "checkpoint_path": mint_ckpt,
        "config_json": mint_json_cfg,
        "repr_layer": 33,
        "embed_dim": 1280,
        "use_multimer": True,
        "max_seq_len": 1024,
    }

    # ---- Fusion MLP head ----
    # mint_dim = n_chains * 1280.  For antibody-antigen (H + L + Ag) = 3 chains:
    #   3 * 1280 = 3840.  For simple dimer (A + B) = 2 chains: 2 * 1280 = 2560.
    # Set this to match your data; it is auto-computed at runtime if needed.
    config["fusion"] = {
        "piston_dim": 16,
        "mint_dim": 3840,  # 1280 * 3 chains (default for antibody-antigen)
        "n_chains": 3,     # number of individual PDB chains
        "hidden_dim": 512,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "max_epochs": 100,
        "patience": 10,
        "seed": 42,
    }

    # ---- Directory layout (mirrors PIsToN) ----
    data_prepare_dir = os.path.join(out_dir, "intermediate_files")
    config["dirs"] = {
        "pdb_dir": pdb_dir,
        "out_dir": out_dir,
        "data_prepare": data_prepare_dir,
        "raw_pdb": os.path.join(data_prepare_dir, "00-raw_pdbs"),
        "protonated_pdb": os.path.join(data_prepare_dir, "01-protonated_pdb"),
        "refined": os.path.join(data_prepare_dir, "02-refined_pdb"),
        "cropped_pdb": os.path.join(data_prepare_dir, "03-cropped_pdbs"),
        "chains_pdb": os.path.join(data_prepare_dir, "04-chains_pdbs"),
        "surface_ply": os.path.join(data_prepare_dir, "05-surface_ply"),
        "patches": os.path.join(data_prepare_dir, "06-patches"),
        "grid": os.path.join(out_dir, "grid"),
        "embeddings": os.path.join(out_dir, "embeddings"),
        "vis": os.path.join(out_dir, "patch_vis"),
        "tmp": os.path.join(out_dir, "tmp"),
        "saved_models": os.path.join(
            _project_root, "saved_models"
        ),
    }

    # ---- PIsToN preprocessing constants ----
    config["ppi_const"] = {
        "contact_d": 5,
        "surf_contact_r": 1,
        "patch_r": 16,
        "crop_r": 17,  # patch_r + 1
        "points_in_patch": 400,
    }

    config["interact_feat"] = {"atom_dist": True, "dssp": True}

    config["mesh"] = {"mesh_res": 1.0}

    # ---- Standardization constants (from PIsToN training set) ----
    config["scaling"] = {
        "feature_mean": [
            0.06383528408485302, 0.043833505848899605, -0.08456032982438057,
            0.007828608135306595, -0.06060602411612203, 0.06383528408485302,
            0.043833505848899605, -0.08456032982438057, 0.007828608135306595,
            -0.06060602411612203, 11.390402735801011, 0.1496338245579665,
            0.1496338245579665,
        ],
        "feature_std": [
            0.4507792893174703, 0.14148081793902434, 0.16581325050002976,
            0.28599861830017204, 0.6102229371168204, 0.4507792893174703,
            0.14148081793902434, 0.16581325050002976, 0.28599861830017204,
            0.6102229371168204, 7.265311558033949, 0.18003612950610695,
            0.18003612950610695,
        ],
        "energy_mean": [
            -193.1392953586498, -101.97838818565408, 264.2099535864983,
            -17.27086075949363, 16.329959915611877, -102.78101054852341,
            36.531006329113836, -27.1124789029536, 16.632626582278455,
            -8.784924050632918, -6.206329113924051, -1.8290084388185655,
            -11.827215189873417,
        ],
        "energy_std": [
            309.23521244706757, 66.75799437657368, 9792.783784373369,
            25.384427268309658, 7.929941961525389, 94.05055841984323,
            47.22518557457095, 24.392679889433445, 17.57399925906454,
            7.041949880295568, 6.99554122803362, 2.557571754303165,
            13.666329541281653,
        ],
    }

    return config


def load_mint_cfg(json_path):
    """Load the MINT ESM2 model configuration from JSON."""
    cfg = argparse.Namespace()
    with open(json_path) as f:
        cfg.__dict__.update(json.load(f))
    return cfg


def ensure_dirs(config):
    """Create all directories in config['dirs'] if they don't exist."""
    for d in config["dirs"].values():
        os.makedirs(d, exist_ok=True)

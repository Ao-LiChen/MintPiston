"""
Thin wrapper around PIsToN's preprocessing pipeline.

Calls the original PIsToN data_prepare.preprocess() function, adding
the fusion project's directory layout on top.
"""

import sys
import os


def run_piston_preprocessing(ppi_list, config):
    """
    Run PIsToN's full preprocessing pipeline for a list of PPIs.

    This function sets up the Python path so PIsToN modules can be imported,
    then delegates to data_prepare.preprocess().

    Args:
        ppi_list: list of PPI identifiers (format: PID_chain1_chain2)
        config: unified configuration dict (from config.default_config)
    """
    piston_root = config["piston"]["root"]

    # Make PIsToN importable
    if piston_root not in sys.path:
        sys.path.insert(0, piston_root)

    from data_prepare.data_prepare import preprocess
    from data_prepare.get_structure import download

    # Build the PIsToN-compatible config dict
    piston_config = {
        "dirs": config["dirs"],
        "ppi_const": config["ppi_const"],
        "interact_feat": config["interact_feat"],
        "mesh": config["mesh"],
    }

    # Download PDB files if not already present
    download(ppi_list, piston_config)

    # Run the full pipeline: protonate -> refine -> crop -> triangulate
    #                         -> compute_patches -> map_patch_atom -> convert_to_images
    preprocess(ppi_list, piston_config)

    print(f"PIsToN preprocessing complete for {len(ppi_list)} complexes.")

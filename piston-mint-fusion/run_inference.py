#!/usr/bin/env python
"""
CLI entry point for inference with the PIsToN + MINT fusion model.

Supports N-chain complexes (e.g. antibody H+L + antigen).

Usage:
    # Single antibody-antigen complex (3 chains: H, L, C)
    python run_inference.py --pdb path/to/complex.pdb --side1 HL --side2 C \
        --mint_ckpt path/to/mint.ckpt \
        --fusion_ckpt saved_models/fusion_classifier_best.pth \
        --out_dir ./output

    # Simple dimer (2 chains: A, B)
    python run_inference.py --pdb path/to/complex.pdb --side1 A --side2 B \
        --mint_ckpt path/to/mint.ckpt

    # Batch mode with pre-processed data
    python run_inference.py --list ppi_list.txt --pdb_dir ./pdbs/ \
        --skip_preprocessing --out_dir ./output --mint_ckpt mint.ckpt
"""

import argparse
import os
import sys

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch


def main():
    parser = argparse.ArgumentParser(
        description="PIsToN + MINT Fusion Model Inference"
    )

    # Input specification (mutually exclusive: single PDB or list)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb", type=str, help="Path to a single PDB file")
    group.add_argument("--list", type=str,
                       help="Text file with PPI IDs (one per line, format: PID_side1_side2)")

    parser.add_argument("--side1", type=str,
                        help="Side 1 chain letters (e.g. 'HL' for antibody, 'A' for monomer). "
                             "Required with --pdb.")
    parser.add_argument("--side2", type=str,
                        help="Side 2 chain letters (e.g. 'C' for antigen). Required with --pdb.")
    parser.add_argument("--pdb_dir", type=str, default="./pdbs",
                        help="Directory with PDB files (for --list mode)")

    # Model checkpoints
    parser.add_argument("--piston_ckpt", type=str, default=None,
                        help="Path to PIsToN model .pth")
    parser.add_argument("--mint_ckpt", type=str, required=True,
                        help="Path to MINT model .ckpt")
    parser.add_argument("--mint_config", type=str, default=None,
                        help="Path to MINT ESM2 config .json")
    parser.add_argument("--fusion_ckpt", type=str, default=None,
                        help="Path to trained fusion classifier .pth")

    # Output
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory")

    # Options
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip PIsToN preprocessing (use existing grid files)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available, else cpu)")

    args = parser.parse_args()

    if args.pdb and (not args.side1 or not args.side2):
        parser.error("--side1 and --side2 are required with --pdb")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build config
    from config.default_config import get_default_config, ensure_dirs

    config = get_default_config(
        pdb_dir=args.pdb_dir,
        out_dir=args.out_dir,
        piston_ckpt=args.piston_ckpt,
        mint_ckpt=args.mint_ckpt,
        mint_json_cfg=args.mint_config,
    )
    ensure_dirs(config)

    # Build pipeline
    from inference.pipeline import FusionPipeline

    pipeline = FusionPipeline(config, fusion_ckpt_path=args.fusion_ckpt, device=device)

    # Run inference
    if args.pdb:
        result = pipeline.predict(
            args.pdb, args.side1, args.side2,
            skip_preprocessing=args.skip_preprocessing,
        )
        n_chains = len(args.side1) + len(args.side2)
        print(f"\nResult for {result['ppi']} ({n_chains} chains):")
        print(f"  Native probability: {result['probability']:.4f}")
        print(f"  Raw logit:          {result['logit']:.4f}")

        # Write output CSV
        out_csv = os.path.join(args.out_dir, "fusion_scores.csv")
        with open(out_csv, "w") as f:
            f.write("PPI,probability,logit\n")
            f.write(f"{result['ppi']},{result['probability']:.6f},{result['logit']:.6f}\n")
        print(f"\nScores saved to {out_csv}")

    else:
        ppi_list = [line.strip() for line in open(args.list) if line.strip()]
        print(f"Running inference on {len(ppi_list)} complexes...")

        results = pipeline.predict_batch(
            ppi_list,
            pdb_dir=args.pdb_dir,
            skip_preprocessing=args.skip_preprocessing,
        )

        out_csv = os.path.join(args.out_dir, "fusion_scores.csv")
        with open(out_csv, "w") as f:
            f.write("PPI,probability,logit\n")
            for r in results:
                if r.get("probability") is not None:
                    f.write(f"{r['ppi']},{r['probability']:.6f},{r['logit']:.6f}\n")
                else:
                    f.write(f"{r['ppi']},NA,NA\n")

        n_success = sum(1 for r in results if r.get("probability") is not None)
        print(f"\nInference complete. {n_success}/{len(ppi_list)} succeeded.")
        print(f"Scores saved to {out_csv}")


if __name__ == "__main__":
    main()

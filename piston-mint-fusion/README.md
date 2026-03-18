# PIsToN-MINT Fusion

Core framework for the MintPiston project. See the [project README](../README.md) for full documentation, benchmark results, installation, data downloads, and reproduction instructions.

## Quick Reference

- **MaSIF-test benchmark**: `python run_benchmark.py --data_root /path/to/masif_test_copy --mint_ckpt saved_models/mint.ckpt --device cuda`
- **SAbDab benchmark**: `python benchmarks/sabdab/run_benchmark_v2.py --pipeline_out ../pipeline_out --device cpu`
- **Inference**: `python run_inference.py --pdb complex.pdb --side1 A --side2 B --mint_ckpt saved_models/mint.ckpt --fusion_ckpt saved_models/Fusion_best.pth`

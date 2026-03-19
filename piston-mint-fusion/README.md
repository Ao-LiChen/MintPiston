# PIsToN-MINT Fusion

Core framework for the MintPiston project. See the [project README](../README.md) for full documentation, installation, data downloads, and reproduction instructions.

## Benchmark Results

### MaSIF-test (262 test samples)

| Model       |   AUC  |   Acc  |    F1  |  Prec  |   Rec  |   MCC  |
|-------------|--------|--------|--------|--------|--------|--------|
| PIsToN-only | 0.9565 | 0.8817 | 0.8755 | 0.9237 | 0.8321 | 0.7671 |
| MINT-only   | 0.8209 | 0.7023 | 0.7383 | 0.6587 | 0.8397 | 0.4208 |
| **Fusion**  | **0.9573** | **0.8931** | **0.8939** | 0.8872 | **0.9008** | **0.7864** |

### SAbDab antibody-antigen (1,676 test samples)

| Model       |   AUC  |   Acc  |    F1  |  Prec  |   Rec  |   MCC  |   N  | Pos  | Neg |
|-------------|--------|--------|--------|--------|--------|--------|------|------|-----|
| PIsToN-only | 0.6672 | 0.6510 | 0.7570 | 0.8660 | 0.6723 | 0.1897 | 1676 | 1355 | 321 |
| MINT-only   | 0.7628 | 0.5811 | 0.6692 | **0.9257** | 0.5240 | 0.2736 | 1676 | 1355 | 321 |
| **Fusion**  | **0.8127** | **0.7440** | **0.8267** | 0.9134 | **0.7550** | **0.3784** | 1676 | 1355 | 321 |

## Quick Reference

- **MaSIF-test benchmark**: `python run_benchmark.py --data_root /path/to/masif_test_copy --mint_ckpt saved_models/mint.ckpt --device cuda`
- **SAbDab benchmark**: `cd .. && python run_benchmark_v2.py --device cuda --max_epochs 100 --patience 15`
- **Inference**: `python run_inference.py --pdb complex.pdb --side1 A --side2 B --mint_ckpt saved_models/mint.ckpt --fusion_ckpt saved_models/Fusion_best.pth`

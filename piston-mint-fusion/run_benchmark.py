#!/usr/bin/env python
"""
Benchmark script: train and compare PIsToN-only, MINT-only, and Fusion MLPs
on the masif_test dataset.

Runs:
  1. Phase 1: Extract PIsToN (16-dim) and MINT (2560-dim) embeddings for all PPIs
  2. Phase 2: Train three separate MLP classifiers:
     - PIsToN-only:  16-dim  -> MLP -> binary
     - MINT-only:    2560-dim -> MLP -> binary
     - Fusion:       2576-dim -> MLP -> binary
  3. Evaluate all three on the held-out test set and print comparison table

Usage:
    python run_benchmark.py \
        --data_root "C:/Users/liche/Downloads/masif_test.tar/masif_test/masif_test_copy" \
        --mint_ckpt saved_models/mint.ckpt \
        --out_dir ./output \
        --device cuda
"""

import argparse
import os
import sys
import csv
import json

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_labels(path):
    labels = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


def load_list(path):
    return [l.strip() for l in open(path) if l.strip()]


def train_classifier(name, train_dataset, val_dataset, input_dim, config, device):
    """Train one MLP classifier and return the best model path."""
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from models.fusion_model import FusionClassifier

    fusion_cfg = config["fusion"]
    seed = fusion_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=fusion_cfg["batch_size"],
        shuffle=True, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=fusion_cfg["batch_size"],
        shuffle=False, num_workers=0,
    )

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
    save_dir = config["dirs"]["saved_models"]
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"{name}_best.pth")

    for epoch in range(fusion_cfg["max_epochs"]):
        # Training
        classifier.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            emb = batch["emb"].to(device)
            labels = batch["label"].to(device)
            logits = classifier(emb).squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.shape[0]
            n_train += labels.shape[0]

        avg_train_loss = train_loss / max(n_train, 1)

        # Validation
        classifier.eval()
        val_loss = 0.0
        n_val = 0
        all_labels = []
        all_probs = []

        import torch
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device)
                labels_b = batch["label"].to(device)
                logits = classifier(emb).squeeze(-1)
                loss = loss_fn(logits, labels_b)
                val_loss += loss.item() * labels_b.shape[0]
                n_val += labels_b.shape[0]
                probs = torch.sigmoid(logits)
                all_labels.append(labels_b.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        import numpy as np
        from sklearn.metrics import roc_auc_score
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        if len(np.unique(all_labels)) > 1:
            val_auc = roc_auc_score(all_labels, all_probs)
        else:
            val_auc = 0.0

        scheduler.step(val_auc)

        print(
            f"  [{name}] Epoch {epoch + 1:3d}/{fusion_cfg['max_epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss / max(n_val, 1):.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(classifier.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{name}] Early stopping at epoch {epoch + 1}")
                break

    print(f"  [{name}] Best val AUC: {best_auc:.4f}")
    return best_path, input_dim


def evaluate_on_test(name, model_path, input_dim, test_dataset, config, device):
    """Load a trained MLP and evaluate on the test set."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        precision_score, recall_score, matthews_corrcoef,
    )

    from models.fusion_model import FusionClassifier

    classifier = FusionClassifier(
        input_dim=input_dim,
        hidden_dim=config["fusion"]["hidden_dim"],
        dropout=config["fusion"]["dropout"],
    ).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    test_loader = DataLoader(
        test_dataset, batch_size=config["fusion"]["batch_size"],
        shuffle=False, num_workers=0,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            emb = batch["emb"].to(device)
            labels = batch["label"].to(device)
            logits = classifier(emb).squeeze(-1)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            n_samples += labels.shape[0]
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "model": name,
        "input_dim": input_dim,
        "loss": total_loss / max(n_samples, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "mcc": matthews_corrcoef(all_labels, all_preds),
    }

    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = 0.0

    return metrics


class SingleModalityDataset:
    """Wraps cached embeddings for a single modality or the fusion of both."""

    def __init__(self, ppi_list, labels, embeddings_dir, mode="fusion"):
        """
        Args:
            mode: "piston" | "mint" | "fusion"
        """
        import numpy as np
        import torch

        self.mode = mode
        self.labels = labels
        self.embeddings_dir = embeddings_dir

        self.valid_ppis = [
            ppi for ppi in ppi_list
            if (
                os.path.exists(os.path.join(embeddings_dir, f"{ppi}_piston.npy"))
                and os.path.exists(os.path.join(embeddings_dir, f"{ppi}_mint.npy"))
            )
        ]

    def __len__(self):
        return len(self.valid_ppis)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        ppi = self.valid_ppis[idx]
        piston_emb = np.load(
            os.path.join(self.embeddings_dir, f"{ppi}_piston.npy")
        )
        mint_emb = np.load(
            os.path.join(self.embeddings_dir, f"{ppi}_mint.npy")
        )
        label = float(self.labels.get(ppi, 0))

        if self.mode == "piston":
            emb = torch.from_numpy(piston_emb).float()
        elif self.mode == "mint":
            emb = torch.from_numpy(mint_emb).float()
        else:  # fusion
            emb = torch.from_numpy(
                np.concatenate([piston_emb, mint_emb])
            ).float()

        return {
            "emb": emb,
            "label": torch.tensor(label, dtype=torch.float32),
            "ppi": ppi,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: PIsToN vs MINT vs Fusion on masif_test"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to masif_test_copy directory")
    parser.add_argument("--mint_ckpt", type=str, required=True)
    parser.add_argument("--piston_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--device", type=str, default=None)

    # Phase control
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip embedding extraction (use cached)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Paths ---
    data_root = args.data_root
    grid_dir = os.path.join(data_root, "prepare_energies_16R", "07-grid")
    pdb_dir = os.path.join(data_root, "PDB")
    data_dir = os.path.join(_project_root, "data")

    # Verify data files
    for f in ["all_ppis.txt", "train.txt", "val.txt", "test.txt", "labels.csv"]:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run prepare_masif_data.py first.")
            sys.exit(1)

    # --- Config ---
    from config.default_config import get_default_config, ensure_dirs

    config = get_default_config(
        pdb_dir=pdb_dir,
        out_dir=args.out_dir,
        piston_ckpt=args.piston_ckpt,
        mint_ckpt=args.mint_ckpt,
    )
    config["dirs"]["grid"] = grid_dir
    config["dirs"]["pdb_dir"] = pdb_dir
    config["fusion"]["n_chains"] = 2
    config["fusion"]["mint_dim"] = 2560
    config["fusion"]["lr"] = args.lr
    config["fusion"]["batch_size"] = args.batch_size
    config["fusion"]["max_epochs"] = args.max_epochs
    config["fusion"]["patience"] = args.patience
    config["fusion"]["hidden_dim"] = args.hidden_dim
    config["fusion"]["dropout"] = args.dropout
    config["fusion"]["seed"] = args.seed
    ensure_dirs(config)

    # --- Load data ---
    all_ppis = load_list(os.path.join(data_dir, "all_ppis.txt"))
    train_ppis = load_list(os.path.join(data_dir, "train.txt"))
    val_ppis = load_list(os.path.join(data_dir, "val.txt"))
    test_ppis = load_list(os.path.join(data_dir, "test.txt"))
    labels = load_labels(os.path.join(data_dir, "labels.csv"))

    print(f"Train: {len(train_ppis)} | Val: {len(val_ppis)} | Test: {len(test_ppis)}")

    # =============================================
    # Phase 1: Extract embeddings
    # =============================================
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("PHASE 1: Extracting PIsToN + MINT embeddings")
        print("=" * 60)

        from training.train import extract_and_cache_embeddings
        extract_and_cache_embeddings(all_ppis, labels, config, device=device)
    else:
        print("\nSkipping embedding extraction (--skip_extract)")

    embeddings_dir = config["dirs"]["embeddings"]

    # =============================================
    # Phase 2: Train 3 classifiers
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 2: Training classifiers")
    print("=" * 60)

    # Detect embedding dims from first sample
    import numpy as np
    sample_ppi = train_ppis[0]
    piston_dim = np.load(
        os.path.join(embeddings_dir, f"{sample_ppi}_piston.npy")
    ).shape[0]
    mint_dim = np.load(
        os.path.join(embeddings_dir, f"{sample_ppi}_mint.npy")
    ).shape[0]
    fusion_dim = piston_dim + mint_dim
    print(f"Embedding dims: PIsToN={piston_dim}, MINT={mint_dim}, Fusion={fusion_dim}")

    models_info = [
        ("PIsToN-only", "piston", piston_dim),
        ("MINT-only", "mint", mint_dim),
        ("Fusion", "fusion", fusion_dim),
    ]

    trained_models = {}

    for name, mode, dim in models_info:
        print(f"\n--- Training {name} (input_dim={dim}) ---")
        train_ds = SingleModalityDataset(train_ppis, labels, embeddings_dir, mode=mode)
        val_ds = SingleModalityDataset(val_ppis, labels, embeddings_dir, mode=mode)

        print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        model_path, input_dim = train_classifier(
            name, train_ds, val_ds, dim, config, device
        )
        trained_models[name] = (model_path, input_dim, mode)

    # =============================================
    # Phase 3: Evaluate on test set
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation on test set")
    print("=" * 60)

    all_metrics = []

    for name, (model_path, input_dim, mode) in trained_models.items():
        test_ds = SingleModalityDataset(test_ppis, labels, embeddings_dir, mode=mode)
        print(f"\n  Evaluating {name} on {len(test_ds)} test samples...")
        metrics = evaluate_on_test(name, model_path, input_dim, test_ds, config, device)
        all_metrics.append(metrics)

    # =============================================
    # Print comparison table
    # =============================================
    print("\n" + "=" * 60)
    print("RESULTS: Test Set Performance Comparison")
    print("=" * 60)

    header = f"{'Model':<16} {'AUC':>7} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'MCC':>7} {'Loss':>7}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        print(
            f"{m['model']:<16} "
            f"{m['auc']:>7.4f} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['mcc']:>7.4f} "
            f"{m['loss']:>7.4f}"
        )

    print("-" * len(header))

    # Highlight best
    best_auc = max(all_metrics, key=lambda x: x["auc"])
    best_f1 = max(all_metrics, key=lambda x: x["f1"])
    print(f"\nBest AUC:  {best_auc['model']} ({best_auc['auc']:.4f})")
    print(f"Best F1:   {best_f1['model']} ({best_f1['f1']:.4f})")

    # Save results to CSV
    results_path = os.path.join(args.out_dir, "benchmark_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nResults saved to: {results_path}")

    # Save results to JSON too
    results_json = os.path.join(args.out_dir, "benchmark_results.json")
    with open(results_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Results saved to: {results_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark v2: PIsToN-only vs MINT-only vs Fusion on SAbDab dataset.

Fixes from v1:
  - Uses train.txt / val.txt / test.txt (not benchmark_* files)
  - Handles variable MINT dimensions (2560 and 3840) via zero-padding
  - Validation set is separate from test set (no data leakage)
  - Class-weighted loss to address native/decoy imbalance
  - Reports sample sizes, class balance, and confidence context
"""
import sys
import os
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_config import (
    OUT_DIR, EMBED_DIR, LABELS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE,
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
)


MAX_MINT_DIM = 3840


def load_list(path):
    if not Path(path).exists():
        return []
    return [l.strip() for l in open(path) if l.strip()]


def load_labels(path):
    labels = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


class MultiModalDataset(Dataset):
    """Dataset that handles variable MINT dimensions via zero-padding."""

    def __init__(self, ppi_list, labels, embeddings_dir, mode="fusion"):
        self.mode = mode
        self.labels = labels
        self.embeddings_dir = Path(embeddings_dir)

        self.valid_ppis = []
        for ppi in ppi_list:
            piston_path = self.embeddings_dir / f"{ppi}_piston.npy"
            mint_path = self.embeddings_dir / f"{ppi}_mint.npy"
            if piston_path.exists() and mint_path.exists():
                self.valid_ppis.append(ppi)

    def __len__(self):
        return len(self.valid_ppis)

    def __getitem__(self, idx):
        ppi = self.valid_ppis[idx]
        piston_emb = np.load(self.embeddings_dir / f"{ppi}_piston.npy")
        mint_emb = np.load(self.embeddings_dir / f"{ppi}_mint.npy")

        if mint_emb.shape[0] < MAX_MINT_DIM:
            padded = np.zeros(MAX_MINT_DIM, dtype=np.float32)
            padded[: mint_emb.shape[0]] = mint_emb
            mint_emb = padded

        label = float(self.labels.get(ppi, 0))

        if self.mode == "piston":
            emb = torch.from_numpy(piston_emb).float()
        elif self.mode == "mint":
            emb = torch.from_numpy(mint_emb).float()
        else:
            emb = torch.from_numpy(
                np.concatenate([piston_emb, mint_emb])
            ).float()

        return {
            "emb": emb,
            "label": torch.tensor(label, dtype=torch.float32),
            "ppi": ppi,
        }


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_metrics(all_labels, all_probs):
    all_preds = (all_probs >= 0.5).astype(int)
    metrics = {
        "n_samples": len(all_labels),
        "n_positive": int(all_labels.sum()),
        "n_negative": int((1 - all_labels).sum()),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(all_labels, all_preds)),
    }
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
    else:
        metrics["auc"] = 0.0
    return metrics


def train_and_evaluate(
    name,
    train_ds,
    val_ds,
    test_ds,
    device,
    save_dir,
    hidden_dim=512,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=100,
    patience=15,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sample = train_ds[0]
    input_dim = sample["emb"].shape[0]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = MLPClassifier(input_dim, hidden_dim, dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    train_labels = np.array(
        [train_ds.labels.get(p, 0) for p in train_ds.valid_ppis]
    )
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], dtype=torch.float32
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    patience_counter = 0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{name}_best.pth"

    print(f"\n{'='*50}")
    print(f"Training: {name} (input_dim={input_dim})")
    print(f"  Train: {len(train_ds)} (pos={int(n_pos)}, neg={int(n_neg)})")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")
    print(f"  pos_weight: {pos_weight.item():.3f}")
    print(f"{'='*50}")

    for epoch in range(max_epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for batch in train_loader:
            emb = batch["emb"].to(device)
            labels_b = batch["label"].to(device)
            logits = model(emb)
            loss = loss_fn(logits, labels_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels_b.shape[0]
            n_train += labels_b.shape[0]

        model.eval()
        val_labels_list, val_probs_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device)
                logits = model(emb)
                val_labels_list.append(batch["label"].numpy())
                val_probs_list.append(
                    torch.sigmoid(logits).cpu().numpy()
                )

        val_labels_arr = np.concatenate(val_labels_list)
        val_probs_arr = np.concatenate(val_probs_list)
        val_m = compute_metrics(val_labels_arr, val_probs_arr)
        scheduler.step(val_m["auc"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{max_epochs} | "
                f"Loss: {train_loss/max(n_train,1):.4f} | "
                f"Val AUC: {val_m['auc']:.4f} | "
                f"Val MCC: {val_m['mcc']:.4f}"
            )

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"  Best val AUC: {best_auc:.4f}")

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    model.eval()
    test_labels_list, test_probs_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            emb = batch["emb"].to(device)
            logits = model(emb)
            test_labels_list.append(batch["label"].numpy())
            test_probs_list.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    test_labels_arr = np.concatenate(test_labels_list)
    test_probs_arr = np.concatenate(test_probs_list)
    test_metrics = compute_metrics(test_labels_arr, test_probs_arr)
    test_metrics["model"] = name
    test_metrics["input_dim"] = input_dim
    test_metrics["best_val_auc"] = best_auc

    return test_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SAbDab Benchmark v2"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    train_ppis = load_list(TRAIN_FILE)
    val_ppis = load_list(VAL_FILE)
    test_ppis = load_list(TEST_FILE)
    labels = load_labels(LABELS_FILE)

    print(
        f"Split sizes: train={len(train_ppis)}, "
        f"val={len(val_ppis)}, test={len(test_ppis)}"
    )
    embeddings_dir = str(EMBED_DIR)
    save_dir = OUT_DIR / "saved_models"

    models_info = [
        ("PIsToN-only", "piston"),
        ("MINT-only", "mint"),
        ("Fusion", "fusion"),
    ]

    all_metrics = []
    for name, mode in models_info:
        train_ds = MultiModalDataset(
            train_ppis, labels, embeddings_dir, mode=mode
        )
        val_ds = MultiModalDataset(
            val_ppis, labels, embeddings_dir, mode=mode
        )
        test_ds = MultiModalDataset(
            test_ppis, labels, embeddings_dir, mode=mode
        )

        if len(train_ds) == 0:
            print(f"\nERROR: No training samples for {name}")
            continue
        if len(val_ds) == 0:
            print(f"\nERROR: No validation samples for {name}")
            continue

        metrics = train_and_evaluate(
            name,
            train_ds,
            val_ds,
            test_ds,
            device,
            str(save_dir),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        all_metrics.append(metrics)

    if not all_metrics:
        print(
            "\nERROR: No models trained. "
            "Check data availability."
        )
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS - Test Set")
    print("=" * 80)

    header = (
        f"{'Model':<14} {'AUC':>7} {'Acc':>7} {'F1':>7} "
        f"{'Prec':>7} {'Rec':>7} {'MCC':>7} "
        f"{'N':>6} {'Pos':>5} {'Neg':>5}"
    )
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        print(
            f"{m['model']:<14} "
            f"{m['auc']:>7.4f} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['mcc']:>7.4f} "
            f"{m['n_samples']:>6d} "
            f"{m['n_positive']:>5d} "
            f"{m['n_negative']:>5d}"
        )

    print("-" * len(header))

    if all_metrics:
        best_auc = max(all_metrics, key=lambda x: x["auc"])
        best_mcc = max(all_metrics, key=lambda x: x["mcc"])
        print(
            f"\nBest AUC: {best_auc['model']} "
            f"({best_auc['auc']:.4f})"
        )
        print(
            f"Best MCC: {best_mcc['model']} "
            f"({best_mcc['mcc']:.4f})"
        )

    results_path = OUT_DIR / "benchmark_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

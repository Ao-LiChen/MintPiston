#!/usr/bin/env python3
"""
Benchmark v2: PIsToN-only vs MINT-only vs Fusion on SAbDab dataset.

This script consumes a precomputed `pipeline_out/` directory and compares
three MLP heads trained on top of frozen embeddings:
  - PIsToN-only (16-dim)
  - MINT-only (2560/3840-dim; padded to 3840)
  - Fusion (16 + 3840 = 3856-dim)

Key properties:
  - Uses train/val/test splits (no leakage)
  - Handles variable MINT dimensions via zero-padding (2560 -> 3840)
  - Uses class-weighted BCE loss (pos_weight) to mitigate imbalance

Expected directory layout:
  <MintPiston>/pipeline_out/
    labels.csv
    train.txt, val.txt, test.txt
    embeddings/{ppi}_piston.npy and {ppi}_mint.npy
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


MAX_MINT_DIM = 3840


def load_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


class MultiModalDataset(Dataset):
    def __init__(self, ppi_list, labels, embeddings_dir: Path, mode: str):
        self.mode = mode
        self.labels = labels
        self.embeddings_dir = embeddings_dir
        self.valid_ppis: list[str] = []

        for ppi in ppi_list:
            p_path = self.embeddings_dir / f"{ppi}_piston.npy"
            m_path = self.embeddings_dir / f"{ppi}_mint.npy"
            if p_path.exists() and m_path.exists():
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
            emb = torch.from_numpy(np.concatenate([piston_emb, mint_emb])).float()

        return {"emb": emb, "label": torch.tensor(label, dtype=torch.float32), "ppi": ppi}


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
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


def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray):
    all_preds = (all_probs >= 0.5).astype(int)
    metrics = {
        "n_samples": int(len(all_labels)),
        "n_positive": int(all_labels.sum()),
        "n_negative": int((1 - all_labels).sum()),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(all_labels, all_preds)),
    }
    metrics["auc"] = float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.0
    return metrics


def train_and_evaluate(
    name: str,
    train_ds: MultiModalDataset,
    val_ds: MultiModalDataset,
    test_ds: MultiModalDataset,
    device: str,
    save_dir: Path,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_dim = int(train_ds[0]["emb"].shape[0])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MLPClassifier(input_dim, hidden_dim, dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    train_labels = np.array([train_ds.labels.get(p, 0) for p in train_ds.valid_ppis], dtype=np.float32)
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{name}_best.pth"

    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'='*50}")
    print(f"Training: {name} (input_dim={input_dim})")
    print(f"  Train: {len(train_ds)} (pos={int(n_pos)}, neg={int(n_neg)})")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")
    print(f"  pos_weight: {pos_weight.item():.3f}")
    print(f"{'='*50}")

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            emb = batch["emb"].to(device)
            labels_b = batch["label"].to(device)
            logits = model(emb)
            loss = loss_fn(logits, labels_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(labels_b.shape[0])
            n_train += int(labels_b.shape[0])

        model.eval()
        val_labels_list, val_probs_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device)
                logits = model(emb)
                val_labels_list.append(batch["label"].numpy())
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())

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
            best_auc = float(val_m["auc"])
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"  Best val AUC: {best_auc:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    test_labels_list, test_probs_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            emb = batch["emb"].to(device)
            logits = model(emb)
            test_labels_list.append(batch["label"].numpy())
            test_probs_list.append(torch.sigmoid(logits).cpu().numpy())

    test_labels_arr = np.concatenate(test_labels_list)
    test_probs_arr = np.concatenate(test_probs_list)
    test_metrics = compute_metrics(test_labels_arr, test_probs_arr)
    test_metrics["model"] = name
    test_metrics["input_dim"] = input_dim
    test_metrics["best_val_auc"] = best_auc
    return test_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAbDab Benchmark v2 (embeddings-based)")
    parser.add_argument("--pipeline_out", type=str, default=None, help="Path to pipeline_out (default: auto-detect)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Default pipeline_out = ../../pipeline_out relative to this file.
    here = Path(__file__).resolve()
    default_pipeline_out = here.parents[3] / "pipeline_out"
    pipeline_out = Path(args.pipeline_out) if args.pipeline_out else default_pipeline_out

    labels_path = pipeline_out / "labels.csv"
    train_path = pipeline_out / "train.txt"
    val_path = pipeline_out / "val.txt"
    test_path = pipeline_out / "test.txt"
    embeddings_dir = pipeline_out / "embeddings"

    train_ppis = load_list(train_path)
    val_ppis = load_list(val_path)
    test_ppis = load_list(test_path)
    labels = load_labels(labels_path)

    print(f"Split sizes: train={len(train_ppis)}, val={len(val_ppis)}, test={len(test_ppis)}")

    save_dir = pipeline_out / "saved_models"

    models_info = [("PIsToN-only", "piston"), ("MINT-only", "mint"), ("Fusion", "fusion")]

    all_metrics = []
    for name, mode in models_info:
        train_ds = MultiModalDataset(train_ppis, labels, embeddings_dir, mode=mode)
        val_ds = MultiModalDataset(val_ppis, labels, embeddings_dir, mode=mode)
        test_ds = MultiModalDataset(test_ppis, labels, embeddings_dir, mode=mode)

        if len(train_ds) == 0:
            print(f"\nERROR: No training samples for {name}")
            continue
        if len(val_ds) == 0:
            print(f"\nERROR: No validation samples for {name}")
            continue

        metrics = train_and_evaluate(
            name=name,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
            save_dir=save_dir,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        all_metrics.append(metrics)

    if not all_metrics:
        raise SystemExit("No models were trained. Check embeddings availability.")

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

    results_path = pipeline_out / "benchmark_v2_results.json"
    results_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()


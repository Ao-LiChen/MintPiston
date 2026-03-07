"""
Evaluation utilities for the fusion model.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def evaluate(classifier, dataloader, device, loss_fn=None):
    """
    Evaluate the fusion classifier on a dataset.

    Args:
        classifier: FusionClassifier (MLP head)
        dataloader: DataLoader yielding dicts with piston_emb, mint_emb, label
        device: torch device
        loss_fn: loss function (optional, for computing val loss)

    Returns:
        dict with metrics: auc, accuracy, f1, precision, recall, loss
    """
    classifier.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            piston_emb = batch["piston_emb"].to(device)
            mint_emb = batch["mint_emb"].to(device)
            labels = batch["label"].to(device)

            combined = torch.cat([piston_emb, mint_emb], dim=1)
            logits = classifier(combined).squeeze(-1)

            if loss_fn is not None:
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * labels.shape[0]

            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            n_samples += labels.shape[0]

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / max(n_samples, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }

    # AUC requires at least two classes present
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = 0.0

    return metrics


def print_metrics(metrics, prefix=""):
    """Pretty-print evaluation metrics."""
    print(f"{prefix}Loss:      {metrics['loss']:.4f}")
    print(f"{prefix}AUC:       {metrics['auc']:.4f}")
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}F1:        {metrics['f1']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")

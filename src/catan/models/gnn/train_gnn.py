"""
Train the BoardGAT model on the pre-built PyG dataset.

Usage:
    python -m catan.models.gnn.train_gnn [--data data/gnn] [--out data/gnn] [--epochs 60]

Saves:
    data/gnn/model_gat.pt   — best checkpoint (by val AUC)
    data/gnn/train_log.csv  — epoch-by-epoch metrics
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from catan.models.gnn.gnn_model import BoardGAT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── collate helper: add batch-global v1/v2 node indices ──────────────────────

def collate_with_global_idx(data_list: list[Data]) -> Data:
    """Standard PyG batching. v1_local/v2_local/seat are auto-concatenated."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(data_list)


def make_loader(dataset, indices, batch_size, shuffle):
    subset = [dataset[i] for i in indices]
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_with_global_idx,
        num_workers=0,
    )


# ── loss ─────────────────────────────────────────────────────────────────────

def compute_loss(win_logit, rank_pred, data, rank_weight=0.3):
    """
    Multi-task loss:
      L = BCE(win_logit, y_win) + rank_weight * MSE(rank_pred, y_rank)
    """
    y_win = data.y.view(-1, 1)
    bce = F.binary_cross_entropy_with_logits(win_logit, y_win)

    if rank_pred is not None and hasattr(data, 'y_rank'):
        y_rank = data.y_rank.view(-1, 1)
        mse = F.mse_loss(rank_pred, y_rank)
        return bce + rank_weight * mse
    return bce


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n = 0

    for data in loader:
        data = data.to(device)
        logit, rank_pred = model(data)
        loss = compute_loss(logit, rank_pred, data)
        total_loss += loss.item() * data.num_graphs
        n += data.num_graphs

        all_logits.append(logit.squeeze(-1).cpu())
        all_labels.append(data.y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    proba  = 1 / (1 + np.exp(-logits))

    auc = roc_auc_score(labels, proba)
    return total_loss / n, auc


# ── train ─────────────────────────────────────────────────────────────────────

def train(
    data_dir: Path,
    out_dir:  Path,
    epochs:   int   = 60,
    batch:    int   = 512,
    lr:       float = 5e-4,
    hidden:   int   = 32,
    heads:    int   = 4,
    layers:   int   = 3,
    dropout:  float = 0.2,
    rank_w:   float = 0.3,
    patience: int   = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset from %s ...", data_dir / "dataset.pt")
    dataset = torch.load(data_dir / "dataset.pt", weights_only=False)
    split   = torch.load(data_dir / "split.pt",   weights_only=False)
    logger.info("Dataset: %d graphs  train=%d  val=%d  test=%d",
                len(dataset), len(split["train"]), len(split["val"]), len(split["test"]))

    train_loader = make_loader(dataset, split["train"], batch, shuffle=True)
    val_loader   = make_loader(dataset, split["val"],   batch, shuffle=False)
    test_loader  = make_loader(dataset, split["test"],  batch, shuffle=False)

    # Model
    model = BoardGAT(hidden=hidden, heads=heads, n_layers=layers,
                     dropout=dropout, multitask=(rank_w > 0)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d parameters", n_params)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch","train_loss","val_loss","val_auc","lr"])

    best_val_auc = 0.0
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logit, rank_pred = model(data)
            loss = compute_loss(logit, rank_pred, data, rank_weight=rank_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            n += data.num_graphs

        scheduler.step()
        train_loss = total_loss / n
        val_loss, val_auc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info("Epoch %3d/%d  train=%.4f  val=%.4f  AUC=%.4f  lr=%.2e  %.1fs",
                    epoch, epochs, train_loss, val_loss, val_auc,
                    optimizer.param_groups[0]["lr"], elapsed)
        log_writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                              f"{val_auc:.5f}", f"{optimizer.param_groups[0]['lr']:.2e}"])
        log_file.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "val_auc":   val_auc,
                "config":    dict(hidden=hidden, heads=heads, layers=layers,
                                  dropout=dropout, multitask=(rank_w > 0)),
            }, out_dir / "model_gat.pt")
            logger.info("  ✓ Saved best model (val AUC %.4f)", best_val_auc)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    log_file.close()

    # Final test evaluation
    ckpt = torch.load(out_dir / "model_gat.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_auc = evaluate(model, test_loader, device)
    logger.info("Test AUC: %.4f  (best val AUC: %.4f)", test_auc, best_val_auc)

    # Compare to baselines
    logger.info("\n=== Final Results ===")
    logger.info("  BoardGAT (GNN):          test AUC = %.4f", test_auc)
    logger.info("  Logistic regression:     AUC ~ 0.508  (from earlier run)")
    logger.info("  Gradient boosting:       AUC ~ 0.510  (from earlier run)")
    logger.info("  Pip count heuristic:     AUC ~ 0.503")

    return model, test_auc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/gnn")
    p.add_argument("--out",     default="data/gnn")
    p.add_argument("--epochs",  type=int,   default=60)
    p.add_argument("--batch",   type=int,   default=512)
    p.add_argument("--lr",      type=float, default=5e-4)
    p.add_argument("--hidden",  type=int,   default=32)
    p.add_argument("--heads",   type=int,   default=4)
    p.add_argument("--layers",  type=int,   default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--rank-w",  type=float, default=0.3,
                   help="weight for rank regression loss (0 = win-only)")
    p.add_argument("--patience",type=int,   default=10)
    args = p.parse_args()

    train(
        data_dir = Path(args.data),
        out_dir  = Path(args.out),
        epochs   = args.epochs,
        batch    = args.batch,
        lr       = args.lr,
        hidden   = args.hidden,
        heads    = args.heads,
        layers   = args.layers,
        dropout  = args.dropout,
        rank_w   = args.rank_w,
        patience = args.patience,
    )

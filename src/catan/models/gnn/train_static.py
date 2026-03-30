"""
Fast training for StaticBoardGCN.

Converts the dataset to flat tensors once, then trains with plain PyTorch
minibatches — no PyG DataLoader overhead, ~20x faster on CPU vs GAT.

Usage:
    python -m catan.models.gnn.train_static [--data data/gnn] [--epochs 80]
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
from sklearn.metrics import roc_auc_score
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from catan.models.gnn.static_gcn import StaticBoardGCN, dataset_to_tensors, build_norm_adj, N_VERTS
from catan.models.gnn.board_to_graph import _get_edge_index, _sorted_vertices
from catan.board.board import CatanBoard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_adj():
    """Build A_hat from a canonical board (topology is fixed for all games)."""
    board = CatanBoard.random(seed=0)
    ei = _get_edge_index(board)
    return build_norm_adj(ei, N_VERTS)


def evaluate(model, X, v1, v2, seat, y_win, y_rank, batch_size=4096):
    model.eval()
    N = X.shape[0]
    all_logits, all_preds, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, N, batch_size):
            sl = slice(start, start + batch_size)
            logit, rank_pred = model(X[sl], v1[sl], v2[sl], seat[sl])
            y = y_win[sl].unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if rank_pred is not None:
                loss = loss + 0.3 * F.mse_loss(rank_pred, y_rank[sl].unsqueeze(1))
            total_loss += loss.item() * (sl.stop - sl.start if sl.stop else N - start)
            all_logits.append(logit.squeeze(1).cpu())
            all_labels.append(y_win[sl].cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    proba  = 1 / (1 + np.exp(-logits))
    auc    = roc_auc_score(labels, proba)
    return total_loss / N, auc


def train(
    data_dir:  Path,
    out_dir:   Path,
    epochs:    int   = 80,
    batch:     int   = 2048,
    lr:        float = 1e-3,
    hidden:    int   = 64,
    layers:    int   = 3,
    dropout:   float = 0.2,
    rank_w:    float = 0.3,
    patience:  int   = 12,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load and convert dataset
    logger.info("Loading dataset ...")
    t0 = time.time()
    dataset = torch.load(data_dir / "dataset.pt", weights_only=False)
    split   = torch.load(data_dir / "split.pt",   weights_only=False)
    logger.info("Loaded %d graphs in %.1fs", len(dataset), time.time() - t0)

    cache_path = data_dir / "tensors_cache.pt"
    if cache_path.exists():
        logger.info("Loading tensor cache ...")
        t0 = time.time()
        cache = torch.load(cache_path, weights_only=True)
        all_X    = cache["X"].to(device)
        all_v1   = cache["v1"].to(device)
        all_v2   = cache["v2"].to(device)
        all_seat = cache["seat"].to(device)
        all_y    = cache["y_win"].to(device)
        all_yr   = cache["y_rank"].to(device)
        logger.info("Cache loaded in %.1fs  — X shape: %s", time.time()-t0, all_X.shape)
    else:
        logger.info("Converting to tensors (first run — will cache for next time) ...")
        t0 = time.time()
        all_X, all_v1, all_v2, all_seat, all_y, all_yr = dataset_to_tensors(dataset, "cpu")
        logger.info("Tensors ready in %.1fs  — X shape: %s", time.time()-t0, all_X.shape)
        logger.info("Saving tensor cache to %s ...", cache_path)
        torch.save({"X": all_X, "v1": all_v1, "v2": all_v2, "seat": all_seat,
                    "y_win": all_y, "y_rank": all_yr}, cache_path)
        logger.info("Cache saved.")
        all_X    = all_X.to(device)
        all_v1   = all_v1.to(device)
        all_v2   = all_v2.to(device)
        all_seat = all_seat.to(device)
        all_y    = all_y.to(device)
        all_yr   = all_yr.to(device)

    # Split
    tr  = torch.tensor(split["train"], dtype=torch.long)
    val = torch.tensor(split["val"],   dtype=torch.long)
    tst = torch.tensor(split["test"],  dtype=torch.long)

    trX, trV1, trV2, trS, trY, trYR = (all_X[tr], all_v1[tr], all_v2[tr],
                                        all_seat[tr], all_y[tr], all_yr[tr])

    # Adjacency
    A_hat = get_adj().to(device)

    # Model
    model = StaticBoardGCN(hidden=hidden, n_layers=layers,
                           dropout=dropout, multitask=(rank_w > 0)).to(device)
    model.set_adj(A_hat)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d parameters", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log_static.csv"
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch","train_loss","val_loss","val_auc","lr","time_s"])

    best_val_auc = 0.0
    no_improve   = 0
    N_tr = trX.shape[0]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(N_tr, device=device)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N_tr, batch):
            idx = perm[start:start + batch]
            optimizer.zero_grad()
            logit, rank_pred = model(trX[idx], trV1[idx], trV2[idx], trS[idx])
            y  = trY[idx].unsqueeze(1)
            yr = trYR[idx].unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if rank_pred is not None and rank_w > 0:
                loss = loss + rank_w * F.mse_loss(rank_pred, yr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_loss = total_loss / n_batches

        val_loss, val_auc = evaluate(model, all_X[val], all_v1[val], all_v2[val],
                                     all_seat[val], all_y[val], all_yr[val])
        elapsed = time.time() - t0

        logger.info("Epoch %3d/%d  train=%.4f  val=%.4f  AUC=%.4f  lr=%.2e  %.1fs",
                    epoch, epochs, train_loss, val_loss, val_auc,
                    optimizer.param_groups[0]["lr"], elapsed)
        writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                         f"{val_auc:.5f}", f"{optimizer.param_groups[0]['lr']:.2e}",
                         f"{elapsed:.1f}"])
        log_file.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "val_auc": val_auc,
                "A_hat":   A_hat,
                "config":  dict(hidden=hidden, layers=layers, dropout=dropout),
            }, out_dir / "model_static_gcn.pt")
            logger.info("  ✓ Best model saved (AUC %.4f)", best_val_auc)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    log_file.close()

    # Test
    ckpt = torch.load(out_dir / "model_static_gcn.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_auc = evaluate(model, all_X[tst], all_v1[tst], all_v2[tst],
                                   all_seat[tst], all_y[tst], all_yr[tst])

    logger.info("\n=== Results ===")
    logger.info("  StaticBoardGCN (GNN):    test AUC = %.4f", test_auc)
    logger.info("  Logistic regression:     AUC ~ 0.508")
    logger.info("  Gradient boosting:       AUC ~ 0.510")
    logger.info("  Pip count heuristic:     AUC ~ 0.503")

    return model, test_auc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/gnn")
    p.add_argument("--out",     default="data/gnn")
    p.add_argument("--epochs",  type=int,   default=80)
    p.add_argument("--batch",   type=int,   default=2048)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden",  type=int,   default=64)
    p.add_argument("--layers",  type=int,   default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--rank-w",  type=float, default=0.3)
    p.add_argument("--patience",type=int,   default=12)
    args = p.parse_args()
    train(Path(args.data), Path(args.out), args.epochs, args.batch, args.lr,
          args.hidden, args.layers, args.dropout, args.rank_w, args.patience)

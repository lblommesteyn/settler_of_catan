"""
Data pipeline script for Catan Opening Intelligence.

Usage:
  # Step 1: Inspect dataset to validate schema (always do this first)
  python pipeline.py inspect --data path/to/games/

  # Step 2: Process dataset and save features
  python pipeline.py process --data path/to/games/ --out data/processed/

  # Step 3: Train and evaluate models
  python pipeline.py train --input data/processed/dataset.npz

  # Step 4: Generate synthetic data (no real dataset needed)
  python pipeline.py simulate --boards 200 --sims 50 --out data/synthetic/

  # Step 5: Score an opening
  python -m catan.scorer.cli --model logreg --model-path data/processed/model.pkl --rank-all
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_inspect(args):
    from catan.data.loader import inspect_sample_files
    source = Path(args.data)
    if not source.exists():
        print(f"Not found: {source}")
        sys.exit(1)
    inspect_sample_files(source, n=args.n)


def cmd_process(args):
    from catan.data.loader import build_training_dataset, save_dataset_numpy
    source = Path(args.data)
    if not source.exists():
        print(f"Not found: {source}")
        sys.exit(1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing dataset from %s ...", source)
    records = build_training_dataset(
        source,
        max_games=args.max_games,
        compute_features=True,
    )
    logger.info("Got %d records", len(records))

    out_path = out_dir / "dataset.npz"
    save_dataset_numpy(records, out_path)
    print(f"Saved to {out_path}")


def cmd_train(args):
    import numpy as np
    from catan.data.loader import load_dataset_numpy
    from catan.models.ml_model import compare_all_models, train_and_evaluate
    from catan.features.opening_features import FEATURE_NAMES

    input_path = Path(args.input)
    X, y = load_dataset_numpy(input_path)
    logger.info("Loaded X=%s y=%s (win rate=%.1f%%)", X.shape, y.shape, y.mean()*100)

    # Compare all models
    print("\n=== Model Comparison ===")
    results = compare_all_models(X, y)
    for name, r in results.items():
        auc = r.get("auc_roc", 0)
        print(f"  {name:12s}: AUC-ROC = {auc:.4f}")

    # Train best model and save
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        for model_type in ["logreg", "gbc"]:
            r = train_and_evaluate(X, y, model_type=model_type)
            model = r["model"]
            path = out_dir / f"model_{model_type}.pkl"
            model.save(path)
            logger.info("Saved %s → %s (AUC=%.4f)", model_type, path, r["auc_roc"])

        # Feature importance for logistic model
        logreg_r = train_and_evaluate(X, y, model_type="logreg")
        imps = logreg_r["model"].feature_importances()
        if imps:
            print("\n=== Top 10 Features (Logistic) ===")
            sorted_feats = sorted(imps.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            for feat, imp in sorted_feats:
                print(f"  {feat:35s}: {imp:+.4f}")


def cmd_simulate(args):
    from catan.simulation.simulator import simulate_dataset
    import numpy as np
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Simulating %d boards × %d openings × %d sims ...",
        args.boards, args.openings_per_board, args.sims,
    )
    rows = simulate_dataset(
        n_boards=args.boards,
        openings_per_board=args.openings_per_board,
        n_sims_per_opening=args.sims,
        seed=args.seed,
    )
    logger.info("Generated %d simulation rows", len(rows))

    # Pack into numpy
    feature_arrays = [r["features"] for r in rows if r.get("features") is not None]
    if feature_arrays:
        X = np.stack(feature_arrays)
        # Use win_rate as a continuous label; binarize at 0.25 (expected rate)
        y = np.array([float(r["win_rate"]) for r in rows if r.get("features") is not None])
        y_binary = (y > y.mean()).astype(int)

        out_path = out_dir / "synthetic_dataset.npz"
        np.savez(out_path, X=X, y=y, y_binary=y_binary)
        logger.info("Saved %d synthetic records → %s", len(rows), out_path)

        # Quick analysis
        import pandas as pd
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in rows])
        print("\n=== Synthetic Dataset Stats ===")
        print(df[["combined_pip_count", "unique_resource_count", "expansion_vertex_count",
                   "win_rate", "archetype"]].describe().round(2))
        print("\nWin rate by archetype:")
        print(df.groupby("archetype")["win_rate"].mean().sort_values(ascending=False).round(3))


def main():
    parser = argparse.ArgumentParser(description="Catan Opening Intelligence Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Inspect
    p_inspect = sub.add_parser("inspect", help="Inspect sample game files")
    p_inspect.add_argument("--data", required=True, help="Path to games/ directory")
    p_inspect.add_argument("--n", type=int, default=3, help="Number of files to inspect")

    # Process
    p_proc = sub.add_parser("process", help="Process dataset → features + labels")
    p_proc.add_argument("--data", required=True, help="Path to games/ directory")
    p_proc.add_argument("--out", required=True, help="Output directory for .npz")
    p_proc.add_argument("--max-games", type=int, default=None)

    # Train
    p_train = sub.add_parser("train", help="Train and evaluate models")
    p_train.add_argument("--input", required=True, help="Path to dataset.npz")
    p_train.add_argument("--out", default=None, help="Directory to save trained models")

    # Simulate
    p_sim = sub.add_parser("simulate", help="Generate synthetic data via simulation")
    p_sim.add_argument("--boards", type=int, default=100)
    p_sim.add_argument("--openings-per-board", type=int, default=20)
    p_sim.add_argument("--sims", type=int, default=50)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--out", default="data/synthetic/")

    args = parser.parse_args()
    {"inspect": cmd_inspect, "process": cmd_process,
     "train": cmd_train, "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    main()

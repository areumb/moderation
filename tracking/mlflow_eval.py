"""Thin MLflow wrapper around the existing evaluation pipeline.

Does NOT modify hs_generalization. It shells out to `hs_generalization.test`
exactly like hs_generalization/run_many.py does, then reads the JSON the
research code already writes (`pipeline.output_predictions`) and logs params
and metrics to a local MLflow store (./mlruns). Each (variant, seed,
checkpoint) becomes one tracked, comparable run.

Example:
    python -m tracking.mlflow_eval \
        -c configs/test/example.json \
        --dataset davidson --eval-mode 3class --train-mode 3class \
        --seed 5 --checkpoint outputs/davidson/RoBERTa-base/3class/seed5_RoBERTa-base_7.pt \
        --variant ternary

View: mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def per_class_f1_from_confusion(cm: list[list[int]]) -> dict[str, float]:
    """Per-class F1 computed from the confusion matrix the research code saves
    (rows = references, cols = predictions)."""
    n = len(cm)
    out = {}
    for k in range(n):
        tp = cm[k][k]
        fp = sum(cm[r][k] for r in range(n)) - tp
        fn = sum(cm[k][c] for c in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"f1_class_{k}"] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--dataset", default="davidson", choices=["davidson", "hatecheck_xr"])
    parser.add_argument("--eval-mode", default="3class")
    parser.add_argument("--train-mode", default="3class")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--variant", default=None,
        help="Run name for MLflow, e.g. 'ternary' or 'hate_vs_nonhate'. Defaults to train-mode.",
    )
    parser.add_argument("--hatecheck-csv", default="datasets/hatecheck-xr/hatecheck-xr.csv")
    parser.add_argument("--experiment", default="hs-generalization")
    args = parser.parse_args()

    import mlflow  # imported here so the research requirements stay untouched

    # 1) Run the UNMODIFIED research evaluator as a subprocess.
    cmd = [
        sys.executable, "-m", "hs_generalization.test",
        "-c", args.config,
        "--dataset", args.dataset,
        "--eval-mode", args.eval_mode,
        "--train-mode", args.train_mode,
        "--seed", str(args.seed),
        "--checkpoint", args.checkpoint,
    ]
    if args.dataset == "hatecheck_xr":
        cmd += ["--hatecheck-csv", args.hatecheck_csv]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # 2) Read the JSON the research code wrote (pipeline.output_predictions).
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    out_path = config["pipeline"]["output_predictions"].format(seed=args.seed)
    with open(out_path, encoding="utf-8") as f:
        saved = json.load(f)

    # 3) Log to MLflow.
    mlflow.set_tracking_uri(f"file://{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment(args.experiment)
    variant = args.variant or args.train_mode
    with mlflow.start_run(run_name=f"{variant}-seed{args.seed}"):
        mlflow.log_params(
            {
                "variant": variant,
                "dataset": args.dataset,
                "eval_mode": args.eval_mode,
                "train_mode": args.train_mode,
                "seed": args.seed,
                "checkpoint": args.checkpoint,
                "model_name": config["task"].get("model_name", ""),
                "learning_rate": config.get("optimizer", {}).get("learning_rate", ""),
                "n_epochs": config.get("pipeline", {}).get("n_epochs", ""),
            }
        )
        metrics = {"eval_loss": saved["average_loss"]}
        for name, value in saved["results"].items():
            if isinstance(value, (int, float)):
                metrics[name] = value  # includes eval_macro_f1 / eval_micro_f1 etc.
        metrics.update(per_class_f1_from_confusion(saved["confusion_matrix"]))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(out_path)

    print(f"[mlflow] Logged run '{variant}-seed{args.seed}' to {REPO_ROOT / 'mlruns'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

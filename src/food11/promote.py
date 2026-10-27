from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a Food-11 model through a statistical quality gate")
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--model-name", default="food11")
    parser.add_argument("--metric", default="final_val_accuracy")
    parser.add_argument("--test-size", type=int)
    return parser.parse_args()


def infer_test_size() -> int:
    evaluation = Path("data/food11_processed_mini/evaluation")
    count = sum(1 for path in evaluation.rglob("*") if path.is_file()) if evaluation.is_dir() else 0
    if count <= 0:
        raise ValueError("Could not infer test size; pass --test-size explicitly")
    return count


def metric_for_version(client: MlflowClient, name: str, version: str, metric: str) -> float:
    model_version = client.get_model_version(name, version)
    if not model_version.run_id:
        raise ValueError(f"Model {name} version {version} has no source run")
    metrics = client.get_run(model_version.run_id).data.metrics
    if metric not in metrics:
        raise ValueError(f"Run {model_version.run_id} does not contain metric {metric!r}")
    return float(metrics[metric])


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    candidate = client.get_model_version(args.model_name, str(args.version))
    champion = client.get_model_version_by_alias(args.model_name, "champion")
    candidate_accuracy = metric_for_version(client, args.model_name, candidate.version, args.metric)
    champion_accuracy = metric_for_version(client, args.model_name, champion.version, args.metric)
    test_size = args.test_size or infer_test_size()
    standard_error = math.sqrt(champion_accuracy * (1.0 - champion_accuracy) / test_size)
    threshold = champion_accuracy + 2.0 * standard_error

    if candidate_accuracy >= threshold:
        decision = "promoted_to_champion"
        client.set_registered_model_alias(args.model_name, "champion", candidate.version)
    else:
        decision = "assigned_to_challenger"
        client.set_registered_model_alias(args.model_name, "challenger", candidate.version)

    tags = {
        "promotion.decision": decision,
        "promotion.metric": args.metric,
        "promotion.candidate_accuracy": f"{candidate_accuracy:.8f}",
        "promotion.champion_accuracy": f"{champion_accuracy:.8f}",
        "promotion.standard_error": f"{standard_error:.8f}",
        "promotion.threshold": f"{threshold:.8f}",
        "promotion.test_size": str(test_size),
    }
    for key, value in tags.items():
        client.set_model_version_tag(args.model_name, candidate.version, key, value)

    print(
        f"{decision}: candidate={candidate_accuracy:.4f}, champion={champion_accuracy:.4f}, "
        f"2*se={2 * standard_error:.4f}, threshold={threshold:.4f}"
    )


if __name__ == "__main__":
    main()

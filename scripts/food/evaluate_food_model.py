import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_label(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", default="Data-Pipeline/data/processed/food_predictions.jsonl", type=str)
    parser.add_argument("--out_metrics", default="Data-Pipeline/data/processed/food_eval_metrics.json", type=str)
    parser.add_argument("--out_results_csv", default="Data-Pipeline/data/processed/food_eval_results.csv", type=str)
    args = parser.parse_args()

    preds_path = Path(args.predictions_jsonl)
    out_metrics = Path(args.out_metrics)
    out_results_csv = Path(args.out_results_csv)

    records = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    df["true_label_norm"] = df["true_label"].apply(normalize_label)
    df["predicted_food_norm"] = df["predicted_food"].apply(normalize_label)

    df["correct"] = df["true_label_norm"] == df["predicted_food_norm"]

    total = len(df)
    correct = df["correct"].sum()
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "total_samples": int(total),
        "correct_predictions": int(correct),
        "accuracy": float(accuracy)
    }

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_results_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df.to_csv(out_results_csv, index=False)

    print("Evaluation complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
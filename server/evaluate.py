"""
Evaluate a saved fake review detection model on one or more CSV datasets.

Example:
    python evaluate.py --model_dir model --data_paths data/clean_dataset.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import build_tokenized_dataset, compute_metrics, load_datasets, split_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fake review detector")
    parser.add_argument("--model_dir", default="model", help="Path to trained model directory")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=["data/clean_dataset.csv"],
        help="One or more CSV files used for evaluation",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized length")
    parser.add_argument("--test_size", type=float, default=0.15, help="Final test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation ratio taken from train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_datasets(args.data_paths, sample_size=0, seed=args.seed)
    _, _, test_df = split_dataframe(
        loaded.dataframe,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    test_dataset = build_tokenized_dataset(test_df, tokenizer, args.max_length)

    logits = []
    labels = []
    for record in test_dataset:
      labels.append(record["labels"])
      inputs = {
          "input_ids": torch.tensor([record["input_ids"]], device=device),
          "attention_mask": torch.tensor([record["attention_mask"]], device=device),
      }
      with torch.no_grad():
          outputs = model(**inputs)
      logits.append(outputs.logits.cpu().numpy()[0])

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    predictions = np.argmax(logits, axis=1)

    metrics = compute_metrics((logits, labels))
    print("[INFO] Metrics:", metrics)
    print("[INFO] Confusion matrix:")
    print(confusion_matrix(labels, predictions))
    print("[INFO] Classification report:")
    print(classification_report(labels, predictions, target_names=["genuine", "fake"], digits=4))
    print(f"[INFO] Evaluated model from: {os.path.abspath(args.model_dir)}")


if __name__ == "__main__":
    main()

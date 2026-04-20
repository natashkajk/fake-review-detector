"""
Calibrate a trained classification model with temperature scaling.

This script uses the validation split reconstructed from the provided datasets
and saves `temperature.json` into the model directory.

Example:
    python calibrate_temperature.py --model_dir model_mixed --data_paths data/clean_dataset.csv data/fake_reviews_prepared.csv
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import build_tokenized_dataset, load_datasets, split_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate model confidence with temperature scaling")
    parser.add_argument("--model_dir", default="model_mixed", help="Path to trained model directory")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=["data/clean_dataset.csv"],
        help="Training datasets used to reconstruct the validation split",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized length")
    parser.add_argument("--test_size", type=float, default=0.15, help="Final test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation ratio taken from train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def collect_validation_logits(model, val_dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits = []
    labels = []

    model.eval()
    for record in val_dataset:
        labels.append(record["labels"])
        inputs = {
            "input_ids": torch.tensor([record["input_ids"]], device=device),
            "attention_mask": torch.tensor([record["attention_mask"]], device=device),
        }
        with torch.no_grad():
            outputs = model(**inputs)
        logits.append(outputs.logits.squeeze(0).cpu())

    return torch.stack(logits), torch.tensor(labels, dtype=torch.long)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=0.5, max=5.0).item())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_datasets(args.data_paths, sample_size=0, seed=args.seed)
    _, val_df, _ = split_dataframe(
        loaded.dataframe,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    val_dataset = build_tokenized_dataset(val_df, tokenizer, args.max_length)

    logits, labels = collect_validation_logits(model, val_dataset, device)
    temperature = fit_temperature(logits, labels)

    output_path = os.path.join(args.model_dir, "temperature.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"temperature": temperature}, handle, indent=2)

    print(f"[SUCCESS] Saved temperature to: {os.path.abspath(output_path)}")
    print(f"[INFO] temperature={temperature:.4f}")


if __name__ == "__main__":
    main()

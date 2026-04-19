"""
Fake Review Detector - Evaluation Script
========================================
Evaluates a trained model on a test dataset and prints detailed metrics.

Usage:
    python evaluate.py --data_path data/test.csv --model_dir model
    python evaluate.py --generate-sample-data --model_dir model
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Import shared utilities from train.py
from train import generate_sample_data, load_csv_data, set_seed

DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42


def evaluate_model(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    texts: List[str],
    labels: List[int],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Dict:
    """
    Run model evaluation on a list of texts and labels.
    Returns dictionary with predictions, metrics, and per-sample results.
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_attentions = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=["fake", "real"], digits=4
        ),
        "predictions": [
            {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "true_label": int(true_lbl),
                "predicted_label": int(pred_lbl),
                "confidence": float(max(prob)),
                "probabilities": [float(p) for p in prob],
            }
            for text, true_lbl, pred_lbl, prob in zip(
                texts, all_labels, all_preds, all_probs
            )
        ],
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fake Review Detector")
    parser.add_argument("--data_path", type=str, help="Path to test CSV")
    parser.add_argument("--model_dir", type=str, default="model", help="Path to saved model")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic test data")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        print("        Run train.py first to train a model.")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load data
    if args.generate_sample_data:
        data_path = generate_sample_data("./test_data", n_samples=200)
    else:
        data_path = args.data_path

    if not data_path or not os.path.exists(data_path):
        print("[ERROR] No data found. Provide --data_path or use --generate-sample-data")
        sys.exit(1)

    print(f"[INFO] Loading test data from: {data_path}")
    texts, labels = load_csv_data(data_path)
    print(f"[INFO] Test samples: {len(texts)}")

    # Load model and tokenizer
    print(f"[INFO] Loading model from: {args.model_dir}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    # Evaluate
    print("[INFO] Running evaluation...")
    metrics = evaluate_model(
        model, tokenizer, texts, labels, device, args.batch_size, args.max_length
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                 Predicted")
    print(f"                 fake   real")
    print(f"  Actual fake    {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"  Actual real    {cm[1][0]:4d}   {cm[1][1]:4d}")
    print("-" * 60)
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("=" * 60)

    # Show sample predictions
    print("\nSample Predictions:")
    for pred in metrics["predictions"][:5]:
        label_name = "real" if pred["predicted_label"] == 1 else "fake"
        status = "✓" if pred["true_label"] == pred["predicted_label"] else "✗"
        print(
            f"  [{status}] '{pred['text']}' -> {label_name} "
            f"(confidence: {pred['confidence']:.2%})"
        )


if __name__ == "__main__":
    main()

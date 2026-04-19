"""
Fake Review Detector - Training Script
======================================
Trains a DistilBERT-based classifier to detect fake reviews.
Supports custom CSV datasets and HuggingFace datasets format.

Usage:
    python train.py --data_path data/reviews.csv --output_dir model
    python train.py --generate-sample-data --output_dir model
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 3
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Sample data generator (for quick MVP testing)
# ---------------------------------------------------------------------------

FAKE_REVIEWS = [
    "This product is absolutely amazing!!! Best purchase ever!!!",
    "I love it so much!!! Five stars!!! Highly recommend to everyone!!!",
    "OMG this is the best thing I have ever bought in my entire life!!!",
    "Incredible quality!!! Must buy!!! You won't regret it!!!",
    "Best product ever!!! Amazing!!! Love it!!!",
    "This is a scam. I can't believe anyone would buy this.",
    "Terrible quality. Complete waste of money. Do not buy.",
    "Awful product. Broke after one day. Zero stars if I could.",
    "Horrible experience. Customer service was useless.",
    "Cheap junk. Fell apart immediately. Total ripoff.",
]

REAL_REVIEWS = [
    "The product works as described. Battery life is around 6 hours.",
    "Decent quality for the price. The instructions were a bit confusing.",
    "I've been using this for 3 weeks. So far so good, no complaints.",
    "It's smaller than I expected but functions well. Shipping was fast.",
    "Good build quality. The buttons feel responsive. Would recommend.",
    "Average performance. Does what it says but nothing exceptional.",
    "Had some issues initially but customer service resolved them.",
    "The color is slightly different from the photos but I don't mind.",
    "Works fine for my needs. Not premium but acceptable quality.",
    "Took a while to arrive but the product is solid. Four stars.",
]


def generate_sample_data(output_dir: str, n_samples: int = 1000) -> str:
    """
    Generate a synthetic dataset for quick MVP testing.
    Returns path to the generated CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sample_data.csv")

    texts: List[str] = []
    labels: List[int] = []

    for _ in range(n_samples):
        if random.random() < 0.5:
            # Fake review (label: 0)
            base = random.choice(FAKE_REVIEWS)
            # Add some variation
            if random.random() < 0.3:
                base += " " + random.choice(FAKE_REVIEWS)
            texts.append(base)
            labels.append(0)
        else:
            # Real review (label: 1)
            base = random.choice(REAL_REVIEWS)
            if random.random() < 0.2:
                base += " " + random.choice(REAL_REVIEWS)
            texts.append(base)
            labels.append(1)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("text,label\n")
        for text, label in zip(texts, labels):
            # Escape quotes in text
            safe_text = text.replace('"', '""')
            f.write(f'"{safe_text}",{label}\n')

    print(f"[INFO] Generated {n_samples} sample reviews -> {csv_path}")
    print(f"       Fake: {labels.count(0)}, Real: {labels.count(1)}")
    return csv_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_data(data_path: str) -> Tuple[List[str], List[int]]:
    """Load text and labels from a CSV file (columns: text, label)."""
    import csv

    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(int(row["label"]))
    return texts, labels


def create_dataset(texts: List[str], labels: List[int]) -> DatasetDict:
    """Split data and create HuggingFace DatasetDict."""
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=DEFAULT_SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=DEFAULT_SEED, stratify=temp_labels
    )

    return DatasetDict({
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "validation": Dataset.from_dict({"text": val_texts, "label": val_labels}),
        "test": Dataset.from_dict({"text": test_texts, "label": test_labels}),
    })


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(
    dataset: DatasetDict, tokenizer: DistilBertTokenizer, max_length: int
) -> DatasetDict:
    """Tokenize all splits in the dataset."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_fn, batched=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
        "f1": f1_score(labels, predictions, average="binary"),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Run one evaluation epoch. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Fake Review Detector")
    parser.add_argument("--data_path", type=str, help="Path to CSV dataset (text,label columns)")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic data for testing")
    parser.add_argument("--output_dir", type=str, default="model", help="Directory to save model")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.data_path and not args.generate_sample_data:
        print("[ERROR] Provide --data_path or use --generate-sample-data")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    if args.generate_sample_data:
        data_path = generate_sample_data("./data", n_samples=1000)
    else:
        data_path = args.data_path

    print(f"[INFO] Loading data from: {data_path}")
    texts, labels = load_csv_data(data_path)
    dataset = create_dataset(texts, labels)
    print(f"[INFO] Dataset splits: { {k: len(v) for k, v in dataset.items()} }")

    # ------------------------------------------------------------------
    # Tokenizer & Model
    # ------------------------------------------------------------------
    print(f"[INFO] Loading tokenizer and model: {args.model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=True,  # Required for explainability
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    print("[INFO] Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(
        tokenized_dataset["train"], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"], batch_size=args.batch_size
    )
    test_loader = DataLoader(
        tokenized_dataset["test"], batch_size=args.batch_size
    )

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"[INFO] New best model saved (F1={best_f1:.4f})")

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    print("\n[INFO] Loading best model for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(args.output_dir)
    model.to(device)

    test_loss, test_metrics = eval_epoch(model, test_loader, device)
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print("=" * 50)

    print(f"\n[SUCCESS] Model saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

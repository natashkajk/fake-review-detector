"""
Training script for fake review detection.

This version is designed for continued fine-tuning on one or more datasets.
It supports:
  - multiple CSV sources
  - automatic text/label column detection
  - label normalization across common fake/genuine naming schemes
  - train/validation/test split
  - weighted loss for class imbalance
  - richer evaluation metrics
  - loading the best checkpoint at the end

Example:
    python train.py --data_paths data/clean_dataset.csv

    python train.py --data_paths data/clean_dataset.csv data/ott_reviews.csv ^
        --output_dir model --num_train_epochs 4 --max_length 256
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = "./model"
DEFAULT_TEXT_COLUMNS = [
    "text",
    "text_",
    "review_text",
    "review",
    "review_body",
    "content",
    "body",
]
DEFAULT_LABEL_COLUMNS = [
    "label",
    "labels",
    "class",
    "target",
    "spam",
    "is_fake",
    "generated",
]
FAKE_LABELS = {"1", "fake", "spam", "cg", "deceptive", "generated", "filtered", "yes", "y", "true"}
GENUINE_LABELS = {"0", "real", "genuine", "truthful", "or", "original", "human", "recommended", "no", "n", "false"}
ID2LABEL = {0: "genuine", 1: "fake"}
LABEL2ID = {"genuine": 0, "fake": 1}


@dataclass
class LoadedData:
    dataframe: pd.DataFrame
    sources: List[str]


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross entropy."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fake review detector")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=["data/clean_dataset.csv"],
        help="One or more CSV files to use for training",
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Base model or local checkpoint")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Where to save the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized sequence length")
    parser.add_argument("--num_train_epochs", type=float, default=4.0, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Per-device eval batch size")
    parser.add_argument("--test_size", type=float, default=0.15, help="Final test split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation ratio taken from train split")
    parser.add_argument("--sample_size", type=int, default=0, help="Optional downsampling size for quick experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Stop when validation F1 does not improve for N evaluation rounds",
    )
    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        help="Freeze encoder weights and train only the classifier head",
    )
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    return " ".join(str(text).split()).strip()


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    for column in DEFAULT_TEXT_COLUMNS:
        if column in df.columns:
            return column

    summary_candidates = [col for col in ("summary", "review_headline", "title") if col in df.columns]
    body_candidates = [col for col in ("review_body", "body", "content") if col in df.columns]
    if summary_candidates or body_candidates:
        return None

    return None


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for column in DEFAULT_LABEL_COLUMNS:
        if column in df.columns:
            return column
    return None


def normalize_label(value: object) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
        return None

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        return None

    normalized = str(value).strip().lower()
    if normalized in FAKE_LABELS:
        return 1
    if normalized in GENUINE_LABELS:
        return 0
    return None


def build_text_column(df: pd.DataFrame) -> pd.Series:
    direct_text_column = detect_text_column(df)
    if direct_text_column:
        return df[direct_text_column].map(normalize_text)

    headline = df["review_headline"].map(normalize_text) if "review_headline" in df.columns else ""
    summary = df["summary"].map(normalize_text) if "summary" in df.columns else ""
    body = df["review_body"].map(normalize_text) if "review_body" in df.columns else ""

    if isinstance(headline, str) and isinstance(summary, str) and isinstance(body, str):
        raise ValueError("Could not find a usable text column in dataset")

    combined = []
    for values in zip(
        headline if not isinstance(headline, str) else [""] * len(df),
        summary if not isinstance(summary, str) else [""] * len(df),
        body if not isinstance(body, str) else [""] * len(df),
    ):
        parts = [part for part in values if part]
        combined.append(" ".join(parts).strip())

    return pd.Series(combined)


def load_single_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_column = detect_label_column(df)
    if label_column is None:
        raise ValueError(f"Could not detect label column in {path}")

    text_series = build_text_column(df)
    label_series = df[label_column].map(normalize_label)

    normalized_df = pd.DataFrame(
        {
            "text": text_series,
            "label": label_series,
            "source": Path(path).name,
        }
    )
    normalized_df = normalized_df.dropna(subset=["text", "label"])
    normalized_df["text"] = normalized_df["text"].astype(str).str.strip()
    normalized_df["label"] = normalized_df["label"].astype(int)
    normalized_df = normalized_df[normalized_df["text"].str.len() >= 10]
    normalized_df = normalized_df.drop_duplicates(subset=["text"])
    return normalized_df.reset_index(drop=True)


def load_datasets(data_paths: Sequence[str], sample_size: int, seed: int) -> LoadedData:
    frames: List[pd.DataFrame] = []
    resolved_sources: List[str] = []

    for path in data_paths:
        frame = load_single_csv(path)
        frames.append(frame)
        resolved_sources.append(os.path.abspath(path))

    if not frames:
        raise ValueError("No datasets were loaded")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if sample_size and sample_size > 0 and len(combined) > sample_size:
        combined = combined.sample(sample_size, random_state=seed).reset_index(drop=True)

    return LoadedData(dataframe=combined, sources=resolved_sources)


def split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_tokenized_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "label"])
    return dataset


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def build_class_weights(labels: Iterable[int]) -> torch.Tensor:
    labels = list(labels)
    class_counts = np.bincount(labels, minlength=2)
    total = class_counts.sum()
    weights = [total / (2 * count) if count else 1.0 for count in class_counts]
    return torch.tensor(weights, dtype=torch.float)


def maybe_freeze_encoder(model) -> None:
    base_model = getattr(model, model.base_model_prefix, None)
    if base_model is None:
        return
    for parameter in base_model.parameters():
        parameter.requires_grad = False


def save_training_summary(
    output_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    sources: Sequence[str],
) -> None:
    summary = {
        "sources": list(sources),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "train_label_distribution": train_df["label"].value_counts().sort_index().to_dict(),
        "validation_label_distribution": val_df["label"].value_counts().sort_index().to_dict(),
        "test_label_distribution": test_df["label"].value_counts().sort_index().to_dict(),
        "metrics": metrics,
        "training_args": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
        },
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading datasets: {args.data_paths}")

    loaded = load_datasets(args.data_paths, sample_size=args.sample_size, seed=args.seed)
    df = loaded.dataframe
    print(f"[INFO] Loaded {len(df)} unique reviews from {len(loaded.sources)} source(s)")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().sort_index()}")

    train_df, val_df, test_df = split_dataframe(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    if args.freeze_base_model:
        maybe_freeze_encoder(model)
        print("[INFO] Base encoder frozen; training classifier head only")

    train_dataset = build_tokenized_dataset(train_df, tokenizer, args.max_length)
    val_dataset = build_tokenized_dataset(val_df, tokenizer, args.max_length)
    test_dataset = build_tokenized_dataset(test_df, tokenizer, args.max_length)
    class_weights = build_class_weights(train_df["label"].tolist())

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=use_fp16,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    print("[INFO] Evaluating best checkpoint on held-out test split...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(f"[INFO] Test metrics: {test_metrics}")

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_training_summary(
        output_dir=args.output_dir,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        metrics=test_metrics,
        args=args,
        sources=loaded.sources,
    )

    print(f"[SUCCESS] Model saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

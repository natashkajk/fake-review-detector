"""
Prepare an external fake review dataset into the normalized training format:

    text,label

Supported input patterns:
  - text/text_/review_text/review/review_body/content/body
  - review_headline + review_body
  - label/labels/class/target/spam/is_fake/generated
  - labels such as CG/OR, fake/genuine, deceptive/truthful, spam/not spam, 0/1

Examples:
    python prepare_external_dataset.py --input_path data/raw/fake_reviews.csv --output_path data/salminen_prepared.csv

    python prepare_external_dataset.py --input_path data/raw/fake_reviews.csv --output_path data/salminen_prepared.csv --sample_size 20000
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from train import build_text_column, detect_label_column, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize external fake review dataset")
    parser.add_argument("--input_path", required=True, help="Path to source CSV file")
    parser.add_argument("--output_path", required=True, help="Where to save normalized CSV")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Optional downsampling size after cleaning",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=10,
        help="Minimum review text length to keep",
    )
    return parser.parse_args()


def prepare_external_dataset(
    input_path: str,
    output_path: str,
    sample_size: int = 0,
    min_text_length: int = 10,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    label_column = detect_label_column(df)
    if label_column is None:
        raise ValueError(
            "Could not detect a label column. Supported names: "
            "label, labels, class, target, spam, is_fake, generated."
        )

    normalized = pd.DataFrame(
        {
            "text": build_text_column(df),
            "label": df[label_column].map(normalize_label),
        }
    )

    normalized = normalized.dropna(subset=["text", "label"])
    normalized["text"] = normalized["text"].astype(str).str.strip()
    normalized["label"] = normalized["label"].astype(int)
    normalized = normalized[normalized["text"].str.len() >= min_text_length]
    normalized = normalized.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if sample_size and sample_size > 0 and len(normalized) > sample_size:
        normalized = normalized.sample(sample_size, random_state=42).reset_index(drop=True)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    normalized.to_csv(output_path, index=False, encoding="utf-8")
    return normalized


def main() -> None:
    args = parse_args()
    prepared = prepare_external_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        min_text_length=args.min_text_length,
    )

    label_counts = prepared["label"].value_counts().sort_index().to_dict()
    print(f"[SUCCESS] Saved normalized dataset to: {os.path.abspath(args.output_path)}")
    print(f"[INFO] Rows kept: {len(prepared)}")
    print(f"[INFO] Label distribution: {label_counts}")


if __name__ == "__main__":
    main()

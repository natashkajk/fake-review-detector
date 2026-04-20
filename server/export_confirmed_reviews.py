"""
Export human-labeled reviews from SQLite into a CSV suitable for retraining.

Output format:
    text,label

Only rows with human_label in {"fake", "genuine"} are exported by default.

Example:
    python export_confirmed_reviews.py
    python export_confirmed_reviews.py --output_path data/confirmed_reviews.csv
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_DB_PATH = "reviews.db"
DEFAULT_OUTPUT_PATH = "data/confirmed_reviews.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export human-labeled reviews for retraining")
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH, help="Path to SQLite reviews database")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="Where to save exported CSV")
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Optional minimum model confidence filter for exported rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with sqlite3.connect(args.db_path) as conn:
        query = """
            SELECT
                review_text AS text,
                CASE
                    WHEN human_label = 'genuine' THEN 0
                    WHEN human_label = 'fake' THEN 1
                END AS label,
                human_label,
                confidence,
                notes,
                created_at
            FROM review_analyses
            WHERE human_label IN ('fake', 'genuine')
              AND confidence >= ?
            ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn, params=(args.min_confidence,))

    if df.empty:
        print("[INFO] No confirmed rows matched the export criteria.")
        return

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(output_path, index=False, encoding="utf-8")

    counts = df["label"].value_counts().sort_index().to_dict()
    print(f"[SUCCESS] Exported confirmed reviews to: {os.path.abspath(output_path)}")
    print(f"[INFO] Rows kept: {len(df)}")
    print(f"[INFO] Label distribution: {counts}")


if __name__ == "__main__":
    main()

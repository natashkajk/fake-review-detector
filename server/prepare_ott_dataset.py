"""
Prepare the Ott Deceptive Opinion Spam Corpus (P11-1032.Datasets) into CSV.

Output format:
    text,label

Notes:
  - deceptive MTurk reviews are taken from raw `.txt` files
  - truthful TripAdvisor reviews are available only as feature text, so this
    script uses `.uni.txt` files for truthful examples
  - `.bi.txt` files are ignored

Usage:
    python prepare_ott_dataset.py

    python prepare_ott_dataset.py --input_path data/P11-1032.Datasets --output_path data/ott_prepared.csv
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT_PATH = "data/P11-1032.Datasets"
DEFAULT_EXTRACT_DIR = "data/op_spam_v3"
DEFAULT_OUTPUT_PATH = "data/ott_prepared.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Ott deceptive opinion spam corpus")
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, help="Path to P11-1032.Datasets file")
    parser.add_argument("--extract_dir", default=DEFAULT_EXTRACT_DIR, help="Where to extract the archive")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="Where to save ott_prepared.csv")
    return parser.parse_args()


def ensure_extracted(input_path: str, extract_dir: str) -> Path:
    extract_path = Path(extract_dir)
    readme_path = extract_path / "README"
    mturk_dir = extract_path / "MTurk"
    tripadvisor_dir = extract_path / "TripAdvisor"

    if readme_path.exists() and mturk_dir.exists() and tripadvisor_dir.exists():
        return extract_path

    extract_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_path) as archive:
        archive.extractall(path=extract_path.parent)

    return extract_path


def read_text_file(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8", errors="ignore").split()).strip()


def collect_records(base_dir: Path) -> List[dict]:
    records: List[dict] = []

    mturk_dir = base_dir / "MTurk"
    for path in mturk_dir.rglob("*.txt"):
        if path.name.endswith(".uni.txt") or path.name.endswith(".bi.txt"):
            continue
        text = read_text_file(path)
        if text:
            records.append({
                "text": text,
                "label": 1,
                "source": "ott_mturk_deceptive",
                "file": str(path.relative_to(base_dir)),
            })

    tripadvisor_dir = base_dir / "TripAdvisor"
    for path in tripadvisor_dir.rglob("*.uni.txt"):
        text = read_text_file(path)
        if text:
            records.append({
                "text": text,
                "label": 0,
                "source": "ott_tripadvisor_truthful_features",
                "file": str(path.relative_to(base_dir)),
            })

    return records


def prepare_ott_dataset(input_path: str, extract_dir: str, output_path: str) -> pd.DataFrame:
    base_dir = ensure_extracted(input_path, extract_dir)
    records = collect_records(base_dir)

    if not records:
        raise ValueError("No Ott records were collected. Check the archive contents.")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    output = df[["text", "label"]]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    prepared = prepare_ott_dataset(
        input_path=args.input_path,
        extract_dir=args.extract_dir,
        output_path=args.output_path,
    )

    counts = prepared["label"].value_counts().sort_index().to_dict()
    print(f"[SUCCESS] Saved Ott dataset to: {os.path.abspath(args.output_path)}")
    print(f"[INFO] Rows kept: {len(prepared)}")
    print(f"[INFO] Label distribution: {counts}")


if __name__ == "__main__":
    main()

# Model Lifecycle

## Overview

The project follows a full machine learning lifecycle instead of a one-time training script.

## Stage 1. Raw Data Collection

The system starts from several sources:

- prepared fake review datasets
- cleaned local datasets
- Ott deceptive opinion corpus
- manually created `hard_cases.csv`

Relevant files:

- [server/data/fake_reviews_prepared.csv](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/data/fake_reviews_prepared.csv)
- [server/data/clean_dataset.csv](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/data/clean_dataset.csv)
- [server/data/ott_prepared.csv](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/data/ott_prepared.csv)
- [server/data/hard_cases.csv](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/data/hard_cases.csv)

## Stage 2. Dataset Normalization

Raw external datasets are transformed into a common `text,label` format.

Scripts:

- [server/prepare_external_dataset.py](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/prepare_external_dataset.py)
- [server/prepare_ott_dataset.py](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/prepare_ott_dataset.py)

## Stage 3. Model Training

Training is performed with:

- multi-dataset input
- train/validation/test split
- transformer fine-tuning
- class imbalance handling
- metric tracking

Script:

- [server/train.py](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/train.py)

## Stage 4. Evaluation

The project compares several candidate models using:

- accuracy
- precision
- recall
- F1 score
- qualitative manual checking on real examples

Script:

- [server/evaluate.py](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/evaluate.py)

## Stage 5. Confidence Calibration

Raw model probabilities are often overconfident. To reduce this, the project uses temperature scaling.

Script:

- [server/calibrate_temperature.py](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/calibrate_temperature.py)

## Stage 6. Deployment

The preferred model is loaded by the FastAPI backend and used by the Chrome extension in real time.

Default deployment candidate:

- `server/model_mixed`

## Stage 7. Human Feedback Collection

Every analyzed review is stored in SQLite and can be relabeled by a human reviewer.

This turns the application into a `human-in-the-loop` system rather than a static classifier.

## Stage 8. Retraining

Confirmed feedback examples can be exported and later merged into the next training cycle.

Script:

- [server/export_confirmed_reviews.py](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/server/export_confirmed_reviews.py)

## Why This Matters Academically

This lifecycle demonstrates:

- practical ML engineering
- data preparation and normalization
- model selection
- explainability
- confidence calibration
- human feedback integration
- iterative dataset improvement

That makes the project significantly stronger as a diploma work than a simple standalone classifier.

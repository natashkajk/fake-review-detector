# Fake Review Detector

Fake Review Detector is a diploma-scale system for detecting suspicious product reviews in the browser. It combines a Chrome extension, a FastAPI backend, a transformer-based classifier, explainability outputs, and a human-in-the-loop feedback pipeline for collecting cleaner retraining data.

## What This Project Includes

- Chrome extension for selecting and checking review text directly on web pages
- FastAPI backend for model inference and explainable responses
- Fine-tuned DistilBERT-based text classifier for `fake` vs `genuine` review detection
- Evidence extraction that highlights the most suspicious or most relevant text span
- Manual review dashboard for labeling model outputs
- SQLite storage for analyzed reviews and reviewer feedback
- Retraining pipeline for extending the dataset with confirmed examples

## System Architecture

The project is organized as a complete ML application, not only as a model:

1. User highlights review text in Chrome
2. Extension sends the text to the FastAPI backend
3. Backend runs model inference and explainability logic
4. Result is returned as verdict, confidence, evidence text, and explanation
5. Each analyzed review is stored in SQLite
6. Human reviewer can confirm or correct the label in the `/reviews` dashboard
7. Confirmed reviews can be exported and reused for retraining

Detailed architecture: [docs/ARCHITECTURE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/ARCHITECTURE.md)

## Project Structure

```text
fake-review-detector/
├── client/                         Chrome extension used in the current build
├── server/                         FastAPI backend, training, evaluation, calibration
│   ├── data/                       Datasets and prepared data files
│   ├── main.py                     Main API and review dashboard
│   ├── train.py                    Training pipeline
│   ├── evaluate.py                 Evaluation script
│   ├── prepare_external_dataset.py External dataset normalizer
│   ├── prepare_ott_dataset.py      Ott corpus converter
│   ├── calibrate_temperature.py    Confidence calibration
│   └── export_confirmed_reviews.py Export confirmed feedback to CSV
├── manifest.json                   Root manifest for loading the extension from repo root
├── QUICKSTART.md                   Setup notes
└── docs/                           Diploma-oriented documentation
```

## Core Features

- `Fake` / `Original` verdict for highlighted review text
- Explainable output with suspicious evidence text and explanation
- Confidence shown in user-friendly form such as `Moderate (67%)`
- Manual labeling workflow through [http://localhost:8000/reviews](http://localhost:8000/reviews)
- Export of confirmed labels into a retraining-ready CSV
- Ability to compare multiple trained models such as `model_mixed` and `model_final`

## Quick Start

### 1. Start the backend

```powershell
cd "C:\Users\37369\OneDrive\Рабочий стол\fake-review-detector\server"
.\venv\Scripts\python.exe main.py
```

Useful endpoints:

- [http://localhost:8000/health](http://localhost:8000/health)
- [http://localhost:8000/docs](http://localhost:8000/docs)
- [http://localhost:8000/reviews](http://localhost:8000/reviews)

### 2. Load the extension

1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click `Load unpacked`
4. Select the project root:
   `C:\Users\37369\OneDrive\Рабочий стол\fake-review-detector`

The root manifest is configured to use the UI from `client/`.

### 3. Test the system

1. Open a page with reviews
2. Highlight review text
3. Click `Check Review`
4. Open the popup and inspect:
   - verdict
   - confidence
   - evidence text
   - explanation

## Model Training and Data Preparation

### Supported training workflow

The training pipeline supports:

- multiple CSV datasets in a single run
- automatic text and label column detection
- label normalization from formats such as `CG/OR`, `fake/genuine`, `0/1`
- train/validation/test split
- weighted loss for class imbalance
- metrics: accuracy, precision, recall, F1

### Example training command

```powershell
cd "C:\Users\37369\OneDrive\Рабочий стол\fake-review-detector\server"
.\venv\Scripts\python.exe train.py --data_paths data\fake_reviews_prepared.csv data\clean_dataset.csv data\ott_prepared.csv data\hard_cases.csv --output_dir model_final --num_train_epochs 4 --max_length 256
```

### Dataset preparation commands

Prepare generic external CSV:

```powershell
.\venv\Scripts\python.exe prepare_external_dataset.py --input_path data\fake_reviews_dataset.csv --output_path data\fake_reviews_prepared.csv
```

Prepare Ott corpus:

```powershell
.\venv\Scripts\python.exe prepare_ott_dataset.py
```

Calibrate model confidence:

```powershell
.\venv\Scripts\python.exe calibrate_temperature.py --model_dir model_mixed --data_paths data\fake_reviews_prepared.csv data\clean_dataset.csv
```

Model lifecycle details: [docs/MODEL_LIFECYCLE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/MODEL_LIFECYCLE.md)

## Human-in-the-Loop Feedback

The project includes a feedback loop suitable for a diploma demonstration:

- every analyzed review is stored in `server/reviews.db`
- reviewer can inspect analyses at [http://localhost:8000/reviews](http://localhost:8000/reviews)
- reviewer can set `fake`, `genuine`, or `unknown`
- confirmed reviews can be exported into a new training dataset

Export confirmed feedback:

```powershell
cd "C:\Users\37369\OneDrive\Рабочий стол\fake-review-detector\server"
.\venv\Scripts\python.exe export_confirmed_reviews.py --output_path data\confirmed_reviews.csv
```

Or use the `Export Confirmed CSV` button in the `/reviews` dashboard.

## Current Experimental Results

The project already contains several model iterations.

| Model | Training Data | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `model_single` | `fake_reviews_prepared.csv` only | 0.9827 | 0.9750 | 0.9908 | 0.9828 |
| `model_mixed` | `fake_reviews_prepared + clean_dataset` | 0.9786 | 0.9653 | 0.9927 | 0.9788 |
| `model_final` | `fake_reviews_prepared + clean_dataset + ott + hard_cases` | 0.9780 | 0.9656 | 0.9913 | 0.9783 |

Interpretation:

- `model_single` performs best on its own held-out split but is more likely to overfit one dataset style
- `model_mixed` is currently the preferred deployment candidate
- `model_final` did not improve global F1, but its purpose was to reduce dataset bias

Full analysis: [docs/RESULTS_AND_ERROR_ANALYSIS.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/RESULTS_AND_ERROR_ANALYSIS.md)

## Explainability

The backend returns:

- `prediction`
- `confidence`
- `evidence_text`
- `evidence_label`
- `evidence_reason`
- `explanation`

This helps present the system as an explainable decision-support tool rather than a black-box classifier.

## Known Limitations

- The model is still sensitive to dataset bias
- Short reviews remain difficult to classify reliably
- Confidence is calibrated but still not a perfect real-world probability
- Explainability combines model output with heuristic evidence extraction
- Multilingual reviews may be handled less reliably than English reviews

These limitations are documented explicitly because they are important in an academic project.

## Documents for Diploma Presentation

- Architecture: [docs/ARCHITECTURE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/ARCHITECTURE.md)
- Model lifecycle: [docs/MODEL_LIFECYCLE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/MODEL_LIFECYCLE.md)
- Results and error analysis: [docs/RESULTS_AND_ERROR_ANALYSIS.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/RESULTS_AND_ERROR_ANALYSIS.md)
- Demo script: [docs/DEMO_SCRIPT.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/DEMO_SCRIPT.md)
- Suggested thesis chapter structure: [docs/THESIS_STRUCTURE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/THESIS_STRUCTURE.md)

## Recommended Final Demo Flow

1. Show the extension analyzing a highlighted review
2. Show explanation and evidence text in the popup
3. Open `/reviews` and demonstrate manual correction
4. Export confirmed reviews
5. Explain how those labels become new retraining data

## License

This project is intended for educational and diploma use.

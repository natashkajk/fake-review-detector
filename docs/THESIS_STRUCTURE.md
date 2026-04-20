# Suggested Thesis Structure

## 1. Introduction

- relevance of fake review detection
- impact on e-commerce trust and decision-making
- goal of the diploma project
- object and subject of research
- tasks of the work

## 2. Problem Analysis

- what fake reviews are
- why they are difficult to detect
- overview of NLP methods for review classification
- limitations of purely rule-based systems

## 3. Design Of The Proposed System

- system requirements
- architecture of the browser extension
- architecture of the FastAPI backend
- explainability and feedback subsystem
- choice of DistilBERT-based model

Reference:

- [docs/ARCHITECTURE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/ARCHITECTURE.md)

## 4. Data Preparation And Model Training

- source datasets
- normalization into unified format
- train/validation/test split
- training configuration
- calibration of confidence
- hard cases and bias correction

Reference:

- [docs/MODEL_LIFECYCLE.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/MODEL_LIFECYCLE.md)

## 5. Experimental Results

- metrics for all candidate models
- comparison table
- interpretation of precision, recall, and F1
- qualitative review examples
- error analysis

Reference:

- [docs/RESULTS_AND_ERROR_ANALYSIS.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/RESULTS_AND_ERROR_ANALYSIS.md)

## 6. Software Implementation

- extension implementation
- backend implementation
- SQLite storage and manual labeling interface
- export and retraining loop

## 7. Limitations And Future Work

- remaining bias
- multilingual limitations
- confidence calibration limitations
- possible use of larger or better domain-specific models
- larger manually validated datasets

## 8. Conclusion

- summary of the achieved result
- scientific and practical significance
- possible practical deployment

## Appendices

- screenshots of extension UI
- screenshots of `/reviews` dashboard
- API examples
- training commands
- demo script

Reference:

- [docs/DEMO_SCRIPT.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/DEMO_SCRIPT.md)

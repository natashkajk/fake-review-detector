# Demo Script

## Goal

This script helps present the system clearly during a diploma defense or project demo.

## Demo Scenario 1. Real Review

### What to show

1. Open a page with a realistic product review
2. Highlight the text
3. Click `Check Review`
4. Show:
   - verdict
   - confidence
   - evidence text
   - explanation

### What to say

This demonstrates the real-time integration between the browser extension and the FastAPI inference backend.

## Demo Scenario 2. Suspicious Review

### What to show

1. Highlight a more templated or suspicious review
2. Run analysis
3. Point to the suspicious evidence text and explanation

### What to say

This demonstrates that the system does not only return a class label, but also exposes a human-readable justification for the decision.

## Demo Scenario 3. Human Feedback Loop

### What to show

1. Open [http://localhost:8000/reviews](http://localhost:8000/reviews)
2. Show the stored analyzed reviews
3. Filter by prediction or mismatches
4. Set a manual label and save it
5. Click `Export Confirmed CSV`

### What to say

This demonstrates the human-in-the-loop design, where incorrect or uncertain model outputs can be reviewed and converted into cleaner retraining data.

## Demo Scenario 4. Research Value

### What to show

Open the following documents:

- [README.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/README.md)
- [docs/RESULTS_AND_ERROR_ANALYSIS.md](C:/Users/37369/OneDrive/Рабочий%20стол/fake-review-detector/docs/RESULTS_AND_ERROR_ANALYSIS.md)

### What to say

The project includes model comparison, bias analysis, confidence calibration, and data collection for iterative retraining, which makes it suitable as a diploma project rather than only a prototype extension.

## Recommended Defense Order

1. Briefly explain the problem of fake review detection
2. Show the system architecture
3. Run the extension demo on a real review
4. Run the extension demo on a suspicious review
5. Show the reviewer dashboard and feedback loop
6. Present metrics and limitations
7. Conclude with future improvements

## Short Defense Summary

If time is limited, use this one-paragraph summary:

The project is an end-to-end system for fake review detection in the browser. It combines a Chrome extension, a FastAPI backend, a transformer-based classifier, explainability output, confidence calibration, SQLite logging, and a human feedback pipeline that allows continuous improvement of the dataset and model.

# Results And Error Analysis

## Quantitative Results

Three main model variants were tested.

| Model | Dataset Mix | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `model_single` | `fake_reviews_prepared.csv` only | 0.9827 | 0.9750 | 0.9908 | 0.9828 |
| `model_mixed` | `fake_reviews_prepared + clean_dataset` | 0.9786 | 0.9653 | 0.9927 | 0.9788 |
| `model_final` | `fake_reviews_prepared + clean_dataset + ott + hard_cases` | 0.9780 | 0.9656 | 0.9913 | 0.9783 |

## Interpretation Of Results

### `model_single`

Strengths:

- highest F1 on its own held-out split
- strong precision and recall

Risks:

- may overfit one dataset style
- more likely to learn shortcuts tied to the source corpus

### `model_mixed`

Strengths:

- best compromise between generalization and strong metrics
- currently the preferred deployment candidate

Why it is important:

- mixed training reduces dependence on a single dataset style

### `model_final`

Purpose:

- reduce bias through Ott data and manually created hard cases

Observed outcome:

- global F1 did not improve
- practical behavior should still be checked qualitatively because aggregate metrics may not capture all bias changes

## Qualitative Findings

Manual testing showed several recurring issues:

- short reviews can be classified with very high confidence even when evidence is weak
- some generic but genuine reviews are flagged as fake
- evidence extraction may highlight stylistic fragments that are not truly suspicious
- multilingual reviews are more fragile than English ones

## Typical Error Categories

### False Positives

Examples of likely false positives:

- short genuine reviews with little detail
- gift-related reviews
- calm but generic real reviews

Why they happen:

- the model may associate shortness or generic wording with deception

### False Negatives

Examples of likely false negatives:

- fluent fake reviews with smooth, natural language
- longer fake reviews containing concrete details

Why they happen:

- the model may have learned some surface cues better than deeper deception patterns

## Biases Identified During Development

The following biases were observed and partially addressed:

- `long review = genuine`
- `caps = suspicious`
- `strong emotion = suspicious`
- `short review = fake`

To weaken these shortcuts, the project added:

- mixed-dataset training
- `hard_cases.csv`
- confidence calibration
- human feedback collection

## Why Manual Review Is Still Needed

Even with strong metrics, fake review detection remains noisy because:

- labels in public datasets are imperfect
- datasets may not fully reflect real marketplace reviews
- confidence is not equal to absolute truth

Therefore, the system is best presented as:

- an explainable review analysis assistant
- not a fully autonomous moderation engine

## Suggested Table For Thesis Text

You can reuse the following structure in the diploma:

| Criterion | Observation |
|---|---|
| Best raw F1 | `model_single` |
| Best deployment balance | `model_mixed` |
| Best research direction for bias control | `model_final` |
| Main unresolved issue | overconfidence on some short or generic reviews |
| Most important future improvement | more real-world labeled data with human confirmation |

## Main Conclusion

The project already demonstrates a technically valid and academically defensible pipeline. The strongest research conclusion is not that one model achieved perfect accuracy, but that the system evolved from a simple classifier into an explainable, feedback-aware review analysis platform.

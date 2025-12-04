# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Census Income Classifier
Version: 1.0.0
Developed by: Bronwyn Worwood

## Intended Use
For educational purposes to demonstrate scalable ML pipelines with FastAPI

## Training Data
Dataset: Census Income dataset
Size: ~32,500 rows, 15 features
Features: Age, education, nationality, income, hours per week, marital status, etc.

## Evaluation Data
Dataset: Held‑out portion of the UCI Census Income dataset (20% test split).
Size: ~6,400 rows.
Features: Same preprocessing pipeline as training (one‑hot encoding, normalization).
Rationale: Ensures evaluation reflects real‑world generalization by testing on unseen data.
Additional Checks: Stratified sampling to preserve class balance; fairness checks across gender and race subgroups.

## Metrics
Precision: 0.7478 - About 75% of the individuals predicted to earn >50K actually do.
Recall: 0.6359 - The model correctly identified 64% of individuals who absolutely earn >50k
F1 Score: 0.6873 - Moderate overall performance

## Ethical Considerations
There may be biases present in the census data. It is important to do a bias audit before any decision-making to make sure there is fairness.

## Caveats and Recommendations
To be able to watch for biases, it is important to retrain the model with new census data when available. These predictions shopuld be used as supporting information, not the sole information for decision-making.
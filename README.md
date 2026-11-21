# Binary classification: Spambase and Diabetes

We apply the same classification workflow to two datasets: spam detection and the binary Diabetes Health Indicators dataset. The notebooks compare classifiers, study class weighting, test feature selection and evaluate the selected models.

## Datasets

| Dataset | Samples | Features | Positive class |
| --- | ---: | ---: | ---: |
| Spambase | 4,601 | 57 | Spam, about 39.4% |
| Diabetes Health Indicators | 253,680 | 21 | `Diabetes_binary = 1`, about 13.9% |

Sources: [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) and [Diabetes Health Indicators on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

For Diabetes, we use `diabetes_binary_health_indicators_BRFSS2015.csv`, which keeps the original class imbalance. The balanced and three-class versions are not used.

## Repository contents

| Path | Contents |
| --- | --- |
| `ml_workflow.py` | Shared data loading, splitting, pipelines, evaluation, model comparison, feature selection and plotting functions |
| `spambase.ipynb` | Spambase experiments, results and comments |
| `diabetes.ipynb` | Diabetes experiments, results and comments |
| `test_ml_workflow.py` | Five unit tests for splitting, missing values and evaluation |
| `requirements.txt` | Python dependencies |
| `data/README.md` | Dataset information |
| `data/spambase/` | `spambase.data`, `spambase.names` and `spambase.DOCUMENTATION` |
| `data/diabetes/` | `diabetes_binary_health_indicators_BRFSS2015.csv` |

## Setup and execution

Clone the repository:

```bash
git clone https://github.com/Rayene365/spam-diabetes-ml.git
cd spam-diabetes-ml
```

From the repository root, create a virtual environment with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1` instead.

Run the unit tests:

```bash
python -m pytest -q
```

The five tests check:

- Class proportions, aligned labels and disjoint training/test indices.
- Reproducible splits with a fixed seed, and a different split when the seed changes.
- Training and prediction with missing feature values.
- Expected metric values when a classifier predicts no positive cases.
- Metric calculation using SVM decision scores, including ROC AUC and average precision.

These tests use small synthetic examples and do not require the datasets or rerun the notebook experiments.

Open the notebooks:

```bash
jupyter notebook
```

Select the virtual environment's Python kernel, then run each notebook from top to bottom. Both notebooks must remain beside `ml_workflow.py`, with the data files in the paths listed above. The Diabetes experiments take longer because the dataset is larger.

## Experimental workflow

1. Inspect dataset dimensions, missing values and class proportions.
2. Create a stratified 80% training / 20% test split. The random seed is **42 for Spambase** and **67 for Diabetes**.
3. Compare logistic regression, logistic regression with balanced class weights, Gaussian Naive Bayes, a linear SVM with balanced class weights, and an MLP with one hidden layer of 32 neurons.
4. Evaluate the models using five-fold stratified cross-validation on the training split. The folds are shuffled with the dataset's seed and preserve approximately the original class proportions.
5. Tune feature selection and regularization for the two balanced linear models. `SelectPercentile(f_classif)` ranks features using the ANOVA F-score. We test **25%, 50%, 75% and 100%** of the features with **C = 0.1, 1 and 10**: 12 configurations per classifier.
6. Select the configuration with the highest mean cross-validation F1-score, refit it on the full training split, and report its test metrics, confusion matrix, ROC curve and precision–recall curve.

Median imputation and standard scaling are included in each pipeline. Neither dataset has missing feature values in the supplied files. During cross-validation, preprocessing and feature selection are fitted separately on each training fold. Model and parameter choices use training cross-validation scores.

We report accuracy, balanced accuracy, precision, recall, F1, ROC AUC and average precision. F1 is the selection criterion; the other metrics help explain the results, particularly for the imbalanced Diabetes dataset.

## Final results

| Dataset | Selected classifier | C | Retained features |
| --- | --- | ---: | ---: |
| Spambase | Linear SVM, balanced class weights | 0.1 | 57 / 57 |
| Diabetes | Logistic regression, balanced class weights | 1.0 | 21 / 21 |

| Metric | Spambase CV mean | Spambase test | Diabetes CV mean | Diabetes test |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.928 | 0.923 | 0.733 | 0.727 |
| Balanced accuracy | 0.926 | 0.921 | 0.747 | 0.741 |
| Precision | 0.902 | 0.895 | 0.313 | 0.306 |
| Recall | 0.916 | 0.912 | 0.766 | 0.761 |
| F1 | 0.909 | 0.903 | 0.444 | 0.437 |
| ROC AUC | 0.970 | 0.969 | 0.824 | 0.818 |
| Average precision | 0.947 | 0.948 | 0.406 | 0.393 |

The CV values above come from the search used to select the configurations. The test columns report performance on the separate test split.

For **Spambase**, the selected SVM achieves high precision and recall. Its test confusion matrix contains 519 true negatives, 39 false positives, 32 false negatives and 331 true positives. Balanced logistic regression has almost the same CV F1, so the results do not establish a clear advantage for the SVM.

For **Diabetes**, balanced logistic regression detects about 76% of positive cases, but its precision is about 31%. There are 5,383 true positives, 1,686 false negatives and 12,187 false positives. Class weighting improves recall compared with the standard logistic regression baseline, at the cost of more false positives.

Keeping **all features** gives the highest mean CV F1 among the tested feature percentages for both classifiers on both datasets. Feature selection was therefore evaluated, but reducing the feature set did not improve F1 in this search.

The test and CV scores are close for both selected models. These results describe one fixed train/test split per dataset and a limited set of configurations; small score differences should not be interpreted as conclusive.

## AI assistance

ChatGPT was used to help with feature selection metrics and unit tests.

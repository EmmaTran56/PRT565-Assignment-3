# Step-by-step project instructions

## Project objective

Build a reproducible classification project that predicts whether a credit-card client will default on the following month's payment. The project must compare the required machine-learning and deep-learning approaches and explain which model is most useful.

This repository uses the **Default of Credit Card Clients** dataset. The current data file contains 30,000 clients and 25 columns, including the target `default payment next month`.

## Required model comparison

Implement at least these four models:

1. Decision Tree
2. Random Forest
3. Naive Bayes, preferably Gaussian Naive Bayes for the numeric/encoded feature matrix
4. Multilayer Perceptron (MLP), which is the required multilayer ANN

The current notebook already contains Decision Tree and Random Forest work. Add Naive Bayes and MLP experiments, then evaluate all four models using the same held-out test set. An optional fifth model, such as Logistic Regression or XGBoost, can be added if the group has enough time to explain it properly.

## Step 1: Create the working environment

From the repository root, create and activate a virtual environment:

```bash
cd "/Users/maxhoang/Desktop/IT Codefair/ASS3/PRT565-Assignment-3"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl joblib jupyter
```

Start Jupyter if required:

```bash
jupyter notebook
```

## Step 2: Inspect the repository and dataset

The main project files are:

- `default of credit card clients.xlsx`: raw dataset
- `eda_preprocess.py`: exploratory analysis and preprocessing script
- `ML_Assignment_3.ipynb`: notebook for the analysis and models
- `eda_outputs/`: saved EDA charts, statistics, and summaries
- `processed/`: train/test files and fitted preprocessing objects

The spreadsheet has a metadata/header row before the real column names. The project therefore reads it with `header=1`. Do not silently change this to `header=0` without checking the resulting columns.

## Step 3: Run exploratory data analysis and preprocessing

Run:

```bash
python eda_preprocess.py
```

Check that the script completes and regenerates the `eda_outputs/` and `processed/` directories. It should verify the following:

- 30,000 rows and 25 columns
- Unique IDs from 1 to 30,000
- No missing values after loading
- Target column `default payment next month`
- Target imbalance: 6,636 defaults (22.12%) and 23,364 non-defaults
- An 80/20 stratified train/test split with `random_state=42`

The script excludes `ID` from model predictors, maps categorical codes to readable labels, combines undocumented education/marriage codes into an `Others` category for modelling, creates `utilization`, `avg_pay`, and `inactive` features, one-hot encodes categorical variables, applies `log1p` to selected skewed variables, and fits the scaler on the training data only.

## Step 4: Review the EDA findings

Use the saved charts and `eda_outputs/eda_summary.txt` to explain the data before modelling. Important findings to discuss include:

- The target is imbalanced, so accuracy alone is not sufficient.
- Recent repayment status, especially `PAY_0`, is a strong predictor of default.
- Lower credit limits are associated with higher default rates.
- Defaulters tend to have lower payment amounts.
- Bill amounts are highly correlated with one another.
- Some education and marriage codes are undocumented and need a transparent treatment.
- Some bill amounts are negative; these should be interpreted as credit/overpayment values rather than automatically treated as data errors.

Do not describe categorical codes such as `SEX`, `EDUCATION`, `MARRIAGE`, or `PAY_*` as continuous measurements without explaining the encoding and its limitations.

## Step 5: Confirm the train/test design

Use a single fixed test set for the final comparison. The split must be stratified by the target so the default proportion is similar in training and test data.

Important leakage controls:

- Fit the encoder on training rows only.
- Fit the scaler on training rows only.
- Do not use the test labels to select hyperparameters.
- Do not report cross-validation scores as final test performance.
- Do not include `ID` as a predictive feature.

If the notebook is run in Google Colab, update the old `/content/drive/MyDrive/ML_Ass_2` paths to the actual project path or mount location. For this local repository, prefer relative paths based on the notebook's project directory.

## Step 6: Train the Decision Tree

Use a `DecisionTreeClassifier` with a fixed `random_state`. Tune a small, defensible parameter grid using stratified cross-validation. Useful parameters include:

- `max_depth`
- `min_samples_leaf`
- `criterion`
- `class_weight="balanced"`

Use F1-score or another explicitly justified metric for model selection because the positive class is the default class and is the minority class. Record the best parameters and the cross-validation score, then evaluate the selected model once on the held-out test set.

## Step 7: Train the Random Forest

Use a `RandomForestClassifier` with a fixed `random_state`. Tune a reproducible search over:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `max_features`
- `class_weight="balanced"`

Save the selected model and plot the most important features. Explain that feature importance indicates predictive usefulness in this fitted model; it does not prove causation.

## Step 8: Train Naive Bayes

Use `GaussianNB` on the encoded feature matrix. Naive Bayes is a useful baseline because it is simple and fast, but its conditional-independence and distribution assumptions may not fit correlated credit and payment variables well.

For a fair comparison:

- Train it on the same training rows.
- Predict the same test rows.
- Generate probabilities using `predict_proba` for ROC-AUC.
- Record all evaluation metrics using the shared evaluation function.

## Step 9: Train the multilayer ANN/MLP

Use `MLPClassifier` from scikit-learn, or an approved deep-learning framework if the group is prepared to document it. The easiest reproducible implementation is `MLPClassifier` with two or more hidden layers, for example:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,
    random_state=42,
)
```

The MLP requires scaled numeric inputs. Use `train_scaled.csv` and `test_scaled.csv`, or reproduce the same train-only preprocessing in a pipeline. Compare a small number of architectures rather than trying many unexplained variations. Record convergence information and mention any early stopping or validation split used.

## Step 10: Evaluate every model consistently

For every model, calculate at least:

- Accuracy
- Precision for the default class
- Recall for the default class
- F1-score for the default class
- ROC-AUC

Also produce:

- A confusion matrix
- A classification report
- A shared ROC curve
- A bar chart comparing the metrics

A suitable results table has this structure:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree |  |  |  |  |  |
| Random Forest |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |
| MLP/ANN |  |  |  |  |  |

Interpret the trade-off. For example, a model with higher recall identifies more potential defaulters, while a model with higher precision produces fewer false alarms among clients flagged as likely defaulters. Select the preferred model based on the project objective, not automatically on accuracy.

## Step 11: Add model interpretation

Include a short interpretation section that connects the results to the EDA:

- Which repayment-status variables mattered most?
- Did the models agree about the importance of credit limit and payment variables?
- Which model produced the best recall or F1-score?
- What are the likely consequences of false positives and false negatives?
- What limitations arise from the dataset, historical context, class imbalance, and model assumptions?

Avoid claiming that the model causes default or that a feature is a direct cause of default.

## Step 12: Update the notebook for reproducibility

Organise `ML_Assignment_3.ipynb` into clear sections:

1. Project title, group details, and objective
2. Imports and configuration
3. Data loading
4. Dataset description and data quality
5. EDA and visualisations
6. Cleaning, feature engineering, and preprocessing
7. Train/test split
8. Decision Tree
9. Random Forest
10. Naive Bayes
11. MLP/ANN
12. Metric comparison and interpretation
13. Conclusion and limitations
14. References

Set `random_state=42` consistently where randomness is used. Run all cells from top to bottom in a fresh kernel and confirm that the notebook runs without manual intervention.

## Step 13: Prepare the report

Use this report order:

1. Cover page with unit, group, members, and campus
2. Project title, motivation, brief description, and public recording link
3. Introduction and problem definition
4. Dataset source and data dictionary
5. EDA findings
6. Preprocessing and feature engineering
7. Model methods and experimental design
8. Results table, confusion matrices, and ROC comparison
9. Interpretation, limitations, and ethical considerations
10. Conclusion
11. APA references

Ensure that every figure has a number, title, and short explanation in the text. Cite the dataset, research papers, software documentation, and any online sources used.

## Step 14: Prepare the presentation

For a maximum 15-minute group presentation, aim for approximately 10-14 focused slides:

1. Title and group information
2. Problem and motivation
3. Dataset and target variable
4. Key EDA findings
5. Preprocessing and feature engineering
6. Required algorithms
7. Experimental design and metrics
8. Results comparison
9. Model interpretation
10. Limitations and conclusion
11. References, if needed

Assign each member a clear section. Rehearse until the full presentation is within 15 minutes and each member stays within six minutes. Speak clearly, maintain an appropriate pace, and ensure that charts and metric values can be read on screen.

## Step 15: Final validation and submission

Before creating the ZIP file, complete this checklist:

- [ ] The report has a complete cover page.
- [ ] The recording link appears on page 2.
- [ ] The recording link works in a private/logged-out browser.
- [ ] The report and `.pptx` presentation are included.
- [ ] Decision Tree, Random Forest, Naive Bayes, and MLP/ANN results are included.
- [ ] All models use the same held-out test set.
- [ ] Accuracy, precision, recall, F1-score, and ROC-AUC are reported.
- [ ] Confusion matrices and a model comparison chart are readable.
- [ ] The notebook runs from a fresh kernel.
- [ ] APA references are complete and consistent.
- [ ] Any extension evidence is included if applicable.
- [ ] The ZIP opens correctly and contains the intended final files.
- [ ] Every group member submits the same ZIP before 22:00 Sydney time on 27 September 2026.

## Current repository status

The repository provides a strong EDA and preprocessing foundation. The original notebook contains Decision Tree and Random Forest tuning/evaluation, while `ML_Assignment_3_Final.ipynb` is the clean submission-ready workbook. It runs the authoritative EDA/preprocessing script and compares all four required models: Decision Tree, Random Forest, Gaussian Naive Bayes, and a multilayer MLP ANN. The remaining assessment work is to prepare the report, presentation, recording, and APA reference list.

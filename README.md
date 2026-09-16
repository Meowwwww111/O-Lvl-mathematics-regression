# O-Level Mathematics Score Prediction

This project predicts students' O-level mathematics examination scores so that a school can identify students who may need support before the examination. The repository follows the same modular structure as the `AIAP_HDB_Price_Predictor` project: configuration is stored in YAML, data cleaning is separated from model training, and `main.py` runs the complete pipeline.

## Objective

The task is a supervised **regression** problem. The target variable is `final_test` (0–100). Three suitable models are evaluated:

1. **Ridge Regression** – a regularised linear baseline that is stable, fast, and relatively interpretable.
2. **Random Forest Regressor** – captures non-linear relationships and feature interactions without assuming a linear relationship.
3. **Gradient Boosting Regressor** – builds trees sequentially to reduce previous errors and is strong on structured/tabular data.

Each candidate is tuned using 5-fold cross-validation on the training set, then compared on the same untouched validation set. **Validation RMSE is the primary selection metric**; MAE and R² provide supporting evidence. Because the school also wants to identify weaker students, the pipeline also reports weak-student precision, recall, and F1 using the configurable threshold in `src/config.yaml` (default: predicted score below 50).

## Verified result on the provided dataset

The complete pipeline was executed against `data/regression_bonus_practice_data.csv` in GitHub Actions after the code was committed. Cleaning removed **139 exact duplicate rows** and **494 rows with missing/invalid target scores**, leaving **15,267 labelled rows**. The data was then split into **10,686 training**, **2,290 validation**, and **2,291 test** rows.

| Model | CV RMSE | Validation MAE | Validation RMSE | Validation R² | Weak precision | Weak recall | Weak F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **5.3745** | **3.7623** | **5.5744** | **0.8374** | 0.8832 | **0.7297** | **0.7992** |
| Gradient Boosting | 6.2308 | 4.7240 | 6.3908 | 0.7863 | **0.9314** | 0.6293 | 0.7512 |
| Ridge Regression | 9.3414 | 7.6805 | 9.5744 | 0.5203 | 0.7009 | 0.2896 | 0.4098 |

### Most suitable model: Random Forest Regressor

**Random Forest is the most suitable of the three evaluated models for this task.** It has the lowest validation RMSE and MAE, the highest validation R², and the highest weak-student recall. This means it predicts examination scores more accurately overall while also identifying a larger proportion of students below the configured intervention threshold.

Gradient Boosting has slightly higher weak-student precision (0.9314 versus 0.8832), so when it flags a student it is somewhat less likely to be a false alarm. However, its lower weak-student recall (0.6293 versus 0.7297) means it misses more genuinely weak students. Since the stated school objective is to identify weaker students before the examination, Random Forest provides the stronger balance between score accuracy and intervention recall.

Ridge Regression is useful as a transparent linear benchmark, but its validation RMSE is substantially higher and its weak-student recall is only 0.2896. This indicates that the relationship between the available student features and final score contains important non-linearities/interactions that a linear model does not capture as effectively.

The selected Random Forest hyperparameters were:

```text
n_estimators = 250
max_depth = 12
max_features = 0.8
min_samples_leaf = 1
```

After model selection, Random Forest was refitted on the combined training + validation data and evaluated once on the untouched test set:

| Final test metric | Value |
|---|---:|
| MAE | **3.5796** |
| RMSE | **5.2133** |
| R² | **0.8609** |
| Weak precision | **0.8594** |
| Weak recall | **0.7782** |
| Weak F1 | **0.8168** |

The held-out results support the validation conclusion: on average the selected model is off by about **3.58 marks**, and its RMSE is about **5.21 marks**. It explains about **86.1%** of the variance in the held-out examination scores and identifies about **77.8%** of students below the configured score threshold.

## Repository structure

```text
O-Lvl-mathematics-regression/
├── .github/
│   └── workflows/
│       └── train-and-evaluate.yml
├── data/
│   └── regression_bonus_practice_data.csv
├── src/
│   ├── config.yaml
│   ├── data_preparation.py
│   └── model_training.py
├── eda.ipynb
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated files are written to `models/` and `reports/` and are intentionally ignored by Git. The GitHub Actions workflow also uploads the evaluation reports as a workflow artifact.

## Data preparation

The raw dataset contains a mixture of numerical, categorical, identifier, and time fields. The pipeline performs the following preparation before training:

- removes the exported `index` column and exact duplicate rows;
- keeps `student_id` only for reporting and never supplies it to a model;
- excludes `bag_color` because it is not a meaningful or actionable academic predictor and may introduce spurious correlations;
- standardises inconsistent values such as `Yes`/`Y`, `No`/`N`, and category capitalisation such as `ARTS`/`Arts`;
- converts numerical columns to numeric values and converts clearly impossible values (for example an O-level student's age outside 13–20) to missing values;
- drops rows where the target score is missing or outside 0–100;
- derives sleep duration and cyclical sine/cosine time features from sleep and wake times;
- creates `class_size = n_male + n_female`;
- imputes missing numerical values with the **training-set median**;
- imputes missing categorical values with the **training-set mode**;
- standardises numerical inputs and one-hot encodes categorical inputs.

Imputation, scaling, and encoding live inside a scikit-learn `Pipeline`, so they are fitted only on training data. This prevents validation/test information from leaking into model preparation.

## Model comparison methodology

The cleaned labelled data is split approximately **70% training / 15% validation / 15% test** using a fixed random seed. Grid search and 5-fold cross-validation happen only inside the training portion. The best tuned version of each model is then scored on the validation portion.

The model with the **lowest validation RMSE** is selected. RMSE is expressed in score points and penalises large errors more strongly than MAE, which is useful when a large prediction error could cause a genuinely weak student to be missed. MAE gives the average absolute error in marks, while R² measures the proportion of score variation explained by the model. Weak-student recall measures the proportion of truly below-threshold students that are successfully flagged.

Only after model selection is the winning model refitted using training + validation data and evaluated once on the untouched test set. This prevents choosing a model based on its test-set performance.

## How to run

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the complete experiment:

```bash
python main.py
```

The run creates:

```text
reports/model_comparison.csv      # all 3 validation results + best hyperparameters
reports/final_test_metrics.json   # final winner + held-out test metrics
reports/test_predictions.csv      # student IDs, actual/predicted scores, weak-student flags
reports/feature_importance.csv    # Random Forest feature importance
models/best_model.joblib          # complete fitted preprocessing + regression pipeline
```

The repository also includes `eda.ipynb` for inspecting missing values, inconsistent categories, the target distribution, descriptive statistics, and numerical correlations.

## Notes for interpretation

A model can be accurate overall but still miss weaker students, so regression metrics and weak-student recall should be considered together. The threshold of 50 is a project setting, **not an assertion about an official O-level pass mark**; change `weak_student_threshold` in `src/config.yaml` to match the school's intervention policy.

The dataset includes attributes such as gender. Before real-world school deployment, performance should be checked across relevant groups for systematic error differences, and predictions should support educators rather than serve as the sole basis for high-stakes decisions.

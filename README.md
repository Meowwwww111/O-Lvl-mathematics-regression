# O-Level Mathematics Score Prediction

This project predicts students' O-level mathematics examination scores so that a school can identify students who may need support before the examination. The repository follows the same modular structure as the `AIAP_HDB_Price_Predictor` project: configuration is stored in YAML, data cleaning is separated from model training, and `main.py` runs the complete pipeline.

## Objective

The task is a supervised **regression** problem. The target variable is `final_test` (0–100). Three suitable models are evaluated:

1. **Ridge Regression** – a regularised linear baseline that is stable, fast, and relatively interpretable.
2. **Random Forest Regressor** – captures non-linear relationships and feature interactions without assuming a linear relationship.
3. **Gradient Boosting Regressor** – builds trees sequentially to reduce previous errors and is often strong on structured/tabular data.

The models are not chosen by intuition alone. Each candidate is tuned using 5-fold cross-validation on the training set, then compared on the same untouched validation set. **Validation RMSE is the primary selection metric**; MAE and R² provide supporting evidence. Because the school also wants to identify weaker students, the pipeline reports weak-student precision, recall, and F1 using the configurable threshold in `src/config.yaml` (default: predicted score below 50).

## Repository structure

```text
O-Lvl-mathematics-regression/
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

Generated files are written to `models/` and `reports/` and are intentionally ignored by Git.

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
- imputes missing numerical values with the training-set median;
- imputes missing categorical values with the training-set mode;
- standardises numerical inputs and one-hot encodes categorical inputs.

Imputation, scaling, and encoding live inside a scikit-learn `Pipeline`, so they are fitted only on training data. This avoids leakage from the validation or test sets.

## Model comparison and selection

The cleaned labelled data is split approximately **70% training / 15% validation / 15% test** using a fixed random seed. Grid search and 5-fold cross-validation happen only inside the training portion. The best tuned version of each model is then scored on the validation portion.

The model with the **lowest validation RMSE** is selected as the most suitable score estimator. If two models are very close, MAE, R², weak-student recall, model complexity, and interpretability can be considered before deployment. After selection, the winning model is refitted on training + validation data and evaluated once on the untouched test set.

Why RMSE is the primary metric: it is expressed in score points and penalises large prediction errors more strongly than MAE. That matters here because a large error can cause a genuinely weak student to be missed. MAE remains useful because it answers the intuitive question, “on average, how many marks is the model off by?” R² shows how much score variation is explained by the model. Weak-student recall answers a separate operational question: “of the students actually below the chosen threshold, how many did the model flag?”

The code deliberately does **not** hard-code a winner before seeing the data. When `python main.py` is run, it prints the comparison table, selects the model with the lowest validation RMSE, and saves the exact results. This makes the conclusion evidence-based and reproducible rather than assuming that one algorithm must always be best.

## How to run

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the complete experiment from the repository root:

```bash
python main.py
```

The run creates:

```text
reports/model_comparison.csv      # all 3 validation results + best hyperparameters
reports/final_test_metrics.json   # final winner + held-out test metrics
reports/test_predictions.csv      # student IDs, actual/predicted scores, weak-student flags
reports/feature_importance.csv    # feature importance / absolute coefficients
models/best_model.joblib          # complete fitted preprocessing + regression pipeline
```

`reports/model_comparison.csv` is the table to use when explaining which model is more suitable. Choose the first row (lowest validation RMSE), quote its RMSE/MAE/R², compare those values with the other two models, and also discuss weak-student recall because that directly relates to the school's intervention objective.

## Notes for interpretation

A model can be accurate overall but still miss weaker students, so regression metrics and weak-student recall should be considered together. The threshold of 50 is a project setting, **not an assertion about an official O-level pass mark**; change `weak_student_threshold` in `src/config.yaml` to match the school's intervention policy.

The dataset includes attributes such as gender. Before real-world school deployment, performance should be checked across relevant groups for systematic error differences, and predictions should be used to support educators rather than as the sole basis for high-stakes decisions.

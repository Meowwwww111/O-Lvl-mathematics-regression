"""Model training, tuning, comparison, and reporting for student score regression."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


class ModelTraining:
    """Train and compare three suitable regression algorithms fairly."""

    def __init__(self, config: dict[str, Any], preprocessor: Any):
        self.config = config
        self.preprocessor = preprocessor
        self.random_state = int(config.get("random_state", 42))

    def split_data(self, df: pd.DataFrame):
        """Create 70/15/15 train/validation/test partitions by default."""
        target = self.config["target_column"]
        X = df.drop(columns=[target])
        y = df[target]

        test_size = float(self.config.get("test_size", 0.15))
        validation_size = float(self.config.get("validation_size", 0.15))
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
        )
        relative_validation_size = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_validation_size,
            random_state=self.random_state,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _model_specs(self):
        return {
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                random_state=self.random_state,
                loss="squared_error",
            ),
        }

    def _regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        predictions = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
        threshold = float(self.config.get("weak_student_threshold", 50.0))
        actual_weak = np.asarray(y_true) < threshold
        predicted_weak = predictions < threshold
        return {
            "MAE": float(mean_absolute_error(y_true, predictions)),
            "RMSE": float(root_mean_squared_error(y_true, predictions)),
            "R2": float(r2_score(y_true, predictions)),
            "Weak_Precision": float(
                precision_score(actual_weak, predicted_weak, zero_division=0)
            ),
            "Weak_Recall": float(recall_score(actual_weak, predicted_weak, zero_division=0)),
            "Weak_F1": float(f1_score(actual_weak, predicted_weak, zero_division=0)),
        }

    def train_and_compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """Tune all three models on training folds, then compare on validation data."""
        cv = KFold(
            n_splits=int(self.config.get("cv_folds", 5)),
            shuffle=True,
            random_state=self.random_state,
        )
        tuned_models: dict[str, Pipeline] = {}
        rows: list[dict[str, Any]] = []

        for model_name, regressor in self._model_specs().items():
            LOGGER.info("Tuning %s...", model_name)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(self.preprocessor)),
                    ("regressor", regressor),
                ]
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.config["model_grids"][model_name],
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            tuned_models[model_name] = search.best_estimator_

            val_pred = search.best_estimator_.predict(X_val)
            metrics = self._regression_metrics(y_val, val_pred)
            rows.append(
                {
                    "Model": model_name,
                    "CV_RMSE": float(-search.best_score_),
                    "Validation_MAE": metrics["MAE"],
                    "Validation_RMSE": metrics["RMSE"],
                    "Validation_R2": metrics["R2"],
                    "Weak_Precision": metrics["Weak_Precision"],
                    "Weak_Recall": metrics["Weak_Recall"],
                    "Weak_F1": metrics["Weak_F1"],
                    "Best_Params": json.dumps(search.best_params_, sort_keys=True),
                }
            )

        comparison = pd.DataFrame(rows).sort_values(
            by=["Validation_RMSE", "Validation_MAE"], ascending=[True, True]
        ).reset_index(drop=True)
        best_name = str(comparison.loc[0, "Model"])
        LOGGER.info("Selected model: %s", best_name)
        return tuned_models, comparison, best_name

    def refit_and_evaluate_best(
        self,
        best_model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """Refit the chosen pipeline on train+validation and evaluate once on test."""
        final_model = clone(best_model)
        X_development = pd.concat([X_train, X_val], axis=0)
        y_development = pd.concat([y_train, y_val], axis=0)
        final_model.fit(X_development, y_development)

        raw_predictions = final_model.predict(X_test)
        predictions = np.clip(np.asarray(raw_predictions, dtype=float), 0, 100)
        test_metrics = self._regression_metrics(y_test, predictions)
        return final_model, test_metrics, predictions

    def build_prediction_report(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Create an auditable held-out report, sorted from lowest predicted score."""
        threshold = float(self.config.get("weak_student_threshold", 50.0))
        id_column = self.config.get("id_column", "student_id")
        if id_column in X_test.columns:
            identifiers = X_test[id_column].astype("string").reset_index(drop=True)
        else:
            identifiers = pd.Series(X_test.index.astype(str), name=id_column)

        report = pd.DataFrame(
            {
                id_column: identifiers,
                "actual_score": y_test.reset_index(drop=True),
                "predicted_score": np.asarray(predictions),
            }
        )
        report["absolute_error"] = (
            report["actual_score"] - report["predicted_score"]
        ).abs()
        report["actual_weak"] = report["actual_score"] < threshold
        report["predicted_weak"] = report["predicted_score"] < threshold
        return report.sort_values("predicted_score").reset_index(drop=True)

    @staticmethod
    def save_outputs(
        comparison: pd.DataFrame,
        final_model: Pipeline,
        test_metrics: dict[str, float],
        predictions: pd.DataFrame,
        best_name: str,
    ) -> None:
        """Persist model and evaluation artefacts for reproducibility."""
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        comparison.to_csv("reports/model_comparison.csv", index=False)
        predictions.to_csv("reports/test_predictions.csv", index=False)
        joblib.dump(final_model, "models/best_model.joblib")
        with open("reports/final_test_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"best_model": best_name, **test_metrics},
                handle,
                indent=2,
            )

        preprocessor = final_model.named_steps["preprocessor"]
        regressor = final_model.named_steps["regressor"]
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(regressor, "feature_importances_"):
            importance = np.asarray(regressor.feature_importances_, dtype=float)
        elif hasattr(regressor, "coef_"):
            importance = np.abs(np.ravel(regressor.coef_)).astype(float)
        else:
            importance = np.array([])
        if len(importance) == len(feature_names):
            feature_table = pd.DataFrame(
                {"feature": feature_names, "importance": importance}
            ).sort_values("importance", ascending=False)
            feature_table.to_csv("reports/feature_importance.csv", index=False)

"""Data cleaning, feature engineering, and preprocessing for student scores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


class DataPreparation:
    """Clean raw student records and build a leakage-safe sklearn preprocessor."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

    @staticmethod
    def _normalise_yes_no(series: pd.Series) -> pd.Series:
        mapping = {
            "yes": "Yes",
            "y": "Yes",
            "true": "Yes",
            "1": "Yes",
            "no": "No",
            "n": "No",
            "false": "No",
            "0": "No",
        }
        text = series.astype("string").str.strip().str.lower()
        cleaned = text.map(mapping)
        return cleaned.astype(object).where(cleaned.notna(), np.nan)

    @staticmethod
    def _normalise_category(series: pd.Series, title_case: bool = True) -> pd.Series:
        text = series.astype("string").str.strip()
        text = text.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        cleaned = text.str.title() if title_case else text
        return cleaned.astype(object).where(cleaned.notna(), np.nan)

    @staticmethod
    def _time_to_minutes(value: object) -> float:
        if pd.isna(value):
            return np.nan
        try:
            hour_text, minute_text = str(value).strip().split(":")[:2]
            hour, minute = int(hour_text), int(minute_text)
            if not 0 <= hour <= 23 or not 0 <= minute <= 59:
                return np.nan
            return float(hour * 60 + minute)
        except (TypeError, ValueError, AttributeError):
            return np.nan

    @staticmethod
    def _replace_outside_range(series: pd.Series, lower: float, upper: float) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.where(numeric.between(lower, upper))

    def clean_data(self, df: pd.DataFrame, drop_missing_target: bool = True) -> pd.DataFrame:
        """Return a cleaned dataframe while preserving ``student_id`` for reporting."""
        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]

        # Remove the exported row-number column before duplicate detection.
        if "index" in self.config.get("drop_columns", []) and "index" in df.columns:
            df = df.drop(columns=["index"])

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        LOGGER.info("Removed %d exact duplicate rows.", before - len(df))

        for column in ("direct_admission", "tuition"):
            if column in df.columns:
                df[column] = self._normalise_yes_no(df[column])

        for column in ("CCA", "learning_style", "gender"):
            if column in df.columns:
                df[column] = self._normalise_category(df[column])

        if "mode_of_transport" in df.columns:
            df["mode_of_transport"] = self._normalise_category(df["mode_of_transport"])

        if self.config.get("id_column") in df.columns:
            df[self.config["id_column"]] = (
                df[self.config["id_column"]].astype("string").str.strip()
            )

        # Coerce numeric columns first, then turn clearly impossible values into missing.
        ranges = {
            "number_of_siblings": (0, 20),
            "n_male": (0, 60),
            "n_female": (0, 60),
            "age": (13, 20),
            "hours_per_week": (0, 80),
            "attendance_rate": (0, 100),
            self.config["target_column"]: (0, 100),
        }
        for column, (lower, upper) in ranges.items():
            if column in df.columns:
                df[column] = self._replace_outside_range(df[column], lower, upper)

        if {"n_male", "n_female"}.issubset(df.columns):
            df["class_size"] = df["n_male"] + df["n_female"]

        sleep_minutes = (
            df["sleep_time"].map(self._time_to_minutes)
            if "sleep_time" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        wake_minutes = (
            df["wake_time"].map(self._time_to_minutes)
            if "wake_time" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        # Cyclical encoding avoids treating 23:30 and 00:30 as far apart.
        df["sleep_time_sin"] = np.sin(2 * np.pi * sleep_minutes / 1440)
        df["sleep_time_cos"] = np.cos(2 * np.pi * sleep_minutes / 1440)
        df["wake_time_sin"] = np.sin(2 * np.pi * wake_minutes / 1440)
        df["wake_time_cos"] = np.cos(2 * np.pi * wake_minutes / 1440)

        duration = (wake_minutes - sleep_minutes) % 1440
        duration_hours = duration / 60.0
        df["sleep_duration_hours"] = duration_hours.where(duration_hours.between(2, 14))

        df = df.drop(columns=["sleep_time", "wake_time"], errors="ignore")
        df = df.drop(columns=self.config.get("drop_columns", []), errors="ignore")

        target = self.config["target_column"]
        if drop_missing_target and target in df.columns:
            missing_targets = int(df[target].isna().sum())
            if missing_targets:
                LOGGER.info("Dropping %d rows with missing/invalid target values.", missing_targets)
                df = df.dropna(subset=[target]).reset_index(drop=True)

        LOGGER.info("Cleaned dataset shape: %s", df.shape)
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """Create train-only imputation/encoding pipelines to prevent leakage."""
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["numeric_features"]),
                ("cat", categorical_transformer, self.config["categorical_features"]),
            ],
            remainder="drop",
        )

    def load_and_clean(self) -> pd.DataFrame:
        """Load the configured CSV and clean it for model development."""
        path = Path(self.config["data_path"])
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
        return self.clean_data(pd.read_csv(path), drop_missing_target=True)

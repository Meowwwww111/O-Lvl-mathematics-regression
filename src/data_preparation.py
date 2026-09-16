"""Data cleaning and feature engineering for the O-level mathematics task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


class DataPreparation:
    """Clean the raw student dataset without leaking target information."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def _normalise_yes_no(series: pd.Series) -> pd.Series:
        mapping = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "no": "No", "n": "No", "false": "No", "0": "No"}
        return series.astype("string").str.strip().str.lower().map(mapping).fillna(series)

    @staticmethod
    def _normalise_text(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip()

    @staticmethod
    def _time_to_minutes(value: object) -> float:
        if pd.isna(value):
            return float("nan")
        try:
            hour, minute = str(value).strip().split(":")[:2]
            return int(hour) * 60 + int(minute)
        except (ValueError, AttributeError):
            return float("nan")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned dataframe ready for model training."""
        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]
        df = df.drop_duplicates().reset_index(drop=True)

        for column in ["direct_admission", "tuition"]:
            if column in df:
                df[column] = self._normalise_yes_no(df[column])

        if "CCA" in df:
            df["CCA"] = self._normalise_text(df["CCA"]).str.lower()
            df["CCA"] = df["CCA"].replace({"arts": "Arts", "none": "None", "nan": pd.NA})

        for column in ["learning_style", "gender", "mode_of_transport", "bag_color"]:
            if column in df:
                df[column] = self._normalise_text(df[column])

        for column in ["number_of_siblings", "n_male", "n_female", "age", "hours_per_week", "attendance_rate", "final_test"]:
            if column in df:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if "sleep_time" in df:
            df["sleep_minutes"] = df["sleep_time"].map(self._time_to_minutes)
        if "wake_time" in df:
            df["wake_minutes"] = df["wake_time"].map(self._time_to_minutes)
        if "sleep_minutes" in df and "wake_minutes" in df:
            duration = df["wake_minutes"] - df["sleep_minutes"]
            duration = duration.where(duration >= 0, duration + 24 * 60)
            df["sleep_duration_hours"] = duration / 60.0

        df = df.drop(columns=["sleep_time", "wake_time"], errors="ignore")
        df = df.drop(columns=self.config.get("drop_columns", []), errors="ignore")

        LOGGER.info("Cleaned dataset shape: %s", df.shape)
        LOGGER.info("Missing target rows: %d", df[self.config["target_column"]].isna().sum())
        return df

    def load_and_clean(self) -> pd.DataFrame:
        path = Path(self.config["data_path"])
        return self.clean_data(pd.read_csv(path))

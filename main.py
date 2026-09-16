"""Train and compare O-level mathematics score regression models."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


def load_config(path: str = "src/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config()

    preparation = DataPreparation(config)
    data = preparation.load_and_clean()
    trainer = ModelTraining(config, preparation.preprocessor)

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(data)
    logging.info(
        "Rows -> train: %d | validation: %d | test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    models, comparison, best_name = trainer.train_and_compare_models(
        X_train, y_train, X_val, y_val
    )
    print("\nModel comparison (lower RMSE/MAE is better; higher R2 is better):")
    print(comparison.to_string(index=False))

    final_model, test_metrics, test_predictions = trainer.refit_and_evaluate_best(
        models[best_name],
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )
    report = trainer.build_prediction_report(X_test, y_test, test_predictions)
    trainer.save_outputs(
        comparison,
        final_model,
        test_metrics,
        report,
        best_name,
    )

    threshold = config.get("weak_student_threshold", 50.0)
    print(f"\nSelected model: {best_name}")
    print("Final held-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(
        f"\nStudents with predicted score below {threshold:g} are flagged in "
        "reports/test_predictions.csv."
    )
    print("Saved trained pipeline to models/best_model.joblib.")


if __name__ == "__main__":
    if not Path("src/config.yaml").exists():
        raise SystemExit("Run this script from the repository root: python main.py")
    main()

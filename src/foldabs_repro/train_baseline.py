from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV

from foldabs_repro.data import (
    DatasetSchema,
    fit_label_normalizer,
    load_fitness_csv,
    split_by_group,
)
from foldabs_repro.features import PositionOneHotEncoder
from foldabs_repro.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_from_project(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _prediction_frame(split_name: str, frame: pd.DataFrame, predictions) -> pd.DataFrame:
    output = frame[["sample_id", "heavy", "light", "label"]].copy()
    output.insert(1, "split", split_name)
    output["prediction"] = predictions
    return output


def run(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_config = config["data"]
    schema = DatasetSchema(
        heavy_column=data_config["heavy_column"],
        light_column=data_config["light_column"],
        label_column=data_config["label_column"],
    )
    frame = load_fitness_csv(
        _resolve_from_project(data_config["path"]),
        schema=schema,
        duplicate_reducer=data_config.get("duplicate_reducer", "median"),
    )

    split_config = config["split"]
    splits = split_by_group(
        frame,
        group_column=split_config["group_column"],
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
        seed=int(config["seed"]),
    )

    encoder = PositionOneHotEncoder()
    x_train = encoder.fit_transform(splits.train)
    x_validation = encoder.transform(splits.validation)
    x_test = encoder.transform(splits.test)

    normalizer = fit_label_normalizer(splits.train["label"].to_numpy())
    y_train = normalizer.transform(splits.train["label"].to_numpy())

    model = RidgeCV(alphas=config["model"]["alphas"])
    model.fit(x_train, y_train)

    predictions = {}
    metrics = {}
    for split_name, split_frame, features in (
        ("train", splits.train, x_train),
        ("validation", splits.validation, x_validation),
        ("test", splits.test, x_test),
    ):
        split_predictions = normalizer.inverse_transform(model.predict(features))
        predictions[split_name] = split_predictions
        metrics[split_name] = regression_metrics(
            split_frame["label"].to_numpy(), split_predictions
        )

    output_dir = _resolve_from_project(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment_name": config["experiment_name"],
        "seed": int(config["seed"]),
        "dataset_rows_after_deduplication": int(len(frame)),
        "split_sizes": {
            "train": int(len(splits.train)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "feature_dimension": int(encoder.output_dim),
        "selected_alpha": float(model.alpha_),
        "label_normalizer": {"mean": normalizer.mean, "std": normalizer.std},
        "metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    prediction_frames = [
        _prediction_frame("train", splits.train, predictions["train"]),
        _prediction_frame(
            "validation", splits.validation, predictions["validation"]
        ),
        _prediction_frame("test", splits.test, predictions["test"]),
    ]
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "normalizer": normalizer,
            "config": config,
        },
        output_dir / "model.joblib",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CPU one-hot baseline for a FLAb developability dataset."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


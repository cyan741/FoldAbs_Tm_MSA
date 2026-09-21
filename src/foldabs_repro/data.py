from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ACCEPTED_AA = STANDARD_AA | frozenset("XBZUOJ")


@dataclass(frozen=True)
class DatasetSchema:
    heavy_column: str = "heavy"
    light_column: str = "light"
    label_column: str = "fitness"


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class LabelNormalizer:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


def _clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).upper()


def _validate_sequence(sequence: str, column: str, row_number: int) -> None:
    invalid = sorted(set(sequence) - ACCEPTED_AA)
    if invalid:
        raise ValueError(
            f"Invalid amino-acid characters {invalid} in {column}, row {row_number}."
        )


def load_fitness_csv(
    path: str | Path,
    schema: DatasetSchema,
    duplicate_reducer: str = "median",
) -> pd.DataFrame:
    """Load a FLAb-style CSV and return canonical heavy/light/label columns.

    中文提醒：这是训练时的第二道防御性 H/L 去重；它不做 Fv 截取。对于当前
    ``foldabs_public_dsf_tm.csv``，构建阶段已完成一次相同规则的去重，因此这一步
    应不再改变行数。若未来引入全长序列，必须先独立完成并审计 ANARCI Fv 截取，
    不能误以为本函数的清洗或去重替代了该步骤。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {schema.heavy_column, schema.label_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(frame.columns)}")

    selected_columns: dict[str, object] = {
        "heavy": frame[schema.heavy_column].map(_clean_sequence),
        "light": (
            frame[schema.light_column].map(_clean_sequence)
            if schema.light_column in frame.columns
            else ""
        ),
        "label": pd.to_numeric(frame[schema.label_column], errors="coerce"),
    }
    # ``source`` does not enter the model.  It is retained when supplied so
    # diagnostics can still report which public assay source a prediction came
    # from.  Older generic CSVs need not contain it.
    if "source" in frame.columns:
        selected_columns["source"] = frame["source"].fillna("").astype(str)
    selected = pd.DataFrame(selected_columns)
    selected = selected.dropna(subset=["label"]).reset_index(drop=True)
    selected = selected[selected["heavy"].str.len() > 0].reset_index(drop=True)

    for row_number, row in selected.iterrows():
        _validate_sequence(row["heavy"], "heavy", row_number)
        _validate_sequence(row["light"], "light", row_number)

    if duplicate_reducer != "median":
        raise ValueError("Only duplicate_reducer='median' is currently supported.")

    # 完全相同 H/L 对的重复实验以标签中位数合并；重链相同但轻链不同不合并。
    # 若重复记录来自多个公开来源，保留一个稳定、可审计的来源集合；这只是
    # 诊断元数据，既不参与标签汇总，也不输入模型。
    groups = selected.groupby(["heavy", "light"], as_index=False, dropna=False)
    if "source" in selected.columns:
        selected = groups.agg(
            label=("label", "median"),
            source=(
                "source",
                lambda values: "|".join(
                    sorted({value for value in values.astype(str) if value})
                )
                or "unknown",
            ),
        )
    else:
        selected = groups["label"].median()
    selected = selected.sort_values(["heavy", "light"], kind="stable").reset_index(drop=True)
    selected.insert(0, "sample_id", [f"sample_{i:05d}" for i in range(len(selected))])
    return selected


def split_by_group(
    frame: pd.DataFrame,
    group_column: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplits:
    fractions = np.array(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.0.")
    if group_column not in frame.columns:
        raise ValueError(f"Unknown split group column: {group_column}")
    if frame[group_column].nunique() < 3:
        raise ValueError("At least three unique groups are required for train/val/test.")

    first_split = GroupShuffleSplit(
        n_splits=1, train_size=train_fraction, random_state=seed
    )
    train_idx, remainder_idx = next(
        first_split.split(frame, groups=frame[group_column])
    )
    remainder = frame.iloc[remainder_idx]

    validation_share = validation_fraction / (validation_fraction + test_fraction)
    second_split = GroupShuffleSplit(
        n_splits=1, train_size=validation_share, random_state=seed + 1
    )
    validation_local_idx, test_local_idx = next(
        second_split.split(remainder, groups=remainder[group_column])
    )

    splits = DatasetSplits(
        train=frame.iloc[train_idx].reset_index(drop=True),
        validation=remainder.iloc[validation_local_idx].reset_index(drop=True),
        test=remainder.iloc[test_local_idx].reset_index(drop=True),
    )
    _assert_group_disjoint(splits, group_column)
    return splits


def _assert_group_disjoint(splits: DatasetSplits, group_column: str) -> None:
    train_groups = set(splits.train[group_column])
    validation_groups = set(splits.validation[group_column])
    test_groups = set(splits.test[group_column])
    if train_groups & validation_groups:
        raise AssertionError("Train and validation groups overlap.")
    if train_groups & test_groups:
        raise AssertionError("Train and test groups overlap.")
    if validation_groups & test_groups:
        raise AssertionError("Validation and test groups overlap.")


def fit_label_normalizer(train_labels: np.ndarray) -> LabelNormalizer:
    values = np.asarray(train_labels, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training labels must have non-zero finite standard deviation.")
    return LabelNormalizer(mean=mean, std=std)

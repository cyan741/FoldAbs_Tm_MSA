from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DevelopabilityTaskSpec:
    """Published label protocol for one FoldAbS developability property."""

    name: str
    paper_rows: int
    preprocessing: str
    model_label_unit: str
    raw_label_unit: str

    def inverse_preprocessing(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.preprocessing == "identity":
            return values
        if self.preprocessing == "log1p":
            return np.expm1(values)
        raise ValueError(f"Unsupported preprocessing rule: {self.preprocessing}")


# Appendix B.3 / Tables S4-S5.  Keeping all four tasks here prevents the next
# head from silently inheriting thermostability-specific units or row counts.
DEVELOPABILITY_TASKS = {
    "aggregation": DevelopabilityTaskSpec(
        name="aggregation",
        paper_rows=249,
        preprocessing="log1p",
        model_label_unit="log1p(% non-monomer)",
        raw_label_unit="% non-monomer",
    ),
    "expression": DevelopabilityTaskSpec(
        name="expression",
        paper_rows=773,
        preprocessing="identity",
        model_label_unit="mg/L",
        raw_label_unit="mg/L",
    ),
    "immunogenicity": DevelopabilityTaskSpec(
        name="immunogenicity",
        paper_rows=212,
        preprocessing="log1p",
        model_label_unit="log1p(%ADA)",
        raw_label_unit="%ADA",
    ),
    "thermostability": DevelopabilityTaskSpec(
        name="thermostability",
        paper_rows=430,
        preprocessing="identity",
        model_label_unit="degrees Celsius",
        raw_label_unit="degrees Celsius",
    ),
}


def get_developability_task(name: str) -> DevelopabilityTaskSpec:
    try:
        return DEVELOPABILITY_TASKS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown developability property {name!r}; choose from {sorted(DEVELOPABILITY_TASKS)}"
        ) from error

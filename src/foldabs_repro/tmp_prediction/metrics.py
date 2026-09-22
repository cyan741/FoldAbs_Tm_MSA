from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels, scores) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "auc_0_1": float("nan")}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "auc_0_1": float(roc_auc_score(y, p, max_fpr=0.1)),
    }


def macro_pmhc_metrics(frame: pd.DataFrame) -> dict[str, object]:
    required = {"label", "prediction", "pmhc_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    per_pmhc = {}
    for name, group in frame.groupby("pmhc_id", sort=True):
        if group["label"].nunique() < 2:
            continue
        per_pmhc[name] = {"n": int(len(group)), **binary_metrics(group.label, group.prediction)}
    if not per_pmhc:
        return {"eligible_pmhc": 0, "macro_auc_0_1": float("nan"), "per_pmhc": {}}
    return {
        "eligible_pmhc": len(per_pmhc),
        "macro_auc_0_1": float(np.mean([row["auc_0_1"] for row in per_pmhc.values()])),
        "macro_auroc": float(np.mean([row["auroc"] for row in per_pmhc.values()])),
        "macro_auprc": float(np.mean([row["auprc"] for row in per_pmhc.values()])),
        "per_pmhc": per_pmhc,
    }

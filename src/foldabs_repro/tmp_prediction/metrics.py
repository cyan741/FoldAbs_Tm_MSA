from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels, scores) -> dict[str, float]:
    """Return full ROC AUC, average precision and standardized FPR<=0.1 AUC."""
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
    """Average equally across pMHCs with both labels; omit per-pMHC details."""
    score_column = "logit" if "logit" in frame.columns else "prediction"
    required = {"label", score_column, "pmhc_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    group_metrics = []
    for _, group in frame.groupby("pmhc_id", sort=True):
        if group["label"].nunique() < 2:
            continue
        group_metrics.append(binary_metrics(group.label, group[score_column]))
    return {
        "eligible_pmhc": len(group_metrics),
        **{
            name: float(np.mean([row[key] for row in group_metrics]))
            if group_metrics else float("nan")
            for name, key in (("macro_auc_0_1", "auc_0_1"),
                              ("macro_auroc", "auroc"), ("macro_auprc", "auprc"))
        },
    }

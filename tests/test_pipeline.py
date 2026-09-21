from __future__ import annotations

import numpy as np
import pandas as pd

from foldabs_repro.data import fit_label_normalizer, split_by_group
from foldabs_repro.features import AA_ALPHABET, PositionOneHotEncoder


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append(
            {
                "sample_id": f"sample_{index:05d}",
                "heavy": "ACDEFGHIK" + AA_ALPHABET[index % len(AA_ALPHABET)],
                "light": "LMNPQRSTV",
                "label": float(index),
            }
        )
    return pd.DataFrame(rows)


def test_group_split_has_no_heavy_chain_leakage() -> None:
    splits = split_by_group(
        _synthetic_frame(),
        group_column="heavy",
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )
    assert set(splits.train.heavy).isdisjoint(splits.validation.heavy)
    assert set(splits.train.heavy).isdisjoint(splits.test.heavy)
    assert set(splits.validation.heavy).isdisjoint(splits.test.heavy)


def test_position_encoder_shape_and_label_normalizer() -> None:
    frame = _synthetic_frame()
    encoder = PositionOneHotEncoder()
    encoded = encoder.fit_transform(frame)
    assert encoded.shape == (len(frame), encoder.output_dim)
    assert np.all(encoded.sum(axis=1) > 0)

    normalizer = fit_label_normalizer(frame.label.to_numpy())
    transformed = normalizer.transform(frame.label.to_numpy())
    assert np.isclose(transformed.mean(), 0.0)
    assert np.isclose(transformed.std(), 1.0)
    assert np.allclose(normalizer.inverse_transform(transformed), frame.label)


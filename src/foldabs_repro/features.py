from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


@dataclass
class PositionOneHotEncoder:
    """Fixed-position one-hot encoder used as a transparent CPU baseline."""

    max_heavy_length: int | None = None
    max_light_length: int | None = None

    def fit(self, frame: pd.DataFrame) -> "PositionOneHotEncoder":
        if frame.empty:
            raise ValueError("Cannot fit an encoder on an empty frame.")
        self.max_heavy_length = int(frame["heavy"].str.len().max())
        self.max_light_length = int(frame["light"].str.len().max())
        return self

    @property
    def output_dim(self) -> int:
        self._check_fitted()
        return (self.max_heavy_length + self.max_light_length) * len(AA_ALPHABET)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        heavy = self._encode_chain(frame["heavy"], self.max_heavy_length)
        light = self._encode_chain(frame["light"], self.max_light_length)
        return np.concatenate([heavy, light], axis=1)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _encode_chain(self, sequences: pd.Series, max_length: int) -> np.ndarray:
        channel = {aa: index for index, aa in enumerate(AA_ALPHABET)}
        unknown_index = channel["X"]
        encoded = np.zeros(
            (len(sequences), max_length, len(AA_ALPHABET)), dtype=np.float32
        )
        for row_index, sequence in enumerate(sequences):
            for position, aa in enumerate(sequence[:max_length]):
                encoded[row_index, position, channel.get(aa, unknown_index)] = 1.0
        return encoded.reshape(len(sequences), -1)

    def _check_fitted(self) -> None:
        if self.max_heavy_length is None or self.max_light_length is None:
            raise RuntimeError("Encoder must be fit before transform.")


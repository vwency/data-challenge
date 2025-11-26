from __future__ import annotations

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class TrainingDataSplitter:
    def __init__(self, test_size: float, random_state: int):
        self._test_size = test_size
        self._random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X,
            y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )

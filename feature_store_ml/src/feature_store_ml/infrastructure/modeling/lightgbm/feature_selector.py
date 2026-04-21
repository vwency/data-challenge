from __future__ import annotations

import pandas as pd

from feature_store_ml.domain.registry.features import FeatureRegistry


class FeatureSelector:
    def __init__(self, feature_registry: FeatureRegistry):
        self._feature_registry = feature_registry

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_registry.feature_names

    def select_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing_features = set(self.feature_names) - set(dataframe.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        return dataframe[list(self.feature_names)]

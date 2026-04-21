from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_store_ml.infrastructure.modeling.lightgbm.feature_selector import (
    FeatureSelector,
)
from feature_store_ml.infrastructure.modeling.lightgbm.probability_converter import (
    ProbabilityConverter,
)


class ModelPredictor:
    def __init__(
        self,
        model: lgb.Booster,
        feature_selector: FeatureSelector,
        probability_converter: ProbabilityConverter,
    ):
        self._model = model
        self._feature_selector = feature_selector
        self._probability_converter = probability_converter

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._feature_selector.select_features(X)
        predictions = self._model.predict(X_prepared)
        return self._probability_converter.convert(predictions)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_store_ml.domain.models.dataset import FeatureDataset
from feature_store_ml.domain.models.model_artifact import ModelArtifact
from feature_store_ml.domain.registry.features import FeatureRegistry
from feature_store_ml.infrastructure.config import TrainingConfig
from feature_store_ml.infrastructure.modeling.lightgbm.data_splitter import (
    TrainingDataSplitter,
)
from feature_store_ml.infrastructure.modeling.lightgbm.builder import (
    LightGBMModelBuilder,
)
from feature_store_ml.infrastructure.modeling.lightgbm.evaluator import ModelEvaluator
from feature_store_ml.infrastructure.modeling.lightgbm.persistence import (
    ModelPersistence,
)
from feature_store_ml.infrastructure.modeling.lightgbm.metadata import (
    TrainingMetadataBuilder,
)
from feature_store_ml.infrastructure.modeling.lightgbm.feature_selector import (
    FeatureSelector,
)
from feature_store_ml.infrastructure.modeling.lightgbm.probability_converter import (
    ProbabilityConverter,
)
from feature_store_ml.infrastructure.modeling.lightgbm.predictor import ModelPredictor


class LightGBMTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        artifacts_dir: Path,
        feature_registry: FeatureRegistry,
    ) -> None:
        self._config = config
        self._feature_registry = feature_registry
        self._persistence = ModelPersistence(artifacts_dir)
        self._splitter = TrainingDataSplitter(config.test_size, config.random_state)
        self._builder = LightGBMModelBuilder(
            config.learning_rate, config.max_depth, config.num_boost_round
        )
        self._evaluator = ModelEvaluator()
        self._feature_selector = FeatureSelector(feature_registry)
        self._probability_converter = ProbabilityConverter()
        self._predictor = None

    def train(self, dataset: FeatureDataset) -> ModelArtifact:
        X = self._feature_selector.select_features(dataset.features)
        y = dataset.labels

        X_train, X_test, y_train, y_test = self._splitter.split(X, y)

        model = self._builder.train(X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)
        auc = self._evaluator.calculate_auc(y_test, predictions)

        metadata = TrainingMetadataBuilder.build(
            auc=auc,
            feature_names=self._feature_selector.feature_names,
            num_train=len(X_train),
            num_test=len(X_test),
        )

        self._persistence.save_model(model)
        self._persistence.save_metadata(metadata)

        self._predictor = ModelPredictor(
            model, self._feature_selector, self._probability_converter
        )

        return ModelArtifact(
            model_path=self._persistence.model_path,
            feature_names=self._feature_selector.feature_names,
        )

    def load(self) -> Tuple[lgb.Booster, dict]:
        model = self._persistence.load_model()
        metadata = self._persistence.load_metadata()

        self._predictor = ModelPredictor(
            model, self._feature_selector, self._probability_converter
        )

        return model, metadata

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        return self._predictor.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        return self._predictor.predict(X, threshold)

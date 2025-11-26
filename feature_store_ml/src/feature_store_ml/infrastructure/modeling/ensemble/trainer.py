from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from loguru import logger

from feature_store_ml.domain.models.dataset import FeatureDataset
from feature_store_ml.domain.models.model_artifact import ModelArtifact
from feature_store_ml.domain.registry.features import FeatureRegistry
from feature_store_ml.infrastructure.config import TrainingConfig
from feature_store_ml.infrastructure.modeling.ensemble.data_splitter import (
    TrainingDataSplitter,
)
from feature_store_ml.infrastructure.modeling.ensemble.xboost_builder import (
    XGBoostModelBuilder,
)
from feature_store_ml.infrastructure.modeling.ensemble.catboost_builder import (
    CatBoostModelBuilder,
)
from feature_store_ml.infrastructure.modeling.ensemble.stacking_ensemble import (
    StackingEnsemble,
)
from feature_store_ml.infrastructure.modeling.ensemble.evaluator import (
    ModelEvaluator,
)
from feature_store_ml.infrastructure.modeling.ensemble.model_persistent import (
    ModelPersistence,
)
from feature_store_ml.infrastructure.modeling.ensemble.metadata import (
    TrainingMetadataBuilder,
)


class AdvancedEnsembleTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        artifacts_dir: Path,
        feature_registry: FeatureRegistry,
    ) -> None:
        self._config = config
        self._persistence = ModelPersistence(artifacts_dir)
        self._splitter = TrainingDataSplitter(config.test_size, config.random_state)
        self._xgb_builder = XGBoostModelBuilder(
            config.learning_rate,
            config.max_depth,
            config.num_boost_round,
            config.random_state,
        )
        self._catboost_builder = CatBoostModelBuilder(
            config.learning_rate,
            config.max_depth,
            config.num_boost_round,
            config.random_state,
        )
        self._evaluator = ModelEvaluator()
        self._ensemble = None
        self._feature_registry = feature_registry or FeatureRegistry.default()

    def train(self, dataset: FeatureDataset) -> ModelArtifact:
        feature_names = self._feature_registry.feature_names
        X = dataset.features[list(feature_names)]
        y = dataset.labels

        class_counts = y.value_counts().to_dict()
        logger.info(f"Training data class distribution: {class_counts}")
        logger.info(f"Label 0 (clean): {class_counts.get(0, 0)} samples")
        logger.info(f"Label 1 (artifact): {class_counts.get(1, 0)} samples")

        X_train, X_test, y_train, y_test = self._splitter.split(X, y)

        logger.info("Training XGBoost model...")
        xgb_model = self._xgb_builder.train(X_train, y_train, X_test, y_test)

        logger.info("Training CatBoost model...")
        catboost_model = self._catboost_builder.train(X_train, y_train, X_test, y_test)

        logger.info("Building stacking ensemble...")
        self._ensemble = StackingEnsemble(self._config.random_state)
        self._ensemble.add_model(xgb_model)
        self._ensemble.add_model(catboost_model)

        logger.info("Training meta-model...")
        self._ensemble.fit_meta_model(X_train, y_train)

        predictions = self._ensemble.predict_proba(X_test)
        auc = self._evaluator.calculate_auc(y_test, predictions)

        metadata = TrainingMetadataBuilder.build(
            auc=auc,
            feature_names=feature_names,
            num_train=len(X_train),
            num_test=len(X_test),
            model_type="XGBoost+CatBoost Stacking Ensemble",
            class_distribution=class_counts,
        )

        self._persistence.save_model(self._ensemble)
        self._persistence.save_metadata(metadata)

        logger.info(f"Ensemble AUC: {auc:.4f}")

        return ModelArtifact(
            model_path=self._persistence.model_path,
            feature_names=feature_names,
        )

    def load(self) -> Tuple[StackingEnsemble, dict]:
        model = self._persistence.load_model()
        metadata = self._persistence.load_metadata()
        self._ensemble = model
        return model, metadata

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._ensemble is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        return self._ensemble.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        if self._ensemble is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        return self._ensemble.predict(X, threshold)

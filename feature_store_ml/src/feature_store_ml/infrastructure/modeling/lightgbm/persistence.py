from __future__ import annotations
from pathlib import Path

import joblib
import lightgbm as lgb


class ModelPersistence:
    def __init__(self, artifacts_dir: Path):
        self._artifacts_dir = artifacts_dir
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self._artifacts_dir / "lightgbm_model.pkl"

    @property
    def metadata_path(self) -> Path:
        return self._artifacts_dir / "metadata.pkl"

    def save_model(self, model: lgb.Booster) -> None:
        joblib.dump(model, self.model_path)

    def save_metadata(self, metadata: dict) -> None:
        joblib.dump(metadata, self.metadata_path)

    def load_model(self) -> lgb.Booster:
        return joblib.load(self.model_path)

    def load_metadata(self) -> dict:
        return joblib.load(self.metadata_path)

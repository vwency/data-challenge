from __future__ import annotations

from typing import List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb


class StackingEnsemble:
    def __init__(self, random_state: int):
        self._models: List[Any] = []
        self._meta_model = LogisticRegression(random_state=random_state, max_iter=1000)
        self._random_state = random_state

    def add_model(self, model: Any) -> None:
        self._models.append(model)

    def _get_oof_predictions(
        self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5
    ) -> np.ndarray:
        oof_preds = np.zeros((len(X), len(self._models)))
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self._random_state
        )

        for model_idx, model in enumerate(self._models):
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_fold_train = X.iloc[train_idx]
                y_fold_train = y.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]

                model_clone = self._clone_model(model)
                model_clone.fit(X_fold_train, y_fold_train, verbose=False)

                oof_preds[val_idx, model_idx] = model_clone.predict_proba(X_fold_val)[
                    :, 1
                ]

        return oof_preds

    def _clone_model(self, model: Any) -> Any:
        if isinstance(model, xgb.XGBClassifier):
            return xgb.XGBClassifier(**model.get_params())
        elif isinstance(model, cb.CatBoostClassifier):
            return cb.CatBoostClassifier(**model.get_params())
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

    def fit_meta_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        oof_predictions = self._get_oof_predictions(X, y)
        self._meta_model.fit(oof_predictions, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_predictions = np.column_stack(
            [model.predict_proba(X)[:, 1] for model in self._models]
        )
        return self._meta_model.predict_proba(base_predictions)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

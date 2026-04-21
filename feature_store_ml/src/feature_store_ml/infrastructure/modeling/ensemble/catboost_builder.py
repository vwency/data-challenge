from __future__ import annotations

import pandas as pd
import catboost as cb


class CatBoostModelBuilder:
    def __init__(
        self, learning_rate: float, max_depth: int, n_estimators: int, random_state: int
    ):
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._n_estimators = n_estimators
        self._random_state = random_state

    def build_params(self) -> dict:
        return {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": self._learning_rate,
            "depth": self._max_depth,
            "iterations": self._n_estimators,
            "random_state": self._random_state,
            "verbose": False,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> cb.CatBoostClassifier:
        model = cb.CatBoostClassifier(**self.build_params())
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )
        return model

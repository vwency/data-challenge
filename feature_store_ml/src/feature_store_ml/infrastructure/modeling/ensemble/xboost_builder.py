from __future__ import annotations

import pandas as pd
import xgboost as xgb


class XGBoostModelBuilder:
    def __init__(
        self, learning_rate: float, max_depth: int, n_estimators: int, random_state: int
    ):
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._n_estimators = n_estimators
        self._random_state = random_state

    def build_params(self) -> dict:
        return {
            "objective": "binary:logistic",
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "n_estimators": self._n_estimators,
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": self._random_state,
            "verbosity": 0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(**self.build_params())
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return model

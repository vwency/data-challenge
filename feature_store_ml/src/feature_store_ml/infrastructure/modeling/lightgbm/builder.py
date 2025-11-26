from __future__ import annotations

import lightgbm as lgb
import pandas as pd


class LightGBMModelBuilder:
    def __init__(self, learning_rate: float, max_depth: int, num_boost_round: int):
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._num_boost_round = num_boost_round

    def build_params(self) -> dict:
        return {
            "objective": "binary",
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "metric": "auc",
            "verbosity": -1,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

        booster = lgb.train(
            params=self.build_params(),
            train_set=train_set,
            num_boost_round=self._num_boost_round,
            valid_sets=[valid_set],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        return booster

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ModelEvaluator:
    @staticmethod
    def calculate_auc(y_true: pd.Series, y_pred: np.ndarray) -> float:
        return roc_auc_score(y_true, y_pred)

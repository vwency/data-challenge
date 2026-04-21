from __future__ import annotations

import numpy as np


class ProbabilityConverter:
    @staticmethod
    def convert(predictions: np.ndarray) -> np.ndarray:
        if np.all((predictions >= 0) & (predictions <= 1)):
            return predictions
        return 1 / (1 + np.exp(-predictions))

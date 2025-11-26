from __future__ import annotations


class TrainingMetadataBuilder:
    @staticmethod
    def build(
        auc: float,
        feature_names: tuple[str, ...],
        num_train: int,
        num_test: int,
    ) -> dict:
        return {
            "auc": auc,
            "features": list(feature_names),
            "num_train": num_train,
            "num_test": num_test,
        }

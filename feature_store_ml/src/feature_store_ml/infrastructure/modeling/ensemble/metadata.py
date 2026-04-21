from __future__ import annotations


class TrainingMetadataBuilder:
    @staticmethod
    def build(
        auc: float,
        feature_names: tuple[str, ...],
        num_train: int,
        num_test: int,
        model_type: str,
        class_distribution: dict,
    ) -> dict:
        return {
            "auc": auc,
            "features": feature_names,
            "num_train": num_train,
            "num_test": num_test,
            "model_type": model_type,
            "class_distribution": class_distribution,
        }

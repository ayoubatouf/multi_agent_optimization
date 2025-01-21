from typing import Any
from xgboost import XGBClassifier  # type: ignore
from sklearn.base import BaseEstimator, ClassifierMixin


class SklearnXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs: Any) -> None:
        try:
            if "gpu_id" not in kwargs or kwargs["gpu_id"] == -1:
                kwargs["tree_method"] = "hist"
            else:
                kwargs["tree_method"] = "gpu_hist"
                kwargs["gpu_id"] = 0
        except Exception as e:
            print(f"Error during initialization: {e}")
        self.model = XGBClassifier(**kwargs)

    def fit(self, X: Any, y: Any) -> "SklearnXGBClassifier":
        try:
            self.model.fit(X, y)
        except Exception as e:
            print(f"{e}")
            raise
        return self

    def predict(self, X: Any) -> Any:
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X: Any) -> Any:
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            print(f"Error during predicting probabilities: {e}")
            raise

    def score(self, X: Any, y: Any) -> float:
        try:
            return self.model.score(X, y)
        except Exception as e:
            print(f"Error during scoring: {e}")
            raise

    def set_params(self, **params: Any) -> "SklearnXGBClassifier":
        try:
            self.model.set_params(**params)
        except Exception as e:
            print(f"Error setting parameters: {e}")
            raise
        return self

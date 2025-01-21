from typing import Dict, Optional, Any
from src.parameter_space import ParameterSpace
from src.performance_monitor import PerformanceMonitor
from src.result_sharer import ResultSharer
from src.search_space_updater import SearchSpaceUpdater
from src.sklearn_xgbc_classifier import SklearnXGBClassifier
from skopt import BayesSearchCV  # type: ignore


class Agent:
    def __init__(
        self,
        strategy: str,
        iterations: int,
        param_space: ParameterSpace,
        performance_monitor: PerformanceMonitor,
        result_sharer: ResultSharer,
        search_space_updater: SearchSpaceUpdater,
        max_iterations: int = 50,
    ) -> None:
        self.strategy = strategy
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.model: Optional[BayesSearchCV] = None
        self.best_score: float = -float("inf")
        self.best_params: Optional[Dict[str, Any]] = None
        self.param_space = param_space
        self.performance_monitor = performance_monitor
        self.result_sharer = result_sharer
        self.search_space_updater = search_space_updater
        self.resources_used = 0

    def run(
        self, X_train: Any, y_train: Any, shared_knowledge: Dict[str, Any]
    ) -> float:
        try:
            print(f"Running agent with strategy: {self.strategy}")

            model = BayesSearchCV(
                SklearnXGBClassifier(random_state=42),
                search_spaces=self.param_space.param_space,
                n_iter=self.iterations,
                cv=5,
                n_jobs=1,
            )

            model.fit(X_train, y_train)

            self.model = model
            self.best_score = model.best_score_
            self.best_params = model.best_params_

            self.result_sharer.share(self, shared_knowledge)
            self.search_space_updater.update(self)
            self.performance_monitor.monitor(self, shared_knowledge)

            return self.best_score
        except Exception as e:
            print(f"{self.strategy}: {e}")
            return -float("inf")

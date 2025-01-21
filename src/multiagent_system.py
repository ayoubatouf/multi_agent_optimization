from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.agent import Agent
from src.parameter_space import ParameterSpace
from src.performance_monitor import PerformanceMonitor
from src.result_sharer import ResultSharer
from src.search_space_updater import SearchSpaceUpdater
from skopt.space import Real, Integer  # type: ignore
from joblib import Parallel, delayed


class MultiAgentSystem:
    def __init__(self, param_space: dict = None) -> None:
        self.strategies = [
            "explore",
            "exploit",
            "refine",
            "aggressive_explore",
            "contextual",
            "fine_tune",
            "robust_explore",
        ]
        self.strategy_iterations = {
            "explore": 10,
            "exploit": 7,
            "refine": 5,
            "aggressive_explore": 15,
            "contextual": 10,
            "fine_tune": 3,
            "robust_explore": 8,
        }
        self.shared_knowledge = {"best_score": -float("inf"), "best_params": None}

        if param_space is None:
            param_space = {
                "learning_rate": Real(0.01, 0.3, prior="uniform"),
                "max_depth": Integer(3, 10),
                "n_estimators": Integer(50, 1000),
                "subsample": Real(0.6, 1.0, prior="uniform"),
                "colsample_bytree": Real(0.3, 1.0, prior="uniform"),
                "gamma": Real(0.0, 10.0, prior="uniform"),
                "reg_alpha": Real(0.0, 1.0, prior="uniform"),
                "reg_lambda": Real(0.0, 1.0, prior="uniform"),
                "scale_pos_weight": Integer(1, 10),
                "min_child_weight": Integer(1, 10),
            }

        self.param_space = ParameterSpace(param_space)

        self.performance_monitor = PerformanceMonitor()
        self.result_sharer = ResultSharer()
        self.search_space_updater = SearchSpaceUpdater(self.shared_knowledge)

    def run(self, X: Any, y: Any) -> None:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        agents = [
            Agent(
                strategy,
                self.strategy_iterations[strategy],
                self.param_space,
                self.performance_monitor,
                self.result_sharer,
                self.search_space_updater,
            )
            for strategy in self.strategies
        ]

        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(agent.run)(X_train, y_train, self.shared_knowledge)
            for agent in agents
        )

        for agent in agents:
            print(f"Agent {agent.strategy} Reward: {agent.best_score}")

        best_agent = max(agents, key=lambda agent: agent.best_score)
        print(f"Best agent strategy: {best_agent.strategy}")
        print(f"Best hyperparameters: {best_agent.best_params}")
        print(f"Best reward: {best_agent.best_score}")

        best_model = best_agent.model.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test set accuracy: {accuracy:.4f}")

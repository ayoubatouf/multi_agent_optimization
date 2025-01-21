from typing import Dict, Any
from skopt.space import Real, Integer  # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent import Agent


class SearchSpaceUpdater:
    def __init__(self, shared_knowledge: Dict[str, Any]) -> None:
        self.shared_knowledge = shared_knowledge

    def update(self, agent: "Agent") -> None:
        try:
            if self.shared_knowledge["best_params"] is not None:
                colsample_bytree_low = max(
                    0.3, self.shared_knowledge["best_params"]["colsample_bytree"] * 0.9
                )
                colsample_bytree_high = min(
                    1.0, self.shared_knowledge["best_params"]["colsample_bytree"] * 1.1
                )

                agent.param_space.param_space = {
                    "learning_rate": Real(
                        self.shared_knowledge["best_params"]["learning_rate"] * 0.9,
                        self.shared_knowledge["best_params"]["learning_rate"] * 1.1,
                        prior="uniform",
                    ),
                    "max_depth": Integer(
                        max(3, self.shared_knowledge["best_params"]["max_depth"] - 1),
                        min(10, self.shared_knowledge["best_params"]["max_depth"] + 1),
                    ),
                    "n_estimators": Integer(
                        max(
                            50,
                            self.shared_knowledge["best_params"]["n_estimators"] - 100,
                        ),
                        self.shared_knowledge["best_params"]["n_estimators"] + 100,
                    ),
                    "subsample": Real(
                        self.shared_knowledge["best_params"]["subsample"] * 0.9,
                        self.shared_knowledge["best_params"]["subsample"] * 1.1,
                        prior="uniform",
                    ),
                    "colsample_bytree": Real(
                        colsample_bytree_low,
                        colsample_bytree_high,
                        prior="uniform",
                    ),
                    "gamma": Real(
                        self.shared_knowledge["best_params"]["gamma"] * 0.9,
                        self.shared_knowledge["best_params"]["gamma"] * 1.1,
                        prior="uniform",
                    ),
                    "reg_alpha": Real(
                        self.shared_knowledge["best_params"]["reg_alpha"] * 0.9,
                        self.shared_knowledge["best_params"]["reg_alpha"] * 1.1,
                        prior="uniform",
                    ),
                    "reg_lambda": Real(
                        self.shared_knowledge["best_params"]["reg_lambda"] * 0.9,
                        self.shared_knowledge["best_params"]["reg_lambda"] * 1.1,
                        prior="uniform",
                    ),
                    "scale_pos_weight": Integer(
                        max(
                            1,
                            self.shared_knowledge["best_params"]["scale_pos_weight"]
                            - 1,
                        ),
                        min(
                            10,
                            self.shared_knowledge["best_params"]["scale_pos_weight"]
                            + 1,
                        ),
                    ),
                    "min_child_weight": Integer(
                        max(
                            1,
                            self.shared_knowledge["best_params"]["min_child_weight"]
                            - 1,
                        ),
                        min(
                            10,
                            self.shared_knowledge["best_params"]["min_child_weight"]
                            + 1,
                        ),
                    ),
                }
                print(
                    f"Updated search space based on shared knowledge: {agent.param_space.param_space}"
                )
        except KeyError as e:
            print(f"Key error in search space update: {e}")
        except Exception as e:
            print(f"Error during search space update: {e}")

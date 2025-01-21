from typing import Dict, Any
from skopt.space import Real, Integer  # type: ignore


class ParameterSpace:
    def __init__(self, param_space: Dict[str, Any]) -> None:
        self.param_space = param_space

    def __len__(self) -> int:
        return len(self.param_space)

    def adjust_for_other_agent(
        self, other_param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        adjusted_space = {}
        for param, space in self.param_space.items():
            try:
                if isinstance(space, Real):
                    adjusted_space[param] = Real(
                        space.low,
                        (space.high + other_param_space[param].low) / 2,
                        prior="uniform",
                    )
                elif isinstance(space, Integer):
                    adjusted_space[param] = Integer(
                        space.low, (space.high + other_param_space[param].low) // 2
                    )
                else:
                    adjusted_space[param] = space
            except KeyError as e:
                print(f"Key error during parameter adjustment: {e}")
            except Exception as e:
                print(f"Error during parameter adjustment: {e}")
        return adjusted_space

    def to_dict(self) -> Dict[str, Any]:
        return self.param_space

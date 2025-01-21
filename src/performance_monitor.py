from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.agent import Agent


class PerformanceMonitor:
    def __init__(self) -> None:
        self.no_improvement_count = 0

    def monitor(self, agent: "Agent", shared_knowledge: Dict[str, Any]) -> None:
        try:
            if agent.best_score > shared_knowledge["best_score"]:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count > 3:
                print(f"Agent {agent.strategy} is not improving. Reinitializing...")
                agent.param_space = agent.param_space.param_space
                self.no_improvement_count = 0
        except KeyError as e:
            print(f"Key error in performance monitor: {e}")
        except Exception as e:
            print(f"Error in performance monitor: {e}")

from typing import Dict, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent import Agent


class ResultSharer:
    def __init__(self) -> None:
        self.contribution_score = 0

    def share(self, agent: "Agent", shared_knowledge: Dict[str, Any]) -> None:
        try:
            if agent.best_score > shared_knowledge["best_score"]:
                shared_knowledge["best_score"] = agent.best_score
                shared_knowledge["best_params"] = agent.best_params
                print(
                    f"Shared best score: {agent.best_score} with parameters: {agent.best_params}"
                )
                self.contribution_score += 1
        except KeyError as e:
            print(f"Key error in result sharing: {e}")
        except Exception as e:
            print(f"Error in result sharing: {e}")

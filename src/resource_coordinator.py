from typing import Dict
from src.agent import Agent


class ResourceCoordinator:
    def __init__(self) -> None:
        self.agent_status: Dict[str, int] = {}

    def register_agent(self, agent: Agent) -> None:

        self.agent_status[agent.strategy] = agent.resources_used

    def allocate_resources(self, agents: list[Agent]) -> None:

        for agent in agents:
            if agent.resources_used > agent.max_iterations:
                for other_agent in agents:
                    if other_agent.resources_used < other_agent.max_iterations / 2:
                        agent.param_space = agent.param_space.adjust_for_other_agent(
                            other_agent.param_space.param_space
                        )

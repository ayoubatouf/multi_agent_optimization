from sklearn.datasets import load_breast_cancer
from src.multiagent_system import MultiAgentSystem

if __name__ == "__main__":
    multi_agent_system = MultiAgentSystem()
    X, y = load_breast_cancer(return_X_y=True)
    multi_agent_system.run(X, y)

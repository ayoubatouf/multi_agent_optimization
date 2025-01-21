#  Multi-Agent Hyperparameter Optimization for XGBoost

## Overview

This project implements a multi-agent system that optimizes hyperparameters for an XGBoost classifier using Bayesian optimization. The system uses multiple agents, each with a different strategy, to explore and exploit the hyperparameter space. Agents share knowledge and adjust their search spaces based on the performance of other agents. The results are evaluated using cross-validation, and the best model is tested on the breast cancer dataset to measure its accuracy. This will aims to efficiently explore, refine, and fine-tune hyperparameters in a collaborative manner.

## Why use multi-agent system ?

Using a multi-agent system for hyperparameter optimization offers several advantages over traditional methods. Unlike single-agent approaches, where optimization is typically performed sequentially, a multi-agent system enables parallel exploration of the hyperparameter space using different strategies. This not only speeds up the optimization process but also promotes diversity in the search, increasing the chances of finding a more optimal solution. Agents can share knowledge about their search results, allowing them to collaboratively adjust their strategies and search spaces. This dynamic collaboration helps avoid local optima, as agents can exploit the best-performing regions of the search space while exploring new ones. Additionally, the ability of agents to adjust their behaviors based on the performance of others creates a more adaptive and robust optimization process compared to static methods.

## Usage

To run the code, execute:
```
streamlit run main.py
```

## Requirements

- xgboost: 2.1.3
- scikit-learn: 1.1.3
- scikit-optimize: 0.10.2
- joblib: 1.4.2

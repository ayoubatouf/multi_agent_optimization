import pandas as pd
import streamlit as st
from typing import Any, Dict
from skopt.space import Real, Integer
from src.parameter_space import ParameterSpace
from src.multiagent_system import MultiAgentSystem
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from app.utils import create_slider, load_dataset, get_target_column


class StreamlitApp:
    def __init__(self, param_space: Dict[str, Any]) -> None:
        self.param_space = param_space

    def display(self) -> None:
        st.title("Multi-Agents Hyperparameter Optimization")

        df = load_dataset()
        if df is None:
            return

        target_column = get_target_column(df)
        if target_column is None:
            return

        X = df.drop(columns=[target_column])
        y = df[target_column]

        adjusted_intervals = {}
        for param, space in self.param_space.items():
            slider_min, slider_max = create_slider(param, space)
            adjusted_intervals[param] = (slider_min, slider_max)

        if not adjusted_intervals:
            st.error("Hyperparameter ranges are not defined.")
            return

        if st.button("Start Optimization"):
            self._optimize_model(X, y, adjusted_intervals)

    def _optimize_model(
        self, X: pd.DataFrame, y: pd.Series, adjusted_intervals: Dict[str, Any]
    ) -> None:
        adjusted_intervals = {
            param: Real(interval[0], interval[1], prior="uniform")
            if isinstance(self.param_space[param], Real)
            else Integer(interval[0], interval[1])
            for param, interval in adjusted_intervals.items()
        }

        try:
            param_space_object = ParameterSpace(adjusted_intervals)
            multi_agent_system = MultiAgentSystem(
                param_space=param_space_object.param_space
            )

            with st.spinner("Optimization in progress..."):
                multi_agent_system.run(X, y)

            best_params = multi_agent_system.shared_knowledge["best_params"]
            best_score = multi_agent_system.shared_knowledge["best_score"]

            self._display_best_params(best_params, best_score)
            self._train_and_evaluate_model(X, y, best_params)

        except Exception as e:
            st.error(f"An error occurred during optimization: {e}")

    def _display_best_params(
        self, best_params: Dict[str, Any], best_score: float
    ) -> None:
        st.success("Optimization completed!")
        st.subheader("Best Hyperparameters")
        for param, value in best_params.items():
            st.write(
                f"**{param}**: {value:.4f}"
                if isinstance(value, float)
                else f"**{param}**: {value}"
            )
        st.write(f"Best score: {best_score}")

    def _train_and_evaluate_model(
        self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any]
    ) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            n_estimators=best_params["n_estimators"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            gamma=best_params["gamma"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            scale_pos_weight=best_params["scale_pos_weight"],
            min_child_weight=best_params["min_child_weight"],
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

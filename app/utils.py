import pandas as pd
import streamlit as st
from skopt.space import Real, Integer
from typing import Any, Dict


def create_slider(param: str, space: Any) -> Dict[str, Any]:
    min_value, max_value = float(space.low), float(space.high)
    if isinstance(space, Integer):
        min_value, max_value = int(min_value), int(max_value)
        step = 1
    else:
        step = 0.01

    slider_min, slider_max = st.slider(
        label=f"{param}",
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
        step=step,
    )

    if slider_min == slider_max:
        slider_max = slider_min + step
    return slider_min, slider_max


def load_dataset() -> pd.DataFrame:
    st.subheader("Upload your CSV Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        return None

    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")
        st.write(df.head())
        return df
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None


def get_target_column(df: pd.DataFrame) -> str:
    st.subheader("Specify the Target Variable Name")
    target_column = st.text_input("Enter the name of the target variable")

    if target_column and target_column in df.columns:
        st.write(f"Target variable selected: `{target_column}`")
        return target_column
    elif target_column:
        st.error(f"The column '{target_column}' was not found in the dataset.")
    return None

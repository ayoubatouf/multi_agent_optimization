from app.streamlit_app import StreamlitApp
from app.param_loader import load_param_space_from_yaml

if __name__ == "__main__":
    param_space = load_param_space_from_yaml("params.yaml")
    app = StreamlitApp(param_space)
    app.display()

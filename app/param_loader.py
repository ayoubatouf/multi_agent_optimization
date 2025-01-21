import yaml
from skopt.space import Real, Integer
from typing import Any, Dict


def load_param_space_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        param_data = yaml.safe_load(file)

    param_space = {}
    for param, settings in param_data.items():
        if settings["type"] == "real":
            param_space[param] = Real(
                settings["low"],
                settings["high"],
                prior=settings.get("prior", "uniform"),
            )
        elif settings["type"] == "integer":
            param_space[param] = Integer(settings["low"], settings["high"])

    return param_space

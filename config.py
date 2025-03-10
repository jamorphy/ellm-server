import os
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    def resolve_env_vars(data):
        if isinstance(data, dict):
            return {k: resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [resolve_env_vars(i) for i in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            return os.getenv(data[2:-1], "")  # Extract var name and get env value
        return data

    return resolve_env_vars(config)

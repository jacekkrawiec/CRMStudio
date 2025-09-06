import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "templates", "config.yaml")

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from provided path or default template.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

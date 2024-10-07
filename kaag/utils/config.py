import yaml
from typing import Dict, Any

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config (Dict[str, Any]): Configuration to save.
        file_path (str): Path to save the YAML configuration file.
    """
    with open(file_path, 'w') as f:
        yaml.dump(config, f)
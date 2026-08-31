"""Shared utility functions for the dynamic pricing engine."""

import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    return Path(__file__).parent.parent

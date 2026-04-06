"""Configuration management."""
import yaml
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"

class Config:
    def __init__(self, config_path: str | None = None):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        with open(path) as f:
            self._cfg = yaml.safe_load(f)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        keys = dotted_key.split(".")
        val = self._cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def __getitem__(self, key: str):
        return self._cfg[key]

    @property
    def raw(self) -> dict:
        return self._cfg

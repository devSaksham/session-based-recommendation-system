from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(*paths: str | Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for path in paths:
        config = _merge_dicts(config, load_yaml(path))
    return config

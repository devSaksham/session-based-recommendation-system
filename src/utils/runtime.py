from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


def collect_runtime_info() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


def resolve_device(use_cuda: bool = True) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_workspace_python() -> Path | None:
    candidate = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"
    return candidate if candidate.exists() else None

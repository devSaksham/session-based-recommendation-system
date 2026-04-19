from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.preprocessing import preprocess_from_config
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess YooChoose session data.")
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config.")
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to the shared base config.",
    )
    args = parser.parse_args()

    config = load_config(args.base_config, args.config)
    summary = preprocess_from_config(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

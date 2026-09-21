"""YAML config loading with repo-root-relative path resolution."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(name: str) -> dict:
    """Load configs/<name>.yaml. Raises if the file is missing."""
    p = REPO_ROOT / "configs" / f"{name}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p} did not parse to a mapping")
    return cfg


def resolve(path_str: str) -> Path:
    """Absolute Path for a config value; relative values are repo-root-relative."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()

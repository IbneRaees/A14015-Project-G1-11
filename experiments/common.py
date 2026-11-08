from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "e1_baseline.yaml"


def load_config(path=None, **overrides) -> dict:
    """YAML config with CLI overrides (None values are ignored)."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None

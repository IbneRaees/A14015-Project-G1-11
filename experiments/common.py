from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "e1_baseline.yaml"


def find_floodnet_root() -> Path | None:
    """Auto-detect FloodNet dataset in standard local, Kaggle input, or scratch locations."""
    candidates = [
        Path("data/FloodNet-Supervised_v1.0"),
        Path("/tmp/FloodNet-Supervised_v1.0"),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        try:
            for org in kaggle_input.rglob("train-org-img"):
                candidates.append(org.parent.parent)
        except Exception:
            pass
    for c in candidates:
        if (c / "train" / "train-org-img").is_dir():
            return c.resolve()
    return None


def load_config(path=None, **overrides) -> dict:
    """YAML config with CLI overrides (None or empty string values are ignored)."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in overrides.items() if v is not None and v != ""})

    # Auto-resolve data_root if missing or default doesn't exist
    cur_root = Path(cfg.get("data_root", ""))
    if not (cur_root / "train" / "train-org-img").is_dir():
        found = find_floodnet_root()
        if found:
            cfg["data_root"] = str(found)
    return cfg


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None

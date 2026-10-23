"""Small experiment config loader with explicit inherited dataset overrides."""
from pathlib import Path

from omegaconf import OmegaConf


def load_region_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = OmegaConf.load(path)
    if "base" in cfg:
        base = cfg.pop("base")
        cfg = OmegaConf.merge(load_region_config(base), cfg)
    return OmegaConf.to_container(cfg, resolve=True)

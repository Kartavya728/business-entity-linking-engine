"""Paths and global settings shared by every pipeline stage."""
import os
from pathlib import Path

# Repo root = four levels above this file (code/business_entity_resolution/src/ber/config.py)
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4]))
DATA_DIR = Path(os.environ.get("BER_DATA", ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT", ROOT / "output"))

SEED = 42
SOURCES = ("source1", "source2", "source3")
TARGETS = ("source2", "source3")

for _d in (WORK_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def split_dir(split: str) -> Path:
    """Directory holding intermediate artefacts for a split ('train' or 'test')."""
    d = WORK_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d

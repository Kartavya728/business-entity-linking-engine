"""Paths and global settings shared by every pipeline stage."""
import os
from pathlib import Path

# Repo root = four levels above this file (code/business_entity_resolution/src/ber/config.py)
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4]))
DATA_DIR = Path(os.environ.get("BER_DATA", ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT", ROOT / "output"))

SEED = 42
# Final candidate set: blocking pairs whose stage-1 probability p1 reaches this value. Stage 2 (with
# the cross-encoders and LLM matchers) scores exactly these pairs, and candidate_pairs.tsv lists them.
# 0.01 keeps 4.7 candidates per S1 on test (6.4 at 0.001) at 99.60% train pair recall; v8 accepted
# only 440 of 5.9M test matches below it. BER_S2_MIN_P1 is the former name of the setting.
CAND_MIN_P1 = float(os.environ.get("BER_CAND_MIN_P1", os.environ.get("BER_S2_MIN_P1", "0.01")))
SOURCES = ("source1", "source2", "source3")
TARGETS = ("source2", "source3")

for _d in (WORK_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def split_dir(split: str) -> Path:
    """Directory holding intermediate artefacts for a split ('train' or 'test')."""
    d = WORK_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d

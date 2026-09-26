"""Stage 0: load raw TSVs, normalise every record, cache as parquet.

Usage: python -m ber.prepare --split train|test
"""
import argparse
import time

import polars as pl

from .config import split_dir
from .io import load_ground_truth, load_source
from .admin_areas import canon_admin
from .fillers import apply_fillers, filler_sets
from .normalize import base, normalize_frame


def enc_text(df: pl.DataFrame) -> pl.DataFrame:
    """Text fed to the neural encoders: ASCII-folded 'name | address'."""
    names = [base(x) for x in df["business_name"].to_list()]
    addrs = [base(canon_admin(x)) for x in df["business_address"].to_list()]
    return df.with_columns(pl.Series("text", [f"{n} | {a}" for n, a in zip(names, addrs)]))


def prepare(split: str):
    out = split_dir(split)
    frames = {}
    for src in ("source1", "source2", "source3"):
        t = time.time()
        df = load_source(split, src)
        df = normalize_frame(df)
        df = enc_text(df)
        frames[src] = df
        print(f"[prepare] {split}/{src}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)
    fill = filler_sets(frames)
    for src in ("source1", "source2", "source3"):
        frames[src] = apply_fillers(frames[src], fill)
        frames[src].write_parquet(out / f"{src}.parquet")
    if split == "train":
        gt = load_ground_truth(frames["source1"], frames["source2"], frames["source3"])
        gt.write_parquet(out / "gt.parquet")
        print(f"[prepare] ground truth pairs: {len(gt):,}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    prepare(ap.parse_args().split)

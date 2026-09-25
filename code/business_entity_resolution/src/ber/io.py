"""Loading raw TSVs, ground truth, and writing submission files."""
from pathlib import Path

import numpy as np
import polars as pl

from .config import DATA_DIR


def read_tsv(path: Path) -> pl.DataFrame:
    """Read a challenge TSV as all-string columns.

    Quoting is disabled because names/addresses may contain stray quote characters;
    the files are plain tab-delimited with no quoting.
    """
    df = pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True, truncate_ragged_lines=True,
    )
    return df.with_columns(pl.col(c).fill_null("") for c in df.columns)


def load_source(split: str, source: str) -> pl.DataFrame:
    """Load one source file, adding an int row index `idx` used everywhere internally."""
    df = read_tsv(DATA_DIR / split / f"{split}_{source}.tsv")
    return df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))


def load_ground_truth(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    """Explode the ground truth into (s1_idx, tgt, tgt_idx) rows, tgt in {2, 3}.

    Also returns singleton S1 rows implicitly: they simply have no row here.
    """
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    gt = gt.filter(pl.col("matched_entity_ids").str.len_chars() > 0)
    gt = gt.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "tid"})
    gt = gt.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("s1_idx")), on="s1_id")
    parts = []
    for k, df in ((2, s2), (3, s3)):
        p = gt.filter(pl.col("tid").str.starts_with(f"S{k}-")).join(
            df.select(pl.col("entity_id").alias("tid"), pl.col("idx").alias("tgt_idx")), on="tid")
        parts.append(p.with_columns(pl.lit(k, pl.Int8).alias("tgt")))
    return pl.concat(parts).select("s1_idx", "tgt", "tgt_idx")


def write_id_lists(path: Path, s1_ids: np.ndarray, lists: dict, header: tuple):
    """Write one row per S1 entity: `s1_id \t comma-joined ids` (empty when no ids).

    `lists` maps s1 row index -> list of target entity_id strings.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for i, sid in enumerate(s1_ids):
            ids = lists.get(i)
            if ids:
                ids = list(dict.fromkeys(ids))  # dedupe, keep order
                f.write(f"{sid}\t{','.join(ids)}\n")
            else:
                f.write(f"{sid}\t\n")

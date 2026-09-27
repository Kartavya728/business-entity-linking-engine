"""Data-driven detection of generator 'filler' words, per split and country.

True-match noise in S2/S3 appends filler words to names ('Services', 'Center', 'Holdings'
in the US; 'Participations', 'Developpement', 'Holding' in France) while distractors swap
identity words. Fillers are therefore much more frequent in S2/S3 names than in S1 names.
For every country we compare document frequencies of name tokens in S1 vs S2+S3 (Latin
script names only, so transliterated tokens are not mistaken for fillers) and flag
tokens with frequency ratio >= RATIO and target frequency >= MIN_FREQ. No labels are
used, so the same rule applies to unseen countries (France).

Adds columns: name_f (core name without fillers), name_fk (its skeleton), ctext
(canonical text for the canonical cross-encoder).
"""
import numpy as np
import polars as pl

from .normalize import skeleton

RATIO = 1.8
MIN_FREQ = 0.001
# Countries without training labels (France) also get 'excess' fillers: tokens whose target
# frequency exceeds the S1 frequency times the typical copy shrinkage by EXCESS. This catches
# generator fillers that are also common base-name words ('France', 'Fils', 'Services'),
# which the ratio rule misses because their S1 frequency is already high.
EXCESS = 0.004


def _doc_freq(df: pl.DataFrame) -> pl.DataFrame:
    n = max(1, len(df))
    return (df.select(pl.col("name_n").str.split(" ").list.unique().alias("t")).explode("t")
              .filter(pl.col("t").is_not_null() & (pl.col("t") != "") & ~pl.col("t").str.contains(r"^\d+$"))
              .group_by("t").len("c").with_columns((pl.col("c") / n).alias("f")).drop("c"))


def _excess(s1: pl.DataFrame, tg: pl.DataFrame) -> set:
    a = _doc_freq(s1.select(pl.col("name_c").alias("name_n"))).rename({"f": "f1"})
    b = _doc_freq(tg.select(pl.col("name_c").alias("name_n"))).rename({"f": "f2"})
    r = a.join(b, on="t", how="full", coalesce=True).fill_null(1e-6)
    top = r.filter(pl.col("f1") > 0.005)
    shrink = float(np.median((top["f2"] / top["f1"]).to_numpy()))
    r = r.filter((pl.col("f2") - shrink * pl.col("f1") >= EXCESS) & (pl.col("f2") / pl.col("f1") >= 1.15))
    return set(r["t"].to_list())


def filler_sets(frames: dict, unseen=()) -> dict:
    """frames: {'source1': df, 'source2': df, 'source3': df} -> {country: set(tokens)}
    unseen: countries absent from training, which also get excess fillers."""
    out = {}
    latin = pl.col("business_name").str.contains(r"^[\x00-\x7FÀ-ɏ\s]*$")
    countries = set(frames["source1"]["country"].unique().to_list())
    for c in sorted(countries):
        s1 = frames["source1"].filter((pl.col("country") == c) & latin)
        tg = pl.concat([frames[k].filter((pl.col("country") == c) & latin).select("name_n")
                        for k in ("source2", "source3")])
        r = _doc_freq(s1.select("name_n")).rename({"f": "f1"}).join(
            _doc_freq(tg).rename({"f": "f2"}), on="t", how="full", coalesce=True).fill_null(1e-6)
        r = r.filter((pl.col("f2") >= MIN_FREQ) & (pl.col("f2") / pl.col("f1") >= RATIO))
        out[c] = set(r["t"].to_list())
        print(f"[fillers] {c}: {len(out[c])} filler tokens, e.g. "
              f"{sorted(out[c], key=lambda t: -r.filter(pl.col('t') == t)['f2'][0])[:15]}", flush=True)
        if c in unseen:
            tgc = pl.concat([frames[k].filter((pl.col("country") == c) & latin).select("name_c")
                             for k in ("source2", "source3")])
            extra = _excess(s1, tgc) - out[c]
            out[c] |= extra
            print(f"[fillers] {c}: +{len(extra)} excess fillers {sorted(extra)}", flush=True)
    return out


def apply_fillers(df: pl.DataFrame, fill: dict) -> pl.DataFrame:
    names = df["name_c"].to_list(); ctry = df["country"].to_list()
    legal = df["legal"].to_list(); addr = df["addr_n"].to_list()
    nf, nfk, ct = [], [], []
    for n, c, l, a in zip(names, ctry, legal, addr):
        F = fill.get(c, ())
        toks = [t for t in n.split() if t not in F]
        f = " ".join(toks) if toks else n
        nf.append(f); nfk.append(skeleton(f)); ct.append(f"{f} {l} | {a}".strip())
    return df.with_columns(pl.Series("name_f", nf), pl.Series("name_fk", nfk), pl.Series("ctext", ct))

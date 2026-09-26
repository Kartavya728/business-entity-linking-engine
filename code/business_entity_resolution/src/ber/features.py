"""Stage 3: pair features for (S1, target) candidate pairs.

Groups of features (all country-agnostic; country is never a feature):
  * name string similarities on several normalised views (core, full, skeleton,
    no-space, DBA alternative) via rapidfuzz
  * address string similarities, number/zip agreement logic
  * TF-IDF cosines (word IDF and char 3-gram) for names and addresses, fitted per
    country on the split's own records (unsupervised)
  * legal-form agreement / conflict, domain-name flags, name frequency (chains)
  * retrieval scores/ranks and candidate-group context (competition between S1s for
    the same target, and between targets of the same S1)
"""
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

STR_COLS = ["name_n", "name_c", "name_k", "name_alt", "addr_n", "nums", "zip", "legal",
            "is_dom", "country", "business_name", "name_f", "name_fk"]


def _extra(p: str, q: str):
    """(# S1 skeleton tokens absent from target, # target tokens absent from S1, # shared),
    with a small edit tolerance so typos do not count as missing identity words."""
    a, b = p.split(), q.split()
    sb, sa = set(b), set(a)
    def has(t, S):
        return t in S or (len(t) >= 3 and any(Levenshtein.distance(t, u) <= 1 for u in S))
    m1 = sum(1 for t in sa if not has(t, sb))
    m2 = sum(1 for t in sb if not has(t, sa))
    return m1, m2, len(sa & sb)


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, **kw).astype(np.float32)


def _strip0(t):
    t = t.lstrip("0")
    return t if t else "0"


def _fix(t, u):
    """Number tokens agree up to a dropped leading/trailing digit(s)."""
    return t == u or t.startswith(u) or u.startswith(t) or t.endswith(u) or u.endswith(t)


def _num_feats(args):
    """Number/zip agreement features for one pair (pure python, run in a pool).

    Observed noise on true matches: zero padding (2610 -> 02610), dropped leading digit
    (3890 -> 890), truncation (1914 -> 191), extra inserted numbers (Hn 359 B-8/15).
    Distractors instead carry a different number (8326 vs 8315).
    """
    n1, n2, z1, z2 = args
    a = [_strip0(t) for t in n1.split()]
    b = [_strip0(t) for t in n2.split()]
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    jac = inter / len(sa | sb) if (sa or sb) else -1.0
    za, zb = _strip0(z1) if z1 else "", _strip0(z2) if z2 else ""
    ha = [t for t in a if t != za]; hb = [t for t in b if t != zb]
    if ha and hb:
        h1, h2 = ha[0], hb[0]
        h_eq = float(h1 == h2)
        h_pre = float(h1.startswith(h2) or h2.startswith(h1))
        h_suf = float(h1.endswith(h2) or h2.endswith(h1))
        h_in = float(h1 in sb or h2 in sa)
        h_lev = float(Levenshtein.distance(h1, h2))
        # best agreement of any S1 house-ish number with any target number
        h_any = float(any(t == u for t in ha for u in hb))
        h_anyfix = float(any(_fix(t, u) for t in ha for u in hb))
        big = [t for t in ha if len(t) >= 3]
        h_big_miss = float(sum(1 for t in big if not any(_fix(t, u) for u in hb))) if big else -1.0
    else:
        h_eq = h_pre = h_suf = h_in = h_lev = h_any = h_anyfix = h_big_miss = -1.0
    miss_a = sum(1 for t in sa if not any(_fix(t, u) for u in sb)) if sb else -1
    miss_b = sum(1 for t in sb if not any(_fix(t, u) for u in sa)) if sa else -1
    if z1 and z2:
        z_eq = float(z1 == z2)
        z_p3 = float(z1[:3] == z2[:3])
    else:
        z_eq = z_p3 = -1.0
    return (jac, inter, len(sa), len(sb), h_eq, h_pre, h_suf, h_in, h_lev, h_any, h_anyfix, h_big_miss,
            miss_a, miss_b, z_eq, z_p3)


NUM_NAMES = ["num_jac", "num_inter", "num_n1", "num_n2", "hno_eq", "hno_prefix", "hno_suffix", "hno_in",
             "hno_lev", "hno_any", "hno_anyfix", "hno_big_miss", "num_miss1", "num_miss2", "zip_eq", "zip_p3"]


def _fit_one(key, col_an_kw, t1, t2):
    col, an, kw = col_an_kw
    vec = TfidfVectorizer(analyzer=an, sublinear_tf=True, dtype=np.float32, **kw)
    vec.fit(np.concatenate([t1, t2]))
    return key, vec.transform(t1).tocsr(), vec.transform(t2).tocsr()


class TfidfBank:
    """Per-country TF-IDF matrices for S1 and one target source, fitted once.

    Vectorisers are fitted on that country's S1 + target records of the current split
    (unsupervised, no labels). pair_cos() gives cosine for aligned (s1_idx, tgt_idx) pairs.
    """
    SPECS = {"n_idf": ("name_c", str.split, {}), "n_c3": ("name_c", None, {}),
             "nx_idf": ("name_f", str.split, {}),
             "a_idf": ("addr_n", str.split, {}), "a_c3": ("addr_n", None, {"min_df": 2})}

    def __init__(self, s1: pl.DataFrame, tg: pl.DataFrame, n_jobs: int = 8):
        from joblib import Parallel, delayed
        c1 = s1["country"].to_numpy(); c2 = tg["country"].to_numpy()
        self.countries = sorted(set(c1) & set(c2))
        self.pos1 = np.zeros(len(c1), dtype=np.int64); self.pos2 = np.zeros(len(c2), dtype=np.int64)
        self.cty1 = c1
        jobs = []
        for c in self.countries:
            m1 = np.flatnonzero(c1 == c); m2 = np.flatnonzero(c2 == c)
            self.pos1[m1] = np.arange(len(m1)); self.pos2[m2] = np.arange(len(m2))
            for key, (col, an, kw) in self.SPECS.items():
                jobs.append(delayed(_fit_one)((key, c), (col, an or _char3, kw),
                                              s1[col].to_numpy()[m1], tg[col].to_numpy()[m2]))
        self.mats = {}
        for key, A, B in Parallel(n_jobs=n_jobs)(jobs):
            self.mats[key] = (A, B)

    def pair_cos(self, key: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros(len(a), dtype=np.float32)
        pc = self.cty1[a]
        for c in self.countries:
            m = np.flatnonzero(pc == c)
            if len(m) == 0:
                continue
            A, B = self.mats[(key, c)]
            for s in range(0, len(m), 2_000_000):
                mm = m[s:s + 2_000_000]
                out[mm] = np.asarray(A[self.pos1[a[mm]]].multiply(B[self.pos2[b[mm]]]).sum(1)).ravel()
        return out


def _char3(s):
    s = f" {s} "
    return [s[i:i + 3] for i in range(len(s) - 2)]


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, bank: "TfidfBank",
                  n_proc: int = 40) -> pl.DataFrame:
    """Compute features for candidate rows of ONE target source.

    cand: s1_idx, tgt_idx (+ retrieval columns, carried through)
    s1, tg: normalised source frames (row idx == position)
    """
    a = cand["s1_idx"].to_numpy(); b = cand["tgt_idx"].to_numpy()
    A = {c: s1[c].to_numpy() for c in STR_COLS}
    B = {c: tg[c].to_numpy() for c in STR_COLS}
    g = lambda d, c, i: d[c][i]
    F = {}
    n1, n2 = g(A, "name_c", a), g(B, "name_c", b)
    F["n_ratio"] = _cp(n1, n2, fuzz.ratio)
    F["n_pratio"] = _cp(n1, n2, fuzz.partial_ratio)
    F["n_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    F["n_jw"] = _cp(n1, n2, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cp(n1, n2, Levenshtein.distance)
    F["n_exact"] = (n1 == n2).astype(np.float32)
    nf1, nf2 = g(A, "name_n", a), g(B, "name_n", b)
    F["nf_ratio"] = _cp(nf1, nf2, fuzz.ratio)
    F["nf_tset"] = _cp(nf1, nf2, fuzz.token_set_ratio)
    k1, k2 = g(A, "name_k", a), g(B, "name_k", b)
    F["nk_ratio"] = _cp(k1, k2, fuzz.ratio)
    F["nk_tset"] = _cp(k1, k2, fuzz.token_set_ratio)
    F["nk_exact"] = (k1 == k2).astype(np.float32)
    ns1 = np.char.replace(n1.astype(str), " ", ""); ns2 = np.char.replace(n2.astype(str), " ", "")
    ns1 = ns1.astype(object); ns2 = ns2.astype(object)
    F["nns_ratio"] = _cp(ns1, ns2, fuzz.ratio)
    F["nns_pratio"] = _cp(ns1, ns2, fuzz.partial_ratio)
    raw1 = np.array([s.lower() for s in g(A, "business_name", a)], dtype=object)
    raw2 = np.array([s.lower() for s in g(B, "business_name", b)], dtype=object)
    F["raw_ratio"] = _cp(raw1, raw2, fuzz.ratio)
    alt1, alt2 = g(A, "name_alt", a), g(B, "name_alt", b)
    has_alt = (alt1 != "") | (alt2 != "")
    F["has_alt"] = has_alt.astype(np.float32)
    F["alt_tset"] = np.where(has_alt, np.maximum(_cp(np.where(alt1 != "", alt1, n1), n2, fuzz.token_set_ratio),
                                                _cp(n1, np.where(alt2 != "", alt2, n2), fuzz.token_set_ratio)), -1)
    # token set bookkeeping on core names
    t1 = [set(x.split()) for x in n1]; t2 = [set(x.split()) for x in n2]
    F["n_tok1"] = np.fromiter((len(x) for x in t1), np.float32, len(t1))
    F["n_tok2"] = np.fromiter((len(x) for x in t2), np.float32, len(t2))
    F["n_common"] = np.fromiter((len(x & y) for x, y in zip(t1, t2)), np.float32, len(t1))
    F["n_first_eq"] = np.fromiter(((x.split()[:1] == y.split()[:1]) for x, y in zip(n1, n2)), np.float32, len(n1))
    F["n_len1"] = np.fromiter((len(x) for x in n1), np.float32, len(n1))
    F["n_len2"] = np.fromiter((len(x) for x in n2), np.float32, len(n2))
    # filler-free core names (data-driven fillers, see fillers.py)
    x1, x2 = g(A, "name_f", a), g(B, "name_f", b)
    F["nx_ratio"] = _cp(x1, x2, fuzz.ratio)
    F["nx_tset"] = _cp(x1, x2, fuzz.token_set_ratio)
    F["nx_tsort"] = _cp(x1, x2, fuzz.token_sort_ratio)
    F["nx_jw"] = _cp(x1, x2, JaroWinkler.normalized_similarity)
    F["nx_exact"] = (x1 == x2).astype(np.float32)
    xk1, xk2 = g(A, "name_fk", a), g(B, "name_fk", b)
    F["nxk_ratio"] = _cp(xk1, xk2, fuzz.ratio)
    F["nxk_exact"] = (xk1 == xk2).astype(np.float32)
    with Pool(n_proc) as pool:
        ex = np.asarray(pool.starmap(_extra, zip(xk1, xk2), chunksize=50000), dtype=np.float32)
    F["nx_miss1"], F["nx_miss2"], F["nx_common"] = ex[:, 0], ex[:, 1], ex[:, 2]
    # legal forms
    l1, l2 = g(A, "legal", a), g(B, "legal", b)
    both = (l1 != "") & (l2 != "")
    F["leg_both"] = both.astype(np.float32)
    F["leg_eq"] = np.where(both, (l1 == l2).astype(np.float32), -1)
    F["leg_conflict"] = np.fromiter(
        ((bool(x) and bool(y) and not (set(x.split()) & set(y.split()))) for x, y in zip(l1, l2)), np.float32, len(l1))
    F["dom1"] = (g(A, "is_dom", a) == "1").astype(np.float32)
    F["dom2"] = (g(B, "is_dom", b) == "1").astype(np.float32)
    # address
    ad1, ad2 = g(A, "addr_n", a), g(B, "addr_n", b)
    F["a_empty2"] = (ad2 == "").astype(np.float32)
    F["a_ratio"] = _cp(ad1, ad2, fuzz.ratio)
    F["a_tsort"] = _cp(ad1, ad2, fuzz.token_sort_ratio)
    F["a_tset"] = _cp(ad1, ad2, fuzz.token_set_ratio)
    F["a_pratio"] = _cp(ad1, ad2, fuzz.partial_ratio)
    F["a_jw"] = _cp(ad1, ad2, JaroWinkler.normalized_similarity)
    F["a_len1"] = np.fromiter((len(x) for x in ad1), np.float32, len(ad1))
    F["a_len2"] = np.fromiter((len(x) for x in ad2), np.float32, len(ad2))
    with Pool(n_proc) as pool:
        nums = pool.map(_num_feats, zip(g(A, "nums", a), g(B, "nums", b), g(A, "zip", a), g(B, "zip", b)),
                        chunksize=50000)
    nums = np.asarray(nums, dtype=np.float32)
    for j, nm in enumerate(NUM_NAMES):
        F[nm] = nums[:, j]
    # TF-IDF cosines (per country, fitted on this split's S1 + target records)
    for key in TfidfBank.SPECS:
        F[key] = bank.pair_cos(key, a, b)
    out = cand.with_columns(pl.Series(k, v) for k, v in F.items())
    return out


def name_freq(s1: pl.DataFrame, tg: pl.DataFrame):
    """How common each record's skeleton name is inside its own source & country (chains)."""
    f1 = s1.group_by(["name_k", "country"]).len("f1")
    f2 = tg.group_by(["name_k", "country"]).len("f2")
    r1 = s1.select("idx", "name_k", "country").join(f1, on=["name_k", "country"], how="left")
    r2 = tg.select("idx", "name_k", "country").join(f2, on=["name_k", "country"], how="left")
    return (r1.sort("idx")["f1"].to_numpy().astype(np.float32),
            r2.sort("idx")["f2"].to_numpy().astype(np.float32))


def grp_stats(g: np.ndarray, x: np.ndarray):
    """Per-group statistics of x via one lexsort (fast for ~1e8 rows / ~1e7 groups).

    Returns per-row arrays: group max, 1-based descending rank, group size, and the
    second-largest value in the group (0 when the group has a single row).
    """
    n = len(x)
    order = np.lexsort((-x, g))
    gs = g[order]; xs = x[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    sizes = np.diff(np.r_[starts, n])
    gi = np.repeat(np.arange(len(starts)), sizes)
    gmax = xs[starts]
    second = np.where(sizes > 1, xs[np.minimum(starts + 1, n - 1)], 0).astype(np.float32)
    out = {}
    for name, v in (("max", gmax[gi]), ("rank", (np.arange(n) - starts[gi] + 1).astype(np.float32)),
                    ("n", sizes[gi].astype(np.float32)), ("second", second[gi])):
        r = np.empty(n, dtype=np.float32); r[order] = v; out[name] = r
    return out


def _keys(df: pl.DataFrame):
    s1 = df["s1_idx"].to_numpy().astype(np.int64); t = df["tgt"].to_numpy().astype(np.int64)
    ti = df["tgt_idx"].to_numpy().astype(np.int64)
    return s1 * 4 + t, ti * 4 + t


def group_context(df: pl.DataFrame, score: str, prefix: str) -> pl.DataFrame:
    """Competition features for a pair score within (S1, source) and within target."""
    gs, gt = _keys(df)
    x = df[score].to_numpy().astype(np.float32)
    a = grp_stats(gs, x); b = grp_stats(gt, x)
    return df.with_columns(
        pl.Series(f"{prefix}_s1max", a["max"]), pl.Series(f"{prefix}_s1gap", x - a["max"]),
        pl.Series(f"{prefix}_s1rank", a["rank"]), pl.Series(f"{prefix}_s1n", a["n"]),
        pl.Series(f"{prefix}_tmax", b["max"]), pl.Series(f"{prefix}_tgap", x - b["max"]),
        pl.Series(f"{prefix}_trank", b["rank"]), pl.Series(f"{prefix}_tn", b["n"]),
    )

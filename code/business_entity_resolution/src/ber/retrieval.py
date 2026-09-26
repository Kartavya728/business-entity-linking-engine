"""Candidate retrieval primitives: dense GPU kNN and sparse IDF token matching.

All searches are restricted to records with the same country string (0 cross-country
matches in training); the country is used only as a partition key, never as a feature,
so unseen countries (France) work unchanged.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


# --------------------------------------------------------------------------- dense
@torch.no_grad()
def knn_grouped(q: torch.Tensor, d: torch.Tensor, qg: np.ndarray, dg: np.ndarray, k: int,
                chunk: int = 512):
    """Exact inner-product top-k of q against d restricted to equal group labels.

    q, d: L2-normalised float16 tensors (can live on CPU; moved to GPU per group).
    Returns (idx [nq,k] int32 into d, -1 when fewer than k), score [nq,k] float32.
    """
    nq = q.shape[0]
    out_i = np.full((nq, k), -1, dtype=np.int32)
    out_s = np.full((nq, k), -1.0, dtype=np.float32)
    for g in np.unique(qg):
        qi = np.flatnonzero(qg == g)
        di = np.flatnonzero(dg == g)
        if len(di) == 0:
            continue
        kk = min(k, len(di))
        dgpu = d[torch.from_numpy(di)].cuda()
        di_t = torch.from_numpy(di).cuda()
        for s in range(0, len(qi), chunk):
            rows = qi[s:s + chunk]
            sims = q[torch.from_numpy(rows)].cuda() @ dgpu.T
            sc, ix = sims.topk(kk, dim=1)
            del sims
            out_i[rows, :kk] = di_t[ix].cpu().numpy()
            out_s[rows, :kk] = sc.float().cpu().numpy()
        del dgpu, di_t
        torch.cuda.empty_cache()
    return out_i, out_s


def encode(model, texts, batch_size: int = 1024) -> torch.Tensor:
    """Encode texts with a SentenceTransformer into L2-normalised float16 CPU tensors."""
    emb = model.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb.half().cpu()


# --------------------------------------------------------------------------- sparse
def token_doc(df: pl.DataFrame) -> list:
    """Blocking-key 'documents': typed tokens from name, skeleton, address, zip and
    house-number/street combinations. Each record becomes a whitespace-joined string."""
    exprs = pl.concat_str([
        pl.col("name_c").str.replace_all(r"(\S+)", "n:$1"),
        pl.col("name_k").str.replace_all(r"(\S+)", "k:$1"),
        pl.col("addr_n").str.replace_all(r"(\S+)", "a:$1"),
        pl.when(pl.col("zip") != "").then(pl.lit("z:") + pl.col("zip")).otherwise(pl.lit("")),
        # first number + first alphabetic address token ~ "house number + street"
        pl.when(pl.col("nums") != "").then(
            pl.lit("h:") + pl.col("nums").str.extract(r"^(\S+)") + pl.lit("_")
            + pl.col("addr_n").str.extract(r"\b([a-z]{3,})\b").fill_null("")
        ).otherwise(pl.lit("")),
        # name first token + zip (chains at a specific location)
        pl.when(pl.col("zip") != "").then(
            pl.lit("nz:") + pl.col("name_c").str.extract(r"^(\S+)").fill_null("") + pl.lit("_") + pl.col("zip")
        ).otherwise(pl.lit("")),
    ], separator=" ")
    return df.select(exprs.alias("d"))["d"].to_list()


def sparse_topk(q_docs, d_docs, qg, dg, k: int, max_df: int = 20000, n_threads: int = 48):
    """IDF-weighted token cosine top-k of q_docs vs d_docs within equal groups.

    Tokens with more than `max_df` postings in a group are dropped (they carry little
    evidence and dominate the cost). Returns (idx [nq,k] into d, score [nq,k]).
    """
    q_docs = np.asarray(q_docs, dtype=object)
    d_docs = np.asarray(d_docs, dtype=object)
    nq = len(q_docs)
    out_i = np.full((nq, k), -1, dtype=np.int32)
    out_s = np.zeros((nq, k), dtype=np.float32)
    for g in np.unique(qg):
        qi = np.flatnonzero(qg == g)
        di = np.flatnonzero(dg == g)
        if len(di) == 0:
            continue
        vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, dtype=np.float32,
                              min_df=1, max_df=max_df if len(di) > max_df else 1.0)
        D = vec.fit_transform(d_docs[di])
        Q = vec.transform(q_docs[qi])
        C = sp_matmul_topn(Q, D.T.tocsr(), top_n=k, threshold=1e-6, n_threads=n_threads, sort=True)
        C = C.tocsr()
        for r in range(C.shape[0]):
            a, b = C.indptr[r], C.indptr[r + 1]
            if b > a:
                out_i[qi[r], :b - a] = di[C.indices[a:b]]
                out_s[qi[r], :b - a] = C.data[a:b]
    return out_i, out_s


def topk_to_frame(idx: np.ndarray, score: np.ndarray, name: str, q_is_s1: bool = True) -> pl.DataFrame:
    """Flatten a top-k matrix to (s1_idx, tgt_idx, <name>_s, <name>_r) rows.

    For reverse search (queries are targets), pass q_is_s1=False.
    """
    nq, k = idx.shape
    q = np.repeat(np.arange(nq, dtype=np.int32), k)
    r = np.tile(np.arange(k, dtype=np.int16), nq)
    d = idx.ravel()
    m = d >= 0
    a, b = (q[m], d[m]) if q_is_s1 else (d[m], q[m])
    return pl.DataFrame({"s1_idx": a, "tgt_idx": b,
                         f"{name}_s": score.ravel()[m], f"{name}_r": r[m]})


def token_doc_v2(df: pl.DataFrame) -> list:
    """Typed sparse tokens (v5 blocking): filler-free name + skeleton, address tokens,
    and 'house number x address word' keys (e.g. hs:33_prevert) that pin a specific
    street address. IDF (fitted per country) down-weights generic words automatically."""
    names = df["name_f"].to_list(); sk = df["name_fk"].to_list()
    addr = df["addr_n"].to_list(); nums = df["nums"].to_list()
    out = []
    for n, k, a, m in zip(names, sk, addr, nums):
        at = [t for t in a.split() if not t.isdigit()]
        hn = m.split()[:1]
        toks = [f"n:{t}" for t in n.split()] + [f"k:{t}" for t in k.split()] + [f"a:{t}" for t in at]
        toks += [f"d:{t.lstrip('0') or '0'}" for t in m.split()]
        if hn:
            h = hn[0].lstrip("0") or "0"
            toks += [f"hs:{h}_{t}" for t in at if len(t) >= 3]
        out.append(" ".join(toks))
    return out

"""GPU inverted-index retrieval for typed-token TF-IDF cosine.

Replaces CPU sparse matmul: for a chunk of queries we expand only the postings of the
tokens they contain (tokens with > max_df postings are dropped), reduce (query, doc)
contributions with a sort-based unique + scatter_add, and keep the per-query top-k.
TF-IDF weights are built with polars (multi-threaded): w = (1 + ln tf) * idf, rows L2
normalised, idf = ln((1 + n) / (1 + df)) + 1 computed on the document (target) side.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch


def _explode(docs, name):
    return (pl.DataFrame({"row": np.arange(len(docs), dtype=np.int32), "d": docs})
              .with_columns(pl.col("d").str.split(" ")).explode("d")
              .filter(pl.col("d").is_not_null() & (pl.col("d") != ""))
              .group_by("row", "d").len("tf"))


def tfidf_csr(q_docs, d_docs, max_df: int):
    """Return (Q csr [nq,V], D csr [nd,V]) with shared vocab from d_docs (df <= max_df)."""
    n = len(d_docs)
    D = _explode(d_docs, "d")
    vocab = D.group_by("d").len("df").filter(pl.col("df") <= max_df)
    vocab = vocab.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("tok"),
                               (((1 + n) / (1 + pl.col("df"))).log() + 1).cast(pl.Float32).alias("idf"))
    out = []
    for X, nrows in ((_explode(q_docs, "q"), len(q_docs)), (D, n)):
        X = X.join(vocab.select("d", "tok", "idf"), on="d", how="inner")
        X = X.with_columns(((1 + pl.col("tf").cast(pl.Float32).log()) * pl.col("idf")).alias("w"))
        X = X.with_columns((pl.col("w") / (pl.col("w") ** 2).sum().over("row").sqrt()).alias("w"))
        m = sp.csr_matrix((X["w"].to_numpy(), (X["row"].to_numpy(), X["tok"].to_numpy())),
                          shape=(nrows, len(vocab)), dtype=np.float32)
        out.append(m)
    return out[0], out[1]


@torch.no_grad()
def topk_inverted(Q: sp.csr_matrix, D: sp.csr_matrix, k: int, max_load: int = 40_000_000):
    """Exact top-k of Q @ D.T using a GPU inverted index. Returns (idx, score) [nq, k].

    Query chunks are sized so the number of expanded postings stays <= max_load.
    """
    dev = "cuda"
    DT = D.T.tocsr()  # V x N postings
    p_ptr = torch.from_numpy(DT.indptr.astype(np.int64)).to(dev)
    p_doc = torch.from_numpy(DT.indices.astype(np.int64)).to(dev)
    p_val = torch.from_numpy(DT.data).to(dev)
    N = D.shape[0]
    nq = Q.shape[0]
    out_i = np.full((nq, k), -1, dtype=np.int32)
    out_s = np.zeros((nq, k), dtype=np.float32)
    plen = np.diff(DT.indptr).astype(np.int64)
    Qb = Q.copy(); Qb.data = np.ones_like(Qb.data)
    load = np.cumsum(Qb @ plen)
    bounds, s0 = [], 0
    while s0 < nq:
        base = load[s0 - 1] if s0 else 0
        e = int(np.searchsorted(load, base + max_load, side="right"))
        e = max(e, s0 + 1)
        bounds.append((s0, min(e, nq))); s0 = e
    for s, e in bounds:
        q = Q[s:e].tocoo()
        if q.nnz == 0:
            continue
        qr = torch.from_numpy(q.row.astype(np.int64)).to(dev)
        qt = torch.from_numpy(q.col.astype(np.int64)).to(dev)
        qw = torch.from_numpy(q.data).to(dev)
        st = p_ptr[qt]; ln = p_ptr[qt + 1] - st
        tot = int(ln.sum())
        if tot == 0:
            continue
        rep = torch.repeat_interleave(torch.arange(len(qt), device=dev), ln)
        off = torch.arange(tot, device=dev) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
        pos = st[rep] + off
        key = qr[rep] * N + p_doc[pos]
        val = qw[rep] * p_val[pos]
        uk, inv = torch.unique(key, return_inverse=True)
        sc = torch.zeros(len(uk), device=dev, dtype=torch.float32).scatter_add_(0, inv, val)
        uq = uk // N; ud = uk % N
        # sort by (query asc, score desc): stable sort on score, then on query
        o = torch.argsort(sc, descending=True, stable=True)
        o = o[torch.argsort(uq[o], stable=True)]
        uq, ud, sc = uq[o], ud[o], sc[o]
        first = torch.searchsorted(uq, uq, right=False)
        rank = torch.arange(len(uq), device=dev) - first
        m = rank < k
        r = (uq[m] + s).cpu().numpy(); c = rank[m].cpu().numpy()
        out_i[r, c] = ud[m].cpu().numpy().astype(np.int32)
        out_s[r, c] = sc[m].cpu().numpy()
        del rep, off, pos, key, val, uk, inv, sc, uq, ud, o, first, rank
    del p_ptr, p_doc, p_val
    torch.cuda.empty_cache()
    return out_i, out_s


def sparse_topk_gpu(q_docs, d_docs, qg, dg, k: int, max_df: int = 20000):
    """Grouped (per country) GPU sparse top-k. Also returns per-group CSR for re-scoring."""
    q_docs = np.asarray(q_docs, dtype=object); d_docs = np.asarray(d_docs, dtype=object)
    nq = len(q_docs)
    out_i = np.full((nq, k), -1, dtype=np.int32)
    out_s = np.zeros((nq, k), dtype=np.float32)
    mats = {}
    for g in np.unique(qg):
        qi = np.flatnonzero(qg == g); di = np.flatnonzero(dg == g)
        if len(di) == 0:
            continue
        Q, D = tfidf_csr(q_docs[qi].tolist(), d_docs[di].tolist(), max_df)
        ii, ss = topk_inverted(Q, D, k)
        ok = ii >= 0
        ii2 = np.where(ok, di[np.maximum(ii, 0)], -1)
        out_i[qi] = ii2; out_s[qi] = ss
        mats[g] = (qi, di, Q, D)
    return out_i, out_s, mats


def pair_dot(mats, qg, a, b):
    """Cosine for aligned (query a[i], doc b[i]) pairs using the per-group CSR matrices."""
    out = np.zeros(len(a), dtype=np.float32)
    pg = qg[a]
    for g, (qi, di, Q, D) in mats.items():
        m = np.flatnonzero(pg == g)
        if len(m) == 0:
            continue
        qpos = np.empty(qi.max() + 1, dtype=np.int64); qpos[qi] = np.arange(len(qi))
        dpos = np.full(max(di.max(), b.max()) + 1, -1, dtype=np.int64); dpos[di] = np.arange(len(di))
        for s in range(0, len(m), 2_000_000):
            mm = m[s:s + 2_000_000]
            out[mm] = np.asarray(Q[qpos[a[mm]]].multiply(D[dpos[b[mm]]]).sum(1)).ravel()
    return out

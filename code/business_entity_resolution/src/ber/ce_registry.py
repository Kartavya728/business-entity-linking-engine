"""Registry of cross-encoder variants (kept free of torch imports so stage 2 can read it cheaply).

name -> dict(base checkpoint, output dir, score column, score file, input text, training setup)
  text : 'text'  raw view  ('name | address', ASCII-folded)
         'ctext' canonical view (filler-free name + legal form | normalised address)
         'ftext' field-structured view (raw || canonical)
  prec : 'bf16' autocast or 'fp32' (DeBERTa-v3 diverges under bf16)
  prompt, yesno : decoder prompt style (crossencoder.PROMPTS) and the two answer tokens whose
         LM-head rows initialise the classification head (None = random head)
Every model is trained on encoder-fold (E) S1 entities only, so its scores on the ranker folds
and on test are out-of-sample; stage 2 picks up every model whose score file exists.
All checkpoints are MIT or Apache-2.0 and <= 8B parameters.
"""
from .config import WORK_DIR


def _m(base, tag, col, text="text", bs=128, lr=2e-5, sparse_neg=0, swap=False, max_len=128,
       prec="bf16", score_bs=512, n_s1=150_000, epochs=1, seed=0, lora=None, band=None,
       prompt="plain", yesno=None):
    return dict(base=base, out=WORK_DIR / f"crossencoder{tag}", col=col, file=f"ce_{col[2:] or 's'}.parquet",
                text=text, bs=bs, lr=lr, sparse_neg=sparse_neg, swap=swap, max_len=max_len, prec=prec,
                score_bs=score_bs, n_s1=n_s1, epochs=epochs, seed=seed, lora=lora, band=band,
                prompt=prompt, yesno=yesno)


QA = dict(prompt="qa", yesno=(" Yes", " No"))            # base decoders: "... same business? Answer: Yes"
RERANK = dict(prompt="qwen3rr", yesno=("yes", "no"))     # Qwen3-Reranker native template and answer


# Hard-pair band for the large LLM matchers: only pairs whose stage-1 probability p1 lies in
# [lo, hi) are used for training and scored (the same rule on train and test). Outside the band
# stage 1 is already near-certain; inside it holds ~2.4M train / 2.2M test pairs (vs 12M / 9.5M).
HARD_BAND = (0.02, 0.995)


CE_MODELS = {
    # --- models used by v1..v8 (file names kept for compatibility) ---
    "small": dict(_m("intfloat/multilingual-e5-small", "", "ce", bs=256, lr=3e-5, score_bs=1024), file="ce.parquet"),
    "base": dict(_m("intfloat/multilingual-e5-base", "_base", "ceb"), file="ce_b.parquet"),
    "base2": dict(_m("intfloat/multilingual-e5-base", "_base2", "ceb2", seed=2), file="ce_b2.parquet"),
    "canon": dict(_m("intfloat/multilingual-e5-small", "_canon", "cec", text="ctext", bs=256, lr=3e-5,
                     score_bs=1024), file="ce_c.parquet"),
    "large": dict(_m("intfloat/multilingual-e5-large", "_large", "cel", text="ftext", bs=64, lr=1e-5,
                     sparse_neg=6, swap=True, max_len=160, score_bs=256), file="ce_l.parquet"),
    # --- additional ensemble members (DGX experiments) ---
    # second e5-large: other seed, 2x the training entities
    "large2": dict(_m("intfloat/multilingual-e5-large", "_large2", "cel2", text="ftext", bs=64, lr=1e-5,
                      sparse_neg=6, swap=True, max_len=160, score_bs=256, n_s1=250_000, seed=1),
                   file="ce_l2.parquet"),
    # BGE-M3 (MIT): XLM-R large further trained for multilingual retrieval
    "bgem3": dict(_m("BAAI/bge-m3", "_bgem3", "cebg", text="ftext", bs=64, lr=1e-5, sparse_neg=6,
                     swap=True, max_len=160, score_bs=256, n_s1=250_000), file="ce_bg.parquet"),
    # mDeBERTa-v3 (MIT): EXPERIMENTAL - diverges to NaN after the first update with transformers 5.8
    # (bf16 and fp32 alike); kept for completeness, not used by any run
    "mdeberta": dict(_m("microsoft/mdeberta-v3-base", "_mdeberta", "cemd", text="ftext", bs=64, lr=2e-5,
                        sparse_neg=6, swap=True, max_len=160, prec="fp32", score_bs=512, n_s1=300_000),
                     file="ce_md.parquet"),
    # decoder LLM matcher (Apache-2.0), full fine-tune, last-token pooling
    "qwen15": dict(_m("Qwen/Qwen2.5-1.5B", "_qwen15", "ceq", text="ftext", bs=64, lr=1e-5, sparse_neg=6,
                      swap=True, max_len=192, score_bs=256, n_s1=100_000), file="ce_q.parquet"),
    # Qwen3 (Apache-2.0): reranker-pretrained 0.6B backbone (native yes/no head), and the 1.7B base model
    "qwen3rr": dict(_m("Qwen/Qwen3-Reranker-0.6B", "_qwen3rr", "ceqr", text="text", bs=64, lr=1e-5, sparse_neg=6,
                       swap=True, max_len=256, score_bs=256, n_s1=150_000, **RERANK), file="ce_qr.parquet"),
    "qwen3_17": dict(_m("Qwen/Qwen3-1.7B-Base", "_qwen3_17", "ceq3", text="ftext", bs=64, lr=1e-5, sparse_neg=6,
                        swap=True, max_len=192, score_bs=256, n_s1=100_000), file="ce_q3.parquet"),
    # LLM matchers (raw text: ~60 tokens per pair vs ~110 for the field view; an LLM needs no hand-made
    # canonical view). LoRA on a frozen bf16 backbone, hard-pair band only, head initialised from the
    # LM's yes/no answer. bs 64 amortises per-step overhead (bs 16 ran at 1.1 s/step on a 3g.90gb slice).
    # Qwen2.5-7B (7.6B params, Apache-2.0): the largest allowed class (<= 8B)
    "qwen25_7b": dict(_m("Qwen/Qwen2.5-7B", "_qwen25_7b", "ceq7", text="text", bs=64, lr=2e-4, swap=True,
                         max_len=160, score_bs=256, n_s1=200_000, lora=dict(r=16), band=HARD_BAND, **QA),
                      file="ce_q7.parquet"),
    # Qwen3-Reranker-4B (4.0B params, Apache-2.0): multilingual pair-relevance model, native template
    "qwen3rr_4b": dict(_m("Qwen/Qwen3-Reranker-4B", "_qwen3rr_4b", "ceqr4", text="text", bs=64, lr=2e-4, swap=True,
                          max_len=256, score_bs=256, n_s1=150_000, lora=dict(r=32), band=HARD_BAND,
                          **RERANK),
                       file="ce_qr4.parquet"),
    # Qwen3-4B-Base (4.0B params, Apache-2.0)
    "qwen3_4b": dict(_m("Qwen/Qwen3-4B-Base", "_qwen3_4b", "ceq4", text="text", bs=32, lr=1e-4, swap=True,
                        max_len=160, score_bs=512, n_s1=150_000, lora=dict(r=32), band=HARD_BAND, **QA),
                     file="ce_q4.parquet"),
    # tiny decoders, only for smoke-testing the LLM code paths
    "qwen05": dict(_m("Qwen/Qwen2.5-0.5B", "_qwen05", "ceq05", text="ftext", bs=32, lr=2e-5, max_len=192,
                      score_bs=256, n_s1=2_000), file="ce_q05.parquet"),
    "qwen05_lora": dict(_m("Qwen/Qwen2.5-0.5B", "_qwen05_lora", "ceq05l", text="text", bs=8, lr=1e-4, max_len=160,
                           score_bs=64, n_s1=300, lora=dict(r=8), band=HARD_BAND, **QA), file="ce_q05l.parquet"),
    "qwen3rr_smoke": dict(_m("Qwen/Qwen3-Reranker-0.6B", "_qwen3rr_smoke", "ceqrs", text="text", bs=8, lr=1e-4,
                             swap=True, max_len=256, score_bs=64, n_s1=300, lora=dict(r=8), band=HARD_BAND,
                             **RERANK), file="ce_qrs.parquet"),
}

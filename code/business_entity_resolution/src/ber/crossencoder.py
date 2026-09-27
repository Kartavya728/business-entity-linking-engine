"""Cross-encoder pair classifier used as a stage-2 feature (all backbones MIT / Apache-2.0).

Encoders (multilingual-e5, BGE-M3, mDeBERTa): "<S1 text>" [SEP] "<target text>", [CLS] pooling.
Decoders (Qwen2.5 / Qwen3): one prompt per pair (PROMPTS), left padding and last-token pooling
(the only position that has attended to both records).
  yesno=(yes, no): the linear head starts as the LM-head difference of the two answer tokens, so
  the untrained model already scores logit(yes) - logit(no) for "same business?". Training then
  starts from the LLM's own multilingual judgement instead of a random direction, which matters
  for France (no labels, vocabulary unseen in training).
Trained on encoder-fold (E) S1 entities only, using blocking candidates as hard negatives, so
its scores on ranker folds (R) and on test are out-of-sample.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

CE_BASE = "intfloat/multilingual-e5-small"
DECODER_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "phi3"}
REC_CHARS = 200  # decoder prompts: cap each record so the answer position is never truncated
PROMPTS = {
    "plain": "S1: {a}\nT: {b}\nsame business?",
    "qa": "Record A: {a}\nRecord B: {b}\nDo records A and B describe the same business? Answer:",
    # Qwen3-Reranker template (model card); the answer token follows the empty think block
    "qwen3rr": ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
                "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
                "<Instruct>: Decide whether the Query and the Document describe the same business. Both are noisy "
                "records (name | address) with typos, abbreviations and reordering.\n"
                "<Query>: {a}\n<Document>: {b}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
}


def _answer_rows(name: str, enc, tok, words) -> torch.Tensor:
    """Output-embedding rows of the answer tokens: the tied input embeddings, or lm_head.weight
    read straight from the checkpoint (AutoModel does not load the LM head)."""
    ids = []
    for w in words:
        t = tok(w, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise ValueError(f"answer {w!r} is {len(t)} tokens for {name}; pick a single-token answer")
        ids.append(t[0])
    if getattr(enc.config, "tie_word_embeddings", False):
        return enc.get_input_embeddings().weight[ids].detach().float().cpu()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = name
    if not os.path.isdir(name):
        try:  # the checkpoint is normally in the local cache already (setup_env.sh)
            d = snapshot_download(name, allow_patterns=["*.json", "*.safetensors"], local_files_only=True)
        except Exception:
            d = snapshot_download(name, allow_patterns=["*.json", "*.safetensors"])
    idx = os.path.join(d, "model.safetensors.index.json")
    shard = json.load(open(idx))["weight_map"]["lm_head.weight"] if os.path.exists(idx) else "model.safetensors"
    with safe_open(os.path.join(d, shard), "pt") as f:
        w = f.get_slice("lm_head.weight")
        return torch.cat([w[i:i + 1] for i in ids]).float()


class CrossEncoder(torch.nn.Module):
    def __init__(self, name_or_path: str = CE_BASE, max_len: int = 128, prec: str = "bf16",
                 lora: dict | None = None, adapter: str | None = None, prompt: str = "plain",
                 yesno: tuple | None = None):
        """lora: train LoRA adapters on a frozen bf16 backbone (7B-class models);
        adapter: directory of trained adapters to merge into the backbone for inference;
        prompt: decoder prompt style (PROMPTS); yesno: answer tokens that initialise the head."""
        super().__init__()
        cfg = AutoConfig.from_pretrained(name_or_path)
        self.decoder = cfg.model_type in DECODER_TYPES
        self.tok = AutoTokenizer.from_pretrained(adapter or name_or_path)
        if self.decoder:
            self.tok.padding_side = "left"
            self.tok.truncation_side = "left"  # keep the question / answer position
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        self.base_name, self.lora, self.prompt, self.yesno = name_or_path, lora, prompt, yesno
        rows = None
        if lora or adapter:
            enc = AutoModel.from_pretrained(name_or_path, dtype=torch.bfloat16)
            if yesno and not adapter:
                rows = _answer_rows(name_or_path, enc, self.tok, yesno)
            if adapter:
                from peft import PeftModel
                enc = PeftModel.from_pretrained(enc, adapter).merge_and_unload()
            else:
                from peft import LoraConfig, get_peft_model
                if lora.get("ckpt", True):  # False skips recomputation, but 4B at bs 64 x 256 tok then needs >89 GB
                    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                enc.enable_input_require_grads()
                enc = get_peft_model(enc, LoraConfig(r=lora["r"], lora_alpha=lora.get("alpha", 2 * lora["r"]),
                                                     lora_dropout=0.05, target_modules="all-linear"))
                enc.print_trainable_parameters()
        else:  # full fine-tuning keeps fp32 master weights (bf16 autocast for compute)
            enc = AutoModel.from_pretrained(name_or_path, dtype=torch.float32)
            if yesno and self.decoder:
                rows = _answer_rows(name_or_path, enc, self.tok, yesno)
        self.enc = enc
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        if rows is not None:  # logit = h . (w_yes - w_no): the LM's own yes-vs-no margin
            with torch.no_grad():
                self.head.weight.copy_((rows[0] - rows[1])[None])
                self.head.bias.zero_()
        self.max_len = max_len
        self.prec = prec

    def batch(self, a, b):
        if self.decoder:
            t = PROMPTS[self.prompt]
            txt = [t.format(a=x[:REC_CHARS], b=y[:REC_CHARS]) for x, y in zip(a, b)]
            return self.tok(txt, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return self.tok(a, b, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

    def forward(self, ids, mask):
        kw = {"use_cache": False} if self.decoder else {}  # no KV cache: one forward pass per pair
        h = self.enc(input_ids=ids, attention_mask=mask, **kw).last_hidden_state
        h = h[:, -1] if self.decoder else h[:, 0]  # left padding: last position is the last real token
        return self.head(h.to(self.head.weight.dtype)).squeeze(-1)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.prec == "bf16")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.enc.save_pretrained(path); self.tok.save_pretrained(path)  # LoRA: adapters only
        torch.save(self.head.state_dict(), f"{path}/head.pt")
        json.dump({"base": self.base_name, "lora": self.lora, "prompt": self.prompt, "yesno": self.yesno},
                  open(f"{path}/ber_ce.json", "w"))

    @classmethod
    def load(cls, path, max_len: int = 128, prec: str = "bf16"):
        meta = json.load(open(f"{path}/ber_ce.json")) if os.path.exists(f"{path}/ber_ce.json") else {}
        prompt = meta.get("prompt", "plain")  # models saved before prompts were configurable
        if meta.get("lora"):
            m = cls(meta["base"], max_len=max_len, prec=prec, adapter=path, prompt=prompt)
        else:
            m = cls(path, max_len=max_len, prec=prec, prompt=prompt)
        m.head.load_state_dict(torch.load(f"{path}/head.pt"))
        return m

    @torch.no_grad()
    def predict(self, a, b, batch_size: int = 1024) -> np.ndarray:
        self.eval().cuda()
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
        out = np.empty(len(a), dtype=np.float32)
        t0 = time.time()
        for bi, s in enumerate(range(0, len(a), batch_size)):
            idx = order[s:s + batch_size]
            t = self.batch([a[i] for i in idx], [b[i] for i in idx])
            with self.autocast():
                lo = self(t["input_ids"].cuda(), t["attention_mask"].cuda())
            out[idx] = torch.sigmoid(lo.float()).cpu().numpy()
            if bi % 2000 == 0:
                print(f"  ce predict {s:,}/{len(a):,} {time.time() - t0:.0f}s", flush=True)
        return out


def train_crossencoder(a, b, y, out_path, epochs: int = 1, batch_size: int = 256, lr: float = 3e-5,
                       seed: int = 42, base: str = CE_BASE, swap: bool = False, max_len: int = 128,
                       prec: str = "bf16", lora: dict | None = None, prompt: str = "plain",
                       yesno: tuple | None = None):
    """swap: randomly present the pair as (target, S1) - the match relation is symmetric."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CrossEncoder(base, max_len=max_len, prec=prec, lora=lora, prompt=prompt, yesno=yesno).cuda().train()
    if model.decoder and not lora:
        model.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
    n = len(a); steps = epochs * (n // batch_size); warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    y = np.asarray(y, dtype=np.float32)
    step, t0, bad = 0, time.time(), 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n - batch_size + 1, batch_size):
            idx = perm[s:s + batch_size]
            if swap:
                flip = rng.random(len(idx)) < 0.5
                t = model.batch([b[i] if f else a[i] for i, f in zip(idx, flip)],
                                [a[i] if f else b[i] for i, f in zip(idx, flip)])
            else:
                t = model.batch([a[i] for i in idx], [b[i] for i in idx])
            with model.autocast():
                lo = model(t["input_ids"].cuda(), t["attention_mask"].cuda())
            loss = F.binary_cross_entropy_with_logits(lo.float(), torch.from_numpy(y[idx]).cuda())
            if not torch.isfinite(loss):  # never let one bad batch poison the weights
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad > 50:
                    raise RuntimeError("cross-encoder training diverged (non-finite loss)")
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            opt.step(); sched.step(); step += 1
            if step % 500 == 0 or step in (1, 50, 100):
                print(f"  ce ep{ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.save(out_path)
    return model

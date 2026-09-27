"""Cross-encoder pair classifier used as a stage-2 feature (all backbones MIT / Apache-2.0).

Encoders (multilingual-e5, BGE-M3, mDeBERTa): "<S1 text>" [SEP] "<target text>", [CLS] pooling.
Decoders (Qwen2.5): one prompt "S1: <text>\\nT: <text>\\nsame business?", left padding and
last-token pooling (the only position that has attended to both records).
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


class CrossEncoder(torch.nn.Module):
    def __init__(self, name_or_path: str = CE_BASE, max_len: int = 128, prec: str = "bf16",
                 lora: dict | None = None, adapter: str | None = None):
        """lora: train LoRA adapters on a frozen bf16 backbone (7B-class models);
        adapter: directory of trained adapters to merge into the backbone for inference."""
        super().__init__()
        cfg = AutoConfig.from_pretrained(name_or_path)
        self.decoder = cfg.model_type in DECODER_TYPES
        self.tok = AutoTokenizer.from_pretrained(adapter or name_or_path)
        if self.decoder:
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        self.base_name, self.lora = name_or_path, lora
        if lora or adapter:
            enc = AutoModel.from_pretrained(name_or_path, dtype=torch.bfloat16)
            if adapter:
                from peft import PeftModel
                enc = PeftModel.from_pretrained(enc, adapter).merge_and_unload()
            else:
                from peft import LoraConfig, get_peft_model
                enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                enc.enable_input_require_grads()
                enc = get_peft_model(enc, LoraConfig(r=lora["r"], lora_alpha=lora.get("alpha", 2 * lora["r"]),
                                                     lora_dropout=0.05, target_modules="all-linear"))
                enc.print_trainable_parameters()
        else:  # full fine-tuning keeps fp32 master weights (bf16 autocast for compute)
            enc = AutoModel.from_pretrained(name_or_path, dtype=torch.float32)
        self.enc = enc
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        self.max_len = max_len
        self.prec = prec

    def batch(self, a, b):
        if self.decoder:
            txt = [f"S1: {x}\nT: {y}\nsame business?" for x, y in zip(a, b)]
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
        json.dump({"base": self.base_name, "lora": self.lora}, open(f"{path}/ber_ce.json", "w"))

    @classmethod
    def load(cls, path, max_len: int = 128, prec: str = "bf16"):
        meta = json.load(open(f"{path}/ber_ce.json")) if os.path.exists(f"{path}/ber_ce.json") else {}
        if meta.get("lora"):
            m = cls(meta["base"], max_len=max_len, prec=prec, adapter=path)
        else:
            m = cls(path, max_len=max_len, prec=prec)
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
                       prec: str = "bf16", lora: dict | None = None):
    """swap: randomly present the pair as (target, S1) - the match relation is symmetric."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CrossEncoder(base, max_len=max_len, prec=prec, lora=lora).cuda().train()
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

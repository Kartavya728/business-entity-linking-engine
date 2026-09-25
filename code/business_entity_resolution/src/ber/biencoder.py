"""Bi-encoder for dense blocking: multilingual-e5 fine-tuned with in-batch InfoNCE.

Positive pairs = (S1 text, matched S2/S3 text). Every other target in the batch is a
negative, plus (optionally) mined hard negatives. The model is MIT-licensed
(intfloat/multilingual-e5-small, 118M params).
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "intfloat/multilingual-e5-small"


class BiEncoder(torch.nn.Module):
    def __init__(self, name_or_path: str = BASE_MODEL, max_len: int = 64):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.enc = AutoModel.from_pretrained(name_or_path)
        self.max_len = max_len

    def batch(self, texts):
        return self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt")

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    def save(self, path):
        self.enc.save_pretrained(path)
        self.tok.save_pretrained(path)

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 2048, log_every: int = 1000) -> torch.Tensor:
        """Encode to L2-normalised float16 CPU tensor. Sorts by length for speed."""
        self.eval().cuda()
        order = np.argsort([len(t) for t in texts])
        out = torch.empty((len(texts), self.enc.config.hidden_size), dtype=torch.float16)
        t0 = time.time()
        for bi, s in enumerate(range(0, len(texts), batch_size)):
            idx = order[s:s + batch_size]
            b = self.batch([texts[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = self(b["input_ids"].cuda(non_blocking=True), b["attention_mask"].cuda(non_blocking=True))
            out[torch.from_numpy(idx)] = e.half().cpu()
            if log_every and bi % log_every == 0:
                print(f"  encode {s:,}/{len(texts):,} {time.time() - t0:.0f}s", flush=True)
        return out


def train_biencoder(anchors, positives, out_path, epochs: int = 1, batch_size: int = 512,
                    lr: float = 5e-5, temp: float = 0.05, hard_negs=None, seed: int = 42):
    """Fine-tune with symmetric InfoNCE over in-batch negatives.

    anchors/positives: aligned lists of texts. hard_negs: optional aligned list of texts
    appended as extra negatives for the anchor->target direction.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BiEncoder().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(anchors)
    steps = epochs * (n // batch_size)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    step, t0 = 0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n - batch_size + 1, batch_size):
            idx = perm[s:s + batch_size]
            a = model.batch([anchors[i] for i in idx])
            ptexts = [positives[i] for i in idx]
            if hard_negs is not None:
                ptexts = ptexts + [hard_negs[i] for i in idx]
            p = model.batch(ptexts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ea = model(a["input_ids"].cuda(), a["attention_mask"].cuda())
                ep_ = model(p["input_ids"].cuda(), p["attention_mask"].cuda())
            ea, ep_ = ea.float(), ep_.float()
            logits = ea @ ep_.T / temp
            lab = torch.arange(len(idx), device=logits.device)
            loss = F.cross_entropy(logits, lab)
            loss = 0.5 * (loss + F.cross_entropy(logits[:, :len(idx)].T, lab))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"  biencoder ep{ep} step {step}/{steps} loss {loss.item():.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
    model.save(out_path)
    return model

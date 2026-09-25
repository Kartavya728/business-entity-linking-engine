"""Cross-encoder pair classifier (multilingual-e5, MIT) used as a stage-2 feature.

Input: "<S1 name | address>" [SEP] "<target name | address>" -> P(match).
Trained on encoder-fold (E) S1 entities only, using blocking candidates as hard
negatives, so its scores on ranker folds (R) and on test are out-of-sample.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CE_BASE = "intfloat/multilingual-e5-small"


class CrossEncoder(torch.nn.Module):
    def __init__(self, name_or_path: str = CE_BASE, max_len: int = 128):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.enc = AutoModel.from_pretrained(name_or_path)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        self.max_len = max_len

    def batch(self, a, b):
        return self.tok(a, b, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)

    def save(self, path):
        self.enc.save_pretrained(path); self.tok.save_pretrained(path)
        torch.save(self.head.state_dict(), f"{path}/head.pt")

    @classmethod
    def load(cls, path):
        m = cls(path)
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = self(t["input_ids"].cuda(), t["attention_mask"].cuda())
            out[idx] = torch.sigmoid(lo.float()).cpu().numpy()
            if bi % 2000 == 0:
                print(f"  ce predict {s:,}/{len(a):,} {time.time() - t0:.0f}s", flush=True)
        return out


def train_crossencoder(a, b, y, out_path, epochs: int = 1, batch_size: int = 256, lr: float = 3e-5,
                       seed: int = 42, base: str = CE_BASE):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CrossEncoder(base).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(a); steps = epochs * (n // batch_size); warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    y = np.asarray(y, dtype=np.float32)
    step, t0 = 0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n - batch_size + 1, batch_size):
            idx = perm[s:s + batch_size]
            t = model.batch([a[i] for i in idx], [b[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(t["input_ids"].cuda(), t["attention_mask"].cuda())
            loss = F.binary_cross_entropy_with_logits(lo.float(), torch.from_numpy(y[idx]).cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 500 == 0:
                print(f"  ce ep{ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.save(out_path)
    return model

"""Fine-tune a cross-encoder on the pairs the feature model cannot decide.

The first-stage gradient-boosted model is near-certain about 99% of candidate
pairs and wrong almost only inside a narrow band of uncertainty. Deciding that
band perfectly would move held-out macro F_0.5 from 0.9319 to 0.9753, so the
whole remaining score sits in about 1% of pairs.

A second gradient-boosted model trained on the same features reached 0.7787
AUC inside that band against the first stage's 0.8043 -- worse. The features
have nothing further to give there, which is the argument for a model that
reads the two records as text rather than as 79 summaries of them. These are
pairs sharing a name whose addresses differ, where the question is whether
the address was damaged or belongs to a different branch, and that is a
question about the strings themselves.

MiniLM is Apache-2.0 and 22M parameters, well inside the competition's
MIT/Apache and 8B limits. Only the provided training data is used; the
pretrained weights are a general-purpose language model, not a lookup of any
business.

Attach the exported pairs as a dataset and enable GPU.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def find_csv():
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".csv") and "pair" in f:
                return os.path.join(root, f)
    raise SystemExit("training csv not found under /kaggle/input")


MODEL = os.environ.get("MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "64"))
MAXLEN = int(os.environ.get("MAXLEN", "128"))
LR = float(os.environ.get("LR", "2e-5"))
OUT = os.environ.get("OUT", "/kaggle/working")


class Pairs(Dataset):
    def __init__(self, df, tok):
        self.a = df.text_a.astype(str).tolist()
        self.b = df.text_b.astype(str).tolist()
        self.y = df.label.astype(np.float32).tolist()
        self.tok = tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def collate(batch, tok):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), truncation=True, max_length=MAXLEN,
              padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.float32)
    return enc


def auc(y, p):
    y = np.asarray(y)
    r = np.argsort(np.argsort(np.asarray(p)))
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {dev}  ({torch.cuda.get_device_name(0) if dev=='cuda' else ''})")

    df = pd.read_csv(find_csv())
    log(f"{len(df):,} pairs, {df.label.mean():.1%} positive, "
        f"{int(df.in_band.sum()):,} in band")

    # Split by Source-1 entity, never by pair: several pairs share an entity
    # and a pair-level split would put the same entity on both sides.
    ents = df.s1_id.unique()
    rng = np.random.default_rng(42)
    rng.shuffle(ents)
    va_e = set(ents[int(0.85 * len(ents)):])
    is_va = df.s1_id.isin(va_e)
    tr, va = df[~is_va], df[is_va]
    log(f"{len(tr):,} train / {len(va):,} val pairs")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, ignore_mismatched_sizes=True).to(dev)

    dl_tr = DataLoader(Pairs(tr, tok), batch_size=BATCH, shuffle=True,
                       collate_fn=lambda b: collate(b, tok), num_workers=2)
    dl_va = DataLoader(Pairs(va, tok), batch_size=BATCH * 4, shuffle=False,
                       collate_fn=lambda b: collate(b, tok), num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.1)
    lossf = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    def evaluate():
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for enc in dl_va:
                y = enc.pop("labels")
                enc = {k: v.to(dev) for k, v in enc.items()}
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    out = model(**enc).logits.squeeze(-1)
                ps.append(torch.sigmoid(out.float()).cpu().numpy())
                ys.append(y.numpy())
        model.train()
        return np.concatenate(ys), np.concatenate(ps)

    best = -1.0
    for ep in range(EPOCHS):
        model.train()
        run = 0.0
        for i, enc in enumerate(dl_tr):
            y = enc.pop("labels").to(dev)
            enc = {k: v.to(dev) for k, v in enc.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                out = model(**enc).logits.squeeze(-1)
                loss = lossf(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if i % 200 == 0:
                log(f"epoch {ep+1} step {i}/{len(dl_tr)} loss {run/max(i+1,1):.4f}")
        yv, pv = evaluate()
        band = va.in_band.to_numpy().astype(bool)
        a_all = auc(yv, pv)
        a_band = auc(yv[band], pv[band])
        a_stage1 = auc(yv[band], va.stage1.to_numpy()[band])
        log(f"epoch {ep+1}: AUC all {a_all:.4f} | IN BAND cross-encoder "
            f"{a_band:.4f} vs first stage {a_stage1:.4f}")
        if a_band > best:
            best = a_band
            model.save_pretrained(f"{OUT}/ce_model")
            tok.save_pretrained(f"{OUT}/ce_model")
            log(f"  saved (best in-band AUC {best:.4f})")
    log(f"DONE best in-band AUC {best:.4f}")
    with open(f"{OUT}/result.txt", "w") as fh:
        fh.write(f"best_in_band_auc={best:.4f}\n")


if __name__ == "__main__":
    main()

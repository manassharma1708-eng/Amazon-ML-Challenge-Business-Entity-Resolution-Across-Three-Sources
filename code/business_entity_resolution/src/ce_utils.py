"""Cross-encoder training/scoring (MiniLM, Apache 2.0) and the stage-3 features."""
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MAX_LEN = 96


def train_ce(model_name, a, b, y, out_dir, epochs=2, lr=4e-5, bs=128, device='cuda'):
    tok = AutoTokenizer.from_pretrained(model_name)
    ce = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    data = list(zip(a, b, [float(v) for v in y]))
    def collate(batch):
        aa, bb, yy = zip(*batch)
        enc = tok(list(aa), list(bb), truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor(yy, dtype=torch.float32)
        return enc
    loader = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=2)
    total = len(loader) * epochs
    opt = torch.optim.AdamW(ce.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler('cuda')
    lossf = torch.nn.BCEWithLogitsLoss()
    ce.train(); t0 = time.time()
    for ep in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            yy = batch.pop('labels')
            with torch.autocast('cuda', dtype=torch.float16):
                logits = ce(**batch).logits.squeeze(-1)
            loss = lossf(logits.float(), yy)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if step % 1000 == 0:
                print(f'  epoch {ep+1}/{epochs} step {step}/{len(loader)} loss {loss.item():.4f} {(time.time()-t0)/60:.1f} min', flush=True)
    ce.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return tok, ce


def load_ce(model_dir, device='cuda'):
    return (AutoTokenizer.from_pretrained(model_dir),
            AutoModelForSequenceClassification.from_pretrained(model_dir).to(device))


@torch.no_grad()
def score(tok, ce, a, b, bs=512, device='cuda'):
    ce.eval(); out = []
    for i in range(0, len(a), bs):
        enc = tok(a[i:i+bs], b[i:i+bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(ce(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def stage3_feats(df):
    """Stage-1 prob (p1), stage-2 prob (p3), cross-encoder score (ce, NaN outside the band) + in-shop context."""
    g = df.groupby('source1_entity_id')
    X = pd.DataFrame(index=df.index)
    X['p1'], X['p3'], X['ce'] = df['p1'], df['p3'], df['ce']
    X['p3_rank'] = g['p3'].rank(ascending=False, method='first')
    X['p3_gap'] = df['p3'] - g['p3'].transform('max')
    X['p3_share'] = df['p3'] / g['p3'].transform('sum').clip(lower=1e-6)
    X['ce_rank'] = g['ce'].rank(ascending=False, method='first')
    X['ce_gap'] = df['ce'] - g['ce'].transform('max')
    X['n_scored'] = g['ce'].transform('count')
    return X.astype('float32')

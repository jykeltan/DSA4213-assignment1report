"""Shared utilities for DSA4213 Assignment 1 (Parts I-III).
Models, batching, training loop, evaluation, timing and text generation.
"""
import math, time, copy, random, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, UNK_ID = 0, 1


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


#batching
def make_lm_tensors(ids, seq_len=64, pad_id=PAD_ID):
    """Cut a token stream into non-overlapping chunks.
    x = ids[s : s+L], y = ids[s+1 : s+L+1]  ->  every token except the first is a target exactly once.
    The last chunk is right-padded with PAD; PAD targets are ignored in the loss."""
    n = len(ids) - 1
    n_chunks = math.ceil(n / seq_len)
    X = torch.full((n_chunks, seq_len), pad_id, dtype=torch.long)
    Y = torch.full((n_chunks, seq_len), pad_id, dtype=torch.long)
    for c in range(n_chunks):
        s, e = c * seq_len, min(c * seq_len + seq_len, n)
        X[c, :e - s] = torch.tensor(ids[s:e])
        Y[c, :e - s] = torch.tensor(ids[s + 1:e + 1])
    return X, Y


#models
def _init_embedding(emb, emb_weights, freeze):
    if emb_weights is not None:
        emb.weight.data.copy_(torch.as_tensor(emb_weights, dtype=torch.float))
        emb.weight.data[PAD_ID].zero_()
    emb.weight.requires_grad = not freeze


class RecurrentLM(nn.Module):
    """Vanilla RNN (cell='rnn') or LSTM (cell='lstm') language model."""
    def __init__(self, vocab_size, emb_dim=100, hidden=256, layers=2, dropout=0.3,
                 cell='lstm', emb_weights=None, freeze_emb=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        _init_embedding(self.emb, emb_weights, freeze_emb)
        rnn_cls = nn.LSTM if cell == 'lstm' else nn.RNN
        self.rnn = rnn_cls(emb_dim, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, return_hidden=False):
        h, _ = self.rnn(self.drop(self.emb(x)))          # (B, T, hidden)
        logits = self.out(self.drop(h))
        return (logits, h) if return_hidden else logits


class TransformerLM(nn.Module):
    """Decoder-only Transformer: token emb -> linear to d_model + learned positions -> causal self-attention."""
    def __init__(self, vocab_size, emb_dim=100, d_model=256, nhead=4, layers=2, ff=512,
                 dropout=0.3, max_len=64, emb_weights=None, freeze_emb=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        _init_embedding(self.emb, emb_weights, freeze_emb)
        self.proj = nn.Linear(emb_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x, return_hidden=False):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.proj(self.emb(x)) + self.pos(pos))
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)  # causal
        h = self.norm(self.enc(h, mask=mask, is_causal=True))
        logits = self.out(h)
        return (logits, h) if return_hidden else logits


def build_model(kind, vocab_size, cfg, emb_weights=None, freeze_emb=False):
    if kind in ('rnn', 'lstm'):
        return RecurrentLM(vocab_size, cfg['emb_dim'], cfg['hidden'], cfg['layers'],
                           cfg['dropout'], cell=kind, emb_weights=emb_weights, freeze_emb=freeze_emb)
    if kind == 'transformer':
        return TransformerLM(vocab_size, cfg['emb_dim'], cfg['d_model'], cfg['nhead'], cfg['layers'],
                             cfg['ff'], cfg['dropout'], cfg['seq_len'],
                             emb_weights=emb_weights, freeze_emb=freeze_emb)
    raise ValueError(kind)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# train / evaluate
def run_epoch(model, X, Y, device, opt=None, batch_size=32, clip=1.0):
    """Returns average next-token cross-entropy per (non-PAD) target token."""
    train = opt is not None
    model.train(train)
    order = torch.randperm(len(X)) if train else torch.arange(len(X))
    tot_loss, tot_tok = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), batch_size):
            b = order[i:i + batch_size]
            x, y = X[b].to(device), Y[b].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=PAD_ID, reduction='sum')
            n = (y != PAD_ID).sum().item()
            if train:
                opt.zero_grad()
                (loss / n).backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            tot_loss += loss.item(); tot_tok += n
    return tot_loss / tot_tok


def train_lm(model, Xtr, Ytr, Xva, Yva, device, lr=1e-3, max_epochs=60, patience=3,
             batch_size=32, seed=42, verbose=True):
    """Train with AdamW + early stopping on validation loss. Restores the best-val weights."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist = {'train_loss': [], 'val_loss': [], 'epoch_time': []}
    best, best_state, bad = float('inf'), None, 0
    t_start = time.perf_counter()
    for ep in range(1, max_epochs + 1):
        sync(device); t0 = time.perf_counter()
        tr = run_epoch(model, Xtr, Ytr, device, opt, batch_size)
        sync(device); t_ep = time.perf_counter() - t0
        va = run_epoch(model, Xva, Yva, device, None, 256)
        hist['train_loss'].append(tr); hist['val_loss'].append(va); hist['epoch_time'].append(t_ep)
        if verbose:
            print(f'  epoch {ep:2d} | train {tr:.3f} | val {va:.3f} (ppl {math.exp(va):6.1f}) | {t_ep:5.1f}s')
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f'  early stop: no val improvement for {patience} epochs')
                break
    model.load_state_dict(best_state)
    hist['best_epoch'] = int(np.argmin(hist['val_loss'])) + 1
    hist['epochs_run'] = len(hist['val_loss'])
    hist['train_time_s'] = sum(hist['epoch_time'])
    hist['wall_time_s'] = time.perf_counter() - t_start
    return hist


@torch.no_grad()
def throughput(model, X, Y, device, repeats=3, batch_size=256):
    """Common inference workload: score the whole test set. Returns target tokens / second (median of repeats)."""
    model.eval()
    run_epoch(model, X[:8], Y[:8], device, None, batch_size)          # warm-up
    n_tok = (Y != PAD_ID).sum().item()
    times = []
    for _ in range(repeats):
        sync(device); t0 = time.perf_counter()
        run_epoch(model, X, Y, device, None, batch_size)
        sync(device); times.append(time.perf_counter() - t0)
    return n_tok / float(np.median(times))


# generation
def sample_next(logits, temperature, top_k, banned, gen):
    logits = logits.clone().float()
    logits[banned] = float('-inf')
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen).item())


@torch.no_grad()
def generate_neural(model, prompt_ids, n_tokens, device, temperature=0.8, top_k=40,
                    max_ctx=64, seed=42, banned=(PAD_ID, UNK_ID)):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids)
    for _ in range(n_tokens):
        x = torch.tensor([ids[-max_ctx:]], device=device)
        logits = model(x)[0, -1].cpu()
        ids.append(sample_next(logits, temperature, top_k, list(banned), gen))
    return ids[len(prompt_ids):]


def detok(tokens):
    s = ' '.join(tokens)
    s = re.sub(r' ([.,!?;:])', r'\1', s)
    return s

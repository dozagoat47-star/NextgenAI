"""Nextgen AI - Seq2Seq uretici egitimi (bagimsiz script).

colab/nextgen_seq2seq_colab.ipynb hucresinin birebir tasinmis hali.
Colab'dan farki:
  - kotk dizine bagimlilik YOK (/content yerine script klasoru kullanilir),
  - checkpoint kalici diskte tutulur ve otomatik devralinir (restart/copma
    guvenli),
  - tarayiciya ihtiyac yok: terminalde arka planda calisir.

Kullanim:
  python train_seq2seq.py                 # 250 epoch (varsayilan)
  python train_seq2seq.py --epochs 400    # daha uzun egitim
  SMOKE=1 python train_seq2seq.py         # 2 adim calis ve cik (CPU hiz testi)

Ciktilari:
  <SAVE_DIR>/seq2seq_ckpt.pt   -> kaldigi yerden devam icin
  <SAVE_DIR>/seq2seq_model.json-> yerel model/ klasorune kopyala
SAVE_DIR varsayilani script klasoru; ortam degiskeni ile asilabilir.
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from seqgen import clean_chars, build_vocab, load_pairs
from seq2seq import Seq2Seq, encode_seq2

# ---------------- hiperparametreler (numpy inference ile AYNI mimari)
TARGET_MAX_LEN = 42
D_MODEL = 128
NUM_BLOCKS = 4
NUM_HEADS = 4
FF_MULT = 3
DROPOUT = 0.10
BATCH_SIZE = 64
LR_BASE = 1e-3
LR_MIN = 0.1
WARMUP = 200
PATIENCE = 20
GRAD_CLIP = 5.0
CKPT_FREQ = 5
MAX_PAIRS = 20000
MAX_ENC_LEN = 40
MAX_DEC_LEN = 48
SEED = 7

SAVE_DIR = os.environ.get('SAVE_DIR', BASE)
os.makedirs(SAVE_DIR, exist_ok=True)
CKPT = os.path.join(SAVE_DIR, 'seq2seq_ckpt.pt')
INTENTS = os.path.join(BASE, 'intents.json')


class TorchSeq2Seq(nn.Module):
    """seq2seq.Seq2Seq ile birebir AYNI cebir (PyTorch). Paramt adlari
    numpy 'params' anahtarlariyla eslesecek sekilde duz attribute tutulur.
    Dropout yalnizca training'de; eval'da kapali -> NumPy parity bozulmaz.
    """
    def __init__(self, V, d_model=D_MODEL, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                 ff_mult=FF_MULT, max_enc_len=MAX_ENC_LEN, max_dec_len=MAX_DEC_LEN,
                 drop=DROPOUT):
        super().__init__()
        self.V, self.D = V, d_model
        self.N, self.H = num_blocks, num_heads
        self.hd = d_model // num_heads
        self.ff = ff_mult * d_model
        self.rsqrt = self.hd ** -0.5
        self.drop = drop

        def he(shape, scale=None):
            if scale is None:
                scale = math.sqrt(2.0 / shape[0])
            return torch.randn(*shape) * scale

        with torch.no_grad():
            self.embed = nn.Parameter(torch.randn(V, d_model) * 0.02)
            self.embed.data[0].zero_()
            pe = torch.zeros(max(max_enc_len, max_dec_len), d_model)
            pos = torch.arange(max(max_enc_len, max_dec_len), dtype=torch.float32).unsqueeze(1)
            dim = torch.arange(d_model // 2, dtype=torch.float32)
            div = 10000.0 ** (2.0 * dim / d_model)
            pe[:, 0::2] = torch.sin(pos / div)
            pe[:, 1::2] = torch.cos(pos / div)
            pe = pe / math.sqrt(max(d_model, 1))
        self.register_buffer('pos_enc', pe)

        for i in range(num_blocks):
            for n, dims in [('Wq', (d_model, d_model)), ('Wk', (d_model, d_model)),
                            ('Wv', (d_model, d_model)), ('Wo', (d_model, d_model))]:
                setattr(self, f'b{i}_{n}', nn.Parameter(he(dims)))
                setattr(self, f'b{i}_b{n[1:]}', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'b{i}_ln1_g', nn.Parameter(torch.ones(1, d_model)))
            setattr(self, f'b{i}_ln1_b', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'b{i}_W1', nn.Parameter(he((d_model, self.ff))))
            setattr(self, f'b{i}_b1', nn.Parameter(torch.zeros(1, self.ff)))
            setattr(self, f'b{i}_W2', nn.Parameter(he((self.ff, d_model), scale=0.02)))
            setattr(self, f'b{i}_b2', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'b{i}_ln2_g', nn.Parameter(torch.ones(1, d_model)))
            setattr(self, f'b{i}_ln2_b', nn.Parameter(torch.zeros(1, d_model)))
            for n, dims in [('Wq', (d_model, d_model)), ('Wk', (d_model, d_model)),
                            ('Wv', (d_model, d_model)), ('Wo', (d_model, d_model))]:
                setattr(self, f'db{i}_{n}', nn.Parameter(he(dims)))
                setattr(self, f'db{i}_b{n[1:]}', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'db{i}_ln1_g', nn.Parameter(torch.ones(1, d_model)))
            setattr(self, f'db{i}_ln1_b', nn.Parameter(torch.zeros(1, d_model)))
            for n, dims in [('Wq', (d_model, d_model)), ('Wk', (d_model, d_model)),
                            ('Wv', (d_model, d_model)), ('Wo', (d_model, d_model))]:
                setattr(self, f'db{i}_{n}c', nn.Parameter(he(dims)))
                setattr(self, f'db{i}_b{n[1:]}c', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'db{i}_ln2_g', nn.Parameter(torch.ones(1, d_model)))
            setattr(self, f'db{i}_ln2_b', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'db{i}_W1', nn.Parameter(he((d_model, self.ff))))
            setattr(self, f'db{i}_b1', nn.Parameter(torch.zeros(1, self.ff)))
            setattr(self, f'db{i}_W2', nn.Parameter(he((self.ff, d_model), scale=0.02)))
            setattr(self, f'db{i}_b2', nn.Parameter(torch.zeros(1, d_model)))
            setattr(self, f'db{i}_ln3_g', nn.Parameter(torch.ones(1, d_model)))
            setattr(self, f'db{i}_ln3_b', nn.Parameter(torch.zeros(1, d_model)))

        self.out_ln_g = nn.Parameter(torch.ones(1, d_model))
        self.out_ln_b = nn.Parameter(torch.zeros(1, d_model))
        self.head = nn.Parameter(he((d_model, V), scale=0.02))
        self.head_b = nn.Parameter(torch.zeros(1, V))

    def _ln(self, x, g, b):
        return torch.nn.functional.layer_norm(
            x, (x.size(-1),), g.reshape(-1), b.reshape(-1))

    @staticmethod
    def _attn(q, k, v, rsqrt, heads, hd, mask=None, causal=False,
              drop_p=0.0, training=False):
        B, T, d = q.shape
        S = k.shape[1]

        def split(x):
            return x.reshape(B, x.shape[1], heads, hd).transpose(1, 2)
        Q, K, V = split(q), split(k), split(v)
        scores = (Q @ K.transpose(-1, -2)) * rsqrt
        if mask is not None:
            scores = scores.masked_fill((mask <= 0)[:, None, None, :], -1e9)
        if causal:
            tri = torch.triu(torch.full((T, T), -1e9, device=q.device), 1)
            scores = scores + tri[None, None]
        p = torch.softmax(scores, dim=-1)
        if training and drop_p > 0:
            p = torch.nn.functional.dropout(p, drop_p)
        return (p @ V).transpose(1, 2).reshape(B, T, d)

    def _ffn(self, x, W1, b1, W2, b2):
        return torch.nn.functional.gelu(x @ W1 + b1, approximate='tanh') @ W2 + b2

    def forward(self, enc, dec):
        B, T1 = enc.shape
        T2 = dec.shape[1]
        em = (enc != 0).float()
        x = self.embed[enc] * em.unsqueeze(-1)
        x = x + self.pos_enc[:T1][None] * em.unsqueeze(-1)
        if self.training and self.drop > 0:
            x = torch.nn.functional.dropout(x, self.drop)
        for i in range(self.N):
            pre = self._ln(x, getattr(self, f'b{i}_ln1_g'), getattr(self, f'b{i}_ln1_b'))
            a = self._attn(pre, pre, pre, self.rsqrt, self.H, self.hd, mask=em,
                          drop_p=0.05, training=self.training)
            x = x + a
            pre = self._ln(x, getattr(self, f'b{i}_ln2_g'), getattr(self, f'b{i}_ln2_b'))
            x = x + self._ffn(pre, getattr(self, f'b{i}_W1'), getattr(self, f'b{i}_b1'),
                              getattr(self, f'b{i}_W2'), getattr(self, f'b{i}_b2'))
        enc_out = x

        y = self.embed[dec]
        y = y + self.pos_enc[:T2][None]
        if self.training and self.drop > 0:
            y = torch.nn.functional.dropout(y, self.drop)
        for i in range(self.N):
            pre = self._ln(y, getattr(self, f'db{i}_ln1_g'), getattr(self, f'db{i}_ln1_b'))
            a = self._attn(pre, pre, pre, self.rsqrt, self.H, self.hd,
                          causal=True, drop_p=0.05, training=self.training)
            y = y + a
            pre = self._ln(y, getattr(self, f'db{i}_ln2_g'), getattr(self, f'db{i}_ln2_b'))
            c = self._attn(pre, enc_out, enc_out, self.rsqrt, self.H, self.hd,
                          mask=em, drop_p=0.05, training=self.training)
            y = y + c
            pre = self._ln(y, getattr(self, f'db{i}_ln3_g'), getattr(self, f'db{i}_ln3_b'))
            y = y + self._ffn(pre, getattr(self, f'db{i}_W1'), getattr(self, f'db{i}_b1'),
                              getattr(self, f'db{i}_W2'), getattr(self, f'db{i}_b2'))
        h = self._ln(y, self.out_ln_g, self.out_ln_b)
        return h @ self.head + self.head_b


def seq2_loss(logits, tgt, mask):
    lg = torch.log_softmax(logits, dim=-1)
    nll = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return -(nll * mask).sum() / mask.sum().clamp(min=1.0)


def masked_acc(logits, tgt, mask):
    ok = ((logits.argmax(-1) == tgt) & mask.bool())
    return ok.sum().item() / mask.sum().clamp(min=1.0).item()


def refine_resp(r, maxc=TARGET_MAX_LEN):
    r = (r or '').strip()
    if len(r) < 10:
        return None
    if len(r) > maxc:
        cut = 0
        for i in range(15, min(len(r), maxc + 8)):
            if r[i] in '.?:!':
                cut = i
        r = r[:cut + 1].strip() if cut > 0 else r[:maxc].strip()
    if r and r[-1] not in '.?!...':
        r = r + '.'
    return r if len(r) >= 10 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    args = ap.parse_args()
    EPOCHS = args.epochs

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('PyTorch', torch.__version__, '| device:', DEVICE,
          '| GPU:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '-',
          '| SAVE_DIR:', SAVE_DIR, flush=True)

    # ---------------- veri
    assert os.path.exists(INTENTS), f'intents.json bulunamadi: {INTENTS}'
    pairs = load_pairs(INTENTS, max_pairs=MAX_PAIRS, use_query=True)
    print('egitim cifti (sorgu, yanit):', len(pairs), flush=True)

    pairs = [(ctx, rr) for ctx, r in pairs if (rr := refine_resp(r)) is not None]
    print('rafine hedef sonrasi cift:', len(pairs), flush=True)

    all_text = []
    for ctx, resp in pairs:
        all_text.append(ctx)
        all_text.append(resp)
    vocab = build_vocab(all_text)
    print('karakter sozlugu:', len(vocab), flush=True)

    dummy = Seq2Seq(vocab, d_model=4, num_blocks=1, num_heads=1,
                    max_enc_len=MAX_ENC_LEN, max_dec_len=MAX_DEC_LEN, seed=SEED)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(pairs))
    n_val = max(1, int(0.1 * len(pairs)))
    tr_pairs = [pairs[i] for i in perm[n_val:]]
    va_pairs = [pairs[i] for i in perm[:n_val]]

    def make_batches(pairs_, B, dec_len=MAX_DEC_LEN):
        items = sorted(pairs_, key=lambda pr: len(pr[0]))
        batches = []
        for i in range(0, len(items), B):
            block = items[i:i + B]
            encs, dins, dtgts, dmasks = [], [], [], []
            maxe = 0
            for ctx, resp in block:
                e, d1, d2, m1, m2 = encode_seq2(dummy, ctx, resp, max_dec=dec_len)
                encs.append(e)
                dins.append(d1)
                dtgts.append(d2)
                dmasks.append(m2)
                maxe = max(maxe, len(e))
            E = np.zeros((len(block), maxe), np.int64)
            EM = np.zeros((len(block), maxe), np.float32)
            for j, e in enumerate(encs):
                E[j, :len(e)] = e
                EM[j, :len(e)] = 1.0
            DI = np.stack(dins)
            DT = np.stack(dtgts)
            DM = np.stack(dmasks)
            batches.append((E, DI, DT, EM, DM))
        return batches

    tr = make_batches(tr_pairs, BATCH_SIZE)
    va = make_batches(va_pairs, BATCH_SIZE)
    print('train batch:', len(tr), '| val batch:', len(va), flush=True)
    print('ornek cift:', tr_pairs[0], flush=True)

    trE = [torch.from_numpy(b[0]).long().to(DEVICE) for b in tr]
    trI = [torch.from_numpy(b[1]).long().to(DEVICE) for b in tr]
    trT = [torch.from_numpy(b[2]).long().to(DEVICE) for b in tr]
    trM = [torch.from_numpy(b[4]).float().to(DEVICE) for b in tr]
    vaE = [torch.from_numpy(b[0]).long().to(DEVICE) for b in va]
    vaI = [torch.from_numpy(b[1]).long().to(DEVICE) for b in va]
    vaT = [torch.from_numpy(b[2]).long().to(DEVICE) for b in va]
    vaM = [torch.from_numpy(b[4]).float().to(DEVICE) for b in va]

    # ---------------- model + resume
    model = TorchSeq2Seq(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_BASE)

    best_state = None
    best_val = 1e9
    bad = 0
    start_ep = 0
    step = 0
    tot_steps = EPOCHS * len(trE)

    if os.path.exists(CKPT):
        cp = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(cp['model'])
        opt.load_state_dict(cp['opt'])
        best_val, best_state, start_ep, step = cp['best_val'], cp['best_state'], cp['epoch'], cp['step']
        best_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}
        print('Devam: epoch', start_ep, '| step', step, '| best val:', round(best_val, 4), flush=True)

    # ---------------- egitim
    t0_all = time.time()
    done = False
    for ep in range(start_ep + 1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        order = list(range(len(trE)))
        random.Random(ep).shuffle(order)
        tl = 0.0
        for bi in order:
            step += 1
            cur = LR_BASE
            if step <= WARMUP:
                cur = LR_BASE * (step / WARMUP)
            else:
                prog = (step - WARMUP) / max(1, tot_steps - WARMUP)
                cur = LR_BASE * (LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g['lr'] = cur
            opt.zero_grad()
            loss = seq2_loss(model(trE[bi], trI[bi]), trT[bi], trM[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += loss.item()
            if os.environ.get('SMOKE') and step >= 2:
                print('SMOKE OK:', float(loss.item()), flush=True)
                return 0
        tl /= len(trE)

        model.eval()
        vl = va_acc = 0.0
        with torch.no_grad():
            for e, i, t, m in zip(vaE, vaI, vaT, vaM):
                lg = model(e, i)
                vl += seq2_loss(lg, t, m).item()
                va_acc += masked_acc(lg, t, m)
        vl /= len(vaE)
        va_acc /= len(vaE)
        print(f'epoch {ep:3d}/{EPOCHS} | train {tl:.4f} | val {vl:.4f} | acc {va_acc:.3f} | '
              f'{time.time()-t0:.1f}s | lr {cur:.5f}', flush=True)

        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'[seq2seq] Erken durdurma. Best val: {best_val:.4f}', flush=True)
                done = True
        if ep % CKPT_FREQ == 0 or done:
            torch.save({'epoch': ep, 'step': step, 'model': best_state,
                        'opt': opt.state_dict(), 'best_val': best_val,
                        'best_state': best_state}, CKPT)
            print(f'  checkpoint -> {CKPT}', flush=True)
        if done:
            break

    print('\nToplam egitim suresi: %.1f dk' % ((time.time() - t0_all) / 60), flush=True)

    # ---------------- export (nuimpy inference ile uyumlu JSON)
    model.load_state_dict(best_state)
    model.eval()
    data = {
        'arch': 'seq2seq',
        'V': len(vocab),
        'd_model': D_MODEL, 'num_blocks': NUM_BLOCKS, 'num_heads': NUM_HEADS,
        'ff_mult': FF_MULT,
        'max_enc_len': MAX_ENC_LEN, 'max_dec_len': MAX_DEC_LEN,
        'vocab': vocab,
        'params': {k: v.numpy().tolist() for k, v in best_state.items()},
    }
    dest = os.path.join(SAVE_DIR, 'seq2seq_model.json')
    with io.open(dest, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('model yazildi:', dest, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
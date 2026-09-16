"""Nextgen AI - Decoder-only LLM (llm.py) egitimi (bagimsiz script).

colab ve lightning.ai uyumlu PyTorch egitim scripti. Nucleo inference (llm.py)
ile Torch modeli BIREBIR ayni cebiri calistirir; dropout yalnizca training'de,
eval'da kapali -> parity bozulmaz.

Veri formati (tek zincir, otoregresif):
    <PAD> <BOS> <sorgu> <SEP> <yanit> <EOS>
    <PAD> <BOS> <sorgu> <SEP> <bilgi-parcasi> <SEP> <yanit> <EOS>   (--rag)
  Kayip YALNIZCA yanit pozisyonlarinda; --rag ile bilgi parcasi bağlam
  olur -> model verilen bilgiden OZGUN cumle kurmayi (akil yurutme) ogrenir.

Kullanim:
  python train_llm.py                       # 250 epoch, sorgu-koullu yanit
  python train_llm.py --rag                 # corpus'tan bilgi-parcali egitim
  python train_llm.py --natural 3           # her cevabin 3 dogal varyantiyla buyut
  python train_llm.py --rag --natural 3     # bilgi-parcali + dogal varyant (oneri)
  python train_llm.py --epochs 400          # daha uzun egitim
  python train_llm.py --dry-run [--rag] [--natural K]  # torch'suz veri dogrulama
  SMOKE=1 python train_llm.py               # 2 adim calis ve cik (CPU hiz testi)

Ciktilari:
  <SAVE_DIR>/llm_ckpt.pt    -> kaldigi yerden devam (restart/copma guvenli)
  <SAVE_DIR>/llm_model.json -> yerel model/ klasorune kopyala (llm.load_llm)
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

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    torch = nn = None
    HAVE_TORCH = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from seqgen import clean_chars, load_pairs
from llm import PAD, LLM, build_llm_vocab, encode_llm
from naturalize import naturalize_pairs

# ---------------- hiperparametreler (numpy inference ile AYNI mimari)
D_MODEL = 128
NUM_BLOCKS = 4
NUM_HEADS = 4
FF_MULT = 4
DROPOUT = 0.10
BATCH_SIZE = 64
LR_BASE = 1e-3
LR_MIN = 0.1
WARMUP = 200
PATIENCE = 20
GRAD_CLIP = 5.0
CKPT_FREQ = 5
MAX_PAIRS = 20000
MAX_CTX_LEN = 40
MAX_SEQ_LEN = 160
SEED = 7

SAVE_DIR = os.environ.get('SAVE_DIR', BASE)
os.makedirs(SAVE_DIR, exist_ok=True)
CKPT = os.path.join(SAVE_DIR, 'llm_ckpt.pt')
INTENTS = os.path.join(BASE, 'intents.json')


if HAVE_TORCH:
    class TorchLLM(nn.Module):
        """llm.LLM ile birebir AYNI cebir (PyTorch). Parametre adlari
        numpy 'params' anahtarlariyla eslesecek sekilde duz attribute tutulur
        (embed, b{i}_Wq/bk/bv/bo/ln/W1/W2, out_ln_g/b, head, head_b).
        Dropout yalnizca training'de; eval'da kapali -> NumPy parity bozulmaz.
        """
        def __init__(self, V, d_model=D_MODEL, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                     ff_mult=FF_MULT, max_seq_len=MAX_SEQ_LEN, drop=DROPOUT):
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
                pe = torch.zeros(max_seq_len, d_model)
                pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
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

            self.out_ln_g = nn.Parameter(torch.ones(1, d_model))
            self.out_ln_b = nn.Parameter(torch.zeros(1, d_model))
            self.head = nn.Parameter(he((d_model, V), scale=0.02))
            self.head_b = nn.Parameter(torch.zeros(1, V))

        def _ln(self, x, g, b):
            return torch.nn.functional.layer_norm(
                x, (x.size(-1),), g.reshape(-1), b.reshape(-1))

        @staticmethod
        def _causal_attn(q, k, v, rsqrt, heads, hd, drop_p=0.0, training=False):
            B, T, d = q.shape
            Q = q.reshape(B, T, heads, hd).transpose(1, 2)
            K = k.reshape(B, T, heads, hd).transpose(1, 2)
            V = v.reshape(B, T, heads, hd).transpose(1, 2)
            scores = (Q @ K.transpose(-1, -2)) * rsqrt
            tri = torch.triu(torch.full((T, T), -1e9, device=q.device), 1)
            scores = scores + tri[None, None]
            p = torch.softmax(scores, dim=-1)
            if training and drop_p > 0:
                p = torch.nn.functional.dropout(p, drop_p)
            return (p @ V).transpose(1, 2).reshape(B, T, d)

        def _ffn(self, x, i):
            W1, b1 = getattr(self, f'b{i}_W1'), getattr(self, f'b{i}_b1')
            W2, b2 = getattr(self, f'b{i}_W2'), getattr(self, f'b{i}_b2')
            return torch.nn.functional.gelu(x @ W1 + b1, approximate='tanh') @ W2 + b2

        def forward(self, X):
            """X:(B,T) token ids -> logits:(B,T,V) (tam sekans, otoregresif)."""
            T = X.shape[1]
            x = self.embed[X]
            x = x + self.pos_enc[:T][None]
            if self.training and self.drop > 0:
                x = torch.nn.functional.dropout(x, self.drop)
            for i in range(self.N):
                pre = self._ln(x, getattr(self, f'b{i}_ln1_g'), getattr(self, f'b{i}_ln1_b'))
                q = pre @ getattr(self, f'b{i}_Wq') + getattr(self, f'b{i}_bq')
                k = pre @ getattr(self, f'b{i}_Wk') + getattr(self, f'b{i}_bk')
                v = pre @ getattr(self, f'b{i}_Wv') + getattr(self, f'b{i}_bv')
                a = self._causal_attn(q, k, v, self.rsqrt, self.H, self.hd,
                                      drop_p=0.05, training=self.training)
                x = x + a @ getattr(self, f'b{i}_Wo') + getattr(self, f'b{i}_bo')
                pre = self._ln(x, getattr(self, f'b{i}_ln2_g'), getattr(self, f'b{i}_ln2_b'))
                x = x + self._ffn(pre, i)
            h = self._ln(x, self.out_ln_g, self.out_ln_b)
            return h @ self.head + self.head_b


def llm_loss(logits, tgt, mask):
    """Otoregresif next-token: logits[t] -> tgt[t+1] (KENDI token'i degil)."""
    lg = torch.log_softmax(logits, dim=-1)
    nxt = torch.full_like(tgt, PAD)
    nxt[:, :-1] = tgt[:, 1:]
    nll = lg.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
    return -(nll * mask).sum() / mask.sum().clamp(min=1.0)


def masked_acc(logits, tgt, mask):
    nxt = torch.full_like(tgt, PAD)
    nxt[:, :-1] = tgt[:, 1:]
    ok = ((logits.argmax(-1) == nxt) & mask.bool())
    return ok.sum().item() / mask.sum().clamp(min=1.0).item()


def refine_resp(r, maxc=MAX_SEQ_LEN - MAX_CTX_LEN - 4):
    r = (r or '').strip()
    if len(r) < 10:
        return None
    if len(r) > maxc:
        cut = 0
        for i in range(15, min(len(r), maxc + 8)):
            if r[i] in '.?:!':
                cut = i
        r = r[:cut + 1].strip() if cut > 0 else r[:maxc].strip()
    if not r:
        return None
    return r


def prepare_data(RAG, NATURAL=0):
    """Veri + RAG hattini HAZIRLAR (yalnizca numpy; torch gerektirmez).
    --dry-run bu fonksiyonu calistirip dogrular; egitim de ayni yolu kullanir.

    NATURAL > 0 ise her (sorgu, yanit) cifti, yanitin dogal varyantlariyla
    cogaltilir (naturalize_pairs): model ayni icerigi pek cok dogal sekilde
    ifade etmeyi ogrenip kopya-yerine-canli-sohbet icin veri kazanir."""
    assert os.path.exists(INTENTS), f'intents.json bulunamadi: {INTENTS}'
    pairs = load_pairs(INTENTS, max_pairs=MAX_PAIRS, use_query=True)
    pairs = [(ctx, rr) for ctx, r in pairs if (rr := refine_resp(r)) is not None]
    if NATURAL > 0:
        n_before = len(pairs)
        pairs = naturalize_pairs(pairs, k=NATURAL)
        print(f'dogal varyant: {n_before} -> {len(pairs)} cift'
              f' (varyant: {NATURAL})', flush=True)
    print('egitim cifti (sorgu, yanit):', len(pairs), flush=True)

    all_text = []
    for ctx, resp in pairs:
        all_text.append(ctx)
        all_text.append(resp)
    vocab = build_llm_vocab(all_text)
    print('karakter sozlugu:', len(vocab), flush=True)

    # Dummy numpy modeli yalnizca encode icin (c2i + uzunluk bilgisi)
    dummy = LLM(vocab, d_model=4, num_blocks=1, num_heads=1,
                max_ctx_len=MAX_CTX_LEN, max_seq_len=MAX_SEQ_LEN, seed=SEED)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(pairs))
    n_val = max(1, int(0.1 * len(pairs)))
    tr_pairs = [pairs[i] for i in perm[n_val:]]
    va_pairs = [pairs[i] for i in perm[:n_val]]

    # ---- RAG: her (sorgu, yanit) ciftine corpus'tan ilgili bilgi parcasi
    ctx_map = {}
    corpus = None
    if RAG:
        try:
            from corpus import Corpus
            corpus = Corpus()
            corpus.load()
            print('RAG corpus yuklendi (parca:', len(corpus.chunks), ')', flush=True)
        except Exception as e:
            corpus = None
            print('RAG corpus yuklenemedi, bilgi-parcasiz egitim:', e, flush=True)

    def kb_for(ctx):
        if corpus is None:
            return None
        try:
            chunk = corpus.search(ctx)
            if not chunk:
                return None
            text = ((chunk.get('title') or '') + '. ' +
                    (chunk.get('text') or ''))[:200]
            return text if len(text.strip()) >= 20 else None
        except Exception:
            return None

    if RAG:
        hits = 0
        for ctx, _ in tr_pairs + va_pairs:
            if ctx in ctx_map:
                continue
            ctx_map[ctx] = kb_for(ctx)
            if ctx_map[ctx]:
                hits += 1
        print(f'RAG contextli ornek: {hits}/{len(tr_pairs) + len(va_pairs)}', flush=True)

    def make_batches(pairs_, B):
        items = sorted(pairs_, key=lambda pr: len(pr[0]))
        batches = []
        for i in range(0, len(items), B):
            block = items[i:i + B]
            seqs, masks = [], []
            for ctx, resp in block:
                seq, smask = encode_llm(dummy, ctx, resp,
                                        context=ctx_map.get(ctx))
                seqs.append(seq)
                masks.append(smask)
            X = np.stack(seqs)
            M = np.stack(masks)
            batches.append((X, M))
        return batches

    tr = make_batches(tr_pairs, BATCH_SIZE)
    va = make_batches(va_pairs, BATCH_SIZE)
    print('train batch:', len(tr), '| val batch:', len(va), flush=True)
    print('ornek cift:', (clean_chars(tr_pairs[0][0], 30),
                          clean_chars(tr_pairs[0][1], 30)), flush=True)
    return {'vocab': vocab, 'tr': tr, 'va': va, 'ctx_map': ctx_map}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--rag', action='store_true',
                    help='corpus.jsonl riddaren bilgi-parcalariyla koullu egitim')
    ap.add_argument('--natural', type=int, default=0, metavar='K',
                    help='her cevabin K dogal varyantiyla veriyi buyut (orijinal dahil)')
    ap.add_argument('--dry-run', action='store_true',
                    help='torch olmadan veri/RAG hattini dogrula ve cik')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='erken durdurmada calinmasi gereken iyilesmesiz epoch sayisi '
                         '(ornegin --patience 250 ile neredeyse tamamen devre disi)')
    args = ap.parse_args()
    EPOCHS = args.epochs
    RAG = args.rag
    NATURAL = args.natural
    patience = args.patience

    if args.dry_run:
        d = prepare_data(RAG, NATURAL=NATURAL)
        ex = next((c for c in d['ctx_map'].values() if c), None)
        print('DRY-RUN OK: train batch', len(d['tr']), '| val batch',
              len(d['va']), '| vocab', len(d['vocab']), flush=True)
        if RAG:
            print('RAG ornek baslam:', (ex or '')[:80].replace('\n', ' '), flush=True)
        return 0

    if not HAVE_TORCH:
        print('Eksik: PyTorch kurulu degil. Egitim Colab/Lightning AI ortaminda '
              'calistirilir; burada yalnizca --dry-run ile veri hattini '
              'dogrulayabilirsin.', flush=True)
        return 2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('PyTorch', torch.__version__, '| device:', DEVICE,
          '| GPU:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '-',
          '| SAVE_DIR:', SAVE_DIR, '| RAG:', RAG, '| patience:', patience, flush=True)

    # ---------------- veri
    d = prepare_data(RAG, NATURAL=NATURAL)
    vocab = d['vocab']
    tr, va = d['tr'], d['va']

    trX = [torch.from_numpy(b[0]).long().to(DEVICE) for b in tr]
    trM = [torch.from_numpy(b[1]).float().to(DEVICE) for b in tr]
    vaX = [torch.from_numpy(b[0]).long().to(DEVICE) for b in va]
    vaM = [torch.from_numpy(b[1]).float().to(DEVICE) for b in va]

    # ---------------- model + resume
    model = TorchLLM(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_BASE)

    best_state = None
    best_val = 1e9
    bad = 0
    start_ep = 0
    step = 0
    tot_steps = EPOCHS * len(trX)

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
        order = list(range(len(trX)))
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
            loss = llm_loss(model(trX[bi]), trX[bi], trM[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += loss.item()
            if os.environ.get('SMOKE') and step >= 2:
                print('SMOKE OK:', float(loss.item()), flush=True)
                return 0
        tl /= len(trX)

        model.eval()
        vl = va_acc = 0.0
        with torch.no_grad():
            for x, m in zip(vaX, vaM):
                lg = model(x)
                vl += llm_loss(lg, x, m).item()
                va_acc += masked_acc(lg, x, m)
        vl /= len(vaX)
        va_acc /= len(vaX)
        print(f'epoch {ep:3d}/{EPOCHS} | train {tl:.4f} | val {vl:.4f} | acc {va_acc:.3f} | '
              f'{time.time()-t0:.1f}s | lr {cur:.5f}', flush=True)

        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f'[llm] Erken durdurma. Best val: {best_val:.4f}', flush=True)
                done = True
        if ep % CKPT_FREQ == 0 or done:
            torch.save({'epoch': ep, 'step': step, 'model': best_state,
                        'opt': opt.state_dict(), 'best_val': best_val,
                        'best_state': best_state}, CKPT)
            print(f'  checkpoint -> {CKPT}', flush=True)
        if done:
            break

    print('\nToplam egitim suresi: %.1f dk' % ((time.time() - t0_all) / 60), flush=True)

    # ---------------- export (numpy inference ile uyumlu JSON)
    model.load_state_dict(best_state)
    model.eval()
    data = {
        'arch': 'llm',
        'V': len(vocab),
        'd_model': D_MODEL, 'num_blocks': NUM_BLOCKS, 'num_heads': NUM_HEADS,
        'ff_mult': FF_MULT,
        'max_ctx_len': MAX_CTX_LEN, 'max_seq_len': MAX_SEQ_LEN,
        'vocab': vocab,
        'params': {k: v.numpy().tolist() for k, v in best_state.items()},
    }
    dest = os.path.join(SAVE_DIR, 'llm_model.json')
    with io.open(dest, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('model yazildi:', dest, flush=True)

    # --------- parity dogrulamasi: ayni girdide numpy logits vs torch logits
    x = trX[0][:4]
    with torch.no_grad():
        torch_logits = model(x).float().numpy() if DEVICE == 'cpu' else \
            model(x).cpu().float().numpy()
    np_model = LLM(vocab, d_model=D_MODEL, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                   ff_mult=FF_MULT, max_ctx_len=MAX_CTX_LEN, max_seq_len=MAX_SEQ_LEN,
                   seed=SEED)
    np_model.params = {k: np.asarray(v, np.float32) for k, v in best_state.items()}
    numpy_logits = np_model.forward(x.detach().cpu().numpy())
    diff = float(np.max(np.abs(torch_logits - numpy_logits)))
    print(f'parity max-abs fark: {diff:.6f} (beklenen < 1e-3)', flush=True)
    assert diff < 1e-3, f'Parity bozuk: {diff}'
    return 0


if __name__ == '__main__':
    sys.exit(main())
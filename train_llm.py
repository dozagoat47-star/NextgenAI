"""Nextgen AI - Decoder-only LLM (llm.py) egitimi (bagimsiz script).

Kaggle/Colab/Lightning AI uyumlu PyTorch egitim scripti. Nucleo inference
(llm.py) ile Torch modeli BIREBIR ayni cebiri calistirir; dropout yalnizca
training'de, eval'da kapali -> parity bozulmaz.

Veri formati (tek zincir, otoregresif):
    <PAD> <BOS> <sorgu> <SEP> <yanit> <EOS>
    <PAD> <BOS> <sorgu> <SEP> <bilgi-parcasi> <SEP> <yanit> <EOS>   (--rag)
  Kayip YALNIZCA yanit pozisyonlarinda; --rag ile bilgi parcasi bağlam
  olur -> model verilen bilgiden OZGUN cumle kurmayi (akil yurutme) ogrenir.

Kullanim (yerel dogrulama icin torch gerektirmez):
  python train_llm.py --dry-run [--rag] [--natural K]   # veri hattini dogrula
  SMOKE=1 python train_llm.py                            # 2 adim CPU hiz testi

Kaggle'da egitim (GPU notebook):
  python train_llm.py --rag --natural 3                  # varsayilan buyuk model
  python train_llm.py --rag --natural 3 --d-model 384 --num-blocks 6
  python train_llm.py --rag --natural 3 --max-ctx-len 48 --max-seq-len 192
  python train_llm.py --epochs 400 --patience 40 --batch-size 64

Kapasite flag'leri: --d-model --num-blocks --num-heads --ff-mult
  --max-ctx-len --max-seq-len --batch-size --lr-base. Varsayilani d=256'dir;
  eski (d=128) model dosyalari veri ogesi tasidigi icin geriye donuk yuklenir.

Ciktilari:
  <SAVE_DIR>/llm_ckpt.pt    -> kaldigi yerden devam (restart/copma guvenli)
  <export-dir>/llm_model.json + llm_model_weights.npz
Notebook'ta SAVE_DIR='../working' ayarla; indirilen iki dosyayi yerel
model/ klasorune kopyala (llm.load_llm otomatik agar).
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
from llm import PAD, LLM, build_llm_vocab, encode_llm, load_tokenizer
from naturalize import naturalize_pairs

# ---------------- hiperparametreler (numpy inference ile AYNI mimari)
# Varsayilanlar "buyuk" kapasite icindir (d=256); eski model dosyalari veri
# ogesi oldugu icin geriye donuk yuklenmeye devam eder. Kaggle'da daha
# buyuk model icin:  python train_llm.py --d-model 384 --num-blocks 6
D_MODEL = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
FF_MULT = 4
DROPOUT = 0.10
BATCH_SIZE = 48
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


FUNCTIONAL_OPENERS = frozenset("""
    merhaba selam selamlar selamet hey merhabalar gunaydin gunaydinlar
    iyi iyiyim iyiym hosgeldin hosgeldiniz hosbulduk
    evet tabii tabi tabii ki elbette peki aynen kesinlikle dogru
    harika super guzel cok cok guzel muhtesem bayagi baya
    bir bu su o ben benim biz sana size lutfen rica ederim rica
    paylasayim vereyim anlatayim soyleyeyim yazayim bakayim dusunelim
    vaktim seve seve memnuniyetle elbette ki
    no tamam olur olur tabi evet tabii
""".split())


def stabilize_first_words(pairs, seed=SEED):
    """IŞLEVSEL açılışların İLK KELİMESİNİ moda sabitler (içerik korunur).

    Ocak (pattern) başına yanıtlar farklı IŞLEVSEL açılış kelimeleriyle
    başladığında (merhaba/selam/selamlar, evet/tabii/elbette, harika/süper...)
    model ilk yanıt-tokenini (o=0) ayırt edemez -> ilk-token doğruluğu
    düşer, bu token sampling'i zehirler. Burada YALNIZCA işlevsel açılış
    kelimeleri o pattern için mod işlevsel kelimeye hizalanır; İÇERİK
    taşıyan ilk kelimeler (tarif adı, isim, sayı...) olduğu gibi bırakılır
    (anlam bozulmaz). naturalize ilk kelimeyi zaten korur -> uyumludur.

    intents.json DEĞİŞTİRİLMEZ; dönüşüm yalnızca eğitim çiftlerinde olur.
    """
    by_ctx = {}
    for i, (ctx, _resp) in enumerate(pairs):
        by_ctx.setdefault(ctx, []).append(i)
    changed = 0
    for _ctx, idxs in by_ctx.items():
        func_counts = {}
        for i in idxs:
            w = pairs[i][1].strip().split()
            if not w:
                continue
            w0 = w[0].strip('.,;:!?…"\'()').lower()
            if w0 in FUNCTIONAL_OPENERS:
                func_counts[w0] = func_counts.get(w0, 0) + 1
        if len(func_counts) < 2:
            continue
        mode = max(func_counts, key=func_counts.get)
        for i in idxs:
            parts = pairs[i][1].strip().split()
            if not parts:
                continue
            w0 = parts[0].strip('.,;:!?…"\'()').lower()
            if w0 not in FUNCTIONAL_OPENERS:
                continue
            cap = parts[0][:1].isupper()
            newfirst = mode if not cap else mode[:1].upper() + mode[1:]
            parts[0] = newfirst
            pairs[i] = (pairs[i][0], ' '.join(parts))
            changed += 1
    print(f'islevsel acilis sabitendi ({changed} yanit)', flush=True)
    return pairs


def prepare_data(RAG, NATURAL=0, tokenizer=None, kb_map_path=None,
                 FIRST_WORD_STABILIZE=True, max_ctx_len=MAX_CTX_LEN,
                 max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE):
    """Veri + RAG hattini HAZIRLAR (yalnizca numpy; torch gerektirmez).
    --dry-run bu fonksiyonu calistirip dogrular; egitim de ayni yolu kullanir.

    NATURAL > 0 ise her (sorgu, yanit) cifti, yanitin dogal varyantlariyla
    cogaltilir (naturalize_pairs): model ayni icerigi pek cok dogal sekilde
    ifade etmeyi ogrenip kopya-yerine-canli-sohbet icin veri kazanir.

    kb_map_path verilirse (enrich_intents.py uretimi knowledge_map.jsonl)
    desen->bilgi parcasini CANLI corpus.search yerine ezberlenmis haritadan
    alir: deterministik (RAG hit orani %100'la simrek istedigimiz bilgi
    intent'leri icin guvenli) ve corplar shisha yapisirken hizlidir.

    tokenizer (BPETokenizer) verilirse BPE modu kullanilir (subword vocab);
    yoksa eski karakter sozlugu (build_llm_vocab) kullanilir."""
    assert os.path.exists(INTENTS), f'intents.json bulunamadi: {INTENTS}'
    pairs = load_pairs(INTENTS, max_pairs=MAX_PAIRS, use_query=True)
    pairs = [(ctx, rr) for ctx, r in pairs if (rr := refine_resp(r)) is not None]
    if FIRST_WORD_STABILIZE:
        n_before = len(pairs)
        pairs = stabilize_first_words(pairs)
        print(f'islevsel acilis sabitlendi: {n_before} cift'
              f' (yalnizca islevsel acilislar -> mod; icerik korunur),'
              f' o=0 entropisi dusuruldu', flush=True)
    if NATURAL > 0:
        n_before = len(pairs)
        pairs = naturalize_pairs(pairs, k=NATURAL)
        print(f'dogal varyant: {n_before} -> {len(pairs)} cift'
              f' (varyant: {NATURAL})', flush=True)
    print('egitim cifti (sorgu, yanit):', len(pairs), flush=True)

    vocab = None
    if tokenizer is None:
        all_text = []
        for ctx, resp in pairs:
            all_text.append(ctx)
            all_text.append(resp)
        vocab = build_llm_vocab(all_text)
        print('karakter sozlugu:', len(vocab), flush=True)

    # Dummy modeli yalnizca encode icin (c2i/tokenizer + uzunluk bilgisi)
    dummy = LLM(vocab, d_model=4, num_blocks=1, num_heads=1,
                max_ctx_len=max_ctx_len, max_seq_len=max_seq_len,
                seed=SEED, tokenizer=tokenizer)
    if tokenizer is not None:
        print('BPE tokenizer: vocab =', len(tokenizer), flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(pairs))
    n_val = max(1, int(0.1 * len(pairs)))
    tr_pairs = [pairs[i] for i in perm[n_val:]]
    va_pairs = [pairs[i] for i in perm[:n_val]]

    # ---- RAG: her (sorgu, yanit) ciftine corpus'tan ilgili bilgi parcasi
    ctx_map = {}
    corpus = None
    kb_pre = {}
    if kb_map_path and os.path.exists(kb_map_path):
        with io.open(kb_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get('ctx') and row.get('text'):
                    kb_pre[row['ctx']] = row['text']
        print('kb-map yuklendi (desen):', len(kb_pre), flush=True)
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
        if ctx in kb_pre:
            return kb_pre[ctx]
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

    tr = make_batches(tr_pairs, batch_size)
    va = make_batches(va_pairs, batch_size)
    print('train batch:', len(tr), '| val batch:', len(va), flush=True)
    print('ornek cift:', (clean_chars(tr_pairs[0][0], 30),
                          clean_chars(tr_pairs[0][1], 30)), flush=True)
    return {'vocab': vocab, 'tokenizer': tokenizer,
            'tr': tr, 'va': va, 'ctx_map': ctx_map}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--rag', action='store_true',
                    help='corpus.jsonl riddaren bilgi-parcalariyla koullu egitim')
    ap.add_argument('--kb-map', default=None, metavar='PATH',
                    help='enrich_intents.py uretimi knowledge_map.jsonl; '
                         'RAG desen->parca eslemesini corpus.search yerine '
                         'bu haritadan alir (deterministik, --rag ile birlikte)')
    ap.add_argument('--natural', type=int, default=0, metavar='K',
                    help='her cevabin K dogal varyantiyla veriyi buyut (orijinal dahil)')
    ap.add_argument('--dry-run', action='store_true',
                    help='torch olmadan veri/RAG hattini dogrula ve cik')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='erken durdurmada calinmasi gereken iyilesmesiz epoch sayisi '
                         '(ornegin --patience 250 ile neredeyse tamamen devre disi)')
    ap.add_argument('--d-model', type=int, default=D_MODEL, help='gizli boyut')
    ap.add_argument('--num-blocks', type=int, default=NUM_BLOCKS, help='transformer blok sayisi')
    ap.add_argument('--num-heads', type=int, default=NUM_HEADS, help='dikkat kafa sayisi')
    ap.add_argument('--ff-mult', type=int, default=FF_MULT, help='FFN genisletme carpani')
    ap.add_argument('--max-ctx-len', type=int, default=MAX_CTX_LEN,
                    help='sorgu icin token butcesi (secenek: 48)')
    ap.add_argument('--max-seq-len', type=int, default=MAX_SEQ_LEN,
                    help='toplam sekans uzunlugu (secenek: 192)')
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--lr-base', type=float, default=LR_BASE)
    ap.add_argument('--export-dir', default=None,
                    help='llm_model.json + _weights.npz ciktisi (varsayilan: SAVE_DIR)')
    args = ap.parse_args()
    EPOCHS = args.epochs
    RAG = args.rag
    NATURAL = args.natural
    patience = args.patience
    dm, nb, nh, ff = args.d_model, args.num_blocks, args.num_heads, args.ff_mult
    mxc, mxs = args.max_ctx_len, args.max_seq_len
    bs, lr_base = args.batch_size, args.lr_base
    export_dir = args.export_dir or SAVE_DIR
    if dm % nh != 0:
        raise SystemExit(f'--d-model {dm} --num-heads {nh} ile bolunebilir olmali')
    os.makedirs(export_dir, exist_ok=True)

    print('CONFIG: d_model=%d blocks=%d heads=%d ff_mult=%d '
          'max_ctx=%d max_seq=%d batch=%d lr=%.1e export=%s' % (
              dm, nb, nh, ff, mxc, mxs, bs, lr_base, export_dir), flush=True)

    if args.dry_run:
        d = prepare_data(RAG, NATURAL=NATURAL, tokenizer=load_tokenizer(),
                         kb_map_path=args.kb_map, max_ctx_len=mxc, max_seq_len=mxs,
                         batch_size=bs)
        ex = next((c for c in d['ctx_map'].values() if c), None)
        print('DRY-RUN OK: train batch', len(d['tr']), '| val batch',
              len(d['va']), '| tokenizer', d['tokenizer'].vocab_size
              if d['tokenizer'] else len(d['vocab']), flush=True)
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
    d = prepare_data(RAG, NATURAL=NATURAL, tokenizer=load_tokenizer(),
                     kb_map_path=args.kb_map, max_ctx_len=mxc, max_seq_len=mxs,
                     batch_size=bs)
    vocab = d['vocab']
    tok = d['tokenizer']
    V = tok.vocab_size if tok is not None else len(vocab)
    tr, va = d['tr'], d['va']

    trX = [torch.from_numpy(b[0]).long().to(DEVICE) for b in tr]
    trM = [torch.from_numpy(b[1]).float().to(DEVICE) for b in tr]
    vaX = [torch.from_numpy(b[0]).long().to(DEVICE) for b in va]
    vaM = [torch.from_numpy(b[1]).float().to(DEVICE) for b in va]

    # ---------------- model + resume
    model = TorchLLM(V, d_model=dm, num_blocks=nb, num_heads=nh, ff_mult=ff,
                     max_seq_len=mxs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr_base)

    best_state = None
    best_val = 1e9
    bad = 0
    start_ep = 0
    step = 0
    tot_steps = EPOCHS * len(trX)

    if os.path.exists(CKPT):
        cp = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        arch = cp.get('arch', {})
        same_arch = (arch.get('d_model') == dm and arch.get('num_blocks') == nb
                     and arch.get('num_heads') == nh and arch.get('ff_mult') == ff
                     and arch.get('max_seq_len') == mxs and arch.get('V') == V)
        if not same_arch:
            print('Uyari: mevcut checkpoint baska mimaride '
                  '(beklenen d=%s blk=%s k=%s f=%s mxs=%s V=%s). '
                  'Sifirdan basliyorum (eski dosya korunur).' % (
                      arch.get('d_model'), arch.get('num_blocks'),
                      arch.get('num_heads'), arch.get('ff_mult'),
                      arch.get('max_seq_len'), arch.get('V')), flush=True)
        else:
            model.load_state_dict(cp['model'])
            opt.load_state_dict(cp['opt'])
            best_val, best_state, start_ep, step = cp['best_val'], cp['best_state'], cp['epoch'], cp['step']
            best_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}
            print('Devam: epoch', start_ep, '| step', step,
                  '| best val:', round(best_val, 4), flush=True)

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
            cur = lr_base
            if step <= WARMUP:
                cur = lr_base * (step / WARMUP)
            else:
                prog = (step - WARMUP) / max(1, tot_steps - WARMUP)
                cur = lr_base * (LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * prog)))
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
                        'best_state': best_state,
                        'arch': {'d_model': dm, 'num_blocks': nb, 'num_heads': nh,
                                 'ff_mult': ff, 'max_seq_len': mxs, 'V': V}}, CKPT)
            print(f'  checkpoint -> {CKPT}', flush=True)
        if done:
            break

    print('\nToplam egitim suresi: %.1f dk' % ((time.time() - t0_all) / 60), flush=True)

    # ---------------- export (numpy inference ile uyumlu compact NPZ format)
    model.load_state_dict(best_state)
    model.eval()
    data = {
        'arch': 'llm',
        'V': V,
        'd_model': dm, 'num_blocks': nb, 'num_heads': nh,
        'ff_mult': ff,
        'max_ctx_len': mxc, 'max_seq_len': mxs,
        'weights_file': 'llm_model_weights.npz',
    }
    if tok is not None:
        data['tok_mode'] = 'bpe'
        data['vocab'] = None
        data['tokenizer'] = {
            'specials': tok.specials,
            'chars': tok.chars,
            'merges': [list(m) for m in tok.merges],
        }
    else:
        data['tok_mode'] = 'char'
        data['vocab'] = vocab
    dest = os.path.join(export_dir, 'llm_model.json')
    with io.open(dest, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    weights_dest = os.path.join(export_dir, 'llm_model_weights.npz')
    np.savez(weights_dest,
             **{k: np.asarray(v, dtype=np.float32) for k, v in best_state.items()})
    print('model yazildi:', dest, flush=True)
    print('agirliklar yazildi:', weights_dest, flush=True)

    # --------- parity 1: dogrudan tensor -> numpy (best_state uzerinden)
    x = trX[0][:4]
    with torch.no_grad():
        torch_logits = model(x).float().numpy() if DEVICE == 'cpu' else \
            model(x).cpu().float().numpy()
    np_model = LLM(vocab, d_model=dm, num_blocks=nb, num_heads=nh,
                   ff_mult=ff, max_ctx_len=mxc, max_seq_len=mxs,
                   seed=SEED, tokenizer=tok)
    np_model.params = {k: np.asarray(v, np.float32) for k, v in best_state.items()}
    numpy_logits = np_model.forward(x.detach().cpu().numpy())
    diff = float(np.max(np.abs(torch_logits - numpy_logits)))
    print(f'parity dogrudan: {diff:.6f} (beklenen < 1e-3)', flush=True)
    assert diff < 1e-3, f'Parity bozuk: {diff}'

    # --------- parity 2: export round-trip (data + npz -> from_dict -> forward)
    json_model = LLM(['<PAD>', '<BOS>', '<SEP>', '<EOS>']).from_dict(
        data, weights_path=weights_dest)
    json_logits = json_model.forward(x.detach().cpu().numpy())
    diff2 = float(np.max(np.abs(torch_logits - json_logits)))
    print(f'parity JSON round-trip: {diff2:.6f} (beklenen < 1e-3)', flush=True)

    # --------- agirlik karsilastirmasi: her param key icin max-abs fark
    worst_key, worst_val = '', 0.0
    for k in sorted(set(json_model.params) & set(best_state)):
        d = float(np.max(np.abs(json_model.params[k] -
                                np.asarray(best_state[k].cpu().numpy(), np.float32))))
        if d > worst_val:
            worst_val = d
            worst_key = k
    print(f'agirlik en buyuk fark: {worst_key} = {worst_val:.3e}', flush=True)

    # --------- val-batch karsilastirmasi: ilk val batch'te torch vs json numpy
    if va:
        xv, mv = va[0]
        with torch.no_grad():
            tv = model(torch.from_numpy(xv).long().to(DEVICE))
            tv = tv.cpu().float().numpy()
        nv = json_model.forward(xv)
        diff3 = float(np.max(np.abs(tv - nv)))
        xv_t = torch.from_numpy(xv).long()
        mv_t = torch.from_numpy(mv).float()
        tv_acc = masked_acc(torch.from_numpy(tv), xv_t, mv_t)
        nv_acc = masked_acc(torch.from_numpy(nv), xv_t, mv_t)
        print(f'val-batch: torch acc={tv_acc:.3f} | json-numpy acc={nv_acc:.3f} | logit-fark={diff3:.3e}', flush=True)

    if diff2 >= 1e-3:
        print('[HATA] JSON round-trip parity tutarsiz!', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
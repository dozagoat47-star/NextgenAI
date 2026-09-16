"""
Nextgen AI - LLM: Decoder-only GPT benzeri dil modeli (NumPy inference)
=========================================================================
Sorgu-koullu, karakter-seviye otoregresif uretec. Egitim Colab/Lightning AI'da
PyTorch ile yapilir (train_llm.py), agirliklar model/llm_model.json formatinda
saklanir; bu modul YALNIZCA ileri gecis + ornekleme yapar (backprop yok).

Mimari (GPT stili decoder-only):
  - Paylasilan karakter embedding + sinusoidal konum kodlari (1/sqrt(d))
  - N adet Pre-LN blok: causal multi-head self-attn + GELU-FFN + residual
  - Final Pre-LN + lineer kafa -> karakter logitleri

Sekans formati (tek zincir, otoregresif teacher forcing):
    <PAD> <BOS> <sorgu> <SEP> <yanit> <EOS>
  Kayip YALNIZCA <SEP> sonrasi (yanit) pozisyonlarinda hesaplanir.

Ornekleme: temperature + top-k + EOS durdurma + 6-gram tekrar korunmasi + tekrar cezasi.

Parametre semasi ('params' sozlugu -- train_llm.py export'uyla birebir):
  embed                          (V,d)
  b{i}_Wq/Wk/Wv/Wo, bq,bk,bv,bo  self-attn projeksiyonlari (i=0..N-1)
  b{i}_ln1_g/b                   attention oncesi LayerNorm
  b{i}_W1/b1, b{i}_W2/b2         FFN (d -> ff -> d, GELU)
  b{i}_ln2_g/b                   FFN oncesi LayerNorm
  out_ln_g/b                     final LayerNorm
  head (d,V), head_b (1,V)       karakter logit kafasi
"""

import io
import json
import math
import os
import re

import numpy as np

from seqgen import clean_chars

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'llm_model.json')

# <PAD> <BOS> <SEP> <EOS> (sira build_vocab ile hizali tutulmalidir)
PAD, BOS, SEP, EOS = 0, 1, 2, 3


def softmax(z, axis=-1):
    """Kararli (max-kaydirmali) softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


_GELU_A = np.float32(0.7978845608028654)
_GELU_B = np.float32(0.044715)


def gelu(x):
    """GELU (tanh yaklasikligi, GPT/BERT stili)."""
    t = np.tanh(_GELU_A * (x + _GELU_B * x ** 3))
    return np.float32(0.5) * x * (np.float32(1.0) + t)


def ln(x, gamma, beta, eps=np.float32(1e-5)):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    return gamma * (x - mean) / std + beta


def _f32(a):
    return np.asarray(a, dtype=np.float32)


class LLM:
    """GPT-benzeri decoder-only karakter seviyesi dil modeli (NumPy)."""

    def __init__(self, vocab, d_model=128, num_blocks=4, num_heads=4, ff_mult=4,
                 max_ctx_len=40, max_seq_len=160, seed=7):
        self.vocab = list(vocab)
        self.c2i = {ch: i for i, ch in enumerate(self.vocab)}
        self.i2c = {i: ch for i, ch in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads
        self.ff_dim = int(ff_mult) * self.d_model
        self.max_ctx_len = int(max_ctx_len)
        self.max_seq_len = int(max_seq_len)
        self.rsqrt = np.float32(1.0 / math.sqrt(self.head_dim))
        self.params = self._init_params(seed)
        self.pos_enc = self._sinusoidal(self.max_seq_len)

    # ------------------------------------------------------------ KURULUM
    def _init_params(self, seed):
        rng = np.random.RandomState(seed)
        d, ff, N = self.d_model, self.ff_dim, self.num_blocks

        def he(shape, scale=None):
            if scale is None:
                scale = math.sqrt(2.0 / shape[0])
            return _f32(rng.randn(*shape) * scale)

        p = {'embed': _f32(rng.randn(self.V, d) * 0.02)}
        p['embed'][PAD] = 0.0
        for i in range(N):
            for n in ('Wq', 'Wk', 'Wv', 'Wo'):
                p[f'b{i}_{n}'] = he((d, d))
                p[f'b{i}_b{n[1:]}'] = np.zeros((1, d), np.float32)
            p[f'b{i}_ln1_g'] = np.ones((1, d), np.float32)
            p[f'b{i}_ln1_b'] = np.zeros((1, d), np.float32)
            p[f'b{i}_W1'] = he((d, ff))
            p[f'b{i}_b1'] = np.zeros((1, ff), np.float32)
            p[f'b{i}_W2'] = he((ff, d), scale=0.02)
            p[f'b{i}_b2'] = np.zeros((1, d), np.float32)
            p[f'b{i}_ln2_g'] = np.ones((1, d), np.float32)
            p[f'b{i}_ln2_b'] = np.zeros((1, d), np.float32)
        p['out_ln_g'] = np.ones((1, d), np.float32)
        p['out_ln_b'] = np.zeros((1, d), np.float32)
        p['head'] = he((d, self.V), scale=0.02)
        p['head_b'] = np.zeros((1, self.V), np.float32)
        return p

    def _sinusoidal(self, length):
        d = self.d_model
        pe = np.zeros((length, d), np.float32)
        pos = np.arange(length, dtype=np.float32)[:, None]
        dim = np.arange(d // 2, dtype=np.float32)
        div = np.power(np.float32(10000.0), np.float32(2.0) * dim / np.float32(d))
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        pe *= np.float32(1.0 / math.sqrt(max(d, 1)))
        return pe

    # ------------------------------------------------------------ DİKKAT
    def _attn(self, x, i, causal=True):
        """x:(B,T,d) -> self-attn ciktisi (B,T,d). causal=yukari-ucgen maske."""
        B, T, d = x.shape
        H, hd = self.num_heads, self.head_dim
        p = self.params
        Q = x @ p[f'b{i}_Wq'] + p[f'b{i}_bq']
        K = x @ p[f'b{i}_Wk'] + p[f'b{i}_bk']
        V = x @ p[f'b{i}_Wv'] + p[f'b{i}_bv']
        Qh = Q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        scores = Qh @ Kh.transpose(0, 1, 3, 2) * self.rsqrt
        if causal:
            tri = np.triu(np.ones((T, T), np.float32), k=1)   # j>i yasa
            scores = scores + tri[None, None] * np.float32(-1e9)
        a = softmax(scores, axis=-1)
        out = (a @ Vh).transpose(0, 2, 1, 3).reshape(B, T, d)
        return out @ p[f'b{i}_Wo'] + p[f'b{i}_bo']

    def _ffn(self, x, i):
        p = self.params
        h = gelu(x @ p[f'b{i}_W1'] + p[f'b{i}_b1'])
        return h @ p[f'b{i}_W2'] + p[f'b{i}_b2']

    # ------------------------------------------------------------ İLERİ
    def forward(self, X):
        """X:(B,T) token ids -> logits:(B,T,V) (tam sekans, otoregresif)."""
        p = self.params
        x = p['embed'][X]
        T = X.shape[1]
        x = x + self.pos_enc[:T][None, :, :]
        for i in range(self.num_blocks):
            pre = ln(x, p[f'b{i}_ln1_g'], p[f'b{i}_ln1_b'])
            x = x + self._attn(pre, i)
            pre = ln(x, p[f'b{i}_ln2_g'], p[f'b{i}_ln2_b'])
            x = x + self._ffn(pre, i)
        h = ln(x, p['out_ln_g'], p['out_ln_b'])
        return h @ p['head'] + p['head_b']

    # ------------------------------------------------------------ ORNEKLEME
    def _sample_next(self, logits, temperature, top_k, banned):
        probs = softmax(logits, axis=-1)
        if abs(temperature - 1.0) > 1e-9:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
        ids = np.argsort(probs)[::-1]
        pool = [int(i) for i in ids if int(i) not in banned]
        if not pool:
            return EOS
        pool = pool[:top_k]
        ps = probs[pool]
        ps = ps / max(float(ps.sum()), 1e-12)
        return int(np.random.choice(pool, p=ps))

    def sample(self, context, temperature=0.7, top_k=10, max_len=None,
               knowledge=None, rep_penalty=0.3):
        """Sorgu -> karakter karakter yanit uret (EOS'a ya da max_len'e kadar).

        Baslangic sekansi:
          bilgi yok:    <BOS><sorgu><SEP> | <uretilen yanit>
          bilgi var:    <BOS><sorgu><SEP><bilgi><SEP> | <uretilen yanit>
        Bilgi (knowledge) ile koullandirilir -> RAG tabanli uretim: model
        soguk ezber yerine bilgi metninden yola cikarak yeni cumle kurmayi
        ogrenir (train_llm.py --rag ile bu formatta egitilir).

        rep_penalty: daha once uretilmis token'lardan sonra logit dusurur
        (0.0 = ceza yok; 0.3 = dengeli cesitlilik).
        """
        max_len = max_len or 96
        ctx = clean_chars(context, self.max_ctx_len)
        ids = [self.c2i[ch] for ch in ctx if ch in self.c2i]
        if ctx and not ids:
            ids = [self.c2i.get(' ', PAD)]

        dec = [BOS] + list(ids[:self.max_ctx_len]) + [SEP]
        if knowledge:
            kmid = [self.c2i[ch] for ch in clean_chars(knowledge, self.max_ctx_len)
                    if ch in self.c2i]
            if kmid:
                dec += list(kmid[:self.max_ctx_len]) + [SEP]
        banned = {PAD, BOS}
        out_chars = []
        seen_ngrams = set()
        generated = set()
        for _ in range(max_len):
            logits = self.forward(np.array([dec], np.int64))[0, -1]
            if rep_penalty and generated:
                for idx in generated:
                    logits[idx] -= rep_penalty
            idx = self._sample_next(logits, temperature, top_k, banned)
            if idx == EOS or idx == SEP:
                break
            out_chars.append(idx)
            dec.append(idx)
            generated.add(idx)
            if len(dec) >= self.max_seq_len - 2:
                break
            if len(out_chars) >= 6:
                ngram = tuple(out_chars[-6:])
                if ngram in seen_ngrams:
                    del out_chars[-6:]
                    break
                seen_ngrams.add(ngram)

        return self._decode_clean(''.join(self.i2c[i] for i in out_chars))

    @staticmethod
    def _decode_clean(text):
        text = text.strip()
        if not text:
            return ''
        text = re.sub(r'\s+([,.;:!?…])', r'\1', text)
        text = re.sub(r'([,.;:!?…])(?=[a-zçğıöşü0-9])', r'\1 ', text)
        return text[:1].upper() + text[1:]

    # ------------------------------------------------------------ KAYDET/YUKLE
    def to_dict(self):
        return {
            'arch': 'llm',
            'V': self.V,
            'd_model': self.d_model,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'ff_mult': self.ff_dim // self.d_model,
            'max_ctx_len': self.max_ctx_len,
            'max_seq_len': self.max_seq_len,
            'vocab': self.vocab,
            'params': {k: np.asarray(v, np.float32).tolist()
                       for k, v in self.params.items()},
        }

    def from_dict(self, data):
        self.vocab = list(data['vocab'])
        self.c2i = {ch: i for i, ch in enumerate(self.vocab)}
        self.i2c = {i: ch for i, ch in enumerate(self.vocab)}
        self.V = int(data['V'])
        self.d_model = int(data['d_model'])
        self.num_blocks = int(data['num_blocks'])
        self.num_heads = int(data['num_heads'])
        self.head_dim = self.d_model // self.num_heads
        self.ff_dim = int(data['ff_mult']) * self.d_model
        self.max_ctx_len = int(data.get('max_ctx_len', 40))
        self.max_seq_len = int(data.get('max_seq_len', 160))
        self.rsqrt = np.float32(1.0 / math.sqrt(self.head_dim))
        self.params = {k: _f32(v) for k, v in data['params'].items()}
        self.pos_enc = self._sinusoidal(self.max_seq_len)
        return self


def build_llm_vocab(texts, min_count=2):
    """Veriden karakter sozlugu kurar. 0/1/2/3 = PAD/BOS/SEP/EOS (sirayla)."""
    counts = {}
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1
    vocab = ['<PAD>', '<BOS>', '<SEP>', '<EOS>']
    for ch, cnt in sorted(counts.items()):
        if cnt >= min_count:
            vocab.append(ch)
    return vocab


def encode_llm(model, ctx, resp, context=None, max_seq=None):
    """(sorgu, yanit) ciftini decoder-only egitim girdisine cevirir.

    context verilirse RAG formati kullanilir (bilgi-parcalariyla koullu
    egitim; train_llm.py --rag):
        seq   = [<BOS>]+sorgu+[<SEP>]+bilgi+[<SEP>]+yanit+[<EOS>]
        (bilgi yoksa): [<BOS>]+sorgu+[<SEP>]+yanit+[<EOS>]

    Returns:
        (seq, smask)  (numpy dizileri, uzunluklari esit)
        smask = 1 yalnizca yanit (bilgi <SEP> sonrasi) pozisyonlarinda
    """
    max_seq = int(max_seq or getattr(model, 'max_seq_len', 160))
    max_ctx = int(getattr(model, 'max_ctx_len', 40))
    kb_budget = max(8, max_seq - max_ctx - 8)
    cc = clean_chars(ctx, max_ctx)
    rr = clean_chars(resp, kb_budget)
    ck = clean_chars(context, kb_budget) if context else ''
    enc = [model.c2i[ch] for ch in cc if ch in model.c2i]
    kenc = [model.c2i[ch] for ch in ck if ch in model.c2i]
    renc = [model.c2i[ch] for ch in rr if ch in model.c2i]
    if not enc:
        enc = [model.c2i.get(' ', PAD)]
    if not renc:
        renc = [model.c2i.get('.', PAD)]
    seq = [BOS] + list(enc) + [SEP]
    if kenc:
        seq += list(kenc) + [SEP]
    start = len(seq)                       # yanit baslangici (<SEP> sonrasi)
    renc = renc[:max_seq - start - 1]      # yalnizca yanit kismi kirpilir
    if not renc:
        renc = [model.c2i.get('.', PAD)]
    seq += list(renc) + [EOS]
    seq = seq[:max_seq]
    start = min(start, len(seq))
    arr = np.full(max_seq, PAD, np.int64)
    arr[:len(seq)] = seq
    smask = np.zeros(max_seq, np.float32)
    smask[max(0, start - 1):len(seq) - 1] = 1.0
    return arr, smask


def load_llm(path=MODEL_PATH):
    """model/llm_model.json'dan LLM yukler (yoksa None)."""
    if not os.path.exists(path):
        return None
    with io.open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data or data.get('arch') != 'llm':
        return None
    return LLM(['<PAD>', '<BOS>', '<SEP>', '<EOS>']).from_dict(data)


def save_llm(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(model.to_dict(), f, ensure_ascii=False)
    print(f"[llm] Model kaydedildi: {path}")


if __name__ == '__main__':
    # Yerel smoke test: egitimsiz (rastgele) model hizla yanit verir mi?
    import sys
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    corpus = ['merhaba nasilsin iyiyim sen nasilsin',
              'bugun hava cok guzel yuruyecek misin',
              'nerelisin istanbuldan geliyorum']
    vocab = build_llm_vocab(corpus, min_count=1)
    m = LLM(vocab, d_model=32, num_blocks=2, num_heads=2,
            max_ctx_len=40, max_seq_len=80, seed=7)
    text = m.sample('merhaba nasilsin', max_len=30)
    print('sample:', repr(text))
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), 'llm_test.json')
    save_llm(m, tmp)
    m2 = load_llm(tmp)
    assert m2 is not None and m2.params.keys() == m.params.keys()
    print('round-trip OK:', tmp)
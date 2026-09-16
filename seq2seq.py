"""
Nextgen AI - Seq2Seq: NumPy inference transformer encoder-decoder
==================================================================
(Kullanici sorgusu) -> (dogal Turkce yanit) ureten kosullu uretec.

Hazir kutupname/API YOK; yalnizca numpy. EGITIM Colab'da PyTorch+GPU ile
yapilir (colab/nextgen_seq2seq_colab.ipynb), agirliklar bu modulun anladigi
JSON formatina (`model/seq2seq_model.json`) aktarilir. Bu modul YALNIZCA
ileri gecis + ornekleme yapar (backprop yok).

Mimari (cikti dosyasiyla birebir eslesen adlar):
  - Paylasilan karakter embedding + sinusoidal konum kodlari
  - N adet ENCODER blogu: Pre-LN self-attn + Pre-LN FFN (GELU)
  - N adet DECODER blogu: Pre-LN causal self-attn + Pre-LN cross-attn(enc) + Pre-LN FFN
  - Final Pre-LN + lineer kafa -> karakter softmax

Ornekleme: temperature + top-k + EOS durdurma + 6-gram tekrar korunmasi.

Parametre semasi (dosyadaki 'params' sozlugu):
  embed                      (V,d)
  b{i}_Wq/Wk/Wv/Wo,bq,bk,bv,bo     encoder self-attn (i=0..N-1)
  b{i}_ln1_g/b, b{i}_W1/b1, b{i}_W2/b2, b{i}_ln2_g/b
  db{i}_Wq/.../bo                   decoder self-attn
  db{i}_ln1_g/b
  db{i}_Wqc/Wkc/Wvc/Woc, bqc,bkc,bvc,boc   decoder cross-attn (enc'e)
  db{i}_ln2_g/b
  db{i}_W1/b1, db{i}_W2/b2, db{i}_ln3_g/b
  out_ln_g/b, head (d,V), head_b (V,)
"""

import io
import json
import math
import os
import re

import numpy as np

from seqgen import clean_chars

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'seq2seq_model.json')

PAD, BOS, EOS = 0, 1, 2


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


_GELU_A = np.float32(0.7978845608028654)
_GELU_B = np.float32(0.044715)


def gelu(x):
    t = np.tanh(_GELU_A * (x + _GELU_B * x ** 3))
    return np.float32(0.5) * x * (np.float32(1.0) + t)


def ln(x, gamma, beta, eps=np.float32(1e-5)):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _f32(a):
    return np.asarray(a, dtype=np.float32)


class Seq2Seq:
    """Sorgu-koullu transformer encoder-decoder (NumPy inference)."""

    def __init__(self, vocab, d_model=96, num_blocks=3, num_heads=3, ff_mult=3,
                 max_enc_len=40, max_dec_len=80, seed=7):
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
        self.max_enc_len = int(max_enc_len)
        self.max_dec_len = int(max_dec_len)
        self.rsqrt = np.float32(1.0 / math.sqrt(self.head_dim))
        self.params = self._init_params(seed)
        self.pos_enc = self._sinusoidal(max(self.max_enc_len, self.max_dec_len))

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

            for n in ('Wq', 'Wk', 'Wv', 'Wo'):
                p[f'db{i}_{n}'] = he((d, d))
                p[f'db{i}_b{n[1:]}'] = np.zeros((1, d), np.float32)
            p[f'db{i}_ln1_g'] = np.ones((1, d), np.float32)
            p[f'db{i}_ln1_b'] = np.zeros((1, d), np.float32)
            for n in ('Wq', 'Wk', 'Wv', 'Wo'):
                p[f'db{i}_{n}c'] = he((d, d))
                p[f'db{i}_b{n[1:]}c'] = np.zeros((1, d), np.float32)
            p[f'db{i}_ln2_g'] = np.ones((1, d), np.float32)
            p[f'db{i}_ln2_b'] = np.zeros((1, d), np.float32)
            p[f'db{i}_W1'] = he((d, ff))
            p[f'db{i}_b1'] = np.zeros((1, ff), np.float32)
            p[f'db{i}_W2'] = he((ff, d), scale=0.02)
            p[f'db{i}_b2'] = np.zeros((1, d), np.float32)
            p[f'db{i}_ln3_g'] = np.ones((1, d), np.float32)
            p[f'db{i}_ln3_b'] = np.zeros((1, d), np.float32)

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

    # ------------------------------------------------------------ ATTENTION
    def _attn(self, q, k, v, mask=None, causal=False):
        """q:(B,T,d) k,v:(B,S,d) -> out:(B,T,d). mask: (B,S) key padding."""
        B, T, d = q.shape
        S = k.shape[1]
        H, hd = self.num_heads, self.head_dim
        Q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        K = k.reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        V = v.reshape(B, S, H, hd).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) * self.rsqrt      # (B,H,T,S)
        if mask is not None:
            key_bad = (mask <= 0)[:, None, None, :]            # (B,1,1,S)
            scores = np.where(key_bad, np.float32(-1e9), scores)
        if causal:
            tri = np.triu(np.ones((T, T), np.float32), k=1)   # j>i yasa
            scores = scores + tri[None, None] * np.float32(-1e9)
        p = softmax(scores, axis=-1)
        out = (p @ V).transpose(0, 2, 1, 3).reshape(B, T, d)
        return out

    def _ffn(self, x, i, prefix):
        p = self.params
        h = gelu(x @ p[f'{prefix}{i}_W1'] + p[f'{prefix}{i}_b1'])
        return h @ p[f'{prefix}{i}_W2'] + p[f'{prefix}{i}_b2']

    def _encode(self, X):
        """X:(B,T1) -> enc_out:(B,T1,d), enc_mask:(B,T1)."""
        p = self.params
        B, T = X.shape
        mask = (X != PAD).astype(np.float32)
        x = p['embed'][X] * mask[:, :, None]
        x = x + self.pos_enc[:T][None, :, :] * mask[:, :, None]
        for i in range(self.num_blocks):
            pre = ln(x, p[f'b{i}_ln1_g'], p[f'b{i}_ln1_b'])
            a = self._attn(pre, pre, pre, mask=mask)
            x = x + a
            pre = ln(x, p[f'b{i}_ln2_g'], p[f'b{i}_ln2_b'])
            x = x + self._ffn(pre, i, 'b')
        return x, mask

    def _decoder_logits(self, enc_out, enc_mask, dec_ids):
        """dec_ids:(B,T2) -> logits:(B,T2,V) (autoragresif tam ileri gecis)."""
        p = self.params
        B, T2 = dec_ids.shape
        mask = np.ones((B, T2), np.float32)
        x = p['embed'][dec_ids] * mask[:, :, None]
        x = x + self.pos_enc[:T2][None, :, :]
        for i in range(self.num_blocks):
            pre = ln(x, p[f'db{i}_ln1_g'], p[f'db{i}_ln1_b'])
            a = self._attn(pre, pre, pre, causal=True)
            x = x + a
            pre = ln(x, p[f'db{i}_ln2_g'], p[f'db{i}_ln2_b'])
            c = self._attn(pre, enc_out, enc_out, mask=enc_mask)
            x = x + c
            pre = ln(x, p[f'db{i}_ln3_g'], p[f'db{i}_ln3_b'])
            x = x + self._ffn(pre, i, 'db')
        h = ln(x, p['out_ln_g'], p['out_ln_b'])
        return h @ p['head'] + p['head_b']

    # ------------------------------------------------------------ ORNEKLEME
    def _sample_next(self, logits, temperature, top_k):
        probs = softmax(logits, axis=-1)
        if abs(temperature - 1.0) > 1e-9:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
        ids = np.argsort(probs)[::-1]
        pool = [int(i) for i in ids if int(i) not in (PAD, BOS)]
        if not pool:
            return EOS
        pool = pool[:top_k]
        ps = probs[pool]
        ps = ps / max(float(ps.sum()), 1e-12)
        return int(np.random.choice(pool, p=ps))

    def sample(self, context, temperature=0.6, top_k=6, max_len=None):
        """Sorgu -> karakter karakter yanit uret (EOS'a ya da max_len'e kadar).

        Varsayilan temperature=0.6/top_k=6: kelime-salatasi riskini dusururken
        yine de ozgunlugu korur (0.7/10 fazla dagilim uretiyordu).
        """
        max_len = max_len or self.max_dec_len
        ctx = clean_chars(context, self.max_enc_len)
        ids = [self.c2i[ch] for ch in ctx if ch in self.c2i]
        if ctx and not ids:
            ids = [self.c2i.get(' ', PAD)]

        enc = np.array([ids], np.int64)
        enc_out, enc_mask = self._encode(enc)

        dec = [self.c2i['<BOS>']]
        out_chars = []
        seen_ngrams = set()
        for _ in range(max_len):
            logits = self._decoder_logits(enc_out, enc_mask, np.array([dec], np.int64))
            idx = self._sample_next(logits[0, -1], temperature, top_k)
            if idx == EOS:
                break
            out_chars.append(idx)
            dec.append(idx)
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
            'arch': 'seq2seq',
            'V': self.V,
            'd_model': self.d_model,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'ff_mult': self.ff_dim // self.d_model,
            'max_enc_len': self.max_enc_len,
            'max_dec_len': self.max_dec_len,
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
        self.max_enc_len = int(data.get('max_enc_len', 40))
        self.max_dec_len = int(data.get('max_dec_len', 80))
        self.rsqrt = np.float32(1.0 / math.sqrt(self.head_dim))
        self.params = {k: _f32(v) for k, v in data['params'].items()}
        self.pos_enc = self._sinusoidal(max(self.max_enc_len, self.max_dec_len))
        return self


def encode_seq2(model, ctx, resp, max_dec=None):
    """Bir (sorgu, yanit) ciftini seq2seq egitim girdisine cevirir.

    Returns:
        (enc_in, dec_in, dec_tgt, enc_mask, dec_mask)  (numpy dizileri)
        enc_in  = sorgu karakterleri            (T1,)
        dec_in  = <BOS> + yanit                 (T2,)
        dec_tgt = yanit + <EOS>                 (T2,)   (teacher-forcing hedefi)
    """
    max_dec = max_dec or model.max_dec_len
    cc = clean_chars(ctx, model.max_enc_len)
    rr = clean_chars(resp, max_dec - 2)
    enc = [model.c2i[ch] for ch in cc if ch in model.c2i] or [model.c2i.get(' ', PAD)]
    dec_in = [model.c2i['<BOS>']] + [model.c2i[ch] for ch in rr if ch in model.c2i]
    dec_tgt = dec_in[1:] + [model.c2i['<EOS>']]
    enc_in = np.array(enc, np.int64)
    din = np.array(dec_in, np.int64)
    dtgt = np.array(dec_tgt, np.int64)
    dlen = max(len(din), len(dtgt))
    np_din = np.full(max(dlen, max_dec), PAD, np.int64)
    np_dtgt = np.full(max(dlen, max_dec), PAD, np.int64)
    np_din[:len(din)] = din
    np_dtgt[:len(dtgt)] = dtgt
    dmask = np.zeros(max(dlen, max_dec), np.float32)
    dmask[:len(dtgt)] = 1.0
    emask = np.ones(len(enc_in), np.float32)
    return enc_in, np_din, np_dtgt, emask, dmask


def load_seq2seq(path=MODEL_PATH):
    """model/seq2seq_model.json'dan Seq2Seq yukler (yoksa None)."""
    if not os.path.exists(path):
        return None
    with io.open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data or data.get('arch') != 'seq2seq':
        return None
    return Seq2Seq(['<PAD>', '<BOS>', '<EOS>']).from_dict(data)


def save_seq2seq(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(model.to_dict(), f, ensure_ascii=False)
    print(f"[seq2seq] Model kaydedildi: {path}")


if __name__ == '__main__':
    # Yerel smoke test: egitimsiz (rastgele) model hizla yanit verir mi?
    import sys
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    vocab = ['<PAD>', '<BOS>', '<EOS>'] + list('abcçdefgğhıijklmnoöprsştuüvyz. ')
    m = Seq2Seq(vocab, d_model=32, num_blocks=2, num_heads=2,
                max_enc_len=40, max_dec_len=30, seed=7)
    text = m.sample('merhaba nasilsin', max_len=30)
    print('sample:', repr(text))
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), 'seq2seq_test.json')
    save_seq2seq(m, tmp)
    m2 = load_seq2seq(tmp)
    assert m2 is not None and m2.params.keys() == m.params.keys()
    print('round-trip OK:', tmp)
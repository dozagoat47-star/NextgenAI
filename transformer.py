"""
Nextgen AI - Transformer Encoder (numpy only)
=============================================
Gerçek multi-head self-attention ile niyet sınıflandırıcısı.

Girdi: token indeks dizileri (batch, seq_len) -> PAD ile hizalanmiş.

Mimari:
  - Öğrenilen token embedding + sabit sinusoidal positional encoding
  - N adet Pre-LN transformer bloğu (multi-head self-attention + FFN,
    LayerNorm, dropout, residual)
  - Maskeli ortalama-pooling (PAD'ler devre dışı)
  - Lineer kafa -> softmax (intent olasılıkları)

Eğitim: cross-entropy, Adam (bias/LN parametrelerinde ayrışık weight decay),
gradyan clipping, token/dikkat/FF dropout, cosinüs LR çizelgesi, erken durdurma.

Backprop tamamen NumPy; public API eski brain.NeuralNetwork ile aynıdır.
Tüm tensörler float32'tedir (hız/memory).
"""

import json
import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# Temel işlemler (hepsi float32)
# ---------------------------------------------------------------------------

def softmax(z, axis=-1):
    """Kararlı (max-kaydırmalı) softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + np.float32(1e-9))


_GELU_A = np.float32(0.7978845608028654)   # sqrt(2/pi)
_GELU_B = np.float32(0.044715)


def gelu(x):
    """GELU (tanh yaklaşıklığı, BERT/GPT stili)."""
    t = np.tanh(_GELU_A * (x + _GELU_B * x ** 3))
    return np.float32(0.5) * x * (np.float32(1.0) + t)


def gelu_grad(x):
    t = np.tanh(_GELU_A * (x + _GELU_B * x ** 3))
    sech2 = np.float32(1.0) - t * t
    dtdx = _GELU_A * (np.float32(1.0) + np.float32(3.0) * _GELU_B * x * x)
    return np.float32(0.5) * (np.float32(1.0) + t) + np.float32(0.5) * x * sech2 * dtdx


def ln_forward(x, gamma, beta, eps=np.float32(1e-5)):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mean) / std
    return gamma * xhat + beta, (xhat, std)


def ln_backward(dy, gamma, xhat, std):
    dxhat = dy * gamma
    dx = (np.float32(1.0) / std) * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    axes = tuple(range(dy.ndim - 1))
    dgamma = np.sum(dy * xhat, axis=axes, keepdims=True)
    dbeta = np.sum(dy, axis=axes, keepdims=True)
    # (1,...,1,d) -> (1,d)
    dgamma = dgamma.reshape(1, -1)
    dbeta = dbeta.reshape(1, -1)
    return dx, dgamma, dbeta


def _f32(a):
    """float64 (örn. random) diziyi float32'e çevirir."""
    return np.asarray(a, dtype=np.float32)


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """Scaled dot-product dikkat + 4 doğrusal izdüşüm (NumPy backprop)."""

    def __init__(self, d_model, num_heads, init=None):
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        def _init(shape, scale=None):
            if scale is None:
                scale = np.sqrt(2.0 / shape[0])
            return _f32(np.random.randn(*shape) * scale)

        if init is not None:
            _init = init

        self.Wq = _init((d_model, d_model))
        self.Wk = _init((d_model, d_model))
        self.Wv = _init((d_model, d_model))
        self.Wo = _init((d_model, d_model), scale=np.float32(0.02))
        self.bq = np.zeros((1, d_model), np.float32)
        self.bk = np.zeros((1, d_model), np.float32)
        self.bv = np.zeros((1, d_model), np.float32)
        self.bo = np.zeros((1, d_model), np.float32)
        self.rsqrt = np.float32(1.0 / math.sqrt(self.head_dim))

        self._cache = None      # forward ara değerleri
        self._cache_x = None    # dikkat girdisi (ln1 çıktısı)
        self._eff = None        # LoRA etkin ağırlıklar ({key: mat}) ya da None

    def params(self):
        return {'Wq': self.Wq, 'Wk': self.Wk, 'Wv': self.Wv, 'Wo': self.Wo,
                'bq': self.bq, 'bk': self.bk, 'bv': self.bv, 'bo': self.bo}

    def set_eff(self, eff):
        """LoRA etkin ağırlıklarını kurar ({'Wq': mat, ...}) ya da None ile sıfırlar."""
        self._eff = eff

    # -------------------------------------------------------------- FORWARD
    def forward(self, x, mask, rng, p_drop=0.0, training=False):
        """x: (B,L,d) ; mask: (B,L) float (1=gerçek token, 0=PAD)."""
        B, L, d = x.shape
        H, hd = self.num_heads, self.head_dim

        e = self._eff or {}
        Q = x @ e.get('Wq', self.Wq) + self.bq
        K = x @ e.get('Wk', self.Wk) + self.bk
        V = x @ e.get('Wv', self.Wv) + self.bv

        Qh = Q.reshape(B, L, H, hd).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, L, H, hd).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, L, H, hd).transpose(0, 2, 1, 3)

        scale_scores = Qh @ Kh.transpose(0, 1, 3, 2) * self.rsqrt

        row = mask[:, None, :, None]                  # (B,1,L,1) sorgu maskesi
        valid = row * mask[:, None, None, :]          # (B,1,L,L) anahtar maskesi
        logits = scale_scores + np.where(valid > 0, np.float32(0.0),
                                         np.float32(-1e9))

        p = softmax(logits, axis=-1) * row            # PAD sorgu satırları 0

        dropout_mask = None
        if training and p_drop > 0:
            keep = np.float32(1.0 - p_drop)
            dropout_mask = (rng.rand(*p.shape) < keep).astype(np.float32) / keep
            p = p * dropout_mask

        attn_out = p @ Vh                             # (B,H,L,hd)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, d)
        attn_out = attn_out @ e.get('Wo', self.Wo) + self.bo

        self._cache = (Qh, Kh, Vh, p, dropout_mask)
        self._cache_x = x
        return attn_out

    # ------------------------------------------------------------- BACKWARD
    def backward(self, dout):
        """dout: (B,L,d) dikkat çıktısı gradyanı. (dx, grads) döndürür."""
        B, L, d = dout.shape
        H, hd = self.num_heads, self.head_dim
        Qh, Kh, Vh, p, dropout_mask = self._cache
        x = self._cache_x

        # Wo projeksiyonu
        x_attn = (p @ Vh).transpose(0, 2, 1, 3).reshape(B, L, d)
        dWo = x_attn.reshape(B * L, d).T @ dout.reshape(B * L, d)
        dbo = dout.sum(axis=(0, 1), keepdims=True).reshape(1, -1)

        # Kafa gradyanlari x_attn (Wo oncesi) uzayinda geçer:
        # out = x_attn @ Wo + bo -> dL/dx_attn = dout @ Wo.T
        e = self._eff or {}
        Wo_e = e.get('Wo', self.Wo)
        doit = (dout @ Wo_e.T).reshape(B, L, H, hd).transpose(0, 2, 1, 3)

        dattn = doit @ Vh.transpose(0, 1, 3, 2)                    # (B,H,L,L)
        p_used = p
        if dropout_mask is not None:
            dattn = dattn * dropout_mask
            p_used = p * dropout_mask
        dVh = p_used.transpose(0, 1, 3, 2) @ doit

        dl = p * (dattn - (dattn * p).sum(axis=-1, keepdims=True))  # softmax geri

        dQh = dl @ Kh * self.rsqrt
        dKh = dl.transpose(0, 1, 3, 2) @ Qh * self.rsqrt

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, L, d)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, L, d)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, L, d)

        xr = x.reshape(B * L, d)
        dWq = xr.T @ dQ.reshape(B * L, d)
        dWk = xr.T @ dK.reshape(B * L, d)
        dWv = xr.T @ dV.reshape(B * L, d)
        dbq = dQ.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        dbk = dK.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        dbv = dV.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        dbo = dout.sum(axis=(0, 1), keepdims=True).reshape(1, -1)

        dx = (dQ @ e.get('Wq', self.Wq).T
              + dK @ e.get('Wk', self.Wk).T
              + dV @ e.get('Wv', self.Wv).T)

        grads = {'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo,
                 'bq': dbq, 'bk': dbk, 'bv': dbv, 'bo': dbo}
        return dx, grads


# ---------------------------------------------------------------------------
# TransformerNN
# ---------------------------------------------------------------------------

class TransformerNN:
    """Token-sırası transformer encoder intent sınıflandırıcısı.

    Public API brain.NeuralNetwork ile uyumlu.
    """

    def __init__(self, vocab_size, num_intents, max_seq_len,
                 d_model=128, num_blocks=4, num_heads=4, ff_mult=4,
                 dropout=0.1, attn_dropout=0.05, weight_decay=1e-4,
                 max_grad_norm=5.0, seed=42):
        self.vocab_size = vocab_size
        self.pad_idx = vocab_size
        self.num_intents = num_intents
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ff_dim = ff_mult * d_model
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.training = True

        self._build_arch(seed)
        self._init_optimizer()
        self.reset_buffers()

    # ------------------------------------------------------------ KURULUM
    def apply_lora(self, adapter):
        """LoRA adaptörünü taban ağırlıkların ÜZERİNE uygular (model.json bozulmaz).

        adapter (dict):
          rank, alpha,
          vocab_added (int), head_added (int),
          embed_extra (list[n,d]), whead_extra (list[n,d]), bhead_extra (list[n]),
          deltas: {'b0_Wq': {'A':..., 'B':...}, ...}  (B: (in,r), A: (r,out))
        """
        self.clear_lora()
        self._lora_adapter = adapter
        r = int(adapter.get('rank', 8))
        alpha = float(adapter.get('alpha', 1.0))
        scale = alpha / max(r, 1)

        deltas = adapter.get('deltas') or {}
        for key, pair in deltas.items():
            A = _f32(np.asarray(pair['A'], dtype=np.float32))
            B = _f32(np.asarray(pair['B'], dtype=np.float32))
            deltas[key] = {'A': A, 'B': B, 'delta': scale * (B @ A)}

        # Her bloğun etkin ağırlıklarını kur
        for bi in range(self.num_blocks):
            self._apply_block_eff(bi)

        # Yeni sözcük embed satırları + yeni kafa satırları
        n_v = int(adapter.get('vocab_added', 0))
        if n_v > 0:
            extra = _f32(np.asarray(adapter.get('embed_extra'), dtype=np.float32))
            assert extra.shape[0] == n_v and extra.shape[1] == self.d_model
            self._lora_embed_full = np.concatenate(
                [self.embed, extra, np.zeros((1, self.d_model), np.float32)], axis=0)
            self._lora_pad_idx = self.embed.shape[0] + n_v
        n_h = int(adapter.get('head_added', 0))
        if n_h > 0:
            self._lora_head_w = _f32(np.asarray(adapter.get('whead_extra'),
                                                dtype=np.float32))
            self._lora_head_b = _f32(np.asarray(adapter.get('bhead_extra'),
                                                dtype=np.float32)).reshape(1, n_h)
        return self

    def clear_lora(self):
        """LoRA etkisini kaldırır (taban model.json'e döner)."""
        self._lora_adapter = None
        self._lora_embed_full = None
        self._lora_pad_idx = None
        self._lora_head_w = None
        self._lora_head_b = None
        for blk in self.blocks:
            blk.pop('W1e', None)
            blk.pop('W2e', None)
            blk['attn'].set_eff(None)
        return self

    def _apply_block_eff(self, bi):
        """Bir blok için LoRA deltalarını 'W1e/W2e + MHA.eff' üzerinden kurar."""
        blk = self.blocks[bi]
        deltas = (self._lora_adapter or {}).get('deltas') or {}
        scale = (float(self._lora_adapter['alpha']) / max(int(self._lora_adapter['rank']), 1)
                 if self._lora_adapter else 1.0)

        eff_attn = {}
        for name in ('Wq', 'Wk', 'Wv', 'Wo'):
            pair = deltas.get(f'b{bi}_{name}')
            if pair:
                eff_attn[name] = getattr(blk['attn'], name) + scale * (pair['B'] @ pair['A'])
        blk['attn'].set_eff(eff_attn or None)

        for wkey in ('W1', 'W2'):
            pair = deltas.get(f'b{bi}_{wkey}')
            val = blk[wkey]
            blk[f'{wkey}e'] = val + scale * (pair['B'] @ pair['A']) if pair else val
    def _build_arch(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        d = self.d_model

        def _init(shape, scale=None):
            if scale is None:
                scale = np.sqrt(2.0 / shape[0])
            return _f32(np.random.randn(*shape) * scale)

        embed = _f32(np.random.randn(self.vocab_size + 1, d) * 0.02)
        embed[self.vocab_size] = 0.0
        self.embed = embed
        self.pos = self._sinusoidal(self.max_seq_len)

        self.blocks = []
        blk_init = _init
        for _ in range(self.num_blocks):
            self.blocks.append({
                'attn': MultiHeadAttention(d, self.num_heads, init=blk_init),
                'ln1_g': np.ones((1, d), np.float32), 'ln1_b': np.zeros((1, d), np.float32),
                'ln2_g': np.ones((1, d), np.float32), 'ln2_b': np.zeros((1, d), np.float32),
                'W1': _init((d, self.ff_dim)),
                'b1': np.zeros((1, self.ff_dim), np.float32),
                'W2': _init((self.ff_dim, d), scale=np.float32(0.02)),
                'b2': np.zeros((1, d), np.float32),
            })
        self.Whead = _init((d, self.num_intents), scale=np.float32(0.1))
        self.bhead = np.zeros((1, self.num_intents), np.float32)

        # LoRA durumu (model.json/ss taban ağırlıklardır; adaptör burada tutulur)
        self._lora_adapter = None
        self._lora_embed_full = None
        self._lora_pad_idx = None
        self._lora_head_w = None
        self._lora_head_b = None

    def _sinusoidal(self, length):
        d = self.d_model
        pe = np.zeros((length, d), np.float32)
        pos = np.arange(length, dtype=np.float32)[:, None]
        dim = np.arange(d // 2, dtype=np.float32)
        div = np.power(np.float32(10000.0), np.float32(2.0) * dim / np.float32(d))
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        # Konumsal embedlerin normu ~sqrt(d) oldugundan token embedlerini
        # (std ~0.02) ezer; residual/aktivasyon patlamasini onlemek icin
        # 1/sqrt(d) ile olceklenir -> norm ~ O(1), token sinyaliyle karisabilir.
        pe *= np.float32(1.0 / math.sqrt(max(d, 1)))
        return pe

    def _named_params(self):
        yield 'embed', self.embed
        for bi, blk in enumerate(self.blocks):
            for name, val in blk['attn'].params().items():
                yield f'b{bi}_{name}', val
            yield f'b{bi}_ln1_g', blk['ln1_g']
            yield f'b{bi}_ln1_b', blk['ln1_b']
            yield f'b{bi}_ln2_g', blk['ln2_g']
            yield f'b{bi}_ln2_b', blk['ln2_b']
            yield f'b{bi}_W1', blk['W1']
            yield f'b{bi}_b1', blk['b1']
            yield f'b{bi}_W2', blk['W2']
            yield f'b{bi}_b2', blk['b2']
        yield 'Whead', self.Whead
        yield 'bhead', self.bhead

    def _init_optimizer(self):
        self._m = {}
        self._v = {}
        for name, val in self._named_params():
            self._m[name] = np.zeros_like(val)
            self._v[name] = np.zeros_like(val)
        self._t = 0

    def reset_buffers(self):
        self._cache = {}
        self._last_probs = None

    def _is_bias(self, name):
        """Weight-decay uygulanmayacak (bias/LN) parametre adları."""
        if name in ('bhead',):
            return True
        return any(s in name for s in ('_bq', '_bk', '_bv', '_bo',
                                       '_ln1_b', '_ln2_b', '_b1', '_b2'))

    # -------------------------------------------------------------- FORWARD
    def forward(self, X, apply_dropout=None):
        """X: (B,L) int token indeksleri -> (B, num_intents) olasılıklar."""
        if apply_dropout is None:
            apply_dropout = self.training

        B, L = X.shape
        rng = np.random.RandomState(np.random.randint(1, 2 ** 31))

        embed = self._lora_embed_full if self._lora_embed_full is not None else self.embed
        pad = self._lora_pad_idx if self._lora_embed_full is not None else self.pad_idx
        mask = (X != pad).astype(np.float32)     # (B,L)
        mask_ = mask[:, :, None]                          # (B,L,1)

        E = embed[X] * mask_                      # PAD satırları 0
        x = E + self.pos[None, :L, :] * mask_

        if apply_dropout and self.dropout > 0:
            keep = np.float32(1.0 - self.dropout)
            x = x * ((rng.rand(B, L, self.d_model) < keep).astype(np.float32) / keep)

        caches = {'mask': mask, 'mask_': mask_, 'inputs': X, 'blocks': []}

        for blk in self.blocks:
            c = {}
            ln1_out, (xhat1, std1) = ln_forward(x, blk['ln1_g'], blk['ln1_b'])
            attn_out = blk['attn'].forward(ln1_out, mask, rng,
                                           p_drop=self.attn_dropout,
                                           training=apply_dropout)
            a = x + attn_out

            ln2_out, (xhat2, std2) = ln_forward(a, blk['ln2_g'], blk['ln2_b'])
            z1 = ln2_out @ blk.get('W1e', blk['W1']) + blk['b1']
            h = gelu(z1)
            ff_drop = None
            if apply_dropout and self.dropout > 0:
                keep = np.float32(1.0 - self.dropout)
                ff_drop = ((rng.rand(*h.shape) < keep).astype(np.float32) / keep)
                h = h * ff_drop
            ffn_out = h @ blk.get('W2e', blk['W2']) + blk['b2']
            x = a + ffn_out

            c.update({'ln1_out': ln1_out, 'xhat1': xhat1, 'std1': std1,
                      'attn_out': attn_out, 'a': a,
                      'ln2_out': ln2_out, 'xhat2': xhat2, 'std2': std2,
                      'z1': z1,
                      'h': h, 'ff_drop': ff_drop, 'ffn_out': ffn_out})
            caches['blocks'].append(c)

        denom = np.maximum(mask.sum(axis=1, keepdims=True), np.float32(1.0))
        pooled = (x * mask_).sum(axis=1) / denom

        head_out = pooled @ self.Whead + self.bhead
        if self._lora_head_w is not None:
            head_out = np.concatenate(
                [head_out, pooled @ self._lora_head_w.T + self._lora_head_b], axis=1)
        probs = softmax(head_out, axis=-1)

        caches['pooled'] = pooled
        caches['final_x'] = x
        caches['head_out'] = head_out
        self._cache = caches
        self._last_probs = probs
        return probs

    # ------------------------------------------------------------- BACKWARD
    def compute_loss(self, probs, y_true):
        m = probs.shape[0]
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        return float(np.mean(-np.log(probs[np.arange(m), y_true] + np.float32(1e-8))))

    def _gradients(self, y_true, normalize=True):
        """forward() sonrası parametre gradyanlarını hesaplar (saf, güncellemesiz).

        normalize=True ise ortalama (1/B + weight-decay hariç tam backward gradyanı)
        döner. LoRA etkin (head fazlası / embed fazlası) ise bunların da gradyanları
        _cache['lora_head_grads'] ve _cache['lora_embed_grads'] içine konur.
        """
        B = self._cache['pooled'].shape[0]
        probs = self._last_probs
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        d_head = probs.copy()
        d_head[np.arange(B), y_true] -= np.float32(1.0)
        if normalize:
            d_head /= np.float32(B)

        pooled = self._cache['pooled']
        C_base = self.Whead.shape[1]
        if self._lora_head_w is not None:
            d_head_base = d_head[:, :C_base]
            d_head_extra = d_head[:, C_base:]
            dWhead = pooled.T @ d_head_base
            dbhead = d_head_base.sum(axis=0, keepdims=True)
            dpool = (d_head_base @ self.Whead.T
                     + d_head_extra @ self._lora_head_w)
            self._cache['lora_head_grads'] = {'pooled': pooled,
                                              'd_head_extra': d_head_extra}
        else:
            dWhead = pooled.T @ d_head
            dbhead = d_head.sum(axis=0, keepdims=True)
            dpool = d_head @ self.Whead.T

        mask = self._cache['mask']
        mask_ = self._cache['mask_']
        denom = np.maximum(mask.sum(axis=1, keepdims=True), np.float32(1.0))
        dx = (dpool / denom)[:, None, :] * mask_

        grads = {'Whead': dWhead, 'bhead': dbhead}
        for bi in range(self.num_blocks - 1, -1, -1):
            blk = self.blocks[bi]
            c = self._cache['blocks'][bi]
            dx, bg = self._backward_block(dx, blk, c)
            for name, g in bg.items():
                grads[f'b{bi}_{name}'] = g

        embed = self._lora_embed_full if self._lora_embed_full is not None else self.embed
        dE = np.zeros_like(embed)
        np.add.at(dE, self._cache['inputs'], dx * mask_)
        if self._lora_embed_full is not None:
            base_v = self.embed.shape[0]
            grads['embed'] = dE[:base_v]
            self._cache['lora_embed_grads'] = dE[base_v:-1]  # yeni sözcük satırları
        else:
            grads['embed'] = dE
        return grads

    def grads(self, y_true, normalize=True):
        """Dış eğitmenler (LoRA) için ham parametre gradyanları."""
        return self._gradients(y_true, normalize=normalize)

    def backward(self, y_true, learning_rate):
        grads = self._gradients(y_true, normalize=True)

        # Gradyan clipping
        total_norm = 0.0
        for name in grads:
            total_norm += float(np.sum(grads[name] * grads[name]))
        total_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0
        scale = np.float32(1.0)
        if total_norm > self.max_grad_norm:
            scale = np.float32(self.max_grad_norm / total_norm)

        self._t += 1
        t = self._t
        b1, b2, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
        bc1 = np.float32(1.0) - b1 ** t
        bc2 = np.float32(1.0) - b2 ** t

        for name, val in self._named_params():
            g = grads.get(name, np.zeros_like(val)) * scale
            self._m[name] = b1 * self._m[name] + (1 - b1) * g
            self._v[name] = b2 * self._v[name] + (1 - b2) * g * g
            m_hat = self._m[name] / bc1
            v_hat = self._v[name] / bc2
            decay = self.weight_decay * val if (self.weight_decay > 0
                                                and not self._is_bias(name)) else 0.0
            upd = m_hat / (np.sqrt(v_hat) + eps) + decay
            if upd.shape != val.shape:
                raise ValueError(
                    f"shape uyumsuz: {name} val={val.shape} g={g.shape} "
                    f"m={self._m[name].shape} upd={upd.shape}")
            val -= learning_rate * upd
        return float(scale)

    def _backward_block(self, dout, blk, c):
        """Bir bloğun gradyanları. dx (blok girdi gradyanı) + param gradyanları."""
        # (1) FFN geri
        h = c['h']
        dout_r = dout.reshape(-1, self.d_model)
        h_r = h.reshape(-1, self.ff_dim)
        W1e = blk.get('W1e', blk['W1'])
        W2e = blk.get('W2e', blk['W2'])
        dW2 = h_r.T @ dout_r
        db2 = dout.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        dh = dout @ W2e.T
        if c['ff_drop'] is not None:
            dh = dh * c['ff_drop']
        dh = dh * gelu_grad(c['z1'])
        ln2_r = c['ln2_out'].reshape(-1, self.d_model)
        dW1 = ln2_r.T @ dh.reshape(-1, self.ff_dim)
        db1 = dh.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        d_ln2 = dh @ W1e.T

        # (2) LN2 geri
        d_a_ln2, dg2, db2_ln = ln_backward(d_ln2, blk['ln2_g'],
                                            c['xhat2'], c['std2'])

        # (3) a = x + attn_out : residual
        d_a_total = dout + d_a_ln2
        d_x_a = d_a_total                      # a'nın x'e giden düz yolu
        d_attn_out = d_a_total                 # attention çıktısına giden

        # (4) Attention geri
        dx_attn, attn_grads = blk['attn'].backward(d_attn_out)

        # (5) LN1 geri (attention girdisi üzerinden)
        d_x_ln1, dg1, db1_ln = ln_backward(dx_attn, blk['ln1_g'],
                                           c['xhat1'], c['std1'])

        dx_prev = d_x_a + d_x_ln1

        grads = dict(attn_grads)
        grads.update({'ln1_g': dg1, 'ln1_b': db1_ln,
                      'ln2_g': dg2, 'ln2_b': db2_ln,
                      'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2})
        return dx_prev, grads

    # ---------------------------------------------------------------- TRAIN
    def train(self, X, y, epochs=500, learning_rate=0.001, batch_size=64,
              verbose=True, lr_min_ratio=0.1, early_stop=False, patience=30,
              warmup_steps=200, X_val=None, y_val=None):
        """X: (N,L) int ; y: (N,) veya (N,C).

        X_val/y_val verilirse erken durdurma ve periyodik doğruluk çıktısı
        EĞİTİM seti yerine doğrulama setinde ölçülür (overfit görmek için
        şart; aksi halde accuracy yanıltıcıdır). Verilmezse eski davranış:
        eğitim seti üzerinde ölçülür.

        Adam + cosinüs LR + kısa lineer warmup. Warmup, transformer'larda
        bilinen erken-dönem ıraksamasını (büyük LR'de logit/aktivasyon
        patlaması) önler.
        """
        losses = []
        m = X.shape[0]
        base_lr = learning_rate
        best_acc = 0.0
        best_state = None
        wait = 0
        step = 0
        self.training = True

        eval_X, eval_y = (X_val, y_val) if X_val is not None else (X, y)

        for epoch in range(epochs):
            frac = epoch / max(epochs - 1, 1)
            lr = base_lr * (lr_min_ratio + (1 - lr_min_ratio)
                            * (0.5 * (1 + np.cos(np.pi * frac))))

            indices = np.random.permutation(m)
            epoch_loss = 0.0
            nb = 0
            for start in range(0, m, batch_size):
                idx = indices[start:start + batch_size]
                Xb = X[idx]
                yb = y[idx]
                step += 1
                if warmup_steps and step <= warmup_steps:
                    cur_lr = lr * (step / warmup_steps)
                else:
                    cur_lr = lr
                probs = self.forward(Xb, apply_dropout=True)
                loss = self.compute_loss(probs, yb)
                epoch_loss += loss
                nb += 1
                self.backward(yb, cur_lr)
            avg = epoch_loss / max(1, nb)
            losses.append(avg)

            if early_stop:
                acc = self.evaluate(eval_X, eval_y)
                if acc > best_acc + 1e-4:
                    best_acc = acc
                    best_state = self.get_state()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Erken durdurma: epoch {epoch + 1}, "
                                  f"en iyi accuracy {best_acc:.2%}")
                        if best_state is not None:
                            self.set_state(best_state)
                        break
            if verbose and (epoch + 1) % 25 == 0:
                accuracy = self.evaluate(eval_X, eval_y)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg:.4f} "
                      f"- Accuracy: {accuracy:.2%}")
        return losses

    def predict(self, X):
        self.training = False
        try:
            return np.argmax(self.forward(X), axis=1)
        finally:
            self.training = True

    def predict_proba(self, X):
        self.training = False
        try:
            return self.forward(X)
        finally:
            self.training = True

    def evaluate(self, X, y):
        preds = self.predict(X)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return float(np.mean(preds == y))

    # --------------------------------------------------------- DURUM/SAVE
    def get_state(self):
        return {name: val.copy() for name, val in self._named_params()}

    def set_state(self, state):
        for name, val in self._named_params():
            if name in state:
                val[...] = np.asarray(state[name], dtype=val.dtype)

    def save(self, filepath):
        """Ağırlıkları float32 .npz, hiperparametreleri küçük JSON olarak yazar.

        model.json yalnızca mimari/ayar şemasıdır (Colab export'undan gelen
        büyük 'params' JSON'u yerine). Yerel ve Colab export'ları ortak
        formata düşer; okuma/yazma çok daha hızlı, dosya ~%50 küçük.
        """
        data = {
            'arch': 'transformer',
            'vocab_size': self.vocab_size,
            'num_intents': self.num_intents,
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'attn_dropout': self.attn_dropout,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
        }
        base = os.path.splitext(os.path.basename(filepath))[0]
        weights_path = os.path.join(os.path.dirname(os.path.abspath(filepath)),
                                    base + '_weights.npz')
        data['weights_file'] = os.path.basename(weights_path)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        arrays = {name: np.asarray(val, dtype=np.float32)
                  for name, val in self._named_params()}
        np.savez(weights_path, **arrays)
        print(f"Transformer model saved: {filepath} "
              f"(+ {os.path.basename(weights_path)})")

    def _assign_params(self, param_map):
        """Parametre haritasini mevcut mimariye aktarir (npz veya JSON ortak)."""
        for bi in range(self.num_blocks):
            blk = self.blocks[bi]
            attn = blk['attn']
            for name, val in attn.params().items():
                key = f'b{bi}_{name}'
                if key in param_map:
                    val[...] = param_map[key]
            for name in ('ln1_g', 'ln1_b', 'ln2_g', 'ln2_b', 'W1', 'b1', 'W2', 'b2'):
                val = blk[name]
                key = f'b{bi}_{name}'
                if key in param_map:
                    val[...] = param_map[key]
        if 'embed' in param_map:
            self.embed[...] = param_map['embed']
        for name in ('Whead', 'bhead'):
            if name in param_map:
                getattr(self, name)[...] = param_map[name]

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.vocab_size = data.get('vocab_size', self.vocab_size)
        self.pad_idx = self.vocab_size
        self.num_intents = data.get('num_intents', self.num_intents)
        self.max_seq_len = data.get('max_seq_len', self.max_seq_len)
        self.d_model = data.get('d_model', self.d_model)
        self.num_blocks = data.get('num_blocks', self.num_blocks)
        self.num_heads = data.get('num_heads', self.num_heads)
        self.ff_dim = data.get('ff_dim', self.ff_dim)
        self.dropout = data.get('dropout', self.dropout)
        self.attn_dropout = data.get('attn_dropout', self.attn_dropout)
        self.weight_decay = data.get('weight_decay', self.weight_decay)
        self.max_grad_norm = data.get('max_grad_norm', self.max_grad_norm)

        # Mimariyi (aynı boyutlarla) yeniden inşa et ve ağırlıkları yükle
        self._build_arch(seed=None)

        param_map = None
        weights_file = data.get('weights_file')
        if weights_file:
            npz_path = os.path.join(os.path.dirname(os.path.abspath(filepath)),
                                    weights_file)
            if os.path.exists(npz_path):
                arrs = np.load(npz_path)
                param_map = {name: np.asarray(arrs[name], dtype=np.float32)
                             for name in arrs.files}
        if param_map is None:
            # Eski büyük JSON 'params' formatı (Colab float32 export /
            # geriye dönük uyumluluk). Mevcut model.json bu formattadır.
            param_map = {name: np.asarray(val, dtype=np.float32)
                         for name, val in data.get('params', {}).items()}
        self._assign_params(param_map)

        self._init_optimizer()
        self.training = True
        print(f"Transformer loaded: {filepath}")
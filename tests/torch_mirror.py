"""Nextgen AI - TransformerNN'nin PyTorch karsiligi (parity testleri icin).

`colab/nextgen_transformer_colab.ipynb` icindeki TorchMHA / TorchBlock /
TorchTransformer siniflarinin birebir kopyasidir. Egittigi ayni NumPy
(mnumpy) mimarisinin PyTorch aynasi; Kaggle egerimiyle uretilen agirliklarin
yerel `transformer.py` ile birebir ayni sonuc vermesini dogrulamakta kullanilir.
Sadece torch kurulu ortamlarda anlamlidir; testler torch yoksa skip eder.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchMHA(nn.Module):
    """transformer.py'deki MultiHeadAttention'un birebir ayrisi."""
    def __init__(self, d, heads, attn_dropout=0.05):
        super().__init__()
        self.d, self.heads, self.hd = d, heads, d // heads
        self.attn_dropout = attn_dropout

        def lin(scale):
            l = nn.Linear(d, d)
            with torch.no_grad():
                l.weight.normal_(0, scale)
                l.bias.zero_()
            return l

        s = math.sqrt(2.0 / d)
        self.Wq, self.Wk, self.Wv = lin(s), lin(s), lin(s)
        self.Wo = lin(0.02)
        self.rsqrt = 1.0 / math.sqrt(self.hd)

    def forward(self, x, mask):
        B, L, d = x.shape
        H, hd = self.heads, self.hd
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        Qh = Q.view(B, L, H, hd).transpose(1, 2)
        Kh = K.view(B, L, H, hd).transpose(1, 2)
        Vh = V.view(B, L, H, hd).transpose(1, 2)
        scores = (Qh @ Kh.transpose(-1, -2)) * self.rsqrt
        row = mask[:, None, :, None]
        valid = row * mask[:, None, None, :]
        logits = scores.masked_fill(valid <= 0, -1e9)
        p = F.softmax(logits, dim=-1) * row
        if self.training and self.attn_dropout > 0:
            p = F.dropout(p, self.attn_dropout)
        out = (p @ Vh).transpose(1, 2).contiguous().view(B, L, d)
        return self.Wo(out)


class TorchBlock(nn.Module):
    """Pre-LN blogu: LN1 -> MHA -> +residual ; LN2 -> GELU-FFN -> +residual"""
    def __init__(self, d, heads, ff_dim, dropout=0.1, attn_dropout=0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.attn = TorchMHA(d, heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        s = math.sqrt(2.0 / d)
        self.W1 = nn.Linear(d, ff_dim)
        self.W2 = nn.Linear(ff_dim, d)
        self.dropout = dropout
        with torch.no_grad():
            self.W1.weight.normal_(0, s)
            self.W1.bias.zero_()
            self.W2.weight.normal_(0, 0.02)
            self.W2.bias.zero_()

    def forward(self, x, mask):
        a = x + self.attn(self.ln1(x), mask)
        h = F.gelu(self.W1(self.ln2(a)), approximate='tanh')
        if self.training and self.dropout > 0:
            h = F.dropout(h, self.dropout)
        return a + self.W2(h)


class TorchTransformer(nn.Module):
    """transformer.py'deki TransformerNN'in birebir PyTorch karsiligi."""
    def __init__(self, vocab_size, num_intents, max_seq_len, d=128, blocks=4,
                 heads=4, ff_mult=4, dropout=0.1, attn_dropout=0.05):
        super().__init__()
        self.vocab_size, self.pad_idx = vocab_size, vocab_size
        self.num_intents = num_intents
        self.max_seq_len = max_seq_len
        self.d_model, self.num_blocks, self.num_heads = d, blocks, heads
        self.ff_dim = ff_mult * d
        self.dropout, self.attn_dropout = dropout, attn_dropout
        self.embed = nn.Embedding(vocab_size + 1, d)
        self.register_buffer('pos', self._sinusoidal(max_seq_len))
        self.blocks = nn.ModuleList(
            [TorchBlock(d, heads, self.ff_dim, dropout, attn_dropout)
             for _ in range(blocks)])
        self.Whead = nn.Linear(d, num_intents)
        with torch.no_grad():
            self.embed.weight[:vocab_size].normal_(0, 0.02)
            self.embed.weight[vocab_size].zero_()
            self.Whead.weight.normal_(0, 0.1)
            self.Whead.bias.zero_()

    def _sinusoidal(self, length):
        d = self.d_model
        pe = torch.zeros(length, d)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(d // 2, dtype=torch.float32)
        div = torch.pow(torch.tensor(10000.0, dtype=torch.float32),
                        2.0 * dim / d)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        pe *= (1.0 / math.sqrt(max(d, 1)))
        return pe

    def forward(self, X):
        B, L = X.shape
        mask = (X != self.pad_idx).float()
        mask_ = mask[:, :, None]
        x = self.embed(X) * mask_
        x = x + self.pos[:L][None] * mask_
        if self.training and self.dropout > 0:
            x = F.dropout(x, self.dropout)
        for blk in self.blocks:
            x = blk(x, mask)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (x * mask_).sum(1) / denom
        head_out = self.Whead(pooled)
        return F.softmax(head_out, dim=-1)

    def logits(self, X):
        B, L = X.shape
        mask = (X != self.pad_idx).float()
        mask_ = mask[:, :, None]
        x = self.embed(X) * mask_
        x = x + self.pos[:L][None] * mask_
        if self.training and self.dropout > 0:
            x = F.dropout(x, self.dropout)
        for blk in self.blocks:
            x = blk(x, mask)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.Whead((x * mask_).sum(1) / denom)


def copy_numpy_to_torch(mx, module):
    """NumPy TransformerNN agirliklarini TorchTransformer'a aktarir.

    `mx._named_params()` ciktisindaki her anahtar torch modulune birebir
    oturur: torch Linear.weight, numpy x @ W seklinde kullanildigindan (.T)
    ile; biaslari (1,d) ile numpy -> (d,) torch'a cevirir.
    """
    with torch.no_grad():
        module.embed.weight.copy_(
            torch.from_numpy(np.asarray(mx.embed, dtype=np.float32)))
        for bi in range(mx.num_blocks):
            blk = mx.blocks[bi]
            attn = blk['attn'].params()
            t = module.blocks[bi]
            for nm in ('Wq', 'Wk', 'Wv', 'Wo'):
                w = torch.from_numpy(np.asarray(attn[nm], dtype=np.float32)).T
                b = torch.from_numpy(
                    np.asarray(attn['b' + nm[1:]], dtype=np.float32).reshape(-1))
                getattr(t.attn, nm).weight.copy_(w)
                getattr(t.attn, nm).bias.copy_(b)
            for ln, g, b in (('ln1', 'ln1_g', 'ln1_b'),
                             ('ln2', 'ln2_g', 'ln2_b')):
                getattr(t, ln).weight.copy_(
                    torch.from_numpy(np.asarray(blk[g], dtype=np.float32).reshape(-1)))
                getattr(t, ln).bias.copy_(
                    torch.from_numpy(np.asarray(blk[b], dtype=np.float32).reshape(-1)))
            t.W1.weight.copy_(
                torch.from_numpy(np.asarray(blk['W1'], dtype=np.float32)).T)
            t.W1.bias.copy_(
                torch.from_numpy(np.asarray(blk['b1'], dtype=np.float32).reshape(-1)))
            t.W2.weight.copy_(
                torch.from_numpy(np.asarray(blk['W2'], dtype=np.float32)).T)
            t.W2.bias.copy_(
                torch.from_numpy(np.asarray(blk['b2'], dtype=np.float32).reshape(-1)))
        module.Whead.weight.copy_(
            torch.from_numpy(np.asarray(mx.Whead, dtype=np.float32)).T)
        module.Whead.bias.copy_(
            torch.from_numpy(np.asarray(mx.bhead, dtype=np.float32).reshape(-1)))
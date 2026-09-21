"""Nextgen AI - Sayisal (finite-difference) gradyan dogrulama testi.

`transformer.py`'deki el yazimi backward'in hatasiz oldugunu merkezi
farklarla dogrular. Dropout BIRIKTIRILEN ortamda calistirilir; bu, daha
once var olan iki dropout gradyan hatasini yakalar:

1. MHA attention dropout backward: cache'lenmis `p` zaten maskeli oldugu
   halde maskenin ikinci kez uygulanmasi V-gradyanini 1/(1-p) kat
   oldurmekteydi; ve softmax geri formulunde dropout-sonrasi p yerine
   dropout-oncesi s kullanilmasi gerekiyordu.
2. Embedding dropout backward: `dE` hesabinda token (embedding) dropout
   maskesi carpani eksikti.

Kriter: analitik gradyan (g) ile sayisal gradyan (ng) karsilastririlir;
|g| degeri modelin en buyuk |g|'sine gore cok kucukse (bq/bk gibi softmax
kaydirma-degismezliginden dolayi gercek = 0 olan parametreler) o parametre
atlanir. Aksi halde gozlenen bagil hata FD gurultu duzeyindedir (< ~0.1);
bug'li kodda 0.4-1.2 araligina cikar.

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import numpy as np  # noqa: E402

from transformer import TransformerNN  # noqa: E402

VOCAB, INTENTS, LEN = 12, 3, 6
BATCH = 3
DROP, ATTN_DROP = 0.25, 0.35
FD_EPS = 5e-3
TOL = 0.2
GLOBAL_SEED = 7


def _model():
    return TransformerNN(vocab_size=VOCAB, num_intents=INTENTS,
                         max_seq_len=LEN, d_model=8, num_blocks=2,
                         num_heads=2, ff_mult=2, dropout=DROP,
                         attn_dropout=ATTN_DROP, seed=1)


def _data():
    rng = np.random.RandomState(5)
    X = rng.randint(0, VOCAB, size=(BATCH, LEN))
    X[1, LEN - 2:] = VOCAB          # PAD'li iki ornek
    X[2, LEN - 3:] = VOCAB
    y = rng.randint(0, INTENTS, size=(BATCH,))
    return X, y


class NumericGradCheckTest(unittest.TestCase):

    def _check(self, m, X, y, apply_dropout):
        def loss_of():
            np.random.seed(GLOBAL_SEED)      # dropout maskelerini sabitler
            probs = m.forward(X, apply_dropout=apply_dropout)
            return float(m.compute_loss(probs, y))

        np.random.seed(GLOBAL_SEED)
        m.forward(X, apply_dropout=apply_dropout)
        grads = m.grads(y, normalize=True)
        weights = dict(m._named_params())
        common = sorted(set(grads) & set(weights))
        global_max = max(float(np.abs(grads[k]).max()) for k in common)

        worst, checked = 0.0, 0
        for name in common:
            a = grads[name]
            w = weights[name]
            if float(np.abs(a).max()) < 1e-4 * global_max:
                continue               # dejenere (gercek ≈ 0) gradyan, FD-mez
            wf = w.reshape(-1)
            g = np.zeros(w.size, dtype=np.float32)
            for i in range(wf.size):
                orig = wf[i]
                wf[i] = orig + FD_EPS
                lp = loss_of()
                wf[i] = orig - FD_EPS
                lm = loss_of()
                wf[i] = orig
                g[i] = (lp - lm) / (2 * FD_EPS)
            rel = float(np.abs(g.reshape(w.shape) - a).max()) / float(np.abs(a).max())
            worst = max(worst, rel)
            checked += 1
        self.assertGreater(checked, 0, 'Karsilastirilabilir parametre yok')
        self.assertLess(worst, TOL,
                        f'en kotu bagil gradyan hatasi {worst:.3f} (tol {TOL})')

    def test_gradcheck_dropout_active(self):
        """Dropout ACIK iken tam-model gradyanlari FD ile tutarli."""
        X, y = _data()
        self._check(_model(), X, y, apply_dropout=True)

    def test_gradcheck_dropout_off(self):
        """Dropout kapaliyken gradyanlar FD ile tutarli (temel backward)."""
        X, y = _data()
        self._check(_model(), X, y, apply_dropout=False)


if __name__ == '__main__':
    unittest.main()
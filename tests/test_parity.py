"""Nextgen AI - PyTorch <-> NumPy parity testi.

Kaggle/Colab'de PyTorch ile egitilen (ve NumPy model.json + _weights.npz'e
disa aktarilan) transformer'in yerel `transformer.py` inference'iyle birebir
ayni ciktiyi verdigini dogrular. `tests/torch_mirror.py`, notebook'taki
TorchTransformer'in birebir kopyasidir; bu test ayni agirliklari iki tarafta
da kurar ve ciktilari karsilastirir (dropout kapali / eval modu; notebook'un
kendi parity hucresiyle ayni yontem ve tolerans: < 2e-5).

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

import transformer  # noqa: E402

if torch is not None:
    from tests.torch_mirror import TorchTransformer, copy_numpy_to_torch  # noqa: E402


def _random_batch(model, shape, vocab, seed=3):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, vocab, size=shape)
    for i in range(shape[0]):          # orneklere PAD ekle
        cut = rng.randint(1, shape[1])
        X[i, cut:] = vocab
    return X


@unittest.skipIf(torch is None, 'PyTorch yüklü degil => parity testleri skip')
class ParityTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        cfg = dict(vocab_size=12, num_intents=3, max_seq_len=6,
                   d_model=16, num_blocks=2, num_heads=2, ff_mult=2,
                   dropout=0.1, attn_dropout=0.05)
        self.cfg = cfg
        self.mx = transformer.TransformerNN(seed=11, **cfg)
        self.tm = TorchTransformer(
            vocab_size=cfg['vocab_size'], num_intents=cfg['num_intents'],
            max_seq_len=cfg['max_seq_len'], d=cfg['d_model'],
            blocks=cfg['num_blocks'], heads=cfg['num_heads'],
            ff_mult=cfg['ff_mult'], dropout=cfg['dropout'],
            attn_dropout=cfg['attn_dropout'])
        copy_numpy_to_torch(self.mx, self.tm)
        self.tm.eval()
        self.X = _random_batch(self.mx, (4, cfg['max_seq_len']),
                               cfg['vocab_size'])

    def test_probs_parity(self):
        nprobs = self.mx.predict_proba(self.X)
        with torch.no_grad():
            tprobs = self.tm(torch.from_numpy(self.X)).numpy()
        diff = float(np.abs(tprobs - nprobs).max())
        self.assertLess(diff, 2e-5, f'max |torch-numpy probs| = {diff:.3e}')

    def test_logits_parity(self):
        self.mx.predict_proba(self.X)          # _cache['pooled'] icin forward
        pooled = self.mx._cache['pooled']
        nlogits = pooled @ self.mx.Whead + self.mx.bhead
        with torch.no_grad():
            tlogits = self.tm.logits(torch.from_numpy(self.X)).numpy()
        diff = float(np.abs(tlogits - nlogits).max())
        self.assertLess(diff, 2e-5, f'max |torch-numpy logits| = {diff:.3e}')


MODEL_DIR = os.path.join(BASE, 'model')
_SAVED = (os.path.exists(os.path.join(MODEL_DIR, 'model.json'))
          and os.path.exists(os.path.join(MODEL_DIR, 'model_weights.npz')))


@unittest.skipUnless(torch is not None and _SAVED,
                     'Uretim model.json/weights yok veya PyTorch yüklü degil')
class ProductionParityTest(unittest.TestCase):
    """Gerçek eğitilmiş model dosyalariyla torch-numpy parity (notebook'un
    kendi 64-ornek parity hucresine esdeger, burada rastgele PAD'li batch)."""

    def setUp(self):
        with open(os.path.join(MODEL_DIR, 'model.json'), encoding='utf-8') as f:
            meta = json.load(f)
        numpy_kwargs = dict(vocab_size=meta['vocab_size'],
                            num_intents=meta['num_intents'],
                            max_seq_len=meta['max_seq_len'],
                            d_model=meta['d_model'],
                            num_blocks=meta['num_blocks'],
                            num_heads=meta['num_heads'],
                            ff_mult=meta['ff_dim'] // meta['d_model'],
                            dropout=meta['dropout'],
                            attn_dropout=meta['attn_dropout'])
        self.mx = transformer.TransformerNN(seed=1, **numpy_kwargs)
        self.mx.load(os.path.join(MODEL_DIR, 'model.json'))
        cfg = dict(vocab_size=self.mx.vocab_size,
                   num_intents=self.mx.num_intents,
                   max_seq_len=self.mx.max_seq_len,
                   d=self.mx.d_model, blocks=self.mx.num_blocks,
                   heads=self.mx.num_heads,
                   ff_mult=self.mx.ff_dim // self.mx.d_model,
                   dropout=self.mx.dropout,
                   attn_dropout=self.mx.attn_dropout)
        self.tm = TorchTransformer(**cfg)
        copy_numpy_to_torch(self.mx, self.tm)
        self.tm.eval()
        self.X = _random_batch(self.mx, (32, cfg['max_seq_len']),
                               cfg['vocab_size'], seed=77)

    def test_production_probs_parity(self):
        nprobs = self.mx.predict_proba(self.X)
        with torch.no_grad():
            tprobs = self.tm(torch.from_numpy(self.X)).numpy()
        diff = float(np.abs(tprobs - nprobs).max())
        self.assertLess(diff, 2e-5, f'max |torch-numpy probs| = {diff:.3e}')

    def test_production_logits_parity(self):
        self.mx.predict_proba(self.X)
        pooled = self.mx._cache['pooled']
        nlogits = pooled @ self.mx.Whead + self.mx.bhead
        with torch.no_grad():
            tlogits = self.tm.logits(torch.from_numpy(self.X)).numpy()
        diff = float(np.abs(tlogits - nlogits).max())
        self.assertLess(diff, 2e-5, f'max |torch-numpy logits| = {diff:.3e}')


if __name__ == '__main__':
    unittest.main()
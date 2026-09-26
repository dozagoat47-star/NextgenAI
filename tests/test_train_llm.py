"""train_llm dinamik batch (PAD budama + uzunluk kumeleme) testleri.

Optimizasyonun dogrulugu: PAD kuyrugu budanip benzer uzunluklar paketlense de
1) maske toplami (= egitim kaybinin gordugu isaretler) birebir korunur,
2) transformator onundeki logitler budanan aralikta DEGISMEZ (kesme dogru).
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import PAD, LLM, encode_llm
from train_llm import TorchLLM, _pack_encoded, make_batches

_VOCAB = ['<PAD>', '<BOS>', '<SEP>', '<EOS>',
          'm', 'e', 'r', 'h', 'a', 'b', ' ', 'n', 's', 'i', 'l', 'y',
          'q', 'u', 'g', 'z', 'd', 'o']

# Torch tarafi testleri icin kucuk yapi (V, d_model, num_blocks).
_V, _D, _N = 24, 16, 2


def _dummy(max_seq=24):
    return LLM(_VOCAB, d_model=8, num_blocks=2, num_heads=2,
               max_ctx_len=10, max_seq_len=max_seq, seed=7)


class TestPackEncoded(unittest.TestCase):
    def test_pad_tail_is_trimmed(self):
        m = _dummy(24)
        seq, mask = encode_llm(m, 'merhaba', 'iyiyim')
        full_L = int(np.argmax(seq == PAD)) if (seq == PAD).any() else len(seq)
        self.assertLess(full_L, 24)      # kisa cift PAD'e bakmadan budanir
        batches = _pack_encoded([(seq, mask)], 4)
        self.assertEqual(batches[0][0].shape[1], full_L)
        self.assertEqual(np.asarray(mask).sum(), batches[0][1].sum())

    def test_lengths_bucketed_non_decreasing(self):
        m = _dummy(32)
        pair = [('merhaba nasilsin', 'iyiyim tesekkur ederim sen nasilsin'),
                ('selam', 'merhaba bugun nasilsin'),
                ('nasilsin', 'memnuniyetle yardimci olurum'),
                ('hava', 'bugun hava cok guzel ve gunesli')]
        enc = [encode_llm(m, c, r, max_seq=32) for c, r in pair]
        batches = _pack_encoded(enc, 2)
        widths = [b[0].shape[1] for b in batches]
        # Benzer uzunluklar kumelenir: ilk batch EN KISA sekanslari icerir
        self.assertEqual(widths, sorted(widths))
        for X, M in batches:
            self.assertEqual(X.shape, M.shape)
            self.assertLessEqual(X.shape[1], 32)

    def test_total_mask_preserved(self):
        m = _dummy(20)
        pair = [('merhaba', 'iyiyim tesekkur ederim'),
                ('hava nasil', 'bugun hava cok guzel'),
                ('selam', 'merhaba')]
        enc = [encode_llm(m, c, r) for c, r in pair]
        total_before = sum(np.asarray(mask).sum() for _, mask in enc)
        batches = _pack_encoded(enc, 2)
        total_after = sum(float(M.sum()) for _, M in batches)
        self.assertAlmostEqual(float(total_before), total_after)

    def test_forward_math_equivalent_after_trim(self):
        m = _dummy(24)
        seq, _mask = encode_llm(m, 'merhaba nasilsin', 'iyiyim tesekkur')
        L = int(np.argmax(seq == PAD)) if (seq == PAD).any() else len(seq)
        full = seq[None, :]
        trim = seq[:L][None, :]
        f_full = m.forward(full)
        f_trim = m.forward(trim)
        # Budanan aralik icindeki logitler birebir ayni olmali
        np.testing.assert_allclose(f_full[:, :L], f_trim, atol=1e-6)


class TestMakeBatches(unittest.TestCase):
    def test_end_to_end_shapes(self):
        m = _dummy(32)
        pairs = [('merhaba', 'iyiyim tesekkur ederim'),
                 ('hava nasil', 'bugun hava cok guzel olacak'),
                 ('selam', 'merhaba iyiyim'),
                 ('yunus', 'yapay zeka nedir bilgi'),
                 ('ai', 'yapay zeka gelisiyor')]
        batches = make_batches(pairs, 2, m, {})
        n_rows = sum(b[0].shape[0] for b in batches)
        self.assertEqual(n_rows, len(pairs))
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(b[0].shape[1] <= 32 for b in batches))


class TestTiedEmbeddings(unittest.TestCase):
    """Gomme <-> cikis bagliligi: tasarruf, esdegerlik, uyumluluk."""

    def _tied(self):
        return TorchLLM(_V, d_model=_D, num_blocks=_N, num_heads=2,
                       ff_mult=2, max_seq_len=24, tie_embeddings=True).eval()

    def _untied(self):
        return TorchLLM(_V, d_model=_D, num_blocks=_N, num_heads=2,
                       ff_mult=2, max_seq_len=24, tie_embeddings=False).eval()

    def test_head_not_stored_and_params_saved(self):
        tied, untied = self._tied(), self._untied()
        self.assertNotIn('head', tied.state_dict())
        self.assertIn('head', untied.state_dict())
        n_tied = sum(p.numel() for p in tied.parameters())
        n_untied = sum(p.numel() for p in untied.parameters())
        # Tasarruf tam olarak V*d (cikis matrisi).
        self.assertEqual(n_untied - n_tied, _V * _D)

    def test_matches_untied_with_head_transposed(self):
        """Bagli forward, head=embed^T olan ayri modelle BIT Ayni olmali."""
        tied = self._tied()
        untied = self._untied()
        sd = tied.state_dict()
        sd['head'] = sd['embed'].t().contiguous()
        untied.load_state_dict(sd)
        x = torch.randint(0, _V, (3, 12))
        with torch.no_grad():
            self.assertTrue(torch.equal(tied(x), untied(x)))

    def test_head_actually_used(self):
        """head yok sayilmiyor: rastgele head degisince cikti degismeli."""
        tied = self._tied()
        untied = self._untied()
        sd = tied.state_dict()
        sd['head'] = sd['embed'].t().contiguous()   # esdeger kurulum
        untied.load_state_dict(sd)
        x = torch.randint(0, _V, (2, 10))
        with torch.no_grad():
            before = untied(x)
            sd = untied.state_dict()
            sd['head'] = torch.randn_like(sd['head'])
            untied.load_state_dict(sd)
            self.assertGreater((before - untied(x)).abs().max().item(), 1e-3)

    def test_embed_gets_gradient_from_both_paths(self):
        """Baglilik tek yonlu degil: embed, cikis yolundan da gradyan alir."""
        m = self._tied()
        m.train()
        m.embed.grad = None
        x = torch.randint(0, _V, (2, 8))
        torch.nn.functional.cross_entropy(
            m(x).reshape(-1, _V),
            torch.randint(0, _V, (16,))).backward()
        self.assertIsNotNone(m.embed.grad)
        self.assertGreater(m.embed.grad.norm().item(), 0.0)

    def test_torch_numpy_parity_tied(self):
        """Bagli modelde torch -> numpy parity (export yolu)."""
        m = self._tied()
        np_m = LLM(list(range(_V)), d_model=_D, num_blocks=_N, num_heads=2,
                   ff_mult=2, max_seq_len=24, seed=7)
        np_m.tied_embeddings = True
        np_m.params = {k: np.asarray(v.detach().cpu(), np.float32)
                       for k, v in m.state_dict().items()}
        x = torch.randint(0, _V, (4, 12))
        with torch.no_grad():
            torch_logits = m(x).float().numpy()
        diff = float(np.max(np.abs(torch_logits - np_m.forward(x.numpy()))))
        self.assertLess(diff, 1e-3)

    def test_untied_flag_defaults_false_for_fresh_numpy_model(self):
        """Taze LLM() cagrisi eski (ayri head) davranisini korumali."""
        self.assertFalse(LLM(_VOCAB, d_model=8, num_blocks=2, num_heads=2,
                             max_ctx_len=10, max_seq_len=24, seed=7
                             ).tied_embeddings)


class TestNumpyTiedLoad(unittest.TestCase):
    """llm.py tarafi: bayrak okuma + yanlis eslesme reddi."""

    def _params(self, drop_head):
        m = LLM(list(range(_V)), d_model=_D, num_blocks=_N, num_heads=2,
                ff_mult=2, max_seq_len=16, seed=5)
        p = dict(m.params)
        if drop_head:
            p.pop('head')
        return p

    def _header(self, **extra):
        h = {'V': _V, 'd_model': _D, 'num_blocks': _N, 'num_heads': 2,
             'ff_mult': 2, 'max_ctx_len': 8, 'max_seq_len': 16,
             'tok_mode': 'char', 'vocab': list(range(_V))}
        h.update(extra)
        return h

    def _load(self, params, header):
        tmp = tempfile.mkdtemp()
        wp = os.path.join(tmp, 'w.npz')
        np.savez(wp, **{k: np.asarray(v, np.float32) for k, v in params.items()})
        return LLM(['x']).from_dict(header, weights_path=wp)

    def test_tied_loads_without_head(self):
        m = self._load(self._params(drop_head=True),
                       self._header(tied_embeddings=True))
        self.assertTrue(m.tied_embeddings)

    def test_tied_header_with_untied_weights_rejected(self):
        """Bagli header + ayri NPZ sessizce yanlis uretirdi; reddedilmeli."""
        with self.assertRaises(ValueError) as cm:
            self._load(self._params(drop_head=False),
                       self._header(tied_embeddings=True))
        self.assertIn('head', str(cm.exception))

    def test_untied_still_loads(self):
        m = self._load(self._params(drop_head=False),
                       self._header(tied_embeddings=False))
        self.assertFalse(m.tied_embeddings)

    def test_legacy_header_without_flag_loads(self):
        """Eski deploy edilen model (bayrak yok) bozulmamali."""
        m = self._load(self._params(drop_head=False), self._header())
        self.assertFalse(m.tied_embeddings)

    def test_untied_missing_head_rejected(self):
        with self.assertRaises(ValueError) as cm:
            self._load(self._params(drop_head=True),
                       self._header(tied_embeddings=False))
        self.assertIn('head', str(cm.exception))


class TestTiedExport(unittest.TestCase):
    """Kaggle export yolu: basit dis durum -> NPZ + header -> from_dict."""

    def _trained(self, tie=True, steps=2):
        m = TorchLLM(_V, d_model=_D, num_blocks=_N, num_heads=2, ff_mult=2,
                     max_seq_len=24, tie_embeddings=tie)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        for _ in range(steps):
            opt.zero_grad()
            x = torch.randint(0, _V, (3, 10))
            torch.nn.functional.cross_entropy(
                m(x).reshape(-1, _V), torch.randint(0, _V, (30,))).backward()
            opt.step()
        # Export oncesi dropout KAPALI olmali (train_llm.py: _set_mode(False)).
        # Train modunda kalsak parity rastgele bozulur.
        m.eval()
        return m, {k: v.detach().cpu().clone()
                   for k, v in m.state_dict().items()}

    def test_export_roundtrip_end_to_end(self):
        m, best_state = self._trained(tie=True)
        self.assertNotIn('head', best_state)      # export onayi (Kaggle yolu)
        tmp = tempfile.mkdtemp()
        wp = os.path.join(tmp, 'llm_model_weights.npz')
        np.savez(wp, **{k: np.asarray(v, np.float32)
                        for k, v in best_state.items()})
        header = {
            'arch': 'llm', 'V': _V, 'd_model': _D, 'num_blocks': _N,
            'num_heads': 2, 'ff_mult': 2, 'max_ctx_len': 10, 'max_seq_len': 24,
            'tied_embeddings': bool(m.tie_embeddings),
            'weights_file': 'llm_model_weights.npz',
            'tok_mode': 'char', 'vocab': list(range(_V)),
        }
        loaded = LLM(['x']).from_dict(header, weights_path=wp)
        self.assertTrue(loaded.tied_embeddings)
        x = torch.randint(0, _V, (4, 12))
        with torch.no_grad():
            torch_logits = m(x).float().numpy()
        diff = float(np.max(np.abs(torch_logits - loaded.forward(x.numpy()))))
        self.assertLess(diff, 1e-3)

    def test_untied_export_roundtrip_end_to_end(self):
        m, best_state = self._trained(tie=False)
        self.assertIn('head', best_state)
        tmp = tempfile.mkdtemp()
        wp = os.path.join(tmp, 'llm_model_weights.npz')
        np.savez(wp, **{k: np.asarray(v, np.float32)
                        for k, v in best_state.items()})
        header = {'V': _V, 'd_model': _D, 'num_blocks': _N, 'num_heads': 2,
                  'ff_mult': 2, 'max_ctx_len': 10, 'max_seq_len': 24,
                  'tied_embeddings': False, 'tok_mode': 'char',
                  'vocab': list(range(_V))}
        loaded = LLM(['x']).from_dict(header, weights_path=wp)
        x = torch.randint(0, _V, (4, 12))
        with torch.no_grad():
            torch_logits = m(x).float().numpy()
        diff = float(np.max(np.abs(torch_logits - loaded.forward(x.numpy()))))
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    unittest.main()
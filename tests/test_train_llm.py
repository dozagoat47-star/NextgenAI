"""train_llm dinamik batch (PAD budama + uzunluk kumeleme) testleri.

Optimizasyonun dogrulugu: PAD kuyrugu budanip benzer uzunluklar paketlense de
1) maske toplami (= egitim kaybinin gordugu isaretler) birebir korunur,
2) transformator onundeki logitler budanan aralikta DEGISMEZ (kesme dogru).
"""
import os
import sys
import unittest

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import PAD, LLM, encode_llm
from train_llm import _pack_encoded, make_batches

_VOCAB = ['<PAD>', '<BOS>', '<SEP>', '<EOS>',
          'm', 'e', 'r', 'h', 'a', 'b', ' ', 'n', 's', 'i', 'l', 'y',
          'q', 'u', 'g', 'z', 'd', 'o']


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


if __name__ == '__main__':
    unittest.main()
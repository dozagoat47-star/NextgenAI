"""llm.sample 'konu cekimi' (knowledge_bias) icin regresyon testleri.

knowledge_bias>0 iken bilgi parcasinda gecen icerik token'larina logit bonusu
uygulanir; bonus uretim basinda tam, sona dogru dogrusal sonecek sekilde
azalir. Testler: icerik token sectimi, cekimin gecerliligi, bilgi yokken
davranisin DEGISMEMESI (regresyon).
"""
import os
import sys
import unittest

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import LLM

_CORPUS = ['merhaba nasilsin iyiyim tesekkur', 'bugun hava guzel cok',
           'yapay zeka nedir bilgi', 'merhaba nasilsin iyiyim']


def _tok():
    from bpe import train_bpe
    return train_bpe(_CORPUS, vocab_size=300, min_freq=1, min_word_freq=1)


def _tiny(tok=None):
    return LLM(d_model=16, num_blocks=2, num_heads=2,
               max_ctx_len=40, max_seq_len=80, seed=7, tokenizer=tok)


class TestContentTokens(unittest.TestCase):
    def test_char_mode_keeps_long_tokens(self):
        m = LLM(['<PAD>', '<BOS>', '<SEP>', '<EOS>', 'a', 'ab', 'cde', 'x'],
                d_model=16, num_blocks=1, num_heads=1,
                max_ctx_len=40, max_seq_len=80, seed=7)
        ids = [m.c2i['ab'], m.c2i['cde']]
        sel = m._content_tokens(ids)
        self.assertIn(m.c2i['cde'], sel)
        self.assertNotIn(m.c2i['ab'], sel)

    def test_bpe_mode_filters_short_subwords(self):
        tok = _tok()
        self.assertGreaterEqual(
            len(tok.decode([tok.encode('merhaba')[0]])), 3)
        m = _tiny(tok)
        ids = tok.encode('merhaba guzel')
        sel = m._content_tokens(ids)
        self.assertTrue(sel)
        for i in sel:
            self.assertGreaterEqual(len(tok.decode([i])), 3)
        self.assertIn(tok.encode('merhaba')[0], sel)


class TestKnowledgeBias(unittest.TestCase):
    def test_bias_zero_is_identical_to_legacy(self):
        m = _tiny(_tok())
        np.random.seed(42)
        a = m.sample('abc', knowledge='merhaba hava iyiyim', max_len=16,
                     knowledge_bias=0.0)
        np.random.seed(42)
        b = m.sample('abc', knowledge='merhaba hava iyiyim', max_len=16)
        self.assertEqual(a, b)

    def test_strong_bias_carries_knowledge_word(self):
        m = _tiny(_tok())
        np.random.seed(11)
        text = m.sample('abc', knowledge='merhaba guzel', max_len=16,
                        knowledge_bias=8.0)
        self.assertIn('merhaba', text)

    def test_moderate_bias_changes_generation(self):
        m = _tiny(_tok())
        np.random.seed(23)
        a = m.sample('abc', knowledge='merhaba guzel', max_len=16,
                     knowledge_bias=0.0)
        np.random.seed(23)
        b = m.sample('abc', knowledge='merhaba guzel', max_len=16,
                     knowledge_bias=1.2)
        self.assertNotEqual(a, b)

    def test_no_knowledge_boost_idle(self):
        m = _tiny(_tok())
        np.random.seed(5)
        a = m.sample('abc', knowledge=None, max_len=16, knowledge_bias=8.0)
        np.random.seed(5)
        b = m.sample('abc', knowledge=None, max_len=16, knowledge_bias=0.0)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
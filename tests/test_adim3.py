"""Adim 3 testleri: context genisletme + veri/knowledge zenginlestirme.

1) MAX_CTX_LEN=48 / MAX_SEQ_LEN=192 / CTX_CHARS=64 sabitleri ve refine_resp
   butcesi (140 karakter).
2) encode_llm yeni budceyle 192 token'i, eski 160/40 kombinasyonuyla da 160
   token'i asmaz (geri uyumluluk).
3) kb LUT normalizasyonu: knowledge_map anahtari ile egitim ctx'si AYNI uzayda
   (clean_chars + CTX_CHARS) -> RAG isabeti %65 -> %76. Dosyanin kendisi
   degismez (brain ham sorguya vurmaya devam eder).
4) Yeniden uretilen knowledge_map: tum text'ler <= 300 karakter, satir >= 4000.
5) brain ascii-kesim fallback: 64+ karakterli sorgu da orijinal bilgiyi bulur.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import EOS, PAD, LLM, encode_llm
from seqgen import clean_chars
from train_llm import (CTX_CHARS, KB_TEXT_CHARS, MAX_CTX_LEN, MAX_SEQ_LEN,
                       build_kb_lut, refine_resp)

_VOCAB = ['<PAD>', '<BOS>', '<SEP>', '<EOS>',
          'm', 'e', 'r', 'h', 'a', 'b', ' ', 'n', 's', 'i', 'l', 'y',
          'q', 'u', 'g', 'z', 'd', 'o', 't', 'k', 'c']
_KB = os.path.join(BASE, 'knowledge_map.jsonl')


class TestContextExpansion(unittest.TestCase):
    def test_new_budget_constants(self):
        # Adim 3: sorgu 40 -> 48 token, sekans 160 -> 192
        self.assertEqual(MAX_CTX_LEN, 48)
        self.assertEqual(MAX_SEQ_LEN, 192)
        # sorgu karakter butcesi (load_pairs ctx_len + kb lut trunc)
        self.assertEqual(CTX_CHARS, 64)
        self.assertEqual(KB_TEXT_CHARS, 300)
        # dogal bosluk butcesi: 192 - 48 - 4 = 140 karakter
        self.assertEqual(MAX_SEQ_LEN - MAX_CTX_LEN - 4, 140)

    def test_refine_resp_default_cap_is_140_chars(self):
        long_r = ('x ' * 500).strip()
        out = refine_resp(long_r)
        self.assertIsNotNone(out)
        self.assertLessEqual(len(out), 140)
        self.assertTrue(out)  # bos secim degeri degeri dondurur

    def test_encode_budget_192_not_exceeded(self):
        m = LLM(_VOCAB, d_model=8, num_blocks=2, num_heads=2,
                max_ctx_len=48, max_seq_len=192, seed=7)
        ctx = 'kisaca benefse ve onun tarihi hakkinda bilgi ver lufen'
        resp = ' '.join(['benefse eski bir kasabadir'] * 4)
        kb = ' '.join(['bilgi parcasi iceriyor'] * 30)
        seq, mask = encode_llm(m, ctx, resp, context=kb, max_seq=192)
        L = int(np.argmax(seq == PAD)) if (seq == PAD).any() else len(seq)
        self.assertLessEqual(L, 192)
        self.assertIn(EOS, seq[:L])
        # bilgi + yanit butceye SIGMALI sekilde eslestirmeli (maske tokenlari)
        self.assertGreater(int(np.asarray(mask).sum()), 0)
        self.assertEqual(len(seq), len(mask))

    def test_encode_budget_legacy_160_ok(self):
        # eski 40/160 kombinasyonu geri uyumlu (yeni model yok, args onemli)
        m = LLM(_VOCAB, d_model=8, num_blocks=2, num_heads=2,
                max_ctx_len=40, max_seq_len=160, seed=7)
        seq, _mask = encode_llm(m, 'merhaba nasilsin', 'iyiyim tesekkur ederim',
                                max_seq=160)
        L = int(np.argmax(seq == PAD)) if (seq == PAD).any() else len(seq)
        self.assertLessEqual(L, 160)
        self.assertIn(EOS, seq[:L])


class TestKbLutNormalization(unittest.TestCase):
    def _write_kb(self, rows):
        fd, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(fd)  # Windows: fd kapali degilse dosya kilitli kalir
        with io.open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return path

    def test_keys_normalized_to_training_ctx_space(self):
        # Turkce imlali + 64 karekteri asan desenler de egitim ctx'siyle eslesmeli
        rows = [
            {'ctx': 'yağmur nasıl oluşur', 'text': 'yagmur bilgi'},
            {'ctx': 'kisaca benefse hakkinda bana tum detaylari anlatir misin '
                    'cunku uzun bir aciklama gerekiyor ve sabirliyim', 'text': 'uzun bilgi'},
        ]
        path = self._write_kb(rows)
        try:
            lut = build_kb_lut(path)
            # egitim ctx'siyle birebir ayni uzay: clean_chars(ctx, CTX_CHARS)
            self.assertEqual(lut[clean_chars(rows[0]['ctx'], CTX_CHARS)],
                             rows[0]['text'])
            self.assertIn(clean_chars(rows[1]['ctx'], CTX_CHARS), lut)
            self.assertEqual(lut[clean_chars(rows[1]['ctx'], CTX_CHARS)],
                             rows[1]['text'])
        finally:
            os.remove(path)

    def test_missing_file_returns_empty(self):
        self.assertEqual(build_kb_lut(os.path.join(BASE, 'yok.jsonl')), {})


class TestKnowledgeMapData(unittest.TestCase):
    def test_regenerated_map_within_char_cap(self):
        if not os.path.exists(_KB):
            self.skipTest('knowledge_map.jsonl yok')
        rows = []
        with io.open(_KB, encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        self.assertGreaterEqual(len(rows), 4000)
        for r in rows:
            self.assertTrue(r.get('ctx'))
            self.assertTrue(r.get('text'))
            self.assertLessEqual(len(r['text']), KB_TEXT_CHARS)

    def test_long_patterns_kept_by_ctx64(self):
        # 40-karakter boolean'da kesilen desenler 64'te korunur
        if not os.path.exists(_KB):
            self.skipTest('knowledge_map.jsonl yok')
        long_keys = []
        with io.open(_KB, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if len(r['ctx']) > 40:
                    long_keys.append(r['ctx'])
        self.assertGreater(len(long_keys), 0)
        for k in long_keys:
            self.assertGreater(len(clean_chars(k, CTX_CHARS)), 40)


class TestBrainAsciiFallback(unittest.TestCase):
    @staticmethod
    def _stub():
        from brain import ChatBot
        return SimpleNamespace(_kb_map=None, _kb_ascii=None), ChatBot

    def test_long_query_finds_knowledge_via_truncated_lut(self):
        if not os.path.exists(_KB):
            self.skipTest('knowledge_map.jsonl yok')
        stub, ChatBot = self._stub()
        query = None
        with io.open(_KB, encoding='utf-8') as f:
            for line in f:
                ctx = json.loads(line).get('ctx')
                if ctx and len(ctx) > 64:
                    query = ctx
                    break
        if query is None:
            self.skipTest('64+ karakter desen yok')
        out = ChatBot._external_knowledge(stub, query)
        self.assertIsNotNone(out)
        self.assertTrue(len(out) >= 20)
        # fallback lut bayraklanmali ve kesik formda dogru kayda isaret etmeli
        self.assertTrue(stub._kb_ascii)
        self.assertEqual(stub._kb_ascii[clean_chars(query, CTX_CHARS)], out)

    def test_short_query_exact_hit_unchanged(self):
        if not os.path.exists(_KB):
            self.skipTest('knowledge_map.jsonl yok')
        stub, ChatBot = self._stub()
        with io.open(_KB, encoding='utf-8') as f:
            short_ctx = json.loads(f.readline()).get('ctx')
        self.assertTrue(short_ctx)
        self.assertLessEqual(len(short_ctx), 64)
        out = ChatBot._external_knowledge(stub, short_ctx)
        self.assertEqual(out, stub._kb_map[short_ctx])


if __name__ == '__main__':
    unittest.main()
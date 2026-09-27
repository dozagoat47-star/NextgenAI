# -*- coding: utf-8 -*-
"""clean_intents.py korpus koruma kilidi testleri.

Gercek olay: script elle calistirildi ve corpus.jsonl'i 129.577 ->
94.662 yazdi, 34.915 parca (%27) sessizce silindi. Sebep
clean_corpus_file'daki '10 kelimeden kisa metni ele' kurali.

Artik kayip %5'i asarsa dosyaya DOKUNULMAZ ve komut hata verir.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import clean_intents


def parca(i, kelime):
    return {
        'id': 'k%d' % i,
        'title': 'konu %d' % i,
        'text': ' '.join(['kelime'] * kelime),
        'source': 'test',
    }


class TestCorpusDropGuard(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='korum_')
        self.corpus = os.path.join(self.tmp, 'corpus.jsonl')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _yaz(self, kayitlar):
        with io.open(self.corpus, 'w', encoding='utf-8') as f:
            for r in kayitlar:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def _oku(self):
        with io.open(self.corpus, encoding='utf-8') as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_dangerous_drop_aborts_and_keeps_file(self):
        """%27 kayip: dosya aynen kalmali."""
        self._yaz([parca(i, 5) for i in range(100)])   # hepsi 5 kelime -> elenir
        with self.assertRaises(SystemExit):
            clean_intents.clean_corpus_file(self.corpus)
        self.assertEqual(len(self._oku()), 100,
                         'kilit devreye girmeliydi, dosya korunmaliydi')

    def test_small_drop_proceeds(self):
        """%2 kayip: normal temizlik, yazmali."""
        self._yaz([parca(i, 20 if i < 98 else 5) for i in range(100)])
        clean_intents.clean_corpus_file(self.corpus)
        self.assertEqual(len(self._oku()), 98)

    def test_force_overrides_guard(self):
        """--force bilerek kabul."""
        self._yaz([parca(i, 5) for i in range(100)])
        clean_intents.clean_corpus_file(self.corpus, force=True)
        self.assertEqual(len(self._oku()), 0)

    def test_custom_threshold(self):
        self._yaz([parca(i, 20 if i < 90 else 5) for i in range(100)])
        # %10 kayip, esik %20 -> gecer
        clean_intents.clean_corpus_file(self.corpus, max_drop=0.20)
        self.assertEqual(len(self._oku()), 90)

    def test_no_loss_when_nothing_matches(self):
        self._yaz([parca(i, 30) for i in range(50)])
        clean_intents.clean_corpus_file(self.corpus)
        self.assertEqual(len(self._oku()), 50)

    def test_empty_file_is_safe(self):
        self._yaz([])
        clean_intents.clean_corpus_file(self.corpus)
        self.assertEqual(len(self._oku()), 0)


if __name__ == '__main__':
    unittest.main()

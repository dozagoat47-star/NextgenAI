"""Nextgen AI - Cekirdek test paketi (CI icin).

Model egitiminden BAGIMSIZ calisir: NLP pipeline'i, intents.json
semasini, RAG corpus'unu ve LSTM (seqgen) gradyan sapini dogrular.
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

# Proje koku import yoluna eklenir (tests/ dizini altindayiz)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from brain import ChatBot, STOPWORDS, NEGATION_SIGNALS, NEGATION_SUFFIXES


class TestNLP(unittest.TestCase):
    """Tokenizer / stemming / olumsuzluk algilama / ASCII normalizasyon."""

    def setUp(self):
        self.bot = ChatBot()

    def test_ascii_normalize(self):
        self.assertEqual(self.bot.ascii_normalize('Şükran öğle ılık'),
                         'sukran ogle ilik')

    def test_stopwords_present(self):
        self.assertTrue('ve' in STOPWORDS)
        self.assertTrue('ama' in STOPWORDS)
        self.assertTrue('nasil' in STOPWORDS)

    def test_negation_signals(self):
        self.assertTrue('degil' in NEGATION_SIGNALS)
        self.assertTrue('hayir' in NEGATION_SIGNALS)
        self.assertTrue('asla' in NEGATION_SIGNALS)

    def test_negation_suffixes(self):
        self.assertTrue('miyorum' in NEGATION_SUFFIXES)
        self.assertTrue('maz' in NEGATION_SUFFIXES)

    def test_is_negation_word(self):
        self.assertTrue(self.bot.is_negation_word('sevmiyorum'))
        self.assertTrue(self.bot.is_negation_word('degil'))
        self.assertFalse(self.bot.is_negation_word('kedi'))

    def test_detect_negation(self):
        neg, content = self.bot.detect_negation('futbol sevmiyorum')
        self.assertTrue(neg)
        self.assertIn('futbol', content)

    def test_detect_negation_negative_case(self):
        neg, _ = self.bot.detect_negation('futbol oynuyorum')
        self.assertFalse(neg)

    def test_stem_simple(self):
        # Turkce cokluk eki altka
        self.assertEqual(self.bot.simple_stem('kediler'), 'ked')
        # Olumsuzluk infiksi 'miy' gövdede korunur
        self.assertNotEqual(self.bot.simple_stem('sevmiyim'), 'sev')

    def test_tokenize_punctuation(self):
        toks = self.bot.tokenize('Merhaba, nasilsin?')
        self.assertIsInstance(toks, list)
        self.assertTrue(toks)

    def test_tokenize_bigram(self):
        toks = self.bot.tokenize('yapay zeka nedir')
        self.assertIn('yapay_zeka', toks)

    def test_bag_of_words_requires_vocab(self):
        # Vocabulary yuklenmemisse bos ok bulonur (model'siz guvenli)
        bag = self.bot.bag_of_words(['kedi'])
        self.assertEqual(bag, [])


class TestIntentsSchema(unittest.TestCase):
    """intents.json yapisal tutarlilik (egitim oncesi kalkan)."""

    def setUp(self):
        with open(os.path.join(BASE, 'intents.json'), 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def test_intents_present(self):
        self.assertGreater(len(self.data['intents']), 500)

    def test_each_intent_shape(self):
        seen = set()
        for it in self.data['intents']:
            tag = it.get('tag', '')
            self.assertTrue(len(tag) >= 2, f'kisa tag: {tag!r}')
            self.assertNotIn(tag, seen)
            seen.add(tag)
            self.assertTrue(it.get('patterns'), f'{tag}: kalipsiz')
            self.assertTrue(it.get('responses'), f'{tag}: yanitsiz')
            for p in it['patterns']:
                self.assertTrue(p.strip(), f'{tag}: bos kalip')
            for r in it['responses']:
                self.assertTrue(len(r.strip()) >= 2, f'{tag}: cok kisa yanit')

    def test_tags_latin(self):
        # Tag Latin disinda bir HARF iceremez (Turkce degerler zaten ascii_kodlandi).
        for it in self.data['intents']:
            tag = it['tag']
            bad = [c for c in tag if c.isalpha() and ord(c) > 127]
            self.assertFalse(bad, f'Latin disi karakter: {tag!r}')


class TestCorpus(unittest.TestCase):
    """RAG corpus yukleme + NASIL arama (embedding depolu ama model'siz)."""

    def test_corpus_search(self):
        from corpus import Corpus
        c = Corpus()
        chunks = c.load()
        self.assertGreater(len(chunks), 1000,
                           'corpus.jsonl en az 1000 parca islerligi dogruluyor')
        r = c.search('klorofil ne ise yarar')
        self.assertIsNotNone(r)
        self.assertTrue(r.get('title'))
        self.assertTrue(r.get('text'))
        self.assertGreater(r.get('score', 0), 0)

    def test_corpus_snippet(self):
        from corpus import Corpus
        text = 'Birinci cumle. Ikinci cumle uzun ve ayrintili bicimde devam ediyor.'
        self.assertEqual(Corpus.snippet(text, max_len=20), 'Birinci cumle.')


class TestSeqGenLSTM(unittest.TestCase):
    """seqgen LSTM geri yayılımı küçük veriyle kaybi düsürmeli."""

    @staticmethod
    def _build():
        import seqgen
        texts = ['nasilsin', 'iyiyim', 'tesekkur ederim', 'rica ederim',
                 'python nedir', 'python programlama dilidir',
                 'kedi beslemek', 'kediler evcil hayvandir']
        vocab = seqgen.build_vocab(texts, min_count=1)
        m = seqgen.SeqModel(vocab, hidden=32)
        pairs = [('nasilsin', 'iyiyim'), ('tesekkur ederim', 'rica ederim'),
                 ('python nedir', 'python programlama dilidir'),
                 ('kedi beslemek', 'kediler evcil hayvandir'),
                 ('hava nasil', 'hava bugun cok guzel')]
        mb = seqgen.make_batches(m, pairs, 2)
        return m, mb

    def test_lstm_learns(self):
        m, mb = self._build()
        first = None
        last = None
        for step in range(60):
            tot = 0.0
            for inp, tgt, mask in mb:
                loss, _ = m.train_minibatch(inp, tgt, mask, 0.01)
                tot += loss
            if first is None:
                first = tot / len(mb)
            last = tot / len(mb)
        self.assertLess(last, first,
                        f'LSTM kaybı düşmedi: {first:.4f} -> {last:.4f}')

    def test_sample_runs(self):
        from seqgen import SeqModel, build_vocab
        vocab = build_vocab(['merhaba', 'selamlar', 'iyi gunler'], min_count=1)
        m = SeqModel(vocab, hidden=16)
        out = m.sample('merhaba', max_len=12)
        self.assertIsInstance(out, str)


class TestModuleImports(unittest.TestCase):
    """Proje modülleri hicbir yavas runtime'a takilmadan import edilmeli."""

    def test_imports(self):
        import brain
        import corpus
        import generator
        import seqgen
        import clean_intents
        import scrape_intents
        import train  # egitici komutlari modül olarak da yuklenir
        for mod in (brain, corpus, generator, seqgen,
                    clean_intents, scrape_intents, train):
            self.assertIsNotNone(mod)


if __name__ == '__main__':
    unittest.main(verbosity=2)
"""Nextgen AI - Veri zenginlestirme (FAZ 2) test paketi.

naturalize.py (gercek Turkce imla), enrich_intents.py (yanit cesitlendirme,
6-pattern bilgi kurali, idempotans) ve corpus.search regresyonunu dogrular.

    python -m unittest tests.test_enrich -v
"""

import io
import json
import os
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import corpus as corpus_mod  # noqa: E402
from bpe import clean_text, turkish_lower  # noqa: E402
from enrich_intents import enrich_responses  # noqa: E402
from naturalize import natural_variants  # noqa: E402

TURKISH_SAMPLE = "İstanbul Türkiye'nin en büyük şehridir ve nüfusu yaklaşık 16 milyondur."


class TestNaturalizeProperTurkish(unittest.TestCase):
    """Varyantlar gercek Turkce imlayi korumali (ASCII'ye bozulmamali)."""

    def test_variants_keep_turkish_letters(self):
        sample = "Öğrenciler için şehirde güzel bir çalışma ortamı değil mi?"
        joined = ' '.join(natural_variants(sample, k=3))
        for ch in 'öüşçığ':
            self.assertIn(ch, joined)

    def test_no_ascii_mangling(self):
        for v in natural_variants('İstanbul büyük ve güzel bir şehirdir.', k=4):
            self.assertIn('stanbul', clean_text(v))

    def test_deterministic(self):
        a = natural_variants(TURKISH_SAMPLE, k=3)
        b = natural_variants(TURKISH_SAMPLE, k=3)
        self.assertEqual(a, b)

    def test_lowercased_stable(self):
        lo = turkish_lower(TURKISH_SAMPLE)
        self.assertEqual(natural_variants(lo, k=3)[0], clean_text(lo))


class TestEnrichResponses(unittest.TestCase):
    def _intent(self, patterns, responses):
        return {'tag': 'bilgi_x', 'patterns': patterns, 'responses': responses}

    def test_pattern_counts_untouched(self):
        """Bilgi intent'leri 6 desende kalmalidir (brain.py:874 sohbet kurali)."""
        it = self._intent(
            ['x nedir', 'x hakkında bilgi', 'x ne demek', 'x anlat',
             'bana x hakkında bilgi ver', 'x hakkında konuş'],
            ['x konusunda kısa bir özet.', 'x ve önemi üzerine kısa bilgi.'])
        before = len(it['patterns'])
        enrich_responses([it], k=3)
        self.assertEqual(len(it['patterns']), before)

    def test_response_variety_added(self):
        it = self._intent(
            ['x nedir', 'x hakkında bilgi', 'x ne demek', 'x anlat',
             'bana x hakkında bilgi ver', 'x hakkında konuş'],
            ['x konusunda kısa bir özet.'])
        n0 = len(it['responses'])
        enrich_responses([it], k=4)
        self.assertGreater(len(it['responses']), n0)

    def test_raw_originals_preserved(self):
        orig = 'x konusunda kısa bir özet.'
        it = self._intent(['a nedir', 'b nedir', 'c nedir', 'd nedir',
                           'e nedir', 'f nedir'], [orig])
        enrich_responses([it], k=3)
        self.assertIn(orig, it['responses'])

    def test_clean_dedup(self):
        it = self._intent(['a nedir', 'b nedir', 'c nedir', 'd nedir',
                           'e nedir', 'f nedir'], ['A şehri büyüktür.'])
        enrich_responses([it], k=5)
        keys = [clean_text(r) for r in it['responses']]
        self.assertEqual(len(keys), len(set(keys)))

    def test_variants_are_turkish(self):
        joined = ' '.join(natural_variants('A şehri büyüktür ve ünlüdür.', k=4))
        self.assertIn('ş', joined)
        self.assertIn('ü', joined)

    def test_variants_keep_first_word(self):
        """Tum varyantlar orijinalin ilk kelimesiyle baslamalidir.

        Ilk token kosullu ogrenilebilirligi icin kritik: aksi halde model
        soruya karsilik ilk kelimeyi tahmin edemeyip ezbere dolgu uretir.
        """
        samples = [
            'Merhaba! Ben Nextgen AI\'yim, nasılsın?',
            'Tabii! Sana nasıl yardımcı olabilirim? Buyur dinliyorum!',
            "Uganda A yarı finalde Ruanda'yı saf dışı bırakmıştır ve final oynadı.",
            'Bu kadar geniş alana dağılmış bu kuş türünün alt türleri yoktur.',
        ]
        for s in samples:
            first = clean_text(s).split()[0].strip('.,!?;:')
            for v in natural_variants(s, k=4):
                head = clean_text(v).split()[0].strip('.,!?;:')
                self.assertEqual(head, first, f'ilk kelime degisti: {v!r}')


class TestCorpusSearchRegression(unittest.TestCase):
    """corpus.search latent TypeError'larini (list<=set, set|=tuple) dogrular.

    Gercek 60K corpus yerine kucuk bir temp corpus ile hem desen-kisa-yolu
    hem trigram/LSA yolu tetiklenir ve hata verilmeden sonuc donmelidir.
    """

    def _make_corpus(self):
        chunks = [
            {'id': 'akil', 'title': 'Akil', 'text': 'Akil, dusunme ve kavrama yetisidir.',
             'patterns': 'akil nedir akil ne demek'},
            {'id': 'kemal-ataturk', 'title': 'Mustafa Kemal Ataturk',
             'text': 'Mustafa Kemal Ataturk, Turkiye Cumhuriyeti kurucusudur.',
             'patterns': 'mustafa kemal kimdir ataturk hakkinda'},
            {'id': 'istanbul', 'title': 'Istanbul',
             'text': 'Istanbul, Turkiye nin en kalabalik sehrdir.',
             'patterns': 'istanbul nedir'},
        ]
        p = os.path.join(tempfile.mkdtemp(), 'corpus.jsonl')
        with io.open(p, 'w', encoding='utf-8') as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        return p

    def _search(self, query):
        c = corpus_mod.Corpus(self._make_corpus())
        c.load()
        return c.search(query)

    def test_pattern_fast_path(self):
        r = self._search('akil nedir')
        self.assertIsNotNone(r)
        self.assertIn('Akil', r['title'])

    def test_semantic_trigram_path(self):
        q = 'cumhuriyet kurucusu kimdi'
        r = self._search(q)
        # Yolun tamamlanmasi (TypeError olmamasi) yeterli; sonuc None da olabilir.
        self.assertIsInstance(r, (dict, type(None)))


class TestEnrichMetaIdempotency(unittest.TestCase):
    def test_intents_has_enrich_marker(self):
        p = os.path.join(BASE, 'intents.json')
        if not os.path.exists(p):
            self.skipTest('intents.json yok')
        with io.open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data.get('_meta', {}).get('enrich_k'), 3)

    def test_knowledge_map_exists(self):
        p = os.path.join(BASE, 'knowledge_map.jsonl')
        if not os.path.exists(p):
            self.skipTest('knowledge_map.jsonl yok')
        with io.open(p, 'r', encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn('ctx', row)
            self.assertIn('text', row)


if __name__ == '__main__':
    unittest.main()
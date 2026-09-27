# -*- coding: utf-8 -*-
"""AutoGrow kalite kapisi ve konu rotasyonu testleri.

Gerekce (olcum): korpus medyan 179 krkt, %52'si 200 krktin altinda.
Neden uniform rastgele akis: gozlemsiz basliklar. Kategori tabanli akis
kullaniciya sorulan konulari getirir.
"""
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import autogrow


class TestCorpusQualityGate(unittest.TestCase):
    """corpus_worthy: 2+ cumle ve 200+ krkt."""

    def test_two_sentence_long_passes(self):
        ex = ('Bu konu hakkinda iki cumlelik uzun bir aciklama. '
              'Ikinci cumle konuyu genisletmeye devam eder ve ozet metni '
              'gereken en az iki yuz karakteri rahatlaga asar, boylece '
              'kapinin iki cumle ve iki yuz karakterlik sinirini gecer.')
        self.assertGreater(len(ex), 200)
        self.assertEqual(autogrow._sentence_count(ex), 2)
        self.assertTrue(autogrow.corpus_worthy(ex))

    def test_single_sentence_stub_rejected(self):
        # Rastgele akistan gelen ozetlerin %53'u tek cumle; medyan 111 krkt.
        ex = 'Rubus fecundus, Rubus cinsine ait bir turdur.'
        self.assertFalse(autogrow.corpus_worthy(ex))

    def test_long_but_single_sentence_rejected(self):
        # Uzun olmak yeterli degil: cumle sayisi de tanim sunuyor olmali.
        ex = ('Denizyildizi, hayvanlar alemine ait derisi dikenliler '
              'subesine bagli bir deniz omurgasizidir ve Dunya uzerinde '
              'tropikal bolgelerden kutup denizi sularina kadar yasar, '
              'cogu gecen yuz yildir ve besin zincirinin bir parcasi.')
        self.assertGreater(len(ex), 200)
        self.assertEqual(autogrow._sentence_count(ex), 1)
        self.assertFalse(autogrow.corpus_worthy(ex))

    def test_short_second_sentence_still_counts(self):
        """Kapi split_sentences'in 25 krkt filtresine bagli olmamali:
        kisa bir ikinci cumle gercekten ikinci cumledir."""
        ex = ('Bu konu, X uzerine ayrintili bir aciklama sunar ve okurken '
              'cok sayada ayrinti, ornek ve tarihsel bilgi verir; boylece '
              'konunun kapsami oldukca genis bir sekilde anlatilir ve okur '
              'kendisine saglam bir cerceve kazandir. Bu yeterlidir.')
        self.assertGreater(len(ex), 200)
        self.assertEqual(autogrow._sentence_count(ex), 2)
        self.assertTrue(autogrow.corpus_worthy(ex))

    def test_two_sentences_but_too_short_rejected(self):
        ex = 'Bu birinci cumle. Bu ikinci cumle.'
        self.assertFalse(autogrow.corpus_worthy(ex))

    def test_empty_and_none_rejected(self):
        self.assertFalse(autogrow.corpus_worthy(''))
        self.assertFalse(autogrow.corpus_worthy(None))

    def test_thresholds_match_measurement(self):
        """Esikler olcumden geliyor; degistirilirse yorum da guncellenmeli."""
        self.assertEqual(autogrow.MIN_CORPUS_TEXT, 200)
        self.assertEqual(autogrow.MIN_CORPUS_SENTENCES, 2)


class TestTopicCategories(unittest.TestCase):
    """Buyume ondalik kategorilerden gelmeli."""

    def test_categories_are_curated_and_valid(self):
        cats = autogrow.TOPIC_CATEGORIES
        self.assertGreaterEqual(len(cats), 8)
        for c in cats:
            self.assertTrue(c.startswith('Kategori:'), c)

    def test_round_budget_is_bounded(self):
        """Wikipedia kota uyguluyor: tur basina kategori sayisi sinirli
        kalmali, yoksa tum liste cekilip 429 yiyor."""
        self.assertLessEqual(autogrow.TOPIC_CATEGORIES_PER_ROUND,
                             len(autogrow.TOPIC_CATEGORIES))
        self.assertGreaterEqual(autogrow.TOPIC_CATEGORIES_PER_ROUND, 2)

    def test_sample_stays_within_list(self):
        import random as _r
        for _ in range(50):
            s = autogrow.random.sample(autogrow.TOPIC_CATEGORIES,
                                       autogrow.TOPIC_CATEGORIES_PER_ROUND)
            self.assertEqual(len(s), autogrow.TOPIC_CATEGORIES_PER_ROUND)
            for c in s:
                self.assertIn(c, autogrow.TOPIC_CATEGORIES)


class TestPipelineCaps(unittest.TestCase):
    """Tavanlar arasi invaryant: ikinci darbogaz birakilmamali.

    Zincir: intents.json (AutoGrow, tavan AUTOGROW_MAX_INTENTS) ->
    knowledge_map.jsonl (enrich_intents, tavan --kb-limit) ->
    LLM egitim verisi (train_llm --kb-map).

    Intent tarani kaldirilip kb-limit eski degerde kalirsa knowledge_map
    6.000'da doyar ve yeni bilgi yine modele giremez - sessiz darbogaz.
    """

    def test_intent_cap_is_not_a_blocker(self):
        # 800 idi ve doluydu -> AutoGrow intents.json'a hic yazamiyordu.
        self.assertGreater(autogrow.AUTOGROW_MAX_INTENTS, 2000)

    def test_intent_cap_env_overridable(self):
        import importlib

        eski = os.environ.get('AUTOGROW_MAX_INTENTS')
        try:
            os.environ['AUTOGROW_MAX_INTENTS'] = '12345'
            mod = importlib.reload(autogrow)
            self.assertEqual(mod.AUTOGROW_MAX_INTENTS, 12345)
        finally:
            if eski is None:
                os.environ.pop('AUTOGROW_MAX_INTENTS', None)
            else:
                os.environ['AUTOGROW_MAX_INTENTS'] = eski
            importlib.reload(autogrow)

    def test_kb_limit_covers_max_intent_patterns(self):
        import enrich_intents

        n = enrich_intents._default_kb_limit()
        en_fazla_desen = autogrow.AUTOGROW_MAX_INTENTS * autogrow.MAX_PATTERNS
        self.assertGreaterEqual(
            n, en_fazla_desen,
            'kb-limit ikinci darbogaz: %d desen uretilebilir ama tavan %d'
            % (en_fazla_desen, n))

    def test_kb_limit_budget_is_measurable(self):
        """Tavan buyutuldugunde sure olcumle izlenir: build_knowledge_map
        29.1 ms/desen (4.584 desen = 133 sn) + corpus.load 189 sn."""
        import enrich_intents

        n = enrich_intents._default_kb_limit()
        saniye = n * 0.0291
        self.assertLess(
            saniye, 30 * 60,
            'kb-limit gunluk ise sigmaz: %.0f dk' % (saniye / 60))


class TestGateImpactOnCorpusGrowth(unittest.TestCase):
    """Kapinin gecis orani olculmus degerlere yakin mi?

    Gercek akis olcumu: featured %100, rastgele akista 200 krkt ustu %37
    ve 2+ cumle filtresi bunu daha da dusurur. Burada sentetik ozetlerle
    kapi yalnizca cumle/uzunluk kurallarina uyar.
    """

    def test_gate_rejects_exactly_one_sentence(self):
        tek = ['X, Y cinsine ait bir turdur.' for _ in range(50)]
        gecen = [e for e in tek if autogrow.corpus_worthy(e)]
        self.assertEqual(gecen, [])

    def test_gate_accepts_substantial_text(self):
        cok = [('Birinci cumle konuyu anlatiyor. ' * 8 +
                'İkinci cumle burada genisletmeye devam ediyor.')
               for _ in range(50)]
        gecen = [e for e in cok if autogrow.corpus_worthy(e)]
        self.assertEqual(len(gecen), 50)


if __name__ == '__main__':
    unittest.main()

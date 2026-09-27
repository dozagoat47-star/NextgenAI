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

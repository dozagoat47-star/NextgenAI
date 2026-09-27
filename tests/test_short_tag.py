# -*- coding: utf-8 -*-
"""Kisa etiket filtresi: 'l nedir' gibi kalipsiz intent uretilmesin.

Yakalandi: intents.json'a tag='l' girmisti (Wikipedia'nin "L" maddesi).
Uretici iki yerde onlemiyordu:
  - autogrow.build_intent   -> yeni intent uretebiliyordu
  - clean_intents.merge_by_tag -> temizleyici elenmiyordu
Sonuc: egitim oncesi kalkan olan test_core.TestIntentsSchema dustu
(tag >= 2 istiyor) ve bot'a 'l nedir' diye sorulabilir hale geldi.

Bu test, kapinin gercekten kapandigini ve gecerli intent'lerin
DOKUNULMADIGINI gosterir.
"""
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import autogrow
import clean_intents


class TestShortTagRejected(unittest.TestCase):

    def test_min_tag_chars_defined_consistently(self):
        self.assertEqual(autogrow.MIN_TAG_CHARS, clean_intents.MIN_TAG_CHARS)
        # test_core.TestIntentsSchema en az 2 istiyor; 3 daha sert.
        self.assertGreaterEqual(
            clean_intents.MIN_TAG_CHARS, 2,
            'test_core etiket >= 2 istiyor, daha sert olmali')

    def test_build_intent_rejects_one_letter_topic(self):
        """Wikipedia harf maddeleri konu degildir."""
        ozet = ('Bu madde, konu hakkinda uzun ve ayrintili bir aciklama '
                'icerir. Ikinci cumle konuyu genisletir ve ozetin yeterli '
                'uzunluga ulasmasini saglar. Ucuncu cumle destekler.')
        self.assertIsNone(autogrow.build_intent('L', ozet))
        self.assertIsNone(autogrow.build_intent('W', ozet))

    def test_build_intent_rejects_two_letter_topic(self):
        ozet = ('Bu madde, konu hakkinda uzun ve ayrintili bir aciklama '
                'icerir. Ikinci cumle konuyu genisletir ve ozetin yeterli '
                'uzunluga ulasmasini saglar. Ucuncu cumle destekler.')
        self.assertIsNone(autogrow.build_intent('AB', ozet))

    def test_build_intent_accepts_normal_topic(self):
        """Kapı normal konuları engellememeli."""
        ozet = ('Fizik, dogadaki temel kuvvetleri ve maddelerin ozelliklerini '
                'inceleyen bilim dalidir. Hareket, kuvvet ve enerji '
                'arasindaki iliskileri arastirir. Deney ve gozlem yontemini '
                'kullanir.')
        it = autogrow.build_intent('Fizik', ozet)
        self.assertIsNotNone(it)
        self.assertGreaterEqual(len(it['tag']), clean_intents.MIN_TAG_CHARS)


class TestCleanerDropsShortTags(unittest.TestCase):

    def test_merge_by_tag_drops_short_tag(self):
        giris = [
            {'tag': 'l', 'patterns': ['l nedir'], 'responses': ['x y z']},
            {'tag': 'fizik', 'patterns': ['fizik nedir'],
             'responses': ['x y z']},
        ]
        sonuc = clean_intents.merge_by_tag(giris)
        taglar = [i['tag'] for i in sonuc]
        self.assertIn('fizik', taglar)
        self.assertNotIn('l', taglar)

    def test_merge_keeps_normal_tags_intact(self):
        giris = [
            {'tag': 'matematik', 'patterns': ['matematik nedir', 'matematik anlat'],
             'responses': ['a b c', 'd e f']},
        ]
        sonuc = clean_intents.merge_by_tag(giris)
        self.assertEqual(len(sonuc), 1)
        self.assertEqual(sonuc[0]['patterns'],
                         ['matematik nedir', 'matematik anlat'])
        self.assertEqual(len(sonuc[0]['responses']), 2)


if __name__ == '__main__':
    unittest.main()

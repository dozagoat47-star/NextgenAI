# -*- coding: utf-8 -*-
"""_tag_shares_content_word kapi regresyon testleri.

KAPI NEDEN BOZUKTU (25.09 olcumu):
    intents.json'da tag 'spor' -> patterns icinde "futbol nedir" VAR
    yanit[0] = "Futbol dunyanin en populer sporudur, 11 kisilik takimlar oynar!"
    siniflandirici            -> tag='spor', prob=0.999   (dogru)
    kapisi                    -> _tag_shares_content_word('spor', 'futbol nedir')
                                 = False
    cevap                     -> "Bu konuda bilgim yok..."  (yanit elde varken)

    sebep: kapi sorgunun icerik kelimelerini tag'in ADIYLA karsilastiriyordu
           ({'futbol'} & {'spor'} = {}). Tag bir ETIKETTIR; icerik yanitlarda.

OLCULEN ETKI: "X nedir" sorularinin %90'i "bilgim yok" donuyordu
              (9/10: futbol, python, ekonomi, turkiye, siber guvenlik,
               fizik, kuantum, islam, kadin, fotografi).

Bu testler kapinin (a) dogru cevabi GECIRDIGINI ve (b) alakasiz konuyu
HALE KAPATTIgini birlikte guvenceye alir.
"""
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from brain import ChatBot


# gercek veriden kesilmis, kucuk dilim
_INTENTS = {
    'spor': ['Futbol dunyanin en populer sporudur, 11 kisilik takimlar oynar!',
             'Basketbol 5 kisilik takimlarla oynanan bir toptur.',
             'Tenis tek kişilik oynanan bir raket sporudur.'],
    'programlama': ['Programlama bilgisayara talimat vermektir.',
                    'Python bir programlama dilidir.',
                    'Java da bir programlama dilidir.'],
    'teknoloji': ['Teknoloji alet ve makine uretimiyle ugrasan alandir.',
                  'Telefon, bilgisayar gibi cihazlar teknoloji urunleridir.',
                  'Internet arama motorlari teknoloji aracidir.'],
    'musiki': ['Muzik ritmik seslerden olusan sanattir.',
               'Popüler muzik turlari dinleyicileri sever.',
               'Klasik muzik bestecilerin yarattigi eserlerdir.'],
}


def _bot():
    b = ChatBot()
    b.intents = dict(_INTENTS)
    b.intent_kws = {}
    return b


class TestKapiDuzeltmesi(unittest.TestCase):
    def test_kapi_yanitlari_karsilastirir(self):
        """Asil regresyon: 'futbol nedir' -> spor intent'i gecmeli."""
        b = _bot()
        self.assertTrue(b._tag_shares_content_word('spor', 'futbol nedir'))

    def test_eski_hata_belgeli(self):
        """Kok neden: tag adi 'futbol' icermiyordu. Test bunu sabitler ki
        biri bir gun _tag_content_words'a geri dönüp hatayi getirmesin."""
        b = _bot()
        self.assertNotIn('futbol', b._content_words('spor'))
        self.assertIn('futbol', b._tag_content_words('spor'))

    def test_alakasiz_konu_kapiyi_kapatir(self):
        """Beklenen regresyon korumasi: konu iliskisizse kapali kalmali.
        'suyun formulu nedir' -> teknoloji secilmesin diye kapi vardi."""
        b = _bot()
        self.assertFalse(b._tag_shares_content_word('teknoloji',
                                                    'suyun formulu nedir'))

    def test_tag_adi_eslesmesi_hala_gecerli(self):
        """Geriye donuk uyum: sorgu tag'in adini dogrudan soyluyorsa gecmeli."""
        b = _bot()
        self.assertTrue(b._tag_shares_content_word('spor', 'spor nedir'))

    def test_tek_intent_kelimesi_eslesmesi(self):
        b = _bot()
        for kelime in ('basketbol', 'tenis', 'python', 'muzik'):
            self.assertTrue(
                b._tag_shares_content_word(
                    {'basketbol': 'spor', 'tenis': 'spor', 'python':
                     'programlama', 'muzik': 'musiki'}[kelime],
                    '%s nedir' % kelime),
                '%s icin kapi kapali kalmamali' % kelime)

    def test_sadece_kisitlayici_kelime_sorgusu_kapali(self):
        """Soru yalnizca stopword iceriyorsa eslesme sayilmaz."""
        b = _bot()
        self.assertFalse(b._tag_shares_content_word('spor', 'bu nedir'))

    def test_bos_sorgu_kapali(self):
        b = _bot()
        self.assertFalse(b._tag_shares_content_word('spor', ''))

    def test_bilinmeyen_tag_kapali(self):
        b = _bot()
        self.assertFalse(b._tag_shares_content_word('olmayan_tag', 'futbol'))

    def test_anahtar_kelime_seti_de_kullanilir(self):
        """intent_kws varsa o da icerik sayilir (pattern kokenleri).

        DIKKAT: intent_kws KOK saklar ('kaleci' -> 'kalec'), soru kelimeleri
        de koklendigi icin ikisinin de ayni normalizasyondan gecmesi sart.
        """
        b = _bot()
        b.intent_kws = {'spor': {'kalec'}}        # koklu hali
        self.assertTrue(b._tag_shares_content_word('spor', 'kaleci kimdir'))
        b.intent_kws = {'spor': {'kaleci'}}      # ham hali de eslenmeli
        self.assertTrue(b._tag_shares_content_word('spor', 'kaleci kimdir'))

    def test_anahtar_kelimesi_stopword_ise_katki_yok(self):
        b = _bot()
        b.intent_kws = {'spor': {'nedir', 'ne'}}
        self.assertFalse(b._tag_shares_content_word('spor', 'kaleci kimdir'))


class TestGercekVeri(unittest.TestCase):
    """intents.json varsa KESILMIS GERCEK vaka dogrulanir (yoksa atlanir)."""

    @classmethod
    def setUpClass(cls):
        import io
        import json
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            raise unittest.SkipTest('intents.json yok')
        with io.open(yol, encoding='utf-8') as f:
            veri = json.load(f)
        cls.intents = {i['tag']: i['responses'] for i in veri['intents']}
        cls.patterns = {i['tag']: i.get('patterns', [])
                        for i in veri['intents']}
        cls.tags = [t for t, p in cls.patterns.items()
                    if 'futbol nedir' in p]

    def test_futbol_nedir_bir_intent_patterni(self):
        """Veride 'futbol nedir' deyim bir sohbet intent'inin pattern'i."""
        self.assertTrue(self.tags,
                        "intents.json'da 'futbol nedir' pattern'i yok")

    def test_gercek_veride_kapi_ayakta(self):
        b = _bot()
        b.intents = self.intents
        for tag in self.tags:
            self.assertTrue(
                b._tag_shares_content_word(tag, 'futbol nedir'),
                "gercek veride %s kapisi 'futbol nedir'i reddediyor" % tag)

    def test_gercek_veride_kapi_ayirt_edici(self):
        """Kapi hem BULMALI hem DARALTMALI (precision + recall).

        Olcum (25.09): 'futbol nedir' 10.340 intent'in %0,53-10'unu kabul
        etmeli ve dogru olan 'spor' icinde olmali. 'suyun formulu nedir'
        gercekten su/formul gecen ~55 intent'te gecmeli (%0,53) - yani
        sifir degil, ama dar.
        """
        b = _bot()
        b.intents = self.intents
        ornek = list(self.intents)[::20]          # hizli olsun diye seyreltilmis
        eslesen = [t for t in ornek
                   if b._tag_shares_content_word(t, 'futbol nedir')]
        self.assertIn(self.tags[0], eslesen,
                      'dogru intent gercek veride elendi')
        self.assertLess(len(eslesen), len(ornek) * 0.10,
                        'kapi %%%d intent kabul etti - ayirt edici degil'
                        % (100.0 * len(eslesen) / max(1, len(ornek))))

    def test_gercek_veride_alakasiz_soru_dar(self):
        """Beklenen regresyon korumasi: alakasiz olgu sorusu cogu intent'i
        kabul etmemeli (eski 'suyun formulu nedir' -> teknoloji vakasi)."""
        b = _bot()
        b.intents = self.intents
        ornek = list(self.intents)[::20]
        eslesen = [t for t in ornek
                   if b._tag_shares_content_word(t, 'suyun formulu nedir')]
        self.assertLess(len(eslesen), len(ornek) * 0.02,
                        '%d/%d intent alakasiz soruyu kabul etti'
                        % (len(eslesen), len(ornek)))


if __name__ == '__main__':
    unittest.main()

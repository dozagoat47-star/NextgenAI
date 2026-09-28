# -*- coding: utf-8 -*-
"""Etiket Latin kapi testleri.

Gercek olay: AutoGrow Wikipedia basliklarindan U+01C1 (tik sesi),
U+0111 (Bosna-Hersek d'si) ve U+02BB (okina) iceren etiketler
uretti. ascii_normalize bunlari NFD ile AYRISTIRAMADIGI icin
gecirip intents.json'a yazdi:

    tag='ǁkaras bolgesi'
    tag='đakovo'
    tag='liliʻuokalani'

Sonuc: test_core.TestIntentsSchema.test_tags_latin elendi diye
egitim oncesi kalkan dustu ve CI kirmiziydi.

Asil neden: ascii_normalize once ceviri tablosunu uygular, sonra
NFD ayristirmasiyla aksanlari atar. Bu ucu Latin HARFI oldugu icin
ayristirilamaz; hicbir adim onlari yakalamaz.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from clean_intents import (MIN_TAG_CHARS, ascii_letters_only,
                           ascii_normalize, is_latin_tag,
                           merge_by_tag, non_latin_letters,
                           normalize_tag_latin)
from scrape_intents import merge_intents

# CI'da gercekten dusuren uc etiket (log'dan)
GERCEK = ['ǁkaras bolgesi', 'đakovo', 'liliʻuokalani']


def ascii_harf_mi(s):
    return [c for c in s if c.isalpha() and ord(c) > 127]


class TestAsciiNormalizeGebeginAlani(unittest.TestCase):
    """ascii_normalize'in kactigi karakterler."""

    def test_gercek_etiketler_ascii_normalize_den_gecer(self):
        """Bugunun kaniti: normalize bu ucunu GECIRIYOR.

        Test, kapinin neden gerekli oldugunu sabitler. Bir gun
        ascii_normalize duzelirse bu test kizilir ve o zaman kapinin
        gereksiz oldugu dogrulanir - bilerek kirmizi birakmamak icin
        burada acikca yorumlanir.
        """
        for et in GERCEK:
            norm = ascii_normalize(et)
            self.assertTrue(
                ascii_harf_mi(norm),
                '%r ascii_normalize sonrasi temizlendi; kapinin gerekceyi '
                'tazele (non_latin_letters/ascii_letters_only cagrisini '
                'gözden gecir)' % et)

    def test_aksanli_harfler_zaten_cozuluyor(self):
        """Kontrol: NFD'nin calistigi durum (sozde kotu degil)."""
        for et, beklenen in [('fotosentez', 'fotosentez'),
                             ('ışık', 'isik'),
                             ('şeker', 'seker')]:
            self.assertEqual(ascii_normalize(et), beklenen)


class TestIsLatinTag(unittest.TestCase):
    def test_temiz_etiketler_gecer(self):
        for et in ['fizik', 'call of duty 4: modern warfare', 'djl 2024',
                   'liri', 'okul_is', '1994 kis olimpiyatlari']:
            self.assertTrue(is_latin_tag(et), '%r yanlis reddedildi' % et)

    def test_gercek_kotu_etiketler_reddedilir(self):
        for et in GERCEK:
            self.assertFalse(is_latin_tag(et), '%r gecmemis olmamali' % et)

    def test_rakam_tire_nokta_serbest(self):
        """Yalnizca HARF bakar; isaret karakterleri serbest."""
        self.assertTrue(is_latin_tag('superstar 83'))
        self.assertTrue(is_latin_tag('agar.io'))
        self.assertTrue(is_latin_tag("1994 kis olimpiyatlari'nda"))
        self.assertTrue(is_latin_tag('stalag 381'))


class TestNonLatinLetters(unittest.TestCase):
    def test_harfleri_soyler(self):
        self.assertEqual(non_latin_letters('đakovo'),
                         ['đ'], 'U+0111 bildirilmeli')
        self.assertEqual(len(non_latin_letters('liliʻuokalani')), 1)

    def test_bosluk_sonuc_dondurur(self):
        self.assertEqual(non_latin_letters('fizik'), [])

    def test_sadece_harfleri_listeler(self):
        """Harf olmayan ASCII disi karakterler harf sayilmaz."""
        self.assertEqual(non_latin_letters('a—b'), [], 'uzun tire harf degil')
        self.assertEqual(non_latin_letters('a€b'), [], 'euro harf degil')


class TestAsciiLettersOnly(unittest.TestCase):
    def test_duzeltir(self):
        self.assertEqual(ascii_letters_only('đakovo'), 'akovo')
        self.assertEqual(ascii_letters_only('liliʻuokalani'), 'liliuokalani')
        self.assertEqual(ascii_letters_only('ǁkaras bolgesi'), 'karas bolgesi')

    def test_zaten_temiz_etiketi_bozmaz(self):
        for et in ['fizik', 'call of duty 4: modern warfare', 'djl 2024']:
            self.assertEqual(ascii_letters_only(et), et)

    def test_bos_veya_cok_kisa_sonuc_none(self):
        """Cikarilan harfler etiketi bosaltirsa None (kapi gecmez)."""
        self.assertIsNone(ascii_letters_only('đ'))
        self.assertIsNone(ascii_letters_only('ǁ'))
        self.assertIsNone(ascii_letters_only(''))
        self.assertIsNone(ascii_letters_only(None))

    def test_bosluklar_toplanir(self):
        self.assertEqual(ascii_letters_only('đ a k o v o'), 'a k o v o')


class TestNormalizeTagLatin(unittest.TestCase):
    def _kutu(self, etiketler):
        return [{'tag': t, 'patterns': [t + ' nedir'], 'responses': ['r']}
                for t in etiketler]

    def test_gercek_ucunu_duzeltir(self):
        temiz, duzeltilen, dusurulen = normalize_tag_latin(self._kutu(GERCEK))
        self.assertEqual(len(dusurulen), 0, 'hepsi duzeltilebilmeli')
        self.assertEqual(len(duzeltilen), 3)
        self.assertEqual(len(temiz), 3, 'konu kaybi olmamali')
        for it in temiz:
            self.assertEqual(ascii_harf_mi(it['tag']), [])

    def test_temiz_etiketlere_dokunmaz(self):
        temiz, d, x = normalize_tag_latin(self._kutu(['fizik', 'kimya']))
        self.assertEqual([i['tag'] for i in temiz], ['fizik', 'kimya'])
        self.assertEqual((d, x), ([], []))

    def test_dusurulebilen_etiket_duser(self):
        """Harfler atilinca etiket bosalirsa dusurulur."""
        temiz, d, x = normalize_tag_latin(self._kutu(['đ', 'fizik']))
        self.assertEqual([i['tag'] for i in temiz], ['fizik'])
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0][0], 'đ')

    def test_dusurulen_etiket_nedenini_bildirir(self):
        _, _, x = normalize_tag_latin(self._kutu(['đ']))
        etiket, harfler = x[0]
        self.assertEqual(harfler, ['đ'], 'hangi harf yuzunden bilinmeli')

    def test_cakisan_etiketler_birlestirilir(self):
        """Duzeltilmis etiket zaten varsa tek intent olur."""
        kutu = self._kutu(['đakovo', 'akovo'])
        temiz, d, x = normalize_tag_latin(kutu)
        self.assertEqual(len(temiz), 1, 'ayni konu iki intent olmamali')
        self.assertEqual(temiz[0]['tag'], 'akovo')
        self.assertEqual(len(temiz[0]['patterns']), 2,
                         'iki tarafin kaliplari birlestirilmeli')
        self.assertEqual((d, x), ([('đakovo', 'akovo')], []))

    def test_gercek_intents_json_temizlenir(self):
        """Sinir testi: butun intents.json tarandiginda 0 kotu kalmali.

        Girdi olarak depodaki gercek intents.json kullanilir ama
        CALISTIRILMAZ - sadece okunur. Boylece ileride uretilen
        baska bir kotu etiket de yakalanir.
        """
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        with io.open(yol, encoding='utf-8') as f:
            data = json.load(f)
        temiz, d, x = normalize_tag_latin(data['intents'])
        kalan = [i['tag'] for i in temiz if ascii_harf_mi(i['tag'])]
        self.assertEqual(
            kalan, [],
            'intents.json\'da hala ASCII disi harfli etiket var: %s' % kalan[:5])


class TestMergeByTagKapi(unittest.TestCase):
    def test_kotu_etiket_birlestirmeye_girmez(self):
        """clean_intents yolu da kapidan gecmeli."""
        gir = [{'tag': 'đakovo', 'patterns': ['a'], 'responses': ['r']},
               {'tag': 'fizik', 'patterns': ['b'], 'responses': ['r']}]
        sonuc = merge_by_tag(gir)
        etiketler = [i['tag'] for i in sonuc]
        self.assertNotIn('đakovo', etiketler)
        self.assertIn('fizik', etiketler)

    def test_min_tag_chars_korunuyor(self):
        """Mevcut kisa etiket kapisi bozulmamis olmali."""
        gir = [{'tag': 'ab', 'patterns': ['a'], 'responses': ['r']}]
        self.assertEqual(merge_by_tag(gir), [])
        gir = [{'tag': 'a' * MIN_TAG_CHARS, 'patterns': ['a'],
                'responses': ['r']}]
        self.assertEqual(len(merge_by_tag(gir)), 1)


class TestMergeIntentsKendiniIyilestirir(unittest.TestCase):
    """AutoGrow'un yazma yolu kendi ciktisini temizlemeli.

    intents.json'a elle mudahale gerekmesin diye yazma noktasinda
    temizlik yapilir. Bu test, GELECEK bir AutoGrow kosusunun
    temiz intents.json birakacagini dogrular.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='merge_latin_')
        self.yol = os.path.join(self.tmp, 'intents.json')
        self.veri = {'intents': [
            {'tag': 'ǁkaras bolgesi', 'patterns': ['a'], 'responses': ['r1']},
            {'tag': 'đakovo', 'patterns': ['b'], 'responses': ['r2']},
            {'tag': 'liliʻuokalani', 'patterns': ['c'], 'responses': ['r3']},
            {'tag': 'fizik', 'patterns': ['d'], 'responses': ['r4']},
        ]}
        with io.open(self.yol, 'w', encoding='utf-8') as f:
            json.dump(self.veri, f, ensure_ascii=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_merge_sonrasi_kotu_etiket_kalmaz(self):
        cikti, added, updated = merge_intents(self.yol, [])
        kalan = [i['tag'] for i in cikti['intents'] if ascii_harf_mi(i['tag'])]
        self.assertEqual(kalan, [],
                         'merge etiketleri duzeltmedi: %s' % kalan)
        self.assertEqual(len(cikti['intents']), 4,
                         'duzeltilen konular KAYBOLMAMALI')

    def test_merge_yeni_intent_ekleyebilir(self):
        """Temizlik eklemeyi engellememeli."""
        yeni = [{'tag': 'deneme_konu', 'patterns': ['e'], 'responses': ['r']}]
        cikti, added, updated = merge_intents(self.yol, yeni)
        self.assertEqual(added, 1)
        self.assertTrue(any(i['tag'] == 'deneme_konu'
                            for i in cikti['intents']))

    def test_gercek_veri_dosyasina_dokunmaz(self):
        """Bu test intents.json'a YAZMAMALI.

        Kontrol git durumu ile degil, dosyanin ICERIK OZETI ile yapilir:
        calisma agaci zaten kirli olabilir (uretim scriptleri
        intents.json'u degistirmis olabilir) ve test, kendi yazmadigini
        kanitlamak icin o duruma baglanmamalidir.
        """
        import hashlib
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')

        def ozet():
            with io.open(yol, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()

        once = ozet()
        merge_intents(self.yol, [])
        self.assertEqual(ozet(), once,
                         'test gercek intents.json dosyasini degistirdi')


if __name__ == '__main__':
    unittest.main()

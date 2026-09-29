# -*- coding: utf-8 -*-
"""Internet ogrenme yolu artik HICBIR KAYIT EZMIYOR.

OLCULEN HATA (29.09): app.py ogrenme yolunda id = ascii_normalize(title)
diyordu ve `Corpus.append_many` ayni id'yi GUNCELLIYOR (corpus.py:1109).
Sonuc: "matematik nedir" sorusunda AutoGrow'un 3.240 baytlik taniminin
uzerine internetten gelen tek cumle yazildi. Ayni hata her benzer soruda
tekrarliyordu ve kayip kaliciydi.

Iki kural birlikte denetleniyor:
  1) baslik zaten korpussa HIC yazilmaz (ogrenme eksigi tamamlar, var
     olani degistirmez)
  2) yazilacaksa id metin hash'i tasir: ayni metin -> ayni id (buyume
     yok), farkli metin -> yeni kayit (ezme yok)

Bu testler gercek corpus.jsonl'a DOKUNMAZ: corpus nesnesi yalnizca
_slug/_slug_idx/chunks uzerinden sahte durumla kurulur, yazma ise
A.Corpus yerine gecen bir kaydediciye yonlendirilir.
"""
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import app as A
from corpus import Corpus

_YAZILAN = []


class _Kaydedici:
    """A.Corpus yerine gecer: append_many cagrilarini toplar, dosyaya yazmaz."""

    @staticmethod
    def append_many(chunks, path=None):
        _YAZILAN.extend(chunks)
        return len(chunks)


def _sahte_corpus(basliklar=(), idler=()):
    """Gercek Corpus._slug'i, sahte baslik dizini.

    __init__ agir (ChatBot kurar) oluyor, o yuzden nesne atlanip sadece
    _slug'in gercekten ihtiyac duydugu `tokenizer` baglaniyor. Boylece
    test, uretimde kullanilan slug fonksiyonunun kendisini denetliyor.
    """
    c = Corpus.__new__(Corpus)
    c.tokenizer = A.bot
    c._slug_idx = {c._slug(b): i for i, b in enumerate(basliklar)}
    c.chunks = [{'id': i} for i in idler]
    c.refreshed = 0
    c.refresh = lambda: setattr(c, 'refreshed', c.refreshed + 1)
    return c


class TestEzmezId(unittest.TestCase):
    """learned_chunk_id saf fonksiyonu: belirleyici, cakisma sayacli."""

    def test_ayni_metin_ayni_id(self):
        """Ayni olgu iki kez sorulunca id degismemeli (korpus buyumesin)."""
        a = A.learned_chunk_id('matematik', 'sayilar ve felsefe', lambda i: False)
        b = A.learned_chunk_id('matematik', 'sayilar ve felsefe', lambda i: False)
        self.assertEqual(a, b)

    def test_farkli_metin_farkli_id(self):
        """Ayni baslik, farkli metin -> farkli kayit. Ikisi de korunur."""
        a = A.learned_chunk_id('matematik', 'birinci metin', lambda i: False)
        b = A.learned_chunk_id('matematik', 'ikinci metin', lambda i: False)
        self.assertNotEqual(a, b)

    def test_id_basliga_degil_metne_bagli(self):
        """Regresyon: eski kod id'yi yalnizca baslikten turetiyordu, yani
        farkli metinler ayni id'yi alip birbirini eziyordu."""
        a = A.learned_chunk_id('fizik', 'kuvvet ve hareket', lambda i: False)
        b = A.learned_chunk_id('fizik', 'madde ve enerji', lambda i: False)
        self.assertNotEqual(a, b)

    def test_cakismada_sayac(self):
        # taban id once uretilir: cakisma sayaci O id'yi kullanmali
        taban = A.learned_chunk_id('x', 't', lambda i: False)
        var = {taban}
        self.assertEqual(A.learned_chunk_id('x', 't', lambda i: i in var),
                         taban + '_2')
        var.add(taban + '_2')
        self.assertEqual(A.learned_chunk_id('x', 't', lambda i: i in var),
                         taban + '_3')

    def test_bos_slug_hash_tek_basina(self):
        """Baslik normalizasyondan sonra bos kalirsa id yine de uretilmeli."""
        i = A.learned_chunk_id('', 'bir metin', lambda x: False)
        self.assertTrue(i.startswith('ogr_'))
        self.assertNotIn('__', i)


class TestOgrenmeEzmez(unittest.TestCase):
    """learn_from_internet: var olan kaydi yazmaz, yenisini ogr_ ile yazar."""

    def setUp(self):
        del _YAZILAN[:]
        self.eski_corpus, self.eski_sinif = A.corpus, A.Corpus
        A.Corpus = _Kaydedici

    def tearDown(self):
        A.corpus, A.Corpus = self.eski_corpus, self.eski_sinif
        del _YAZILAN[:]

    def test_varolan_baslik_ezilmez(self):
        """29.09 regresyonu: zengin AutoGrow kaydi internet cumlesiyle
        degistirilmemeli, dosyaya hicbir sey yazilmamali."""
        A.corpus = _sahte_corpus(basliklar=['Matematik'])
        sonuc = A.learn_from_internet(
            {'title': 'Matematik', 'answer': 'Matematik; sayilar, felsefe.'},
            'matematik nedir')
        self.assertIsNone(sonuc)
        self.assertEqual(_YAZILAN, [], 'mevcut kayit ezilmemeliydi')
        self.assertEqual(A.corpus.refreshed, 0, 'dosya degismedigi icin '
                                                 'index tazelemesi de olmamali')

    def test_eski_kodun_ezdigi_durum_artik_yazmiyor(self):
        """Birebir eski senaryo: id eskiden 'matematik' olurdu ve
        append_many o satiri guncellerdi. Artik 'ogr_matematik_...' olur."""
        A.corpus = _sahte_corpus(basliklar=['Matematik'], idler=['matematik'])
        A.learn_from_internet(
            {'title': 'Matematik', 'answer': 'kisa internet cumlesi'},
            'matematik nedir')
        self.assertEqual(_YAZILAN, [], 'zaten var olan baslik yazilmamali')

    def test_yeni_baslik_ogr_id_ile_yazilir(self):
        A.corpus = _sahte_corpus()
        sonuc = A.learn_from_internet(
            {'title': 'Yeni Konu', 'answer': 'gercekten yeni bir bilgi'},
            'yeni konu nedir')
        self.assertEqual(len(_YAZILAN), 1)
        kayit = _YAZILAN[0]
        self.assertEqual(kayit['id'], sonuc)
        self.assertTrue(kayit['id'].startswith('ogr_yeni_konu_'))
        self.assertNotEqual(kayit['id'], 'yeni_konu')
        self.assertEqual(kayit['source'], 'learned')
        self.assertEqual(kayit['text'], 'gercekten yeni bir bilgi')
        self.assertEqual(kayit['patterns'], 'yeni konu nedir')
        self.assertEqual(A.corpus.refreshed, 1, 'yazma sonrasi index tazelenmeli')

    def test_bos_baslik_yine_de_yazilir(self):
        """Baslik yoksa ogrenme engellenmemeli (bilgi yine de degerli)."""
        A.corpus = _sahte_corpus()
        sonuc = A.learn_from_internet({'title': '', 'answer': 'metin var'},
                                      'ne nedir')
        self.assertIsNotNone(sonuc)
        self.assertTrue(sonuc.startswith('ogr_'))

    def test_bos_cevap_yazilmaz(self):
        """Bos metin kaydedilmemeli: sorulunca "bilgim yok" yerine bos
        cevap doner ve ogrenme hicbir sey katmaz."""
        A.corpus = _sahte_corpus()
        self.assertIsNone(A.learn_from_internet(
            {'title': 'Bos Konu', 'answer': '   '}, 'bos konu nedir'))
        self.assertEqual(_YAZILAN, [])

    def test_bosluk_cevabi_temizlenir(self):
        A.corpus = _sahte_corpus()
        A.learn_from_internet({'title': 'K', 'answer': '  metin  '}, 'k nedir')
        self.assertEqual(_YAZILAN[0]['text'], 'metin')


if __name__ == '__main__':
    unittest.main()

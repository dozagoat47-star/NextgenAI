# -*- coding: utf-8 -*-
"""Sunucu yapilandirmasinin otomatik guncellenmesi testleri.

Iki sikayet giderildi:
  1) /status "Intents: 804" gosteriyordu ama intents.json 1.402 intent'e
     cikti. Sebep: load_model() bilgi kumesini model/bot_data.json'daki
     26.09 anlik goruntusunden aliyor; restart + elle mudahale gerekiyordu.
  2) "LLM yuklu degil" yaziyordu. Gercekte LLM saglamdi; sadece ilk
     soruya kadar tembel yuklenmedigi icin False gorunuyordu.
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

import app as A


class TestRefreshKnowledgeIntents(unittest.TestCase):
    """Bilgi intent'leri intents.json'dan tazelenmeli, siniflar korunmali."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='tazele_')
        # gercek veri dosyalarina dokunmadan: intents.json'in kopyasi
        self.intents = os.path.join(self.tmp, 'intents.json')
        shutil.copy(os.path.join(BASE, 'intents.json'), self.intents)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refresh_picks_up_newer_intents(self):
        """26.09 anlik goruntusunde 764 vardi, intents.json 1.362 yaziyor.

        Uretim yolu: load_model() -> refresh_knowledge_intents().
        """ 
        data = json.load(io.open(self.intents, encoding='utf-8'))
        bilgi = sum(1 for i in data['intents'] if len(i['patterns']) == 6)
        A.bot.load_model(os.path.join(BASE, 'model'))
        eski = len(A.bot.knowledge_intents)
        A.refresh_knowledge_intents(self.intents)
        self.assertEqual(
            len(A.bot.knowledge_intents), bilgi,
            'bilgi kumesi intents.json ile ayni olmali')
        self.assertEqual(len(A.bot.knowledge_intents) + len(A.bot.intent_tags),
                         len(data['intents']))
        self.assertGreater(len(A.bot.knowledge_intents), eski,
                           'bilgi kumesi 764\'te takili kalmamali')

    def test_refresh_is_a_noop_when_already_current(self):
        """Tazeleme iki kez calistirilirsa sayilar degismemeli."""
        A.bot.load_model(os.path.join(BASE, 'model'))
        A.refresh_knowledge_intents(self.intents)
        bir = (len(A.bot.intent_tags), len(A.bot.knowledge_intents))
        A.refresh_knowledge_intents(self.intents)
        iki = (len(A.bot.intent_tags), len(A.bot.knowledge_intents))
        self.assertEqual(bir, iki)

    def test_class_labels_are_preserved(self):
        """Sinif etiketleri egitilmis modelin cikti katmani: BOZULMAZ.

        load_intents() tum tag'leri intent_tags'a yazdigi icin, tazeleme
        sonrasi sinif sayisi 1.402'ye cikardi ve model agirligi ile
        uyumsuz hale gelecekti. refresh geri aliyor.
        """
        A.bot.load_model(os.path.join(BASE, 'model'))
        siniflar = set(A.bot.intent_tags)
        A.refresh_knowledge_intents(self.intents)
        self.assertEqual(set(A.bot.intent_tags), siniflar)
        self.assertEqual(len(A.bot.intent_tags), len(siniflar))

    def test_keyword_weights_rebuilt(self):
        """Anahtar kelime agirliklari tazelenmeli (retrieval bunlari kullanir)."""
        A.bot.load_model(os.path.join(BASE, 'model'))
        A.refresh_knowledge_intents(self.intents)
        self.assertTrue(A.bot.intent_kws, 'intent_kws bos oldu')
        for tag, kws in A.bot.intent_kws.items():
            self.assertIsInstance(kws, set)
        self.assertTrue(hasattr(A.bot, 'keyword_weights'))

    def test_missing_file_falls_back_silently(self):
        """intents.json yoksa cokmemeli, model anlik goruntusu kalmali."""
        A.bot.load_model(os.path.join(BASE, 'model'))
        once = len(A.bot.knowledge_intents)
        A.refresh_knowledge_intents(os.path.join(self.tmp, 'yok.json'))
        self.assertEqual(len(A.bot.knowledge_intents), once)


class TestClassifierVocabularyPreserved(unittest.TestCase):
    """Tazeleme siniflandiricinin SOZLUGUNU ezmemeli.

    Bu test bir calisma hatasi nedeniyle yazildi. Bilgi tazelemesi
    eklenince sunucu HER MESAJDA coktu:

        IndexError: index 4477 is out of bounds for axis 0 with size 3203

    Sebep: load_intents() (brain.py:981-984) sozlugu intents.json'dan
    yeniden kuruyor:

        self.vocabulary   = sorted(set(patterns'daki kelimeler))
        self.vocab_to_idx = {w: i for i, w in enumerate(vocabulary)}
        self.pad_idx      = len(vocabulary)

    intents.json buyudugu icin sozluk 3.202 -> 4.477 oldu, id'ler
    4.476'ya kadar cikti. Ama gomme matrisi (3.203, 128) 3.202 kelimeyle
    EGITILMIS ve degismiyor. Her tokenizasyonda uretilen id gomme
    sinirini astigi icin mesaj gondermek coktu.

    Duzeltme: refresh_knowledge_intents() sozluk / vocab_to_idx /
    pad_idx'i geri koyuyor. Bilgi kumesi buyuyor, siniflandirici
    mali OLDUGU GIBI kaliyor.
    """

    def setUp(self):
        A.bot.load_model(os.path.join(BASE, 'model'))
        self.vocab = list(A.bot.vocabulary)
        self.pad = A.bot.pad_idx
        self.gomme = self._gomme_satiri()

    def _gomme_satiri(self):
        """Gomme matrisinin ilk boyutu (token sayisi)."""
        for ad in ('embed', 'wte', 'embedding'):
            m = getattr(A.bot.model, ad, None)
            if m is not None and hasattr(m, 'shape'):
                return m.shape[0]
        return None

    def test_vocabulary_unchanged_by_refresh(self):
        A.refresh_knowledge_intents(os.path.join(BASE, 'intents.json'))
        self.assertEqual(
            A.bot.vocabulary, self.vocab,
            'sozluk degisti: egitilmis gommeyle uyumsuz token id uretilir')
        self.assertEqual(A.bot.pad_idx, self.pad)

    def test_token_ids_stay_inside_embedding(self):
        """Bu testin varlik sebebi: mesaj gondermek cokuyordu.

        Her kelimenin id'si gomme matrisinin icinde kalmali. Cokerse
        IndexError: index N is out of bounds for axis 0.
        """
        A.refresh_knowledge_intents(os.path.join(BASE, 'intents.json'))
        if self.gomme is None:
            self.skipTest('gomme matrisi bulunamadi')
        enbuyuk = max(A.bot.vocab_to_idx.values())
        self.assertLess(
            enbuyuk, self.gomme,
            'en buyuk token id %d, gomme %d satir: her mesajde cokar'
            % (enbuyuk, self.gomme))

    def test_pad_idx_inside_embedding(self):
        A.refresh_knowledge_intents(os.path.join(BASE, 'intents.json'))
        if self.gomme is None:
            self.skipTest('gomme matrisi bulunamadi')
        self.assertLess(A.bot.pad_idx, self.gomme,
                        'pad_idx gomme disinda')

    def test_classification_still_works_after_refresh(self):
        """Tazeleme sonrasi siniflandirma saglam kalmali."""
        A.refresh_knowledge_intents(os.path.join(BASE, 'intents.json'))
        for soru in ('merhaba', 'tesekkur ederim', 'bugun hava nasil',
                     'fizik nedir'):
            try:
                A.bot._classify(soru)
            except Exception as e:
                self.fail('"%s" siniflandirilamadi: %s: %s'
                          % (soru, type(e).__name__, e))

    def test_knowledge_set_still_grows(self):
        """Guvenlik agi: sozluk sabit kalirken bilgi buyumeye devam etmeli."""
        A.bot.load_model(os.path.join(BASE, 'model'))
        eski = len(A.bot.knowledge_intents)
        A.refresh_knowledge_intents(os.path.join(BASE, 'intents.json'))
        self.assertGreater(len(A.bot.knowledge_intents), eski)
        self.assertEqual(A.bot.vocabulary, self.vocab)


class TestStatusFields(unittest.TestCase):
    """/status hangi sayiyi ne anlama geliyor acikca bildirmeli."""

    def setUp(self):
        A.bot.load_model(os.path.join(BASE, 'model'))
        A.model_loaded = True

    def _status(self):
        return A.app.test_client().get('/status').get_json()

    def test_three_intent_counts_are_separate(self):
        """40 tek basina anlamsiz: toplam/sinif/bilgi ayri ayri."""
        d = self._status()
        for alan in ('intents_total', 'intent_classes', 'knowledge_intents'):
            self.assertIn(alan, d)
        self.assertEqual(
            d['intent_classes'] + d['knowledge_intents'], d['intents_total'],
            '40 + 764 = 804 olmali; tek sayi gostermek kafa karistiriyor')

    def test_reports_running_version(self):
        """Guncelestirme diskte ama sunucu eski kodda: ayirt edilebilsin."""
        d = self._status()
        for alan in ('app_commit', 'app_started', 'intents_updated',
                     'corpus_updated', 'llm_updated'):
            self.assertIn(alan, d)
        self.assertTrue(d['app_commit'])

    def test_legacy_field_kept(self):
        """Harici istemciler kirilmamali."""
        d = self._status()
        self.assertEqual(d['intent_count'], d['intent_classes'])


class TestFrontendUsesNewFields(unittest.TestCase):
    def test_index_html_does_not_read_legacy_field(self):
        """index.html data.intent_count okursa ekranda yine '40' cikar."""
        html = io.open(os.path.join(BASE, 'templates', 'index.html'),
                       encoding='utf-8').read()
        self.assertNotIn('data.intent_count', html)
        self.assertIn('data.intents_total', html)
        self.assertIn('data.knowledge_intents', html)


if __name__ == '__main__':
    unittest.main()

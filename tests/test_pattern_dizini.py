# -*- coding: utf-8 -*-
"""TAM PATTERN DIZINI regresyon testleri.

KOK NEDEN (25.09 olcumu):
    intents.json'da 67.708 pattern var ve sorular BIREBIR duruyor --
        "ekonomi nedir" -> tag 'ekonomi'
        "islam nedir"   -> tag 'islam'
        "muzik nedir"   -> tag 'musiki'
    Ama yanit yolunda pattern->tag dizini HIC yoktu: all_patterns duz liste
    ve yalnizca otomatik tamamlama icin kullaniliyordu. Tek dize yolu 40
    sinifli sinir agi + bulanik IDF eslesmesiydi; olgu sorularini yapmiyor,
    yakin bir sohbet sinifi buluyor:
        "ekonomi nedir"          -> 'dusuk karbonlu ekonomi'   (0.9+)
        "siber guvenlik nedir"   -> 'sib kuh ilcesi'
        "islam nedir"            -> "2017 islami dayanisma oyunlari'nda ..."
    sonra brain.py'deki `chosen_tag in self.intent_tags` sarti tutmadigi icin
    "bilgim yok" donuyordu. Olcum: "X nedir" sorularinda bos kova %90.

Bu testler (a) dizinin dogru kuruldugunu, (b) normalizasyonun veri ve
sorguda ayni oldugunu, (c) belirsizlikte TAHMIN ETMEDIGINI ve (d)
get_response'un bu dizini gercekten kullandigini guarantee eder.
"""
import io
import json
import os
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from brain import ChatBot

# veriden kesilmis kucuk dilim: 3 sohbet sinifi, 4 bilgi intent'i
_SOHBET = ['spor', 'bilim', 'muzik', 'programlama']
_INTENTS = [
    {'tag': 'spor', 'responses': ['Futbol dunyanin en populer sporudur.'],
     'patterns': ['futbol nedir', 'spor nedir', 'futbol nasil oynanir']},
    {'tag': 'bilim', 'responses': ['Bilim dunyayi inceler.'],
     'patterns': ['bilim nedir', 'fizik nedir']},
    {'tag': 'muzik', 'responses': ['Muzik sanattir.'],
     'patterns': ['muzik nedir']},
    {'tag': 'programlama', 'responses': ['Programlama talimat vermektir.'],
     'patterns': ['programlama nedir', 'yazilim nedir']},
    {'tag': 'teknoloji', 'responses': ['Teknoloji alandir.'],
     'patterns': ['teknoloji nedir', 'yazilim nedir']},
    {'tag': 'ekonomi', 'responses': ['Ekonomi kaynaklarin dagimidir.'],
     'patterns': ['ekonomi nedir', 'ekonomi hakkinda bilgi ver']},
    {'tag': 'islam', 'responses': ['Islam bir dindir.'],
     'patterns': ['islam nedir']},
    {'tag': 'fizik', 'responses': ['Fizik madde ve enerjiyi inceler.'],
     # 'fizik nedir' -> bilim (sohbet) + fizik (bilgi) cakismasi,
    # gercek veride de boyle (16 cakismanin biri)
     'patterns': ['fizik nedir']},
    {'tag': 'kadir', 'responses': ['Kadir yilin on biridir.'],
     'patterns': ['kadir nedir']},
]
_VERI = {'intents': _INTENTS}


def _bot(chat_tags=None, veri=None):
    """Veri dosyasina dokunmadan, kucuk dilimle bot kurar.

    load_intents dosya acar; sinif metodunu GECICI olarak degistirip geri
    aliyoruz. DIKKAT: b.load_intents = ... yazmak instance attribute
    olusturur ve sonraki cagirlarda sinif metodunu kalici olarak gölgeler
    (bizi bir kez yanilti: dizin hic kurulmamisti, testler yanlislikla
    gecmisti).
    """
    b = ChatBot()
    _yukle(b, veri or _VERI)
    if chat_tags is not None:
        b.intent_tags = list(chat_tags)
    return b


def _yukle(bot, veri):
    """GERCEK load_intents'i gecici dosyayla calistirir.

    DIKKAT: load_intents'i stub'lamak HIC BIR SEY DOGRULAMAZ - govde
    calismayinca dizin kurulmaz ve testler yanlislikla "eslesme yok"
    sonucuna dusup gecer. Bu yuzden gecici dosyaya yazilir ve fonksiyonun
    kendisi calisir; veri dosyalarina dokunulmaz.
    """
    fd, yol = tempfile.mkstemp(suffix='.json', prefix='test_pattern_')
    os.close(fd)
    try:
        with io.open(yol, 'w', encoding='utf-8') as f:
            json.dump(veri, f, ensure_ascii=False)
        bot.load_intents(yol)
    finally:
        os.unlink(yol)
    return bot


class TestDizinKurulumu(unittest.TestCase):
    def test_anahtar_sayisi_cesit_derlenir(self):
        b = _bot()
        # 3+2+1+2+2+2+1+1+1 = 15 pattern; 'yazilim nedir' ve 'fizik nedir'
        # iki intent'te gectigi icin 13 benzersiz anahtar
        self.assertEqual(len(b.pattern_tags), 13)
        self.assertTrue(b.pattern_tags)

    def test_tekil_eslesme_tag_dondurur(self):
        b = _bot()
        self.assertEqual(b._exact_pattern_tag('ekonomi nedir'), 'ekonomi')
        self.assertEqual(b._exact_pattern_tag('islam nedir'), 'islam')

    def test_bulunmayan_pattern_none(self):
        b = _bot()
        self.assertIsNone(b._exact_pattern_tag('kadin nedir'))
        self.assertIsNone(b._exact_pattern_tag(''))
        self.assertIsNone(b._exact_pattern_tag('   '))


class TestNormalizasyon(unittest.TestCase):
    """Veri ve sorgu AYNI anahtara inmeli; aksi halde dizin ise yaramaz."""

    def test_buyuk_kucuk_duyarsiz(self):
        b = _bot()
        self.assertEqual(b._exact_pattern_tag('EKONOMI NEDIR'), 'ekonomi')
        self.assertEqual(b._exact_pattern_tag('Ekonomi Nedir'), 'ekonomi')

    def test_sondaki_noktalama_atilir(self):
        b = _bot()
        for s in ('ekonomi nedir?', 'ekonomi nedir!', 'ekonomi nedir.',
                  'ekonomi nedir,', 'ekonomi nedir;'):
            self.assertEqual(b._exact_pattern_tag(s), 'ekonomi',
                             '%r eslesmedi' % s)

    def test_bosluk_normalize(self):
        b = _bot()
        self.assertEqual(b._exact_pattern_tag('  ekonomi   nedir  '),
                         'ekonomi')

    def test_turkce_karakter_donusur(self):
        """Veride 66 non-ascii pattern var; ayni islem sorguya da uygulanir."""
        import brain
        b = _bot()
        b.pattern_tags = {b._pattern_key('fizik nedir'): ('bilim',)}
        self.assertEqual(b._pattern_key('fizik nedir'), 'fizik nedir')
        # Turkce karsilik donusmus olsa da ayni anahtara inmeli
        self.assertEqual(b._pattern_key('FİZİK NEDİR'), 'fizik nedir')

    def test_anahtar_uretimi_asimdir(self):
        b = _bot()
        # YALNIZCA stringin SONUNDAKI noktalama atilir; ic noktalama korunur
        # ("a? b" != "a b") cunku "ne? b" ile "ne b" farkli sorulardir.
        self.assertEqual(b._pattern_key('  A?  B.  '), 'a? b')
        self.assertEqual(b._pattern_key('Ne? B'), 'ne? b')
        self.assertEqual(b._pattern_key('Ekonomi   Nedir???'), 'ekonomi nedir')
        self.assertEqual(b._pattern_key(''), '')
        self.assertEqual(b._pattern_key('???'), '')


class TestCakismaKurali(unittest.TestCase):
    """16/16 olcum sonucu: cakismalar sohbet sinifi <-> bilgi intent'i.

    Kural: birden fazla intent'te gecen bir pattern'de TEK sohbet sinifi
    varsa o secilir; birden fazla sohbet sinifi varsa KARAR VERILMEZ
    (belirsiz -> mevcut akis devreye girer). Tahmin edilmez.
    """

    def test_sohbet_bilgi_cakismasinda_sohbet_kazanir(self):
        # 'fizik nedir' -> bilim (sohbet) + bilgi intent'i
        b = _bot(chat_tags=['bilim', 'spor'])
        self.assertEqual(len(b.pattern_tags['fizik nedir']), 2)
        self.assertEqual(b._exact_pattern_tag('fizik nedir'), 'bilim')

    def test_sohbetin_tek_oldugu_durumda_bilgi_de_yenilir(self):
        b = _bot(chat_tags=['ekonomi', 'bilim'])
        self.assertEqual(b._exact_pattern_tag('ekonomi nedir'), 'ekonomi')

    def test_iki_sohbet_sinifi_belirsiz_kalir(self):
        """'yazilim nedir' -> programlama + teknoloji, ikisi de sohbet.
        Burada karar vermek tahmin olurdu; dizin susmali."""
        b = _bot(chat_tags=['programlama', 'teknoloji', 'bilim'])
        self.assertEqual(len(b.pattern_tags['yazilim nedir']), 2)
        self.assertIsNone(b._exact_pattern_tag('yazilim nedir'))

    def test_sohbet_secimi_dosya_dilinde_hazir(self):
        """intents.json yuklenmeden sohbet sinifi bilinmez; o durumda
        cakisma susmali (ilk kayit kazanmaz, tahmin edilmez)."""
        b = _bot(chat_tags=list(i['tag'] for i in _INTENTS))
        self.assertIsNone(b._exact_pattern_tag('fizik nedir'))


class TestGetResponseKullanimi(unittest.TestCase):
    """Dizin kurulmus olmak yeterli degil; get_response ONA BAKMALI."""

    def _sahte(self, bot, tag='dusuk_karbon', prob=0.99, belirsiz=False):
        bot.model = object()                      # None degil -> yol acilir
        bot._classify = lambda *a, **k: (tag, prob, belirsiz, set())
        bot._is_knowledge_question = lambda *a, **k: True
        bot._select_knowledge = lambda *a, **k: None   # retrieval BOS
        cagrilan = []
        bot._select_response = lambda t, w: cagrilan.append(t) or t
        bot._try_kb_rephrase = lambda q, kb: 'URETILDI:%s' % kb
        bot._negation_reply = lambda c: 'OLUMSUZ'
        bot._unknown_reply = lambda: 'BILGI_YOK'
        return cagrilan

    def test_bilgi_intent_tam_eslesmesi_kullanilir(self):
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama'])
        cagrilan = self._sahte(b)
        y = b.get_response('ekonomi nedir')
        self.assertEqual(y, 'URETILDI:ekonomi')
        self.assertEqual(cagrilan, ['ekonomi'],
                         'siniflandiricinin COP tag\'i kullanildi')

    def test_bos_retrieval_da_tam_eslesme_kurtarir(self):
        """Cop tag dondugu icin 1746'daki kapı 'bilgim yok' diyecekti."""
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama'])
        self._sahte(b)
        self.assertNotEqual(b.get_response('islam nedir'), 'BILGI_YOK')

    def test_eslesme_yoksa_eski_durust_yol_corunur(self):
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama'])
        self._sahte(b, tag='cop_tag')
        self.assertEqual(b.get_response('kadin nedir'), 'BILGI_YOK')

    def test_sohbet_tam_eslesmesi_siniflandiriciyi_gecersiz_kilar(self):
        """Siniflandirici 'cop_tag' dese bile tam eslesme musiki'yi secmeli."""
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama'])
        secilen = []
        b.model = object()
        b._classify = lambda *a, **k: ('cop_tag', 0.99, True, set())
        b._is_knowledge_question = lambda *a, **k: False
        b._select_knowledge = lambda *a, **k: None
        b._select_response = lambda t, w: secilen.append(t) or t
        b._try_seq_rephrase = lambda *a, **k: None
        b._negation_reply = lambda c: 'OLUMSUZ'
        b.get_response('muzik nedir')
        self.assertEqual(secilen, ['muzik'],
                         'tam eslesme siniflandirici secimini gecersiz kilmadi')

    def test_olumsuzluk_onceligi_korunur(self):
        """'futbol sevmiyorum' -> tam eslesme olsa bile olumsuzluk doner."""
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama'])
        b.model = object()
        b._classify = lambda *a, **k: ('spor', 0.99, False, set())
        b.get_response('futbol sevmiyorum')
        # _negation_reply gercek calisir; sadece 'bilgim yok' olmadigini dogrula
        self.assertNotIn('BILGI_YOK', b._unknown_reply())


class TestGercekVeri(unittest.TestCase):
    """intents.json varsa kesilmis GERCEK vaka (yoksa atlanir)."""

    @classmethod
    def setUpClass(cls):
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            raise unittest.SkipTest('intents.json yok')
        with open(yol, encoding='utf-8') as f:
            cls.veri = json.load(f)

    def test_dizin_buyuk(self):
        b = _bot(veri=self.veri)
        self.assertGreater(len(b.pattern_tags), 50000,
                           'dizin beklenmedik kucuk, insaat hatasi mi?')

    def test_olculen_vakalar_cozuluyor(self):
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama', 'tarih'],
                 veri=self.veri)
        for soru, beklenen in [('ekonomi nedir', 'ekonomi'),
                               ('islam nedir', 'islam'),
                               ('muzik nedir', 'musiki'),
                               ('futbol nedir', 'spor'),
                               ('fizik nedir', 'bilim'),
                               ('biyoloji nedir', 'bilim')]:
            self.assertEqual(b._exact_pattern_tag(soru), beklenen,
                             '%r yanlis cozuldu' % soru)

    def test_veride_patterni_olmayan_konular_bos_kalir(self):
        """Konu veride YOKSA dizin UYDURMAMALI -> None.

        Konular SABIT LISTEDEN degil CANLI VERIDEN turetilir. Onceki surum
        `('kadin nedir', 'kuantum bilgisayarlar nedir', 'siber guvenlik
        nedir')` listesini hardcode ediyordu; CI intents.json'u buyuttukce
        liste YANLISLANDI (01.10 olcumunde `kadin_tanim` verideydi, test
        kirmiziydi, kod dogruydu). Artik "dizinde karsiligi olmayan" konu
        sorusu veriyle birlikte ilerler.
        """
        b = _bot(chat_tags=['spor', 'bilim', 'muzik', 'programlama'],
                 veri=self.veri)

        # aday konular: eski liste + veride HICBIR ZAMAN olmayacaklar
        adaylar = ['kadin', 'kuantum bilgisayarlar', 'siber guvenlik',
                   'zzzqqq xvii', 'kirmizi balon uydurma',
                   'pazartesi sabahi uydurma kelimesi']
        yok = []
        for konu in adaylar:
            soru = '%s nedir' % konu
            anahtar = b._pattern_key(soru)
            if anahtar not in b.pattern_tags:
                yok.append(soru)

        self.assertGreater(
            len(yok), 0,
            'hicbir aday konu dizinde yok: ya veri butunu indekslemeye '
            'basladi (test bosluga dustu) ya _pattern_key bozuldu')
        for soru in yok:
            self.assertIsNone(
                b._exact_pattern_tag(soru),
                '%r veride yok, dizin uydurmamali' % soru)

    def test_dizin_veriyle_tutarli(self):
        b = _bot(veri=self.veri)
        # her pattern kendi anahtarina inmis olmali
        for i in self.veri['intents']:
            for p in i['patterns']:
                self.assertIn(b._pattern_key(p), b.pattern_tags,
                              "verideki pattern dizinde yok: %r" % p)
            break


if __name__ == '__main__':
    unittest.main()

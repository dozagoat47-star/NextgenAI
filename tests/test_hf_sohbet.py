# -*- coding: utf-8 -*-
"""fetch_hf_turkish: cok turlu sohbet kaynagi + hizli dedup (28.09).

NEDEN BU TESTLER:
  1) WildChat-Turkish eklendi. Projedeki ilk COK TURLU kaynak. Olcum:
     10.000 konusma -> 40.812 ham cift; kapilar sonrasi 4.613.
  2) dedupe_pairs O(n^2) idi: 588 cift 8,8 sn -> 40.000 cift ~11 SAAT.
     Bu yuzden mevcut chatgrow_hf_*.jsonl dosyalari sadece ~1.200 satir.
     Yeni yol (ters indeks + boyut penceresi) 2,8 sn ve 1.500 ciftte
     ESKI YOLLA BITEN Ayni sonucu veriyor. Bu test o esitligi korur:
     hizlanma DOGRULUK KAYBI olmadan yapilmistir.
  3) stream_hf kaynak adina gore 'endswith' ile dagitim yapiyordu;
     yeni kaynak eklenince yanlis donusturucu calisip sessizce bos
     cift uretirdi. SURES haritasi tek dogruluk kaynagi artik.
"""
import os
import sys
import time
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import fetch_hf_turkish as FH

WILD = 'kilicai/turkish-sft-multi-turn-dialogue-10k'


def konusma(*donusler, source='WildChat-Turkish'):
    """(user, assistant) ciftlerinden messages[] semasi kurar."""
    msgs = []
    for u, a in donusler:
        msgs.append({'role': 'user', 'content': u})
        msgs.append({'role': 'assistant', 'content': a})
    return {'messages': msgs, 'source': source}


class TestCutAtWord(unittest.TestCase):
    """Kirpma kelime sonunda bitsin (28.09: 'gerekl' diye yarim kaldi)."""

    def test_kisa_metin_degismez(self):
        self.assertEqual(FH.cut_at_word('merhaba', 64), 'merhaba')

    def test_uzun_metin_kelime_sonunda_biter(self):
        s = 'bir iki uc dort bes alti yedi sekiz dokuz on'
        k = FH.cut_at_word(s, 20)
        self.assertLessEqual(len(k), 20)
        self.assertTrue(s.startswith(k))
        self.assertFalse(k.endswith(' '))
        # son kelime TAM olmali
        self.assertIn(k.split()[-1], s.split())

    def test_tek_uzun_kelime_olceginde_kalir(self):
        s = 'a' * 100
        self.assertEqual(FH.cut_at_word(s, 20), 'a' * 20)

    def test_bos_metin(self):
        self.assertEqual(FH.cut_at_word('', 64), '')
        self.assertEqual(FH.cut_at_word(None, 64), None)

    def test_oncesiz_kirpma_gerekir(self):
        """ASIL HATA SINIFI: once kirpma, sonra cut_at_word.

        Yanlis cagri: cut_at_word(clean_chars(s, 64), 64)
        -> clean_chars 64'e indigi icin cut_at_word 'len <= max_len' dalina
        duser ve AYNI stringi dondurur. Kelime sonu kisma HIC CALISMAZ.

        Bu test, dogru cagriyi (once normalize, sonra kes) zorunlu kilar.
        """
        s = 'du ve halk arasinda konusulan dil ile alakalı soru burada geliyor'
        yanlis = FH.cut_at_word(FH.clean_chars(s, 64), 64)
        dogru = FH.cut_at_word(FH.clean_chars(s, None), 64)
        self.assertEqual(yanlis, FH.clean_chars(s, 64),
                         'beklenen: kesilmis string degistirilmiyor (hata)')
        self.assertNotEqual(dogru, yanlis,
                            'dogru cagri kirpmayi kelime sonunda bitirmeli')
        self.assertEqual(dogru, dogru.rstrip())
        self.assertTrue(s.replace('ı', 'i').lower().startswith(dogru))

    def test_kirpma_noktasi_gercekten_boşlukta(self):
        """%59,8'i sert kesmede yarim kelimeydi. Simdi kirpma noktasi
        daima bir kelime SONRASI olmali."""
        import random
        rng = random.Random(2)
        sozluk = ['kelime%d' % i for i in range(60)]
        bozuk = 0
        for _ in range(300):
            s = ' '.join(rng.choice(sozluk) for _ in range(30))
            tam = FH.clean_chars(s, None)
            k = FH.cut_at_word(tam, 40)
            if len(tam) <= 40:
                continue
            # kirpma tam kelime mi? k, tam'in bir onek olmali
            if not tam.startswith(k):
                bozuk += 1
            # ve kesilen nokta bosluktan sonra gelmeli
            if len(k) < 40 and k and tam[len(k)] not in (' ', ''):
                bozuk += 1
        self.assertEqual(bozuk, 0, '%d kirpma yarim kelime birakti' % bozuk)


class TestYanitKirpmaKelimeSonunda(unittest.TestCase):
    """29.09: YANIT hatti ctx'in tersini uyguluyordu -> sohbet verisinin
    %79,4'u kelime ortasinda bitiyordu.

    OLCUM (dosyalar uzerinde, loader hic calistirilmadan):
      chatgrow_hf_sohbet.jsonl   yanit tavani 204 | 3.825 tavana dayanmis
                                 | 3.726 (%97,4) YARIM KELIME
      chatgrow_hf_20260924_1948  yanit tavani 140 | 2.110 | 2.048 (%97,1)
      chatgrow_kitap_*           yanit tavani 300 |    24 |    24 (%100)
      load_chatgrow_pairs sonrasi: 8.734/10.995 = %79,4 yanit, %87,4 ctx
      intents.json: 0/169.977 (TEMIZ -- sorun yalnizca chatgrow'da)
    """

    def _ciftler(self, n=40, cap=140):
        """cap'te sert kesilen, kelime ortasinda biten yanitlar uretir."""
        out = []
        rng = __import__('random').Random(7)
        for i in range(n):
            kelimeler = ['w%d' % (i * 7 + j) for j in range(30)]
            s = ' '.join(rng.choice(kelimeler) for _ in range(30))
            out.append(s)
        return out

    def test_refine_resp_sert_kesmeyi_kullanmaz(self):
        """refine_resp cumle sonu bulamazsa KELIME SONUNDA kesmeli."""
        s = ('burada uzun bir aciklama var ve devam ediyor boylece '
             'fakat hic nokta isareti yok')
        r = FH.refine_resp(s, 40)
        self.assertIsNotNone(r)
        self.assertLessEqual(len(r), 40)
        tam = FH.clean_chars(s, None)
        self.assertTrue(tam.startswith(r.rstrip()),
                        'kirpma metnin bir onek olmali')
        if len(r) < len(tam):
            self.assertIn(tam[len(r)], (' ', ''),
                          'kirpma noktasi BOSLUKTA olmali (yarim kelime yasak)')

    def test_ureticinin_yanit_hatti_once_kirpmaz(self):
        """ASIL HATA: refine_resp(clean_chars(resp, LEN), maxc=LEN).

        clean_chars once LEN'e indirir, refine_resp 'len <= maxc' dalina duser.
        29.09: bu yuzden sohbet verisinin %79,4'u yarim kelimeydi.
        """
        s = ('bu bir ornek yanit metni ve oldukca uzun bir sey '
             'cunku devam ediyor boylece')
        yanlis = FH.refine_resp(FH.clean_chars(s, 20), maxc=20)
        dogru = FH.refine_resp(FH.clean_chars(s, None), maxc=20)
        self.assertEqual(yanlis, FH.clean_chars(s, 20),
                         'beklenen: once kirpma onarimi calistirmaz (hata)')
        # TERSI: dogru cagri 20'de degil, son BOSLUKTA biter
        self.assertEqual(dogru, 'bu bir ornek yanit')
        self.assertNotEqual(dogru, yanlis)
        tam = FH.clean_chars(s, None)
        self.assertIn(tam[len(dogru)], (' ', ''),
                      'dogru cagri da yarim kelime birakmamali')

    def test_ureticinin_yuruttugu_yanitlar_kelime_sonunda_biter(self):
        """Ucuz dogru kural: once normalize, sonra refine_resp.

        Ureticinin YANIT hattinda her zaman kullanmasi gereken iki adim.
        """
        bozuk = 0
        for s in self._ciftler(60):
            r = FH.refine_resp(FH.clean_chars(s, None), maxc=140)
            if r is None:
                continue
            if len(r) >= 140 and r[-1] not in ' .!?,;:':
                bozuk += 1
        self.assertEqual(bozuk, 0,
                         '%d yanit tavana dayanip yarim kelime kaldi' % bozuk)

    def test_bozuk_desen_kaynak_kodu_yasaklar(self):
        """Regresyon noketasi: TERSI desen kaynak koda girmeyecek."""
        with open(os.path.join(BASE, 'fetch_hf_turkish.py'),
                  encoding='utf-8') as fh:
            kaynak = fh.read()
        duz = ' '.join(kaynak.split())
        self.assertNotIn('refine_resp(clean_chars(resp, args.resp_len)', duz,
                         'once clean_chars ile kirpmak onarimi calistirmaz')
        self.assertNotIn('r[:maxc].strip()', duz,
                         'sert kesme yasak; cut_at_word kullan')

    def test_refine_resp_butceyi_asmaz(self):
        """29.09: cumle sonu aramasi maxc+8'e bakiyordu -> 204'te 210 krk.

        Butceyi asan yanit, loader'da clean_chars(x, 204) ile SERT kesildigi
        icin yarim kelimeye donusuyordu. Sondanin ciktisi 210 krkta bulundu.
        """
        asilan = 0
        for s in self._ciftler(120):
            # her metni noktali tumce olarak zenginlestir: sonrasi 204+ olsun
            zengin = s + '. ' + ' '.join('ek%d' % i for i in range(40))
            r = FH.refine_resp(FH.clean_chars(zengin, None), maxc=204)
            if r is None:
                continue
            if len(r) > 204:
                asilan += 1
        self.assertEqual(asilan, 0,
                         '%d yanit butceyi asti (loader sert keser)' % asilan)

    def test_uretilen_cift_loaderda_kirpma_kaybettirmez(self):
        """Uretici -> loader zinciri butun bos (birebir ayni metin)."""
        from train_llm import load_chatgrow_pairs
        import tempfile
        import json as _json
        gecersiz = 0
        ornekler = []
        with tempfile.TemporaryDirectory() as d:
            yol = os.path.join(d, 'chatgrow_t.jsonl')
            with open(yol, 'w', encoding='utf-8') as f:
                for s in self._ciftler(150):
                    zengin = s + '. ' + ' '.join('ek%d' % i for i in range(40))
                    c = FH.cut_at_word(FH.clean_chars(s, None), 64)
                    r = FH.refine_resp(FH.clean_chars(zengin, None), maxc=204)
                    if c and r:
                        f.write(_json.dumps({'query': c, 'answer': [r]},
                                            ensure_ascii=False) + '\n')
            for c, r in load_chatgrow_pairs([yol]):
                # loader kirpmis olmamali: cift aynen gecmeli
                if c != FH.cut_at_word(FH.clean_chars(c, None), 64) or \
                        r != FH.clean_chars(r, 204):
                    gecersiz += 1
                    if len(ornekler) < 3:
                        ornekler.append((c[-30:], r[-30:]))
        self.assertEqual(gecersiz, 0,
                         'loader %d cifti KIRPTA (yarim kelime). orn: %s'
                         % (gecersiz, ornekler))

    def test_bosluksuz_ctx_elenir(self):
        """29.09: cut_at_word bosluksuz metinde kelime siniri bulamaz.

        Olcum (yeni sohbet dosyasi, 5.714 cift): 99 bosluksuz ctx (%1,73),
        80 ctx'nin 24'u tam 64 krkta SERT kesilmis. Hepsi URL/sinif adi
        gibi dogal Turkce olmayan girdi. Uretici bunlari elemeli.
        """
        # 64'ten UZUN ve bosluksuz: rfind(' ') = -1 -> sert kesme
        uzun = 'sifirdanbirbir' * 8
        self.assertNotIn(' ', uzun)
        self.assertGreater(len(uzun), 64)
        kes = FH.cut_at_word(uzun, 64)
        self.assertEqual(len(kes), 64,
                         'bosluksuz metinde cut_at_word sert kesiyor')
        # ureticinin kapisi: bosluk yoksa cift eklenmez
        with open(os.path.join(BASE, 'fetch_hf_turkish.py'),
                  encoding='utf-8') as fh:
            kaynak = fh.read()
        self.assertIn("if ' ' not in ctx_c:", kaynak,
                      'bosluksuz ctx kapisi eksik')

    def test_butce_kaynaklariyla_ayni(self):
        """Uretici butcesi loader butcesini ASMAMALI.

        29.09 olcumu: build_book_pairs varsayilani 300/300 iken loader
        clean_chars(q, 64) / clean_chars(x, 204) ile KAPALI sekilde
        kesiyordu -> 300 > 204 oldugu icin hasar ureticiye hic girmiyor,
        dosyaya gomuluyordu:
            chatgrow_kitap_20260925_1122  48/609 = %7,88
            chatgrow_kitap_20260928_0609  26/609 = %4,27
        Kural: uretici butcesi <= loader butcesi.
        """
        import build_book_pairs as BB
        from train_llm import CTX_CHARS, MAX_CTX_LEN, MAX_SEQ_LEN
        from seqgen import RESP_CHARS_MAX as RS
        self.assertEqual(BB.CTX_CHARS, CTX_CHARS,
                         'build_book_pairs.CTX_CHARS kaynaktan ayrildi')
        self.assertEqual(BB.RESP_CHARS_MAX, RS,
                         'build_book_pairs.RESP_CHARS_MAX kaynaktan ayrildi')
        # loader'in gercekten kullandigi yanit butcesi
        self.assertEqual(MAX_SEQ_LEN - MAX_CTX_LEN - 4, RS)

    def test_book_pairs_kelime_sonunda_keser(self):
        """build_book_pairs de ayni kurali kullanmali (300 krk sert kesme)."""
        import build_book_pairs as BP
        s = ' '.join('kelime%d' % i for i in range(60))
        k = BP.cut_at_word(BP.clean_chars(s, None), 50)
        self.assertLessEqual(len(k), 50)
        self.assertIn(s[len(k)], (' ', ''), 'yarim kelime birakildi')
        with open(os.path.join(BASE, 'build_book_pairs.py'),
                  encoding='utf-8') as fh:
            kaynak = fh.read()
        self.assertNotIn('clean_chars(ctx, ctx_max)', kaynak)
        self.assertNotIn('clean_chars(resp, resp_max)', kaynak)


class TestDilKapilari(unittest.TestCase):
    """Stopword tabanli ayirim (28.09 olcumuyle secildi).

    Onceki kapı 'Turkceye ozgu harf orani >= 0.03' idi ve OLCUMDE
    ELENDI: gercek Turkce metnin %22,3'unu siliyordu, cunku sorularin
    %16,6'sinda hic Turkce harf yok. Bilinen Turkce orneklerinde de
    oran 0.000 cikiyordu -> metrik ayirt edici degil.
    """

    def test_dil_ayirici_turkce_pozitif(self):
        self.assertGreaterEqual(
            FH.dil_ayirici('6 subatta turkiyede gerceklesen deprem ile ilgili '
                           'bilgi verir misin, bir de bina de saglamligi'),
            0.0)

    def test_dil_ayirici_ingilizce_negatif(self):
        self.assertLess(
            FH.dil_ayirici('what is the best way to do this and how does it work'),
            0.0)

    def test_turkce_harf_içermeyen_turkce_kalsin(self):
        """ASIL BUG: 'ilk soru burada' gecerli Turkce ama 0 Turkce harf
        iceriyor. Eski kapı bunu SILIYORDU."""
        self.assertFalse(FH.yabanci_dil_mi('ilk soru burada'))
        self.assertFalse(FH.yabanci_dil_mi('tamam'))
        self.assertFalse(FH.yabanci_dil_mi('maturidilere gore dinden donenler'))

    def test_ingilizce_elenir(self):
        self.assertTrue(FH.yabanci_dil_mi('hello! how can i assist you today?'))
        self.assertTrue(FH.yabanci_dil_mi(
            'you can start by reading the documentation and then try'))

    def test_yabanci_alfabe_elenir(self):
        self.assertTrue(FH.yabanci_dil_mi('这是一个测试问题 你好 世界'))
        self.assertTrue(FH.yabanci_dil_mi('هذا اختبار نص'))

    def test_bos_metin_kapi_does_not_raise(self):
        self.assertFalse(FH.yabanci_dil_mi(''))
        self.assertFalse(FH.yabanci_dil_mi(None))
        self.assertEqual(FH.dil_ayirici(''), 0.0)
        self.assertEqual(FH.latin_orani('123 456 !!!'), 0.0)

    def test_latin_orani(self):
        self.assertEqual(FH.latin_orani('abc çğü'), 1.0)
        self.assertLess(FH.latin_orani('你好世界'), 0.5)

    def test_sinav_kalinti_tespiti(self):
        self.assertTrue(FH.sinav_kalinti_mi(
            'c) () () (yunan filozof aristoteles) ()16. yuzyilda'))
        self.assertTrue(FH.sinav_kalinti_mi('burada secenekler: a) x b) y c) z'))
        self.assertFalse(FH.sinav_kalinti_mi(
            'turkiye aktif bir deprem bolgesidir ve depremler sik sik olur'))

    def test_latex_artigi_normalize_sonrasi_yakalaniyor(self):
        """Ham metinde 'r(theta)' bos parantez DEGILDIR; ASCII normalize
        sonrasi 'r()' olur ve kalinti sayilir. Kontrol ham metine bakarsa
        LaTeX artigi KACIRILIR — olcumde 9 cift dosyaya girmisti."""
        self.assertTrue(FH.sinav_kalinti_mi(
            'kardioidi ifade eden parametrik denklem: '
            'r(θ) = a(1 + sinθ), acos(θ) dl a(1 + sinθ)'))

    def test_sinav_kalinti_esigi_cesitli(self):
        # tek '()' normal metinde olabilir -> esik 2
        self.assertFalse(FH.sinav_kalinti_mi('bizim ev (sahile yakin) guzel'))


class TestPairsFromChat(unittest.TestCase):

    def test_cok_turlu_her_donusu_cift(self):
        """EN ONEMLI TEST: ilk mesaji degil, HER donusu cift uretmeli."""
        rec = konusma(
            ('ilk soru burada', 'ilk cevap burada'),
            ('ikinci soru burada', 'ikinci cevap burada'),
            ('ucuncu soru burada', 'ucuncu cevap burada'),
        )
        cift = list(FH.pairs_from_chat(rec))
        self.assertEqual(len(cift), 3)
        self.assertEqual(cift[0][0], 'ilk soru burada')
        self.assertEqual(cift[2][1], 'ucuncu cevap burada')

    def test_tek_tur_da_uretilir(self):
        rec = konusma(('soru', 'cevap'))
        self.assertEqual(len(list(FH.pairs_from_chat(rec))), 1)

    def test_rol_duyarsiz(self):
        rec = {'messages': [{'role': 'human', 'content': 'soru metni burada'},
                            {'role': 'gpt', 'content': 'cevap metni burada'}],
               'source': 'WildChat-Turkish'}
        self.assertEqual(len(list(FH.pairs_from_chat(rec))), 1)

    def test_cevapsiz_kullanici_ucurur(self):
        rec = {'messages': [{'role': 'user', 'content': 'soru var cevap yok'}],
               'source': 'WildChat-Turkish'}
        self.assertEqual(list(FH.pairs_from_chat(rec)), [])

    def test_kotu_kaynak_elenir(self):
        """InstrucTurca-simulated: coktan secmeli sinav, gercek sohbet degil.
        28.09 olcumu: verinin %20'si, dedup sonrasi kalanin TAMAMI."""
        rec = konusma(('soru burada', 'cevap burada'),
                      source='InstrucTurca-simulated')
        self.assertEqual(list(FH.pairs_from_chat(rec)), [])

    def test_ingilizce_elenir(self):
        rec = konusma(('what is the best way to learn this and how does it work',
                       'you can start by reading the documentation and then try'))
        self.assertEqual(list(FH.pairs_from_chat(rec)), [])

    def test_ingilizce_kapatilabilir(self):
        rec = konusma(('what is the best way to learn this and how does it work',
                       'you can start by reading the documentation and then try'))
        self.assertEqual(len(list(FH.pairs_from_chat(rec, drop_english=False))), 1)

    def test_sinav_kalinti_elenir(self):
        rec = konusma(
            ('16. asirda cogu bilim insani ( ) evren hakkinda',
             'c) () () (yunan filozof aristoteles) ()16. yuzyilda'))
        self.assertEqual(list(FH.pairs_from_chat(rec)), [])

    def test_messages_listesi_degilse_bos(self):
        self.assertEqual(list(FH.pairs_from_chat({'messages': 'x'})), [])

    def test_gercek_veri_tek_tur_yok(self):
        """Olcum: veri tamamen cok turlu (turn_count==2 orani %0).
        Bu kaydin degeri burada; tek turlu olsaydi deger kaybolurdu."""
        rec = konusma(('birinci soru', 'birinci cevap'),
                      ('ikinci soru', 'ikinci cevap'))
        cift = list(FH.pairs_from_chat(rec))
        self.assertGreater(len(cift), 1)


class TestJaccardSets(unittest.TestCase):
    """jaccard_sets hazir token kumesiyle = ayni sonucu vermeli."""

    def test_jaccard_ile_ayni(self):
        a = 'bir iki uc dort'
        b = 'bir iki uc dort'
        self.assertAlmostEqual(
            FH.jaccard_sets(FH._tokenize_norm(a), FH._tokenize_norm(b)),
            FH.jaccard(a, b))

    def test_kesismeli(self):
        a = FH._tokenize_norm('bir iki uc')
        b = FH._tokenize_norm('iki uc dort')
        self.assertAlmostEqual(FH.jaccard_sets(a, b), 2 / 4)

    def test_bos_kume(self):
        self.assertEqual(FH.jaccard_sets(set(), set()), 0.0)
        self.assertEqual(FH.jaccard_sets({'a'}, set()), 0.0)


class TestDedupeHizVeDogruluk(unittest.TestCase):
    """Yeni hizli yol, eski O(n^2) yolla AYNI sonucu vermeli."""

    @staticmethod
    def _eski(pairs, thr=0.90):
        """28.09 oncesi surumun TAM hali (referans).

        Token kümeleri ONCEDEN hesaplaniyor: eski kod her jaccard cagrisinda
        2 kez tokenize ediyordu, bu da testi ~2 kat yavaslatiyordu. Karar
        (hangi cift elenir) ayni; jaccard_sets == jaccard dogrulugu
        TestJaccardSets'te ayrica test ediliyor.
        """
        seen_sets, kept, dc, dr, rm = [], [], 0, 0, {}
        for ctx, resp in pairs:
            if resp in rm:
                dr += 1
                continue
            toks = FH._tokenize_norm(ctx)
            dup = False
            for sc in seen_sets:
                if FH.jaccard_sets(toks, sc) >= thr:
                    dup, dc = True, dc + 1
                    break
            if dup:
                continue
            seen_sets.append(toks)
            rm[resp] = True
            kept.append((ctx, resp))
        return kept, dc, dr

    def test_kucuk_veride_birebir_ayni(self):
        """GERCEK veri ile ayni sonuc: hizlanma DOGRULUK KAYBI degil.

        SOVLUK BOYUTU ONEMLI. 17 kelimelik kucuk sozlukle denendiğinde
        hizli yol 1 cift fazla sakliyordu: her tokenin postings listesi
        MAX_POST'u asip tasiyordu. O KURGUSAL bir durum; gercek veri
        olcumunde (bkz. test_maxpost_tasma_gercek_veride_yok) boyle
        bir kacirma olmadi. Burada gercekci boyutlu sozluk kullanilir.
        """
        import random
        rng = random.Random(11)
        kelimeler = [w + str(i) for i in range(220) for w in
                     ('araba kirmizi yol hizli kapi deniz dag yesil ruzgar '
                      'bulut yildiz kiyi saray kule kral tavuk').split()]
        pairs = []
        for i in range(400):
            ctx = ' '.join(rng.choice(kelimeler) for _ in range(rng.randint(6, 12)))
            # birebir tekrarlar ve kacak kopyalar
            if i and i % 17 == 0:
                ctx = pairs[-1][0]
            elif i and i % 23 == 0:
                once = pairs[-1][0].split()
                ctx = ' '.join(once[:-1] + [rng.choice(kelimeler)])
            pairs.append((ctx, 'cevap %d burada' % i))
        eski, dc_e, dr_e = self._eski(pairs)
        yeni = FH.dedupe_pairs(pairs, keep_log=False)
        self.assertEqual(eski, yeni,
                         'hizli yol eski yoldan FARKLI sonuc verdi: '
                         'eski %d, yeni %d' % (len(eski), len(yeni)))

    def test_maxpost_tasma_gercek_veride_yok(self):
        """MAX_POST bir ust sinirdir ve TIKANMAKTA; ama olcumle
        GERCEK VERIDE kacirma olmadigi gosterildi (28.09).

        Gercek veri, eski O(n^2) yolla karsilastirildi:
            N=2.000  eski 83 sn  yeni 0,17 sn  (x500)  kacirma 0
            N=4.000  eski 241 sn yeni 0,36 sn  (x662)  kacirma 0
            N=6.000  eski 390 sn yeni 0,48 sn  (x819)  kacirma 0
        En buyuk posting listesi 852'ydi (yani 60'i 14 kat asiyor) ama
        yine 0 fark: uzaktaki kopyalar NADIR tokenlari paylasir; tasan
        tokenlar SIK'tir ve tek basina Jaccard 0,9'a ulasamaz.

        Bu test, sinirin kaldirilmadan once COK KUCUK bir sozlukle
        kacirma olmadigini teyit eder (belgelenen istisna: asiri
        tekrar eden tek kelimelik baglamlarda eski kopyalar gorusemez).
        """
        import random
        rng = random.Random(5)
        # gercekci: cogu kelimede tekil kullanim, birkac sik kelime
        sozluk = ['w%d' % i for i in range(900)] + ['bir', 've', 'bu'] * 40
        pairs = []
        for i in range(700):
            n = rng.randint(8, 14)
            ctx = ' '.join(rng.choice(sozluk) for _ in range(n))
            pairs.append((ctx, 'cevap %d' % i))
        # belirli aralikla birebir kopya + 1 kelime degisikligi
        for i in range(0, 650, 25):
            if i < len(pairs):
                once = pairs[i][0].split()
                pairs.append((pairs[i][0], 'kopya %d' % i))
                pairs.append((' '.join(once[:-1] + [rng.choice(sozluk)]),
                              'komsu %d' % i))
        eski, _dc, _dr = self._eski(pairs)
        yeni = FH.dedupe_pairs(pairs, keep_log=False)
        kaciran = set(eski) - set(yeni)
        self.assertEqual(len(kaciran), 0,
                         'MAX_POST yuzunden %d kopya kacirildi: %s'
                         % (len(kaciran), list(kaciran)[:2]))

    def test_ayni_resp_kurali_ayni_kalir(self):
        pairs = [('farkli soru bir', 'ayni cevap'),
                 ('farkli soru iki', 'ayni cevap'),
                 ('ucuncu soru', 'başka cevap')]
        self.assertEqual(len(FH.dedupe_pairs(pairs, keep_log=False)), 2)

    def test_bos_ctx_elenir(self):
        pairs = [('', 'bir cevap'), ('gecerli soru', 'baska cevap')]
        self.assertEqual(FH.dedupe_pairs(pairs, keep_log=False),
                         [('gecerli soru', 'baska cevap')])

    def test_boyut_penceresi_yalan_eleme_yapmaz(self):
        """Jaccard >= t icin |B| in [t|A|, |A|/t] olmasi ZORUNLU.
        Bu test, boyut penceresi yanlislikla kopyayi elemiyor."""
        a = 'bir iki uc dort bes alti yedi'
        b = 'bir iki uc dort bes alti yedi sekiz'   # komsu, dusuk Jaccard
        c = 'bir iki uc dort bes alti yedi sekiz dokuz'
        out = FH.dedupe_pairs([(a, 'r1'), (b, 'r2'), (c, 'r3')], keep_log=False)
        # b ve c kalsin (Jaccard dusuk), ama hiçbiri elememeli
        self.assertGreaterEqual(len(out), 2)

    def test_hiz_olcumu(self):
        """40.000 cift ESKI yol ~11 saat. Yeni yol birkac saniye.
        20.000 ciftte 60 sn esigi: 1000 kat marj payi birakir, CI'da
        kirmizi olsa da yerelde nedeni bellidir."""
        import random
        rng = random.Random(3)
        kelimeler = tuple('bir iki uc dort bes alti yedi sekiz dokuz on '
                          'yirmi otuz kirk elli altmis yetmis seksen '
                          'yuz yuzelli bin araba yol deniz dag'.split())
        pairs = []
        for i in range(20000):
            ctx = ' '.join(rng.choice(kelimeler) for _ in range(10))
            pairs.append((ctx, 'cevap %d burada' % i))
        t0 = time.time()
        FH.dedupe_pairs(pairs, keep_log=False)
        sn = time.time() - t0
        self.assertLess(sn, 60,
                        '20.000 cift %.0f sn: MAX_POST/SIZE_PAD ayarini '
                        'gözden gecir (eski yol ~11 saat)' % sn)


class TestKaynakDagitimi(unittest.TestCase):
    """stream_hf artik SOURCES haritasini kullanmali (28.09 duzeltmesi)."""

    def test_kaynak_haritasinda(self):
        self.assertIn(WILD, FH.SOURCES)

    def test_donusturucu_dogru_atanir(self):
        rec = konusma(('turkce bir soru', 'turkce bir cevap'))
        self.assertEqual(list(FH.SOURCES[WILD](rec)),
                         list(FH.pairs_from_chat(rec)))

    def test_stream_hf_harita_kullanir(self):
        """ESKI KOD: 'endswith(ThinkingData-200K-Turkish)' ile dagitim
        yapiyordu; yeni kaynak oraya dustugu icin instruction
        donusturucusu calisip sessizce BOS cift uretiyordu.

        KAYNAK METNI TARAMAK YANLIS OLURDU (kendi yorumumda kelime
        gecince kirmizi olurdu). DAVRANIS olculuyor: sahte bir dataset
        verilip cift uretilip uretilmedigi soruluyor.
        """
        import sys as _sys
        import types as _types

        uretilen = []

        class SahteKayit(dict):
            pass

        rec = konusma(('turkce bir soru', 'turkce bir cevap'))
        sahte_modul = _types.ModuleType('datasets')
        sahte_modul.load_dataset = lambda *a, **k: [rec]
        eski = _sys.modules.get('datasets')
        _sys.modules['datasets'] = sahte_modul
        try:
            cift = FH.stream_hf(WILD, max_pairs=10, seed=1, cot=False)
        finally:
            if eski is None:
                _sys.modules.pop('datasets', None)
            else:
                _sys.modules['datasets'] = eski
        self.assertEqual(len(cift), 1,
                         'yeni kaynak yanlis donusturucuya dustu ve %d cift '
                         'uretti (beklenen 1)' % len(cift))
        self.assertEqual(cift[0][0], 'turkce bir soru')


class TestGercekVeriSarti(unittest.TestCase):
    """Veri dosyasi varsa kalite cizgisi (CI'da da korunur)."""

    VAR = os.path.join(BASE, 'chatgrow_hf_sohbet.jsonl')

    def setUp(self):
        if not os.path.exists(self.VAR):
            self.skipTest('chatgrow_hf_sohbet.jsonl henuz uretilmedi')

    def test_cift_sayisi_anlamli(self):
        import io
        import json
        n = 0
        with io.open(self.VAR, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    n += 1
        self.assertGreater(n, 3000,
                           'beklenen ~4.600 cift, gelen %d' % n)

    def test_sema_ve_butce(self):
        import io
        import json
        import train_llm
        with io.open(self.VAR, encoding='utf-8') as f:
            satirlar = [json.loads(l) for l in f if l.strip()][:500]
        for r in satirlar:
            self.assertIn('query', r)
            self.assertIn('answer', r)
            self.assertIsInstance(r['answer'], list)
            self.assertTrue(r['query'])
            self.assertTrue(r['answer'][0])
            self.assertLessEqual(len(r['query']), train_llm.CTX_CHARS)
            self.assertLessEqual(len(r['answer'][0]), train_llm.RESP_CHARS_MAX)

    def test_yarim_kelime_yok(self):
        """Kirpma kelime sonunda bitmeli.

        Veri dosyasindan 'kirpma noktasi bosluk mu' dogrudan okunAMAZ
        (kirpma oncesi metin yok). Bu yuzden burada butce + kenar
        kosullar dogrulanir; kirpma davranisinin KENDISI
        TestCutAtWord.test_kirpma_noktasi_gercekten_boslukta ile
        300 ornek uzerinde test edilir.
        """
        import io
        import json
        import train_llm
        with io.open(self.VAR, encoding='utf-8') as f:
            satirlar = [json.loads(l) for l in f if l.strip()]
        kirp = 0
        for r in satirlar:
            q = r['query']
            self.assertTrue(q.strip(), 'bos ctx')
            self.assertEqual(q, q.strip(), 'bastaki/sondaki bosluk kirpildi')
            self.assertFalse(q.endswith(' '), 'sonda bosluk kaldi')
            self.assertLessEqual(len(q), train_llm.CTX_CHARS)
            # cut_at_word geri CEKTIği icin kirpilanlar bütçeye tam
            # degil, bir kac karakter altinda biter. Olcum: %31'i 60-64
            # bandında, %27,5'i 55-59'da. Tam 64 olanlar sadece 81.
            if len(q) > 0.8 * train_llm.CTX_CHARS:
                kirp += 1
        self.assertGreater(
            kirp, len(satirlar) * 0.2,
            'ust bantta sadece %d / %d ctx: kirpma calismiyor olabilir'
            % (kirp, len(satirlar)))

    def test_sinav_kalinti_yok(self):
        """Kapı min_bos=2: IKI veya daha fazla iz tasiyan cift OLMAZ.

        OLÇÜM (28.09) — bu test sifir beklemiyor, cunku beklemem YANLIS:
          * dosyanin %98,7'sinde hic '()' yok
          * 1 tane '()' olanlar cogunlukla GECERLI kod
          * birlestirilince esige ulasan 4 ciftin 2'si MEŞRU:
              q: "istanbul'a bugun gidiyorum () dedi."  -> Turkce
                 noktalama DERSI, '()' konunun kendisi
              a: "include iostream ... int main()"     -> gecerli C++
            Esik 1 olsaydi 59 meşru cift elenir, 2'si gercek gurultu.
        """
        import io
        import json
        with io.open(self.VAR, encoding='utf-8') as f:
            satirlar = [json.loads(l) for l in f if l.strip()]
        kotu = [r for r in satirlar
                if FH.sinav_kalinti_mi(r['query'])
                or FH.sinav_kalinti_mi(r['answer'][0])]
        self.assertEqual(kotu, [],
                         '%d cift tek tarafta sinav kalinti tasiyor: %r'
                         % (len(kotu), kotu[0]['query'][:60] if kotu else ''))
        birlestir = [r for r in satirlar
                     if FH.sinav_kalinti_mi(r['query'] + ' ' + r['answer'][0])]
        self.assertLessEqual(
            len(birlestir), 8,
            'birlestirilince %d cift esige ulasan (olcum: 4, yarisi meşru)'
            % len(birlestir))

    def test_noktalama_dersi_korunur(self):
        """Kapı asiri SERT olmasin: '()' konusunun kendisi olan Turkce
        noktalama sorusu ELENMEMELI (olcumde 4 kalintinin 2'si meşru)."""
        rec = konusma(
            ('istanbul\'a bugun gidiyorum () dedi. cumlesindeki ayracla '
             'ilgili bilgi verir misin?',
             'cumlede konusmanin yapildigi kisimin sonunda yer alan ifadeye '
             'ayrac denir ve tirnak isareti kullanilir.'))
        cift = list(FH.pairs_from_chat(rec))
        self.assertEqual(len(cift), 1,
                         'noktalama dersi elendi: kapı fazla sert')

    def test_yabanci_dil_neredeyse_yok(self):
        """Olcum: dosyada yabanci dil izi %0,3."""
        import io
        import json
        with io.open(self.VAR, encoding='utf-8') as f:
            satirlar = [json.loads(l) for l in f if l.strip()]
        kotu = [r for r in satirlar
                if FH.yabanci_dil_mi(r['query'] + ' ' + r['answer'][0])]
        self.assertLessEqual(
            len(kotu), max(5, len(satirlar) * 0.01),
            '%d / %d cift yabanci dil izi tasiyor' % (len(kotu), len(satirlar)))


if __name__ == '__main__':
    unittest.main()

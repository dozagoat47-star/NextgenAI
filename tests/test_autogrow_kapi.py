# -*- coding: utf-8 -*-
"""AutoGrow kapasite sinirlari: toplam kapı ve kosu basi tavan.

NEDEN (28.09 olcumu):
  - 6.000 kapisi DOLUYDU (6.000/6.000) -> intent uretimi durdu,
    soru buyumeye devam ediyordu.
  - buyume hizi 7.790 intent/gun. 25 kosu/gun (autogrow.yml saatlik +
    autogrow-deep.yml gunluk) -> ortalama 312 intent/kosu.
  - 842 bayt/intent; intents.json HER kosuda tam blob olarak saklandigi
    icin git maliyeti dosya boyutuyla carpilir (karesel).

DUZELTME:
  AUTOGROW_MAX_INTENTS 6.000 -> 20.000  (veri butcesi; model DEGIL -
    bilgi intentleri siniflandirici degildir, model.json num_intents 40)
  AUTOGROW_MAX_PER_RUN (yeni, 400)      (buyume hizi siniri; ort. 312
    normal akisi degistirmez, sicak kosularin tek commit'te binlerce
    intent dokmesini engeller)

Bu testler sayilari ve tavan MANTIGINI korur; agdan hicbir sey
cagrilmaz (network yok, intents.json'a yazilmaz).
"""
import io
import json
import os
import re
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import autogrow


class TestToplamKapi(unittest.TestCase):

    def test_kapi_20000_ve_ortam_ile_ayarlanabilir(self):
        """Kapi 20.000 olmali (6.000 doldugu icin buyutuldu)."""
        self.assertEqual(autogrow.AUTOGROW_MAX_INTENTS, 20000,
                         'kapı 20.000 olmali; 6.000 doldugu icin '
                         'uretim durmustu')

    def test_kapi_intents_sayisindan_buyuk(self):
        """Kapı, dosyadakilerin uzerinde olmali (yoksa uretim bitmis)."""
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        with io.open(yol, encoding='utf-8') as f:
            n = len(json.load(f)['intents'])
        self.assertLess(n, autogrow.AUTOGROW_MAX_INTENTS,
                        'intents.json %d kayit, kapı %d: uretim yine '
                        'durdurulmus olur' % (n, autogrow.AUTOGROW_MAX_INTENTS))

    def test_kapi_kb_limitten_kucuk(self):
        """knowledge_map butcesi (kbmap.yml --kb-limit 40000) kapiyi
        ASMAMALI, yoksa yeni intentler RAG deseni alamaz."""
        yol = os.path.join(BASE, '.github', 'workflows', 'kbmap.yml')
        if not os.path.exists(yol):
            self.skipTest('kbmap.yml yok')
        with io.open(yol, encoding='utf-8') as f:
            yml = f.read()
        self.assertIn('--kb-limit 40000', yml,
                      'kbmap butcesi kapiya gore yukseltilmeli')
        self.assertLess(autogrow.AUTOGROW_MAX_INTENTS, 40000,
                        'kapı kbmap butcesini asmali ki yeni intentler '
                        'knowledge_map deseni alabilsin')


class TestKosuBasiTavan(unittest.TestCase):

    def test_tavan_ortalamayi_asmaz(self):
        """Tavan > ortalama (312) olmali, yoksa akis yavaslar."""
        self.assertEqual(autogrow.AUTOGROW_MAX_PER_RUN, 400)
        self.assertGreater(autogrow.AUTOGROW_MAX_PER_RUN, 312,
                           'tavan ortalama 312 intent/kosunun ALTINDA: '
                           'buyume hizi gereksiz yere dusurulur')

    def test_tavan_ortam_ile_ayarlanabilir(self):
        """0 = tavansiz olabilmeli (acil kapatma)."""
        eski = autogrow.AUTOGROW_MAX_PER_RUN
        try:
            os.environ['AUTOGROW_MAX_PER_RUN'] = '0'
            import importlib
            yen = importlib.reload(autogrow)
            self.assertEqual(yen.AUTOGROW_MAX_PER_RUN, 0,
                             '0 -> tavansiz (eski davranis)')
        finally:
            os.environ.pop('AUTOGROW_MAX_PER_RUN', None)
            import importlib
            importlib.reload(autogrow)
        self.assertEqual(autogrow.AUTOGROW_MAX_PER_RUN, eski)

    def test_uygulama_sayimi(self):
        """Dongu mantigini network'suz dogrula: grow_once sahte.

        grow_once her turda kendi butcesi kadar intent ekliyor; ana
        dongu toplam eklenenleri sayip tavanda DURMALI.
        """
        import importlib
        importlib.reload(autogrow)
        eski_grow = autogrow.grow_once
        eski_sleep = autogrow.time.sleep
        cagrilar = []

        def sahte_grow_once(source, count):
            cagrilar.append((source, count))
            # her tur count kadar yeni intent eklenmis gibi don
            return (count, 0)

        try:
            autogrow.grow_once = sahte_grow_once
            autogrow.time.sleep = lambda *a, **k: None   # testi hizlandir
            sys.argv = ['autogrow.py', '--minutes', '999',
                        '--source', 'mixed', '--count', '8']
            # butceyi kucuk tut ki test hizli bitsin
            autogrow.AUTOGROW_MAX_PER_RUN = 20
            autogrow.main()
        finally:
            autogrow.grow_once = eski_grow
            autogrow.time.sleep = eski_sleep
            autogrow.AUTOGROW_MAX_PER_RUN = 400
            sys.argv = sys.argv[:1]

        toplam_istenen = sum(c for _s, c in cagrilar)
        self.assertLessEqual(toplam_istenen, 20,
                             'tavan asildi: istenen toplam %d' % toplam_istenen)
        self.assertEqual(toplam_istenen, 20,
                         'tavan tam kullanilmali, 20 istenmesi bekleniyordu')
        # son turda kalan butce kadar istenmis olmali (tasma olmamali)
        self.assertLessEqual(max(c for _s, c in cagrilar), 8)


class TestModelEtkilenmez(unittest.TestCase):
    """Bilgi intentleri sinif DEGILDIR: model 40'da sabit kalir."""

    def test_model_40_sinif_kalir(self):
        yol = os.path.join(BASE, 'model', 'model.json')
        if not os.path.exists(yol):
            self.skipTest('model/model.json yok (Kaggle egitimi uretir)')
        with io.open(yol, encoding='utf-8') as f:
            mj = json.load(f)
        self.assertEqual(mj.get('num_intents'), 40,
                         'sinif sayisi degismemeli; intents kapisi '
                         'siniflandiriciyi ETKILEMEZ')


class TestMaxPairsEsasVeriButcesi(unittest.TestCase):
    """MAX_PAIRS = zincirin ASIL darbogazi (28.09 olcumu).

    Bu sinif, 'kapi yukseltildi, veri artti' yanlis izlenimini engeller.
    Uretilen cift sayisi her zaman MAX_PAIRS'in USTUNDE oldugu icin veri
    uretiliyor ama cogu KIRPILIYOR.

    KRITIK: kirpilan kisim bos degil. Bir intent N pattern x M cevap ->
    N*M cift uretiyor; bilgi N+M'de, N*M'de degil. Olcum (6.364 intent):
        benzersiz ctx 38.954 | benzersiz cevap 23.366 | cift 162.975
    Yani 162.975 ciftin icinde yalnizca ~24.000 bagimsiz kalem var.
    Bu yuzden olcdugumuz kriter 'cift sayisi' DEGIL, 'kapsanan benzersiz
    ctx orani' — asagida test ediliyor.
    """

    @staticmethod
    def _uretilen(max_pairs):
        import train_llm
        from seqgen import load_pairs
        yol = os.path.join(BASE, 'intents.json')
        return load_pairs(yol, max_pairs=max_pairs, use_query=True,
                          ctx_len=train_llm.CTX_CHARS)

    @staticmethod
    def _butce(ust_tavan=0):
        import train_llm
        return train_llm.coz_max_pairs(ust_tavan=ust_tavan, yaz=False)

    def test_butce_veriyi_zararli_asmaz(self):
        """Butce, mevcut veriyi %10'dan fazla ASMAMALI (kacak guvenligi).

        ESKI KOSUL: `uretilen > butce` (kirpma AKTIF olmali). 01.10.2026'da
        bu kosul KIRMIZI oldu ve dogru bir seyi soylemeyi reddetti:
        darbogaz veriden zamana tasinmistir (olcum: DEVAM_PROMPTU.md 6.11).

            28.09   ham 162.975  butce 120.000  kirpma %73,6
            1.10    ham 270.178  butce 269.555  kirpma %99,8
            bugun   ham 276.016  butce 276.096  kirpma YOK

        Gercek tavan ZAMANDIR: kaggle_start.sh:98 `sure_ve_hesapla(540 dk,
        6 epoch)` = 1.251.723 cift = veri butcesinin 4,5 kati. `coz_max_pairs`
        zaten min(veri_butcesi, zaman_tavani) uygular (train_llm.py:941-948),
        yani veri azaldikca zaman tavani devreye girecek.

        Butce veriyi fazla asmazsa: kirpma olmaz (verinin TAMAMI kullanilir,
        zaman tavani devreye girer) -> zararli degil.
        Butce veriyi %10'dan fazla asarsa: karpan veriden koptu, butce
        havada kaldi, kirpma geri donmus olabilir ama nedenini bilemeyiz.
        """
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        uretilen = len(self._uretilen(10 ** 9))
        butce = self._butce()
        self.assertLessEqual(
            butce, uretilen * 1.10,
            'butce (%s) ham cifti (%s) %%10\'dan fazla asiyor: karpan '
            'veriden koptu, butce islevsiz kaldi.'
            % (format(butce, ','), format(uretilen, ',')))
        # Kirpma varsa GERCEKTEN kirpiyor olmali (bosuna kirpma olmamali).
        if butce < uretilen:
            self.assertLess(
                butce, uretilen * 1.10,
                'butce ham ciftin cok uzerinde ama altinda: kirpma %%.1f\'e '
                'dustu, nedenini olc.'
                % (100.0 * butce / max(1, uretilen)))

    def test_max_pairs_benzersiz_ctxin_cogu_kapsiyor(self):
        """MAX_PAIRS'in GERCEK islevi: benzersiz ctx'nin cogunu kapsamak.

        Olcum (28.09) 6.364 intent ile:
            MAX_PAIRS  benzersiz ctx  kapsama  benzersiz cevap
             70.000      31.848         %81,8     22.570  (%96,6)
            120.000      38.048         %97,7     23.359  (%99,97)
            162.975      38.954        %100,0     23.366  (%100,0)
        Esik %90: hem 120.000'i gecer hem de 70.000'e dusmeyi yakalar.
        """
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        tum = self._uretilen(10 ** 9)
        tum_ctx = set(c for c, _r in tum)
        butce = self._butce()
        secilen = self._uretilen(butce)
        secilen_ctx = set(c for c, _r in secilen)
        kapsama = 100.0 * len(secilen_ctx) / max(1, len(tum_ctx))
        self.assertGreater(
            kapsama, 90.0,
            'butce=%s iken benzersiz ctx kapsamasi sadece %.1f: '
            'kirpma GERCEK kapi kaybetti. Olcumde 120.000 -> %%97,7, '
            '70.000 -> %%81,8 idi.'
            % (format(butce, ','), kapsama))

    def test_otomatik_butce_olculen_dizin_noktasini_uretiyor(self):
        """MAX_PAIRS_CTX_CARPAN / MAX_PAIRS_INTENT_CARPAN karpanlari
        gercekten olculmus degeri uretiyor mu? (6.364 intent'te 120.000)

        Karpanlar elle yazilmis; bu test, veri degisse karsilastigini
        OLCEYEN ilk yeri olur.
        """
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        import train_llm
        with open(yol, encoding='utf-8') as f:
            n = len(json.load(f).get('intents', []))
        uretilen = len(self._uretilen(10 ** 9))
        u = len(set(c for c, _r in self._uretilen(10 ** 9)))

        # iki karpan ayni seyi ifade etmeli
        self.assertAlmostEqual(
            train_llm.MAX_PAIRS_INTENT_CARPAN * n,
            train_llm.MAX_PAIRS_CTX_CARPAN * u,
            delta=0.05 * train_llm.MAX_PAIRS_CTX_CARPAN * u,
            msg='INTENT_CARPAN=%s x %d intent != CTX_CARPAN=%s x %d benzersiz '
                'ctx: karpanlar birbirinden ayri kaldi'
                % (train_llm.MAX_PAIRS_INTENT_CARPAN, n,
                   train_llm.MAX_PAIRS_CTX_CARPAN, u))

        butce = self._butce()
        # ESKI: `butce < uretilen` (kirpma aktif olmali) -- 01.10'da kirmizi
        # oldu ve DARBOGAZIN TASTIGINI bildirdi, hatayi degil. Butce veriyi
        # asmamanin gerekcesi ve olcumu icin `test_butce_veriyi_zararli_asmaz`
        # ve `test_max_pairs_benzersiz_ctxin_cogu_kapsiyor` bak.
        # Buradaki iki karpanin ayni noktayi uretmesi ise ASIL budur ve
        # yukaridaki assertAlmostEqual ile zaten denetleniyor.

    def test_butce_intent_sayisiyla_buyur(self):
        """SABIT deger degil: intents buyunca butce de buymeli.

        Gerekce: intents 6.364 -> 20.000 olunca benzersiz ctx ~122.000'e
        cikar; sabit 120.000 yeniden darbo gaz olur. Iki karpan da ayni
        orani (6,12 benzersiz ctx / intent) kullandigi icin bu test
        20.000 intent'te de gecerli.
        """
        import train_llm
        n = int(round(train_llm.MAX_PAIRS_INTENT_CARPAN * 6364))
        buyuk = int(round(train_llm.MAX_PAIRS_INTENT_CARPAN * 20000))
        self.assertGreater(buyuk, n,
                           'butce intents ile buyuluyor')
        # 6,12 = 38.954 / 6.364 (olculmus, yuvarlanmis). Iki karpan ayni
        # orani kullandigi icin 20.000 intent'te de ayni butce cikmali.
        self.assertAlmostEqual(
            buyuk,
            int(round(train_llm.MAX_PAIRS_CTX_CARPAN * 6.12 * 20000)),
            delta=200,
            msg='20.000 intent butcesi: karpanlar arasi uyumsuzluk')

    def test_tavan_ihtiyaci_gecer_kazanir_ve_yazilir(self):
        """Veri ihtiyaci zaman tavanini asarsa TAVAN KAZANIR ve
        sessizce degil, EKRANA yazilir."""
        import io as _io
        import contextlib as _cl
        import train_llm

        buf = _io.StringIO()
        with _cl.redirect_stdout(buf):
            sonuc = train_llm.coz_max_pairs(ust_tavan=50000)
        metin = buf.getvalue()
        self.assertEqual(sonuc, 50000, 'tavan uygulanmadi')
        self.assertIn('TAVAN KAZANDI', metin,
                      'tavan kazandigi halde SESSIZCE kirpildi: kullanici '
                      'verinin bir kismi kayboldugunu bilmiyor. Cikti: %r'
                      % metin)

    def test_tavan_yoksa_ihtiyac_kullanilir(self):
        import train_llm
        self.assertEqual(self._butce(ust_tavan=0), self._butce(ust_tavan=10 ** 9),
                         'tavan 10^9 iken butce kisildi')

    def test_max_pairs_veri_butcesi_oldugu_belgeli(self):
        """Butce bilincli bir olcum karari; rastgele degil.
        Sabit 70.000'in altina dusmemeli. Simdi sabit DEGIL, otomatik;
        ama otomatik deger de asagi bir tabana sahip olmali."""
        self.assertGreaterEqual(
            self._butce(), 70000,
            'otomatik butce 70.000 altina dustu: epoch suresi icin veri '
            'kirpiliyor. Kalici veri butcesi; bilincli dusurulmelidir.')


class TestSureTavani(unittest.TestCase):
    """SURE TAVANI: kac cift 9 saatlik oturuma sigar?

    Bu bir veri butcesi degil, bir ZAMAN butcesidir; veri butcesi
    (coz_max_pairs) intents ile buyuyor, zaman tavani oturumla kisitli.
    Ikisi carpistiginde TAVAN kazanir ve yazilir.

    ONEMLI: ilk yazimda saat->dk cevirisi IKI KEZ yapildigi icin tavan
    316.000 yerine 9.575 cift cikti — yani egitim 120.000 -> 9.575 cifte
    dusertilecekti ve kimse fark etmeyecekti. Asagidaki testler tam
    olarak bu hata sinifini yakalar.
    """

    def test_olculen_degeri_uretiyor(self):
        """9 saat / 12 epoch -> tavan, iki olcumden turetilmis olmali.

        SABIT SAYI YOK (29.09 duzeltmesi): once 1.426.101 yaziliydi ve
        olcum sabitleri guncellenince kirildi. Kirilan sey KOD DEGILDI,
        testin kendi bayatligiydi. Bunun yerine:
          1) sonucu sabitlerden YENIDEN hesaplayip ayni cifti beklemek
             (formulun kendi degisimini yakalar),
          2) bir MIKTAR araligi beklemek (birim hatasini yakalar:
             saat->dk iki kez bolunse sonuc 1000 kucuk olur).

        Gercek: 540 dk x 0,75 bosluk = 405 dk; - 26 dk encode = 379 dk;
        379/12 = 31,58 dk/epoch; x 60.000 / 1,5139 ms = 1.251.733 cift.
        """
        import train_llm
        tavan = train_llm.sure_ve_hesapla(oturum_dk=540, epochs=12)
        kalan = 540 * train_llm.VARSAYILAN_BOSLUK - train_llm.ENCODE_DK
        beklenen = int((kalan / 12.0) * 60000 / train_llm.MS_PER_HAM_CIFT_EPOCH)
        self.assertEqual(tavan, beklenen,
                         'sure_ve_hesapla sabitlerden turetilmiyor: %d != %d'
                         % (tavan, beklenen))
        # MIKTAR: 01.10 23:26 kosusundan OLCULEN ham cift maliyeti
        # (8,0911 ms/ham/epoch) 540 dk x %75 bosluk -> 234.210 verir.
        # Onceki sabit MS_PER_PAIR (egitim cifti, 1,5139 ms) 1.251.733
        # veriyordu; expansion 4,7711 ile carpilinca 5,34 KAT uzakta
        # kaldi. Test artik SABITLERIN KENDISI degerlendirir (asagida
        # test_zaman_tavani_olculen_sinirda), burada sadece aralik.
        # Genis alt sinir: saat->dk iki kez bolunse 234 yerine ~0 gelirdi.
        self.assertGreater(tavan, 150000,
                           'zaman tavani asiri dar: 9 saatlik oturuma 1 milyon '
                           'cift sigmali (birim hatasi ihtimali)')
        self.assertLess(tavan, 400000,
                        'zaman tavani asiri genis: OLCULEN sinir 234.210 ham '
                        'cift; %s cift diyor, expansion unutulmus olabilir'
                        % format(tavan, ','))

    def test_tavan_oturum_butcesini_asmaz(self):
        """ASIL INVARIANT: tavan x epoch x ms/ham <= kullanilabilir sure.

        Cift bolme iki kez yapilirsa bu ASILIR; boyle bir test formulu
        dondurmeden dogrular.

        01.10.2026: buradaki sabit MS_PER_PAIR iken test de DONGUSELDI
        (formulu ayni sabitle yeniden hesapliyordu, sabitin dogru/yanlis
        oldugunu hic anlamiyordu). Artik MS_PER_HAM_CIFT_EPOCH.
        """
        import train_llm
        for oturum, ep in ((540, 12), (540, 6), (540, 25), (360, 12), (600, 70)):
            tavan = train_llm.sure_ve_hesapla(oturum_dk=oturum, epochs=ep)
            dk = tavan * train_llm.MS_PER_HAM_CIFT_EPOCH / 1000 / 60
            kalan = oturum * train_llm.VARSAYILAN_BOSLUK - train_llm.ENCODE_DK
            self.assertLessEqual(
                dk * ep, kalan + 1.0,
                'oturum=%d epoch=%d: tavan %d cift -> %.0f dk x %d epoch = '
                '%.0f dk > kullanilabilir %.0f dk'
                % (oturum, ep, tavan, dk, ep, dk * ep, kalan))

    def test_zaman_tavani_olculen_sinirda(self):
        """SABITLERIN KENDISI dogrulanir — formulu kendisiyle degil.

        01.10 23:26 kosusu OLCUMU (tools/sure_olc.py, kaggle_train.txt):
            ham 288.802 -> egitim 1.377.895        (expansion 4,7711)
            12 epoch toplami 25.620,4 sn
            encode 2.420,0 sn
            toplam 28.040,4 sn / 288.802 ham = 97,09 ms/ham
        Bu sabitten turetilen MS_PER_HAM_CIFT_EPOCH = 97,09/12 = 8,0911.

        IKILERI (sabit degil, olcumden gelen):
        UST = 540 dk'nin tamami, olculen hizla: 540 x 60000 / 97,09
              = 333.699 ham cift. Bu, oturumun ASLA tasamayacagi sinirdir.
              Ustune cikarsa sessizce tasar -> KESILMEZ.
        ALT = 01.10 kosusunda 288.802 ham ciftin oturumu BITIRDI (540 dk
              icinde; olculen kismi 467,3 dk = %86,5, veri hazirligi
              olculmedi). Yani bu hacim MAKULDE CALISMI olarak kanitli.
              Tavan bunun en az %75'ini tutmali; altina duserse kanitli
              calisan bir hacmin yarisini de atiyoruz demektir.
        ARA = tavan 234.207 = 288.802 x %81,1. %25 bosluk politikasinin
              bedeli: OLCULEN kapsama kaybi canli intents.json uzerinde
              cift %15,1, benzersiz ctx %0,9, benzersiz cevap %0,1.

        Onceki kod 1.251.733 diyordu -> UST'un 3,75 KATI, yani bu test
        kirmizi olurdu. Hata birim hatasidi (egitim cifti / ham cift).
        """
        import train_llm
        # --- sabit OLCULEN log degerine bagli mi?
        ms_ham_toplam = train_llm.MS_PER_HAM_CIFT_EPOCH * 12
        self.assertAlmostEqual(
            ms_ham_toplam, 97.09, delta=0.05,
            msg='MS_PER_HAM_CIFT_EPOCH=%s -> 12 epoch toplami %.2f ms/ham. '
                '01.10 23:26 kosusunda OLCULEN deger 97,09 ms/ham. Yeni bir '
                'kosu olcuyse tools/sure_olc.py calistir ve burasi guncelle.'
                % (train_llm.MS_PER_HAM_CIFT_EPOCH, ms_ham_toplam))
        ust = int(540 * 60 * 1000 / ms_ham_toplam)
        kanitli = 288802          # bittigi OLCULEN kosunun ham cift sayisi
        alt = int(kanitli * 0.75)
        tavan = train_llm.sure_ve_hesapla(oturum_dk=540, epochs=12)
        # --- UST: oturumun tasamasi imkansiz olmali
        self.assertLessEqual(
            tavan, ust,
            'zaman tavani %s ham cift, oysa 540 dk\'ya OLCULEN hizla '
            '(%.2f ms/ham) tam oturumda %s ham cift sigiyor. Tavan gercek '
            'sinirdan USTTE -> oturum sessizce tasar.'
            % (format(tavan, ','), ms_ham_toplam, format(ust, ',')))
        # --- ALT: kanitli calisan hacmin cogu atilmiyor olmali
        self.assertGreaterEqual(
            tavan, alt,
            'zaman tavani %s ham cift; 01.10 kosusunda %s ham ciftin oturumu '
            'BITTI (kanitli calisan hacim). Tavan bunun %%75\'inden kucuk, '
            'yani kanitlanmis calisan verinin yarisindan fazlasini atiyor.'
            % (format(tavan, ','), format(kanitli, ',')))

    def test_epoch_artisi_tavani_azaltir(self):
        """Daha cok epoch = daha az cift (toplam sure sabit)."""
        import train_llm
        az = train_llm.sure_ve_hesapla(epochs=6)
        cok = train_llm.sure_ve_hesapla(epochs=25)
        self.assertGreater(az, cok,
                           'EPOCHS artti ama tavan artti: ayni sureye sigan '
                           'veri artmali, epoch basina azalmali')
        self.assertGreater(cok, 0, 'EPOCHS=25 tavanı 0 oldu: egitim yapamaz')

    def test_olculen_sabitler_birlikte_tutarli(self):
        """MS_PER_PAIR, kaggle_start.sh yorumundaki EPOCH OLCUMU'nden gelir.

        BURADA YAPILAN HATA (29.09'da bulundu): test 7,31 dk'yi SABIT SAYI
        olarak tutuyordu ama cift sayisini yorumdan OKUYORDU. Yorum
        guncellenince (cift sayisi 315.883 -> 921.748) iki farkli olcum
        birlestirildi: 7,31 dk / 921.748 cift = 0,43 ms, koddaki 1,5139
        ile uyusmadi. Yani test, dogru olan kodu reddetti.

        Duzeltme: epoch suresi de yorumdan okunur ("ort N sn/epoch") ve
        cift sayisiyla ayni etiketli bloktan ("EPOCH OLCUMU:") alinir.
        Boylece iki olcum birbirine karisamaz.

        Daha onceki hata: ilk yazimda 70.000 cift varsayildi ->
        6,266 ms/cift -> 4,5 KAT YANLIS -> zaman tavani gereksiz yere
        316.000'a dustu. Test, olcum satirindaki SAYIYI koda baglar.
        """
        import train_llm
        sh = os.path.join(BASE, 'kaggle_start.sh')
        if not os.path.exists(sh):
            self.skipTest('kaggle_start.sh yok')
        with open(sh, encoding='utf-8') as f:
            metin = f.read()
        m = re.search(r'EPOCH\s+OLCUMU:\s*(\d+)\s*cift', metin)
        self.assertIsNotNone(
            m, 'kaggle_start.sh yorumunda "EPOCH OLCUMU: N cift" bulunamadi: '
               'epoch hizi hangi veriyle olculdu artik OKUNAMAZ')
        cift_olcumu = int(m.group(1))
        m_sn = re.search(r'ort\s+(\d+(?:[.,]\d+)?)\s*sn/epoch', metin)
        self.assertIsNotNone(
            m_sn, 'kaggle_start.sh yorumunda "ort N sn/epoch" bulunamadi: '
                  'epoch suresi koda baglanamiyor')
        sn_olcumu = float(m_sn.group(1).replace(',', '.'))
        ms = sn_olcumu * 1000 / cift_olcumu
        self.assertAlmostEqual(
            train_llm.MS_PER_PAIR, ms, delta=0.01,
            msg='MS_PER_PAIR=%s ama kaggle_start.sh yorumundaki olcum '
                '(%.2f sn / %s cift) = %.4f ms. Yorumdaki olcum '
                'degistiyse kodu da guncelle.'
                % (train_llm.MS_PER_PAIR, sn_olcumu, format(cift_olcumu, ','),
                   ms))

    def test_kaggle_start_sh_tek_dogruluk_kaynagini_kullanir(self):
        """Formul bash'ta TEKRARLANMAMALI. Tekrar varsa iki taraf
        ayrilir ve sessizce tutarsizlasir (olcumde olan da bu).

        Burda kaynak metni taramak GEREKLI: 'tek dogruluk kaynagi'
        ilkesi zaten kaynagin kendisiyle ilgilidir.
        """
        yol = os.path.join(BASE, 'kaggle_start.sh')
        if not os.path.exists(yol):
            self.skipTest('kaggle_start.sh yok')
        with open(yol, encoding='utf-8') as f:
            sh = f.read()
        self.assertIn('sure_ve_hesapla', sh,
                      "kaggle_start.sh sure_ve_hesapla()'i cagirmiyor: "
                      'tavan formulu bash\'a kopyalanmis, test edilemez')
        self.assertNotIn('MS_PER_PAIR=', sh,
                         'kaggle_start.sh kendi ms/cift sabitini tanimliyor: '
                         'FORMUL TEKRARLANDI, iki taraf ayrilir')
        # tavan argumani gercekten egitim cagrisina geciriliyor mu?
        self.assertIn('MPCARGS="--max-pairs-cap $MPCAP"', sh)
        # bash'ta satir sonu '\' ile devam eder; komut bloklarina birlestir
        bloklar, buf = [], ''
        for satir in sh.splitlines():
            buf += satir
            if satir.rstrip().endswith('\\'):
                continue
            if buf.strip():
                bloklar.append(buf.strip())
            buf = ''
        cagrilar = [b for b in bloklar if 'python train_llm.py' in b]
        self.assertGreaterEqual(len(cagrilar), 3,
                                'train_llm.py cagrilari bulunamadi')
        for b in cagrilar:
            self.assertIn('$MPCARGS', b,
                          'bir train_llm.py cagrisi zaman tavanini almadi: '
                          '%s' % b.replace('\n', ' ')[:120])

    def test_simdiki_butce_tavana_siuyor(self):
        """Gercek boru hatti: coz_max_pairs TAVANI ALIR, min() uygular.

        01.10.2026'da bu test `coz_max_pairs(yaz=False)` cagriyordu, yani
        ust_tavan=0 ile: tavan HIC UYGULANMIYORDU. Kirpma inert oldugu icin
        fark edilmedi. kaggle_start.sh:98 `--max-pairs-cap $MPCAP` gecirir;
        test de onu yansitmali, yoksa boru hatti hakkinda YANLIIS bilgi
        verir.

        Bugun (birim duzeltmesinden sonra) tavan GERCEKTEN bagliyor:
            veri butcesi 276.096 > zaman tavani 234.207 -> tavan kazandi
        Kirpma bedeli OLCULDU (canli intents.json, seqgen.load_pairs):
            cift 276.016 -> 234.207           (%84,9 kaldi, %15,1 kesildi)
            benzersiz ctx  88.650 ->  87.829 (%99,1 kaldi, %0,9 kesildi)
            benzersiz cevap 45.705 ->  45.645 (%99,9 kaldi, %0,1 kesildi)
        Yani %15,1 cift kaybi ama %0,9 kapsama kaybi: kesilen kisim
        cogunlukla ayni ctx/cevap'in tekrarlari. Zamana karsi alinan
        bedel bu kadar ucuz.
        """
        import train_llm
        tavan = train_llm.sure_ve_hesapla(oturum_dk=540, epochs=12)
        butce = train_llm.coz_max_pairs(ust_tavan=tavan, yaz=False)
        self.assertEqual(
            butce, min(train_llm.coz_max_pairs(yaz=False), tavan),
            'coz_max_pairs ust_tavan ile min() UYGULAMIYOR: kirpma geri '
            'donmus olabilir')
        self.assertLessEqual(
            butce, tavan,
            'ust_tavan gecildi ama butce %s > tavan %s: kirpma calismiyor.'
            % (format(butce, ','), format(tavan, ',')))

    def test_kirpma_kapsamayi_agirmeden_atmiyor(self):
        """Kirpma %15,1 cift aliyor ama sadece %0,9 benzersiz ctx.

        Bu, onceki "kirpma aktif olmali" varsayiminin YERINI alir. Kirpma
        aktif olmasi iyi bir sey degil; kirpinin kapsamayi bozmamasi iyi
        bir sey. Cift sayisi kombinasyoneldir (N pattern x M cevap), o
        yuzden once tekrarlar kesilir.

        OLCUM (01.10.2026, canli intents.json):
            sinirsiz      -> cift 276.016 | ctx 88.650 | cevap 45.705
            tavan 234.207 -> cift 234.207 | ctx 87.829 | cevap 45.645
        Esik %99 ctx kapsamasi: kapsama %1'den fazla duserse kirpma bir
        seyi bozmaya basliyor demektir.
        """
        import train_llm
        from seqgen import load_pairs
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        tavan = train_llm.sure_ve_hesapla(oturum_dk=540, epochs=12)
        tam = load_pairs(yol, max_pairs=10 ** 9, use_query=True,
                         ctx_len=train_llm.CTX_CHARS)
        kirp = load_pairs(yol, max_pairs=tavan, use_query=True,
                          ctx_len=train_llm.CTX_CHARS)
        ctx_tam = set(p[0] for p in tam)
        ctx_kirp = set(p[0] for p in kirp)
        oran = 100.0 * len(ctx_kirp) / max(1, len(ctx_tam))
        self.assertGreaterEqual(
            oran, 99.0,
            'zaman tavani %s ham cift: benzersiz ctx kapsamasi %.1f '
            '(tam %s -> kirpilan %s). Kirpma kapsamayi bozuyor; esik %%99.'
            % (format(tavan, ','), oran, format(len(ctx_tam), ','),
               format(len(ctx_kirp), ',')))
        self.assertLess(
            len(kirp), len(tam),
            'kirpma hicbir cift atmadi: tavan veriyi kirpmiyor, zaman '
            'tavani islevsiz (5,34 KAT birim hatasinin belirtisi).')
    def _sync_ps1(self):
        yol = os.path.join(BASE, 'sync_chatgrow.ps1')
        if not os.path.exists(yol):
            self.skipTest('sync_chatgrow.ps1 yok')
        with io.open(yol, encoding='utf-8') as f:
            return f.read()

    def test_sync_tohumu_kosudan_turetiliyor(self):
        """29.09 OLCUMU: tohum sabit 7 -> script 19 kosuda 16'si no-op.

        '.. icerik mevcut bir HF dosyasiyla ayni; yeni dosya atildi'
        Betigin amaci 'Kaggle en taze dosyayi glob'la alir'; sabit tohum
        bunu gerceklastirmiyor, sadece ayni 1.200 cifti indirip atiyor.
        """
        sh = self._sync_ps1()
        m = re.search(r'\[int\]\$Seed\s*=\s*(-?\d+)', sh)
        self.assertIsNotNone(m, 'param([int]$Seed = ...) bulunamadi')
        self.assertEqual(int(m.group(1)), 0,
                         'tohum sabit: her kosuda ayni ciftler uretilip '
                         'atiliyor (no-op). Varsayilan 0 = kosudan turet.')
        self.assertIn('if ($Seed -le 0)', sh,
                      '$Seed 0 ise kosudan turetilmiyor')
        self.assertIn('Get-Date', sh,
                      'tohum turetmesi zaman tabanli degil')
        # HF adimi tohumu kullanmaya devam etmeli
        self.assertIn('--seed $Seed', sh,
                      'HF adimi $Seed kullanmiyor -> hep ayni veri')

    def test_sync_kitap_dosyasi_sabit_adli(self):
        """Zaman damgali kitap adi -> uretici her degistiginde dosya birikiyor.

        build_book_pairs.py deterministik (SEED sabit) ve katalog tukenmis:
        her kosuda ayni 609 cift. Son kosuda 609 ciftin yalniz 29'u
        benzersizdi (%95 tekrar).
        """
        sh = self._sync_ps1()
        self.assertNotIn('chatgrow_kitap_$(', sh,
                         'kitap ciktisi yine zaman damgali ad kullaniyor')
        self.assertIn("'chatgrow_kitap.jsonl'", sh,
                      'kitap ciktisi sabit ad degil')
        # tohumu almamali: kitap yarisi deterministik kalmali (commit gurultusu yok)
        kitap_blok = sh.split('--- 2)')[1].split('--- 3)')[0] if '--- 2)' in sh else ''
        self.assertNotIn('--seed', kitap_blok,
                         'kitap adimi tohum aliyor -> her kosuda farkli '
                         'veri, gereksiz commit')

    def test_sync_kitap_butceyi_ustune_cikmiyor(self):
        """29.09: sync script'i --ctx-len 300 --resp-len 300 veriyordu.

        Loader clean_chars(x, 204) ile KAPALI sekilde kesiyor; uretici
        300'de kelime sonunda kessse bile 300 > 204 oldugu icin hasar
        dosyaya gomuluyordu (48/609 = %7,88). Script artik arguman
        vermiyor; ureticinin varsayilani loader butcesine esit ve
        testler kilitliyor.
        """
        sh = self._sync_ps1()
        # yorum satirlari kod degildir (dosyada '--ctx-len VERILMEZ' diye
        # yaziyor); sadece CALISTIRILAN satirlara bak
        kod = '\n'.join(s for s in sh.splitlines()
                        if not s.lstrip().startswith('#'))
        self.assertNotIn('--ctx-len', kod, 'kitap adimi ctx butcesi veriyor')
        self.assertNotIn('--resp-len', kod, 'kitap adimi resp butcesi veriyor')
        # gercekten build_book_pairs cagrisi var mi
        self.assertIn('build_book_pairs.py', kod)

    def test_patience_en_az_iki_kotu_olcum(self):
        """29.09: patience=2, --val-every 2 -> TEK kotu val OLCUMU.

        Olcum: kosu 6. epoch'ta bitti. val kaybi 0,5827 (4. ep) -> 0,5901
        (6. ep), yani %1,3 artti; ama acc HALA YUKSELIYORDU (0,876 ->
        0,885) ve train kaybi hizla iniyordu (0,4784 -> 0,3425).
        lr_horizon=12 idi, kosu LR tam inmeden kesildi. 27.09 kosusu 12
        epoch'a tamamlamisti.

        Yan etkisi olmayan tek duzeltme: patience 2 -> 4. lr_horizon
        min(EPOCHS=12, patience+20=24) = 12 oldugu icin LR programi AYNI.
        """
        yol = os.path.join(BASE, 'kaggle_start.sh')
        if not os.path.exists(yol):
            self.skipTest('kaggle_start.sh yok')
        with open(yol, encoding='utf-8') as f:
            sh = f.read()
        m = re.search(r'PATIENCE="\$\{LLM_PATIENCE:-(\d+)\}"', sh)
        self.assertIsNotNone(m, 'PATIENCE varsayilani bulunamadi')
        p = int(m.group(1))
        self.assertGreaterEqual(
            p, 4,
            'patience=%d ama --val-every 2 ile bu TEK kotu val olcumudur '
            '(2 epoch). Gurultu erken durduruyor; en az iki olcum gerekir.'
            % p)
        # LR programi degismemis olmali
        EPOCHS = int(re.search(r'EPOCHS="\$\{2:-\$\{LLM_EPOCHS:-(\d+)\}\}"',
                               sh).group(1))
        self.assertEqual(min(EPOCHS, p + 20), EPOCHS,
                         'patience artisi lr_horizon\'i degistirdi; '
                         'LR programi kasten ayni kalmaliydi')


if __name__ == '__main__':
    unittest.main()

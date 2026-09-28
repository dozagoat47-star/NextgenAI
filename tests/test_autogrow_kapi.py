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

    def test_uretilen_cift_max_pairsi_asiyor(self):
        """Tersi durursa darbogaz tasinmis demektir: o zaman bu test
        KIRMIZI olur ve yeni darbogazin nerede oldugunu gosterir."""
        import train_llm

        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        uretilen = len(self._uretilen(10 ** 9))
        self.assertGreater(
            uretilen, train_llm.MAX_PAIRS,
            'uretilen cift (%s) MAX_PAIRS (%s) altinda: kirpma YOK, yani '
            'asil darbogaz degisti (belki MAX_PAIRS, belki intent kapi). '
            'Bu testi guncelle.'
            % (format(uretilen, ','), format(train_llm.MAX_PAIRS, ',')))

    def test_max_pairs_benzersiz_ctxin_cogu_kapsiyor(self):
        """MAX_PAIRS'in GERCEK islevi: benzersiz ctx'nin cogunu kapsamak.

        Olcum (28.09) 6.364 intent ile:
            MAX_PAIRS  benzersiz ctx  kapsama  benzersiz cevap
             70.000      31.848         %81,8     22.570  (%96,6)
            120.000      38.048         %97,7     23.359  (%99,97)
            162.975      38.954        %100,0     23.366  (%100,0)
        Esik %90: hem 120.000'i gecer hem de 70.000'e dusmeyi yakalar.
        """
        import train_llm

        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        tum = self._uretilen(10 ** 9)
        tum_ctx = set(c for c, _r in tum)
        secilen = self._uretilen(train_llm.MAX_PAIRS)
        secilen_ctx = set(c for c, _r in secilen)
        kapsama = 100.0 * len(secilen_ctx) / max(1, len(tum_ctx))
        self.assertGreater(
            kapsama, 90.0,
            'MAX_PAIRS=%s iken benzersiz ctx kapsamasi sadece %.1f: '
            'kirpma GERCEK kapi kaybetti. Olcumde 120.000 -> %%97,7, '
            '70.000 -> %%81,8 idi.'
            % (format(train_llm.MAX_PAIRS, ','), kapsama))

    def test_max_pairs_veri_butcesi_oldugu_belgeli(self):
        """MAX_PAIRS bilincli bir olcum karari; rastgele degil.
        Deger dusturulurse epoch kisalir ama veri kaybolur; sessizce
        kaldirilmasin. 120.000'in gerekcesi train_llm.py yorumunda."""
        import train_llm

        self.assertGreaterEqual(
            train_llm.MAX_PAIRS, 70000,
            'MAX_PAIRS 70.000 altina dustu: epoch suresi icin veri '
            'kirpiliyor. Kalici veri butcesi; bilincli dusurulmelidir.')


if __name__ == '__main__':
    unittest.main()

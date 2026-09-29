# -*- coding: utf-8 -*-
"""tools/ altindaki olcum araclari saglam kalmali.

Bu araclar karar araclari: hangi modelin uretimi secilecegini, hangi
kapi sartinin duzeltilecegini ve TOKEN_PER_PAIR gibi sabitlerin gercek
degerlerini belirliyorlar. Bozuk/rotur bir arac sessizce yanlis karar
verdigi icin (olcum kodu calisir ama baska seyi olcer) derlenme ve
belgelenme denetimi sart.

29.09'da bu araclar %TEMP%\\opencode altinda yasidi; o dizin gecici
oldugu icin kayboldu ve yeni olcum icin bastan yazilmak zorunda kaldi.
Simdi repoda, ve asagidaki kurallar kilitli.
"""
import io
import os
import py_compile
import re
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

ARACLAR = os.path.join(BASE, 'tools')
VARSAYILAN = ('uretim_olc.py', 'uretim_karsilastir.py', 'kapi_ab.py',
              'kendi_cumlesi.py', 'token_olc.py', 'kalite_olc.py')


def _kaynak(ad):
    with io.open(os.path.join(ARACLAR, ad), encoding='utf-8') as f:
        return f.read()


class TestAraclarMevcut(unittest.TestCase):

    def test_beklenen_araclar_yerinde(self):
        """Belgelenen araclarin hepsi repoda olmali; %TEMP%'e kaybolmamalari
        29.09'da gercekten oldu."""
        for ad in VARSAYILAN:
            self.assertTrue(os.path.exists(os.path.join(ARACLAR, ad)),
                            'eksik arac: %s' % ad)

    def test_readme_her_araci_anlatiyor(self):
        with io.open(os.path.join(ARACLAR, 'README.md'), encoding='utf-8') as f:
            readme = f.read()
        for ad in VARSAYILAN:
            self.assertIn(ad, readme,
                          '%s README\'de anlatilmiyor' % ad)


class TestAraclarDerleniyor(unittest.TestCase):

    def test_hepsi_derleniyor(self):
        """Sozde-dilbilgisi hatasi sessizce araci calisamaz hale getirir;
        komut satirindan hic cagrilmasa bile."""
        tmp = tempfile.mkdtemp(prefix='derle_')
        try:
            for ad in VARSAYILAN:
                try:
                    py_compile.compile(os.path.join(ARACLAR, ad),
                                       cfile=os.path.join(tmp, ad + 'c'),
                                       doraise=True)
                except py_compile.PyCompileError as e:
                    self.fail('%s derlenmiyor: %s' % (ad, e))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestAraclarKendiniBelgeliyor(unittest.TestCase):

    def test_her_aracin_docstring_i_var(self):
        """Docstring'i olmayan arac: ne olctugunu bilmeden cagrilir.
        Sonuc ciktisinda ne oldugu ancak olculdugunde gorunur."""
        for ad in VARSAYILAN:
            kaynak = _kaynak(ad)
            self.assertTrue(kaynak.lstrip().startswith('#'),
                            '%s basligi yok' % ad)
            ilk = kaynak.lstrip().splitlines()
            # docstring: ilk yorum satirindan sonra gelen """...""" blogu
            govde = '\n'.join(ilk[1:])
            self.assertIn('"""', govde.split('\n\n')[0],
                          '%s docstring\'i yok (ne olctugu belirsiz)' % ad)

    def test_yol_kendi_konumundan_turetiliyor(self):
        """GitHub Actions'ta repo yolu baska; sabit yol calismaz.
        (test_no_hardcoded_paths yolun YASAK oldugunu denetliyor, bu
        yolun NASIL turetilmeli oldugunu.)"""
        for ad in VARSAYILAN:
            kaynak = _kaynak(ad)
            if '__file__' not in kaynak:
                self.fail('%s yolu __file__\'dan turetmiyor' % ad)


class TestOlcumRaporlariGitignore(unittest.TestCase):
    """Raporlar repoya girMEMELI: dev JSON uretim verisi, gecici olcum."""

    def test_rapor_dizini_ignored(self):
        with io.open(os.path.join(BASE, '.gitignore'), encoding='utf-8') as f:
            satirlar = [s.strip() for s in f]
        self.assertTrue(any(re.match(r'olcum_raporlari/?$', s)
                            for s in satirlar),
                        'olcum_raporlari/ .gitignore\'da degil; olcum '
                        'raporlari repoya girecek')

    def test_araclar_bu_dizini_kullanir(self):
        """Cikti yolu bu dizine isaret etmeli; eskiden %TEMP%'e gidiyordu
        ve raporlar kayboluyordu."""
        kaynak = _kaynak('uretim_olc.py')
        self.assertIn('olcum_raporlari', kaynak)
        self.assertNotIn("os.environ.get('TEMP'", kaynak,
                         'cikti hala gecici dizine yaziliyor')


if __name__ == '__main__':
    unittest.main()

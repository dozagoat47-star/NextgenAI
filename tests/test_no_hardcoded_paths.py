# -*- coding: utf-8 -*-
"""Depoda SABIT MUTLAK YOL olmamali (tasima testi).

Gercek olay: test_train_llm.py icinde kaynak dosya yolu gelistiricinin
kendi Windows makinesine sabitlenmisti. GitHub Actions Linux'ta
calistigi icin o yol yoktu ve test HER KOSUDA FileNotFoundError ile
dustu:

    ERROR: test_seqgen_uses_response_budget_not_literal_70
    FileNotFoundError: [Errno 2] No such file or directory:
        '...<Desktop>\\<Nextgen_API>\\seqgen.py'

Sonuc: CI 26.09'dan beri her push'ta kirmizidi ve telefona "Run failed"
bildirimi gidiyordu. Hatayi 1 gunden fazla kimse fark etmedi.

Kural: yol her zaman os.path.join(BASE, ...) ile tur edilir. BASE
test dosyalarinda zaten tanimli:

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Bu test ihlali build'de yakalar; boylece tek bir satir yazim hatasi
tum hattin sessizce kirmizi olmasina yol acmaz.

NOT: bu dosyanin KENDISI de taranir. Bu yuzden asagidaki ornek
yollar karakter birlesimiyle kurulur; kaynakta hazir bir mutlak yol
bulunmaz (aksi halde test kendini ihbar ederdi).
"""
import io
import os
import re
import shutil
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Surulen dosya turleri. Veri dosyalari (.json/.jsonl) disarida: yol
# taramasi icerik metni oldugu icin yanlis pozitif uretir.
TURLER = ('.py', '.yml', '.yaml', '.sh')

# Atlanan dizinler: .git, model (egitim ciktilari), sanal ortamlar.
ATLANAN = ('.git', '__pycache__', 'model', '.venv', 'venv', 'node_modules')

BS = chr(92)          # tek ters eğik çizgi, kaynanda gomulu kalmamasi icin

# Windows surucu yolu. Iki sart birden:
#   1) harften hemen once baska harf/rakam OLMAZ. Aksi halde Turkce
#      metinlerde "...dustu:\n" gibi kacaktaylar yanlis pozitif uretir
#      (ilk denemede test kendi mesajini ihbar etti).
#   2) iki noktadan sonra AYRIK veya ters egik gelir.
# DIKKAT: sinifta TERS EGIK iki kez yazilir ("[\\/]"). Tek kez
# yazilsa ("[\/]") Python'da '/' kaçisi sayilir ve sinif yalnizca '/'
# ile eslesir; gercek Windows yollari sessizce kacirilir.
SURUCU = re.compile('(?<![A-Za-z0-9_])[A-Za-z]:[' + BS + BS + '/]')
# POSIX kullanicinin ev dizini. Onceki karakter harf/rakam olmamali:
# boylece "https://x.com/home/kullanici/" gibi bir adres eslesmez.
EV = re.compile('(?<![A-Za-z0-9_/:])/(?:home|Users)/[A-Za-z0-9._-]+/')

# Bu dosyada gosterim amacli kurulan ornekler (kaynak temiz kalsin).
ORNEK_SURUCU = 'D:' + BS + 'kullanici' + BS + 'masaustu' + BS + 'dosya.py'
ORNEK_EV = '/' + 'home' + BS.replace(BS, '/') + 'kullanici' + BS.replace(BS, '/') + 'proje'
ORNEK_TEMIZ = 'seqgen.py'


def _dosyalar():
    for kok, dizinler, adlar in os.walk(BASE):
        dizinler[:] = [d for d in dizinler if d not in ATLANAN]
        for ad in adlar:
            if ad.endswith(TURLER):
                yield os.path.join(kok, ad)


class TestNoHardcodedAbsolutePath(unittest.TestCase):
    """Kimse kendi makinesinin yolunu repoya yazmamali."""

    def _tara(self):
        ihlaller = []
        for yol in _dosyalar():
            try:
                with io.open(yol, encoding='utf-8') as f:
                    satirlar = f.read().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            rel = os.path.relpath(yol, BASE)
            for no, satir in enumerate(satirlar, 1):
                if SURUCU.search(satir) or EV.search(satir):
                    ihlaller.append('%s:%d  %s' % (rel, no, satir.strip()[:90]))
        return ihlaller

    def test_hardcoded_absolute_path_yok(self):
        ihlaller = self._tara()
        self.assertEqual(
            ihlaller, [],
            'Depoda sabit mutlak yol var. GitHub Actions Linux\'ta '
            'calistigi icin bu yollar orada YOK ve testler dustu:\n  '
            + '\n  '.join(ihlaller))

    def test_tarama_gercekten_calisiyor(self):
        """Tarayici boz olmamali: ornek yollari yakalamali.

        Aksi halde yukaridaki test sessizce yesil kalir ve ayni hatayi
        bir daha yakalayamaz. Hem pozitif hem negatif dengeyi kurar.
        """
        self.assertTrue(SURUCU.search(ORNEK_SURUCU),
                        'surucu yolu deseni calismiyor')
        self.assertTrue(EV.search(ORNEK_EV),
                        'ev dizini deseni calismiyor')
        self.assertIsNone(SURUCU.search(ORNEK_TEMIZ),
                          'goreli yol yanlis pozitif uretiyor')
        self.assertIsNone(EV.search(ORNEK_TEMIZ),
                          'goreli yol yanlis pozitif uretiyor')

        # Regresyon: Turkce metinde iki nokta + satir sonu gecerli
        # yoldur sanilip ihlal sayilmamali.
        self.assertIsNone(SURUCU.search('olustugu icin testler dustu:' + BS + 'n'),
                          'iki nokta + satir sonu yanlis pozitif uretiyor')
        # Adres icindeki /home/ ihlal sayilmamali.
        self.assertIsNone(
            EV.search('https://ornek.com/home/kullanici/sayfa'),
            'adres icindeki yol yanlis pozitif uretiyor')

        # ucuncu bir dosyaya yazip tarayiciya gercekten girdirilir
        tmp = tempfile.mkdtemp(prefix='yoldenet_')
        try:
            hedef = os.path.join(tmp, 'ornek.py')
            with io.open(hedef, 'w', encoding='utf-8') as f:
                f.write('# gecici\n')
                f.write('p = "%s"\n' % ORNEK_SURUCU)
            with io.open(hedef, encoding='utf-8') as f:
                bulunan = [s for s in f.read().splitlines()
                           if SURUCU.search(s)]
            self.assertEqual(len(bulunan), 1,
                             'dosya ici tarama yolu calismiyor')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

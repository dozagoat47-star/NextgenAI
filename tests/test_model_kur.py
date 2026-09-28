"""model_kur.py guvenli model kurulumu icin regresyon testleri.

27.09 modeli, model/ uzerine yedek alinmadan yazildigi icin kayboldu. bu
testler yedekle -> dogrula -> atomik degistir -> dogrula zincirini kilitler.

ONEMLI: hicbir test gercek model/ klasorune veya veri dosyalarina dokunmaz;
hepsi gecici dizinde calisir (HEDEF_JSON / YEDEK_KOK yerellestirilir).
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import model_kur
from model_kur import DogrulamaHatasi


def _json_oku(yol):
    with io.open(yol, encoding='utf-8') as f:
        return json.load(f)


def _json_yaz(yol, d):
    with io.open(yol, 'w', encoding='utf-8') as f:
        json.dump(d, f)


def _bayt(yol):
    with open(yol, 'rb') as f:
        return f.read()


def _sahte_model(yol, d_model=384, num_blocks=6, heads=8, V=400,
                 max_ctx=48, nanli=False, blok_atla=None, kirik_npz=False,
                 yuklenebilir=True):
    """Kucuk ama GERCEKTEN yuklenebilen bir model klasoru uretir.

    (json_yolu, agirlik_yolu) doner. yuklenebilir=False ise yalnizca yapı
    dogrulaması icin kullanilir (llm.load_llm'in acamayacagi eksik model).
    """
    os.makedirs(yol, exist_ok=True)
    d = {'arch': 'llm', 'd_model': d_model, 'num_blocks': num_blocks,
         'num_heads': heads, 'V': V, 'ff_mult': 4, 'max_ctx_len': max_ctx,
         'max_seq_len': 256, 'tied_embeddings': True,
         'weights_file': 'llm_model_weights.npz'}
    w = {'head_b': np.zeros((1, V), np.float32),
         'pos_enc': np.zeros((256, d_model), np.float32)}
    if yuklenebilir:
        d['vocab'] = ['<PAD>', '<BOS>', '<SEP>', '<EOS>'] + \
                     [chr(ord('a') + (i % 26)) + str(i) for i in range(V - 4)]
        w['embed'] = np.zeros((V, d_model), np.float32)
        w['out_ln_g'] = np.zeros((1, d_model), np.float32)
        w['out_ln_b'] = np.zeros((1, d_model), np.float32)
    bloklar = range(num_blocks)
    if blok_atla is not None:
        bloklar = [b for b in bloklar if b != blok_atla]
    for b in bloklar:
        w['b%d_Wq' % b] = np.zeros((d_model, d_model), np.float32)
        w['b%d_Wk' % b] = np.zeros((d_model, d_model), np.float32)
        w['b%d_Wv' % b] = np.zeros((d_model, d_model), np.float32)
        w['b%d_Wo' % b] = np.zeros((d_model, d_model), np.float32)
        w['b%d_W1' % b] = np.zeros((d_model, d_model * 4), np.float32)
        w['b%d_W2' % b] = np.zeros((d_model * 4, d_model), np.float32)
        w['b%d_bq' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_bk' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_bv' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_bo' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_b1' % b] = np.zeros((1, d_model * 4), np.float32)
        w['b%d_b2' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_ln1_g' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_ln1_b' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_ln2_g' % b] = np.zeros((1, d_model), np.float32)
        w['b%d_ln2_b' % b] = np.zeros((1, d_model), np.float32)
    if nanli:
        w['head_b'][0, 0] = np.nan
    agirlik = os.path.join(yol, 'llm_model_weights.npz')
    if kirik_npz:
        with open(agirlik, 'wb') as f:
            f.write(b'PK\x03\x04 bozuk veri')
    else:
        np.savez(agirlik, **w)
    js = os.path.join(yol, 'llm_model.json')
    with io.open(js, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    return js, agirlik


class _Gecici(unittest.TestCase):
    """Hedef model klasorunu gecici dizine baglar; test sonunda temizler."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='modelkur_')
        self.hedef = os.path.join(self.tmp, 'model')
        os.makedirs(self.hedef)
        self.hedef_json = os.path.join(self.hedef, 'llm_model.json')
        self.yedek_kok = os.path.join(self.hedef, 'yedek')
        self._eski = (model_kur.HEDEF_JSON, model_kur.YEDEK_KOK)
        model_kur.HEDEF_JSON = self.hedef_json
        model_kur.YEDEK_KOK = self.yedek_kok

    def tearDown(self):
        model_kur.HEDEF_JSON, model_kur.YEDEK_KOK = self._eski
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _yerlestir(self, **kw):
        return _sahte_model(os.path.join(self.tmp, 'kaynak'), **kw)


class TestDogrulama(_Gecici):
    def test_gecerli_model_gecer(self):
        js, _ = self._yerlestir()
        ozet = model_kur.dogrula(js)
        self.assertEqual(ozet['d_model'], 384)
        self.assertEqual(ozet['num_blocks'], 6)
        self.assertEqual(ozet['V'], 400)
        self.assertGreater(ozet['parametre'], 0)

    def test_eksik_anahtar_reddedilir(self):
        js, _ = self._yerlestir()
        d = _json_oku(js)
        del d['num_blocks']
        _json_yaz(js, d)
        with self.assertRaises(DogrulamaHatasi) as c:
            model_kur.dogrula(js)
        self.assertIn('num_blocks', str(c.exception))

    def test_yanlis_arch_reddedilir(self):
        js, _ = self._yerlestir()
        d = _json_oku(js)
        d['arch'] = 'bot'
        _json_yaz(js, d)
        with self.assertRaises(DogrulamaHatasi):
            model_kur.dogrula(js)

    def test_agirlik_dosyasi_yoksa_reddedilir(self):
        js, agirlik = self._yerlestir()
        os.remove(agirlik)
        with self.assertRaises(DogrulamaHatasi) as c:
            model_kur.dogrula(js)
        self.assertIn('agirlik dosyasi yok', str(c.exception))

    def test_nan_icolen_model_reddedilir(self):
        js, _ = self._yerlestir(nanli=True)
        with self.assertRaises(DogrulamaHatasi) as c:
            model_kur.dogrula(js)
        self.assertIn('NaN/Inf', str(c.exception))

    def test_kirik_npz_reddedilir(self):
        js, _ = self._yerlestir(kirik_npz=True)
        with self.assertRaises(DogrulamaHatasi):
            model_kur.dogrula(js)

    def test_eksik_blok_reddedilir(self):
        """json 6 blok diyor ama npz'de 5 blok var = sessiz bozulma."""
        js, _ = self._yerlestir(blok_atla=3)
        with self.assertRaises(DogrulamaHatasi) as c:
            model_kur.dogrula(js)
        self.assertIn('blok sayisi tutarsiz', str(c.exception))

    def test_vocab_ucusmasi_reddedilir(self):
        js, _ = self._yerlestir()
        d = _json_oku(js)
        d['V'] = 16001
        _json_yaz(js, d)
        with self.assertRaises(DogrulamaHatasi) as c:
            model_kur.dogrula(js)
        self.assertIn('head_b', str(c.exception))


class TestKucultmeYasagi(_Gecici):
    def test_d_model_kucultulemez(self):
        js, _ = self._yerlestir(d_model=256)
        with self.assertRaises(DogrulamaHatasi) as c:
            model_kur.kucultme_kontrolu({'d_model': 256, 'num_blocks': 6,
                                         'V': 16000},
                                        {'d_model': 384, 'num_blocks': 6,
                                         'V': 16000})
        self.assertIn('kucultulemez', str(c.exception))

    def test_blok_kucultulemez(self):
        with self.assertRaises(DogrulamaHatasi):
            model_kur.kucultme_kontrolu({'d_model': 384, 'num_blocks': 4,
                                         'V': 16000},
                                        {'d_model': 384, 'num_blocks': 6,
                                         'V': 16000})

    def test_vocab_kucultulemez(self):
        with self.assertRaises(DogrulamaHatasi):
            model_kur.kucultme_kontrolu({'d_model': 384, 'num_blocks': 6,
                                         'V': 8000},
                                        {'d_model': 384, 'num_blocks': 6,
                                         'V': 16000})

    def test_ayni_mimari_gecer(self):
        model_kur.kucultme_kontrolu({'d_model': 384, 'num_blocks': 6,
                                     'V': 16000},
                                    {'d_model': 384, 'num_blocks': 6,
                                     'V': 16000})

    def test_yerinde_model_yoksa_gecer(self):
        model_kur.kucultme_kontrolu({'d_model': 384, 'num_blocks': 6,
                                     'V': 16000}, None)


class TestYedekleme(_Gecici):
    def test_yedek_her_iki_dosyayi_alir(self):
        js, _ = _sahte_model(self.hedef)
        yol = model_kur.yedekle(self.hedef_json, '20260929_1801')
        self.assertTrue(os.path.isdir(yol))
        icerik = os.listdir(yol)
        self.assertIn('llm_model.json', icerik)
        self.assertIn('llm_model_weights.npz', icerik)
        # yedek gercekten ayni icerik
        self.assertEqual(
            os.path.getsize(os.path.join(yol, 'llm_model_weights.npz')),
            os.path.getsize(os.path.join(self.hedef,
                                         'llm_model_weights.npz')))

    def test_ayni_zaman_damgasi_ust_ustte_binmez(self):
        """Ayni ts ile iki kez calistirilirsa ikinci yedek mevcut modeli
        DEGIL birincinin yedegini yedeklemeli (aksi halde yedek zinciri
        kendini yer)."""
        js, _ = _sahte_model(self.hedef)
        ilk = model_kur.yedekle(self.hedef_json, '20260929_1801')
        ikinci = model_kur.yedekle(self.hedef_json, '20260929_1801')
        self.assertNotEqual(ilk, ikinci)
        self.assertTrue(os.path.basename(ikinci).startswith('20260929_1801_2'))

    def test_model_yoksa_yedek_alinmaz(self):
        self.assertIsNone(model_kur.yedekle(self.hedef_json, '20260929_1801'))


class TestKurulumGuvenligi(_Gecici):
    def test_kurulum_once_her_sey_dogrulanir(self):
        """ Bozuk kaynak: yedek almaz, hedefe dokunmaz."""
        js, _ = self._yerlestir(nanli=True)
        with self.assertRaises(DogrulamaHatasi):
            model_kur.kur(js, '20260929_1801')
        # hedef hâlâ dokunulmadi, yedek klasoru de olusmadi
        self.assertFalse(os.path.exists(self.hedef_json))
        self.assertFalse(os.path.exists(self.yedek_kok))

    def test_basarisiz_kurulumda_hedefe_dokunulmaz(self):
        js, _ = self._yerlestir(max_ctx=48)          # gecerli kaynak
        # yerinde FARKLI bir model var (max_ctx=64) -> kurulum degistirmeli
        _sahte_model(self.hedef, max_ctx=64)
        eski = _json_oku(self.hedef_json)
        self.assertEqual(eski['max_ctx_len'], 64)
        sonra, yedek = model_kur.kur(js, '20260929_1801')
        self.assertTrue(os.path.isdir(yedek))
        self.assertIn('llm_model.json', os.listdir(yedek))
        # yedek ESKI modeli tasimali (64), hedef YENI olmali (48)
        yedekli = _json_oku(os.path.join(yedek, 'llm_model.json'))
        self.assertEqual(yedekli['max_ctx_len'], 64)
        self.assertEqual(sonra['arch']['max_ctx_len'], 48)

    def test_yerinde_model_yoksa_kurulur(self):
        js, _ = self._yerlestir()
        sonra, yedek = model_kur.kur(js, '20260929_1801')
        self.assertIsNone(yedek)                     # yedeklenecek bir sey yoktu
        self.assertTrue(os.path.exists(self.hedef_json))
        self.assertTrue(os.path.exists(os.path.join(
            self.hedef, 'llm_model_weights.npz')))
        self.assertEqual(sonra['d_model'], 384)

    def test_kurulum_sonrasi_agirlik_adi_hedefle_uyusur(self):
        js, _ = self._yerlestir()
        model_kur.kur(js, '20260929_1801')
        d = _json_oku(self.hedef_json)
        self.assertTrue(os.path.exists(os.path.join(
            self.hedef, d['weights_file'])))

    def test_gecici_klasor_birakilmaz(self):
        js, _ = self._yerlestir()
        model_kur.kur(js, '20260929_1801')
        artik = [d for d in os.listdir(self.hedef) if d.startswith('.kur_')]
        self.assertEqual(artik, [])

    def test_no_backup_yalnizca_istediginde(self):
        js, _ = self._yerlestir()
        _sahte_model(self.hedef)
        model_kur.kur(js, '20260929_1801', yedekle_var=False)
        self.assertFalse(os.path.exists(self.yedek_kok))


class TestGirisNoktasi(_Gecici):
    def test_kaynak_eksikse_kod_1(self):
        import contextlib
        import io as _io
        buf = _io.StringIO()
        sys_argv = sys.argv
        try:
            sys.argv = ['model_kur.py', os.path.join(self.tmp, 'yok.json')]
            with contextlib.redirect_stdout(buf):
                kod = model_kur.main()
        finally:
            sys.argv = sys_argv
        self.assertEqual(kod, 1)
        self.assertIn('hicbir dosya degistirilmedi', buf.getvalue())

    def test_check_modu_yazmaz(self):
        import contextlib
        import io as _io
        js, _ = self._yerlestir()
        _sahte_model(self.hedef)
        once = _bayt(self.hedef_json)
        buf = _io.StringIO()
        sys_argv = sys.argv
        try:
            sys.argv = ['model_kur.py', js, '--check']
            with contextlib.redirect_stdout(buf):
                kod = model_kur.main()
        finally:
            sys.argv = sys_argv
        self.assertEqual(kod, 0)
        self.assertIn('hicbir sey yazilmadi', buf.getvalue())
        self.assertEqual(_bayt(self.hedef_json), once)
        self.assertFalse(os.path.exists(self.yedek_kok))


class TestGercekModelVarsi(_Gecici):
    """Repoda gercek egitilmis model varsa dogrulama onu okuyabilmeli.
    Model yoksa test atlanir (temiz clone'da CI kirmaz)."""

    def test_model_llm_model_json_dogrulanir(self):
        gercek = os.path.join(BASE, 'model', 'llm_model.json')
        if not os.path.exists(gercek):
            self.skipTest('egitilmis model yok')
        ozet = model_kur.dogrula(gercek)
        self.assertEqual(ozet['arch']['arch'], 'llm')
        self.assertGreater(ozet['d_model'], 0)
        self.assertGreater(ozet['parametre'], 0)


if __name__ == '__main__':
    unittest.main()

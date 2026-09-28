# -*- coding: utf-8 -*-
"""NOBETCI: test kosusu veri dosyalarina DOKUNMAMALI.

Yakalandi: 28.09'da bir tam test kosusundan sonra corpus.jsonl 'M'
cikmisti (+1 satir: 'Sirbistan Futbol Federasyonu'). O madde intents.json'da
vardi, corpus'ta yoktu; bir test yolu bu farki gercekten kapatti. Yani
yazim 'bozuk' degildi, ama TEST KOSUSUNUN veri dosyasini degistirmesi
kabul edilemez: kimse ne eklendigini bilmez, sonraki push'a girer.

Bu dosya iki sey korur:
  1) DORT veri dosyasi git HEAD ile birebir ayni mi (en yetkili olcum).
  2) Modulun icinde kendi anlik durumunu degistiren test var mi.

Not: git HEAD'e gore olcer, bu yuzden bu testin KOSULDUGU andaki
dosya durumu degisse bile yakalar. Test sirasi onemli degildir; yalnizca
bu modulun kendisinden SONRA olan testler icin (1) zaten elenmis olur.
"""
import io
import json
import os
import subprocess
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Elle dokunulmayacak dosyalar (proje kurali)
VERI = ['corpus.jsonl', 'corpus_ids.jsonl', 'intents.json',
        'knowledge_map.jsonl']

# Modul yuklenirken anlik durum (2. koruma)
_ANLIK = {}
for _d in VERI:
    _p = os.path.join(BASE, _d)
    _ANLIK[_d] = os.path.getsize(_p) if os.path.exists(_p) else -1


def _git_diff(dosyalar):
    """Verilen dosyalarda calisma agaci git HEAD'den ayiriyor mu?"""
    try:
        p = subprocess.run(['git', 'diff', '--stat', '--'] + dosyalar,
                           cwd=BASE, capture_output=True, timeout=60)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.decode('utf-8', 'replace').strip()


def _ozet(yol):
    import hashlib
    if not os.path.exists(yol):
        return 'yok'
    h = hashlib.md5()
    with open(yol, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:12]


class TestVeriDosyalariDokunulmaz(unittest.TestCase):

    def test_veri_dosyalari_git_head_ile_ayni(self):
        """calisma agacinda bu dort dosya kirli mi? (yetkili olcum)"""
        fark = _git_diff(VERI)
        if fark is None:
            self.skipTest('git erisilemiyor, HEAD karsilastirmasi yapilamadi')
        self.assertEqual(
            fark, '',
            'TEST KOSUSU VERI DOSYASINI DEGISTIRDI. Elle duzelt:\n'
            '    git checkout -- %s\n%s' % (' '.join(VERI), fark))

    def test_bu_modul_icin_degisiklik_yok(self):
        """Bu modulun kendi testleri veri dosyasini degistirmemeli."""
        simdi = {d: (os.path.getsize(os.path.join(BASE, d))
                     if os.path.exists(os.path.join(BASE, d)) else -1)
                 for d in VERI}
        self.assertEqual(simdi, _ANLIK,
                         'bu test dosyalari degistirdi:\n%s' %
                         {d: (_ANLIK[d], simdi[d]) for d in VERI
                          if _ANLIK[d] != simdi[d]})

    def test_ozet_dosyalari_yine_yerinde(self):
        """Dosyalar SILINMEMIS olmali (bazi testler temizlik yapmis olabilir)."""
        for d in VERI:
            self.assertTrue(os.path.exists(os.path.join(BASE, d)),
                            '%s yok! test bir dosyayi silmis olabilir' % d)


class TestVeriDosyalariSema(unittest.TestCase):
    """Verinin yapisi saglam mi (kirlilik bir sonraki kosuda yakalansin)."""

    def test_corpus_satirlari_json_gecerli(self):
        yol = os.path.join(BASE, 'corpus.jsonl')
        if not os.path.exists(yol):
            self.skipTest('corpus.jsonl yok')
        boz = 0
        with io.open(yol, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except ValueError:
                    boz += 1
                    continue
                if not (j.get('id') and j.get('text')):
                    boz += 1
        self.assertEqual(boz, 0, 'corpus.jsonl\'ta %d boz satir var' % boz)

    def test_intents_etiketleri_latin(self):
        """Kirli etiket kapisi: sagdan soldan degil, testten de gecsin."""
        yol = os.path.join(BASE, 'intents.json')
        if not os.path.exists(yol):
            self.skipTest('intents.json yok')
        with io.open(yol, encoding='utf-8') as f:
            veri = json.load(f)
        etiketler = []
        if isinstance(veri, dict):
            for ad, govde in veri.items():
                if isinstance(govde, dict):
                    for t in (govde.get('tags') or []):
                        if isinstance(t, str):
                            etiketler.append(t)
        boz = sorted({t for t in etiketler
                      if any(c.isalpha() and ord(c) > 127 for c in t)})
        self.assertEqual(
            boz, [],
            'ASCII-disi harfli %d etiket var (ornek: %s). Uretici tarafi '
            'kapida elemesi gerekiyor: autogrow.build_intent / '
            'scrape_intents.merge_intents' % (len(boz), boz[:5]))


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""MODEL A/B: iki modeli AYNI veriyle uretim olcumuyle karsilastirir.

Kullanim:
    python tools/model_ab.py <yeni_model.json> <etiket>
    python tools/model_ab.py <yeni_model.json> <etiket> --rapor eski.json

NEDEN VAR:
01.10'da yeni egitim val loss'u %4,4 iyilestirdi ama 50 sabit soruda
ekrana cikan metin %72 -> %52 dustu. Bu tur arac, "hangi modeli
kuralim" sorusunu GOZLE degil OLCMEYLE cevaplar.

KURALLAR (proje kurali: tahmin yapma, olc):
  * Iki model ayni veriyle, ayni soru listesiyle, ayni seed ile olculur.
    Aradaki tek fark agirliklardir.
  * Veri dosyalari arac tarafindan ASLA yazilmaz.
  * Ayni model iki kez olculurse sonuc birebir ayni olmalidir
    (determinizm kontrolu); `--rapor` verilirse bu kontrol yapilir.
  * Arac modeli kurmaz; kurulumu model_kur.py yapar (yedekler).
"""
from __future__ import print_function

import io
import json
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KURULU_JSON = os.path.join(BASE, 'model', 'llm_model.json')
KURULU_NPZ = os.path.join(BASE, 'model', 'llm_model_weights.npz')
RAPOR_DIZIN = os.path.join(BASE, 'olcum_raporlari')


def turkce(s):
    return s


def veri_damgasi():
    """URETIMI ETKILEYEN veri dosyalarinin (boyut, mtime) izi.

    Amac: iki olcumun ayni veriyle yapildigini KANITLAMAK. Raporun icine
    yazilir; sonradan "veri degisti mi" sorusu olcumden cevaplanir.
    """
    desen = ('intents.json', 'knowledge_map.jsonl', 'corpus.jsonl',
             'corpus_ids.jsonl')
    iz = {}
    for ad in os.listdir(BASE):
        if ad in desen or (ad.startswith('chatgrow_')
                           and ad.endswith('.jsonl')):
            y = os.path.join(BASE, ad)
            if os.path.isfile(y):
                iz[ad] = [os.path.getsize(y), int(os.path.getmtime(y))]
    return iz


def model_damgasi():
    """Kurulu modelin (boyut, mtime) izi."""
    d = {}
    for y in (KURULU_JSON, KURULU_NPZ):
        if os.path.exists(y):
            d[os.path.basename(y)] = [os.path.getsize(y),
                                      int(os.path.getmtime(y))]
    return d


def ozet(yol):
    """uretim raporundan tek satirlik ozet."""
    d = json.load(io.open(yol, encoding='utf-8'))
    k = d['kayitlar']
    u = [r for r in k if r['uretilildi']]
    sd = [100 * r['sadakat'] for r in u if isinstance(r.get('sadakat'), float)]
    kab = sum(1 for r in k for x in r['denemeler'] if x['kabul'])
    top = sum(len(r['denemeler']) for r in k)
    return {
        'etiket': d.get('etiket'),
        'seed': d.get('seed'),
        'soru': len(k),
        'ekrana_uretim': len(u),
        'ekrana_uretim_yuzde': 100.0 * len(u) / max(1, len(k)),
        'sadakat_yuzde': (sum(sd) / len(sd)) if sd else None,
        'sadakat_n': len(sd),
        'deneme_kabul': kab,
        'deneme_toplam': top,
        'kabul_yuzde': 100.0 * kab / max(1, top),
        'veri_damgasi': d.get('veri_damgasi'),
        'model_damgasi': d.get('model_damgasi'),
    }


def kosu(etiket):
    """uretim_olc.py'yi bir etiketle calistirir; rapor yolunu dondurur."""
    cikti = os.path.join(RAPOR_DIZIN, 'uretim_%s.json' % etiket)
    print('--- olcum: %s' % etiket)
    sys.stdout.flush()
    rc = subprocess.call(
        [sys.executable, os.path.join(BASE, 'tools', 'uretim_olc.py'),
         etiket, cikti],
        cwd=BASE)
    if rc != 0:
        raise SystemExit('uretim_olc.py basarisiz (rc=%d)' % rc)
    return cikti


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    kaynak = sys.argv[1]
    etiket = sys.argv[2]
    eski_rapor = None
    if '--rapor' in sys.argv:
        eski_rapor = sys.argv[sys.argv.index('--rapor') + 1]

    kaynak = os.path.abspath(kaynak)
    if not os.path.exists(kaynak):
        raise SystemExit('bulunamadi: %s' % kaynak)

    print('=' * 74)
    print('MODEL A/B')
    print('=' * 74)
    print('  aday model : %s' % kaynak)
    print('  etiket     : %s' % etiket)
    print()
    print('  DIKKAT: bu arac modeli KURMAZ. model_kur.py ile kurulmus olmali.')
    print('  Kurulu modeli DOGRULAMAK icin: python model_kur.py "%s" --check'
          % kaynak)
    print()

    veri_once = veri_damgasi()
    model_once = model_damgasi()

    # determinizm kontrolu: ayni model iki kez
    print('1) DETERMINIZM KONTROLU (ayni model, ayni veri, iki kosu)')
    print('   - bu adim olcum aracina iki tam kosu daha ekler.')
    print()
    d1 = kosu('%s_det1' % etiket)
    d2 = kosu('%s_det2' % etiket)
    a = json.load(io.open(d1, encoding='utf-8'))
    b = json.load(io.open(d2, encoding='utf-8'))
    farkli = 0
    alanlar = ['uretilildi', 'yanit', 'sadakat', 'kb', 'bos_kova']
    for x, y in zip(a['kayitlar'], b['kayitlar']):
        if x['soru'] != y['soru'] or any(x.get(f) != y.get(f)
                                         for f in alanlar):
            farkli += 1
    print('   deterministik: %s  (%d/%d soru farkli)'
          % ('EVET' if farkli == 0 else 'HAYIR', farkli, len(a['kayitlar'])))
    print()

    # asil olcum
    print('2) ASIL OLCUM')
    yeni_rapor = kosu(etiket)

    veri_son = veri_damgasi()
    print()
    print('3) VERI DEGISTI MI (iki kosu arasinda)?')
    degisen = [k for k in set(list(veri_once) + list(veri_son))
               if veri_once.get(k) != veri_son.get(k)]
    print('   degisen veri dosyasi: %d %s'
          % (len(degisen), degisen if degisen else ''))
    if degisen:
        print('   !! SONUC GECERSIZ: olcum sirasinda veri degisti.')
    print()

    print('=' * 74)
    print('SONUC')
    print('=' * 74)
    print('%-14s %8s %10s %10s %10s'
          % ('rapor', 'tohum', 'ekrana', 'sadakat', 'kabul'))
    raporlar = []
    if eski_rapor and os.path.exists(eski_rapor):
        raporlar.append(('ESKI', eski_rapor))
    raporlar.append(('ESKI(det)', d1))
    raporlar.append(('YENI', yeni_rapor))
    ozetler = {}
    for ad, y in raporlar:
        o = ozet(y)
        ozetler[ad] = o
        print('%-14s %8s %6d/%-3d %9s %9s'
              % (ad, o['seed'],
                 o['ekrana_uretim'], o['soru'],
                 ('%.1f' % o['sadakat_yuzde'])
                 if o['sadakat_yuzde'] is not None else '-',
                 '%d/%d' % (o['deneme_kabul'], o['deneme_toplam'])))
    print()

    if eski_rapor and 'ESKI' in ozetler:
        e = ozetler['ESKI']
        y = ozetler['YENI']
        print('%-14s %8s %6d/%-3d %9s %9s'
              % ('fark', '', y['ekrana_uretim'] - e['ekrana_uretim'],
                 '', '', '%+d' % (y['deneme_kabul'] - e['deneme_kabul'])))
        print()
        d = y['ekrana_uretim_yuzde'] - e['ekrana_uretim_yuzde']
        print('  ekrana uretim farki: %+.1f puan' % d)
        if d < 0:
            print('  -> YENI MODEL DAHA KOTU. model_kur.py ile ESKI geri alin.')
        elif d > 0:
            print('  -> YENI MODEL DAHA IYI. model_kur.py ile adayi kurun.')
        else:
            print('  -> fark yok.')
        print()
        print('  KARAR verisi: SADAKET ve val loss TEK BASINA yeterli DEGILDIR;')
        print('  ekrana cikan metin sayisi belirleyici olcut olmali.')

    print()
    print('raporlar: %s' % ', '.join(y for _, y in raporlar))
    print('veri izi: %s' % json.dumps(veri_son)[:200])


if __name__ == '__main__':
    main()

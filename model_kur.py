# -*- coding: utf-8 -*-
"""Guvenli model kurulumu: yedekle -> dogrula -> atomik degistir -> dogrula.

27.09 modeli bu yuzden kayboldu: model/ uzerine yazarken yedek alan bir adim
yoktu ve eski agirliklar tek kopyaydi. bu betik yazma oncesi ZORUNLU yedek alir
ve dogrulama basarisizsa hicbir seye dokunmadan durur.

kullanim:
    python model_kur.py model/yeni/llm_model.json          # kur + yedekle
    python model_kur.py model/yeni/llm_model.json --check   # sadece dogrula
    python model_kur.py model/yeni/llm_model.json --ts 20260929_1801

model/ klasoru .gitignore'dadir; yedekler model/yedek/<ts>/ altinda kalir.
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
HEDEF_JSON = os.path.join(BASE, 'model', 'llm_model.json')
YEDEK_KOK = os.path.join(BASE, 'model', 'yedek')

# mimariyi tanimlayan zorunlu json anahtarlari
ZORUNLU = ('arch', 'd_model', 'num_blocks', 'num_heads', 'V', 'weights_file')
# agirlik dosyasinda bulunmasi beklenen tanilayici diziler
BLOK_ONEK = re.compile(r'^b(\d+)_')


class DogrulamaHatasi(Exception):
    """Kurulum oncesi/sonrasi dogrulama basarisiz. hicbir sey degistirilmez."""


def _oku_json(yol):
    try:
        with io.open(yol, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise DogrulamaHatasi('json okunamadi %s: %s' % (yol, e))


def dogrula(yol):
    """Kurulacak modeli tam dogrular; OZET dondurur, hata halinde exception.

    Kontroller:
      1. dosya var ve json okunuyor
      2. arch == 'llm'
      3. zorunlu mimari anahtarlari dolu
      4. weights_file isaret etti dosya mevcut ve bos degil
      5. npz aciliyor, diziler bos degil, NaN/Inf icermiyor
      6. blok sayisi json ile npz blok indeksleri tutarli
      7. bas ve konum dizileri beklenen sekilde
    """
    if not os.path.exists(yol):
        raise DogrulamaHatasi('kaynak bulunamadi: %s' % yol)
    d = _oku_json(yol)
    if d.get('arch') != 'llm':
        raise DogrulamaHatasi("arch 'llm' degil: %r" % d.get('arch'))
    eksik = [k for k in ZORUNLU if not d.get(k)]
    if eksik:
        raise DogrulamaHatasi('json eksik anahtar: %s' % ', '.join(eksik))

    agirlik = os.path.join(os.path.dirname(os.path.abspath(yol)),
                           d['weights_file'])
    if not os.path.exists(agirlik):
        raise DogrulamaHatasi('agirlik dosyasi yok: %s' % agirlik)
    if os.path.getsize(agirlik) < 1024:
        raise DogrulamaHatasi('agirlik dosyasi bozuk kucuk: %d bayt'
                              % os.path.getsize(agirlik))

    V, dm, nb = int(d['V']), int(d['d_model']), int(d['num_blocks'])
    try:
        w = dict(np.load(agirlik))
    except Exception as e:
        raise DogrulamaHatasi('npz acilamadi: %s' % e)
    if not w:
        raise DogrulamaHatasi('npz bos (0 dizi)')
    kotu = [k for k, v in w.items()
            if v.dtype.kind == 'f' and not np.isfinite(v).all()]
    if kotu:
        raise DogrulamaHatasi('NaN/Inf iceren diziler: %s'
                              % ', '.join(sorted(kotu)[:5]))

    indeksler = sorted({int(m.group(1)) for k in w
                        if (m := BLOK_ONEK.match(k))})
    if indeksler != list(range(nb)):
        raise DogrulamaHatasi(
            'blok sayisi tutarsiz: json %d, npz bloklari %s'
            % (nb, indeksler))

    hb = w.get('head_b')
    if hb is not None and tuple(hb.shape) != (1, V):
        raise DogrulamaHatasi('head_b sekli %s, beklenen (1, %d)'
                              % (tuple(hb.shape), V))
    pe = w.get('pos_enc')
    if pe is not None and tuple(pe.shape)[1:] != (dm,):
        raise DogrulamaHatasi('pos_enc sekli %s, d_model %d ile uyusmuyor'
                              % (tuple(pe.shape), dm))

    n = sum(v.size for v in w.values())
    return {'arch': d, 'agirlik': agirlik, 'agirlik_bayt': os.path.getsize(agirlik),
            'parametre': n, 'dizisayisi': len(w), 'd_model': dm,
            'num_blocks': nb, 'V': V}


def _mevcut_ozet():
    """Yerinde duran modelin mimari ozeti (yoksa None)."""
    if not os.path.exists(HEDEF_JSON):
        return None
    try:
        d = _oku_json(HEDEF_JSON)
    except DogrulamaHatasi:
        return None
    if d.get('arch') != 'llm':
        return None
    return {'d_model': int(d.get('d_model', 0)),
            'num_blocks': int(d.get('num_blocks', 0)),
            'V': int(d.get('V', 0))}


def kucultme_kontrolu(yeni, eski):
    """d_model / blok / V kucultulmesini engeller (proje kurali)."""
    if not eski:
        return
    for alan, ad in (('d_model', 'd_model'), ('num_blocks', 'blok sayisi'),
                     ('V', 'vocab')):
        if yeni[alan] and eski[alan] and yeni[alan] < eski[alan]:
            raise DogrulamaHatasi(
                '%s kucultulemez: mevcut %d -> yeni %d (proje kurali)'
                % (ad, eski[alan], yeni[alan]))


def yedekle(hedef_json, ts, kok=None):
    """Mevcut modeli model/yedek/<ts>/ altina tam kopyalar.

    Doner: yedek klasoru yolu, ya da yedeklencek dosya yoksa None.
    """
    # kok iceride cozulur: varsayilan deger parametreye baglanirsa (kok=YEDEK_KOK)
    # testler/yerellestirmeler YEDEK_KOK'u degistiremeyerek GERCEK model/yedek/
    # altina yazabiliyordu.
    kok = kok or YEDEK_KOK
    if not os.path.exists(hedef_json):
        return None
    d = _oku_json(hedef_json)
    agirlik = os.path.join(os.path.dirname(os.path.abspath(hedef_json)),
                           d.get('weights_file', ''))
    # yedek klasoru: yazma HESAPLANIR, boyleca ayni ts iki kez calistirilirsa
    # ikinci calistirma mevcut modeli degil yedegi yedeklemez
    var = os.path.join(kok, ts)
    n = 1
    while os.path.exists(var):
        n += 1
        var = os.path.join(kok, '%s_%d' % (ts, n))
    os.makedirs(var, exist_ok=True)
    shutil.copy2(hedef_json, os.path.join(var, os.path.basename(hedef_json)))
    if agirlik and os.path.exists(agirlik):
        shutil.copy2(agirlik, os.path.join(var, os.path.basename(agirlik)))
    return var


def _geri_al(yedek_yolu, hedef_json):
    """Yedekten modeli geri yukler (dogrulama basarisizliklari icin)."""
    for ad in os.listdir(yedek_yolu):
        kaynak = os.path.join(yedek_yolu, ad)
        hedef = (HEDEF_JSON if ad.endswith('.json') and 'weights' not in ad
                 else os.path.join(os.path.dirname(HEDEF_JSON), ad))
        shutil.copy2(kaynak, hedef)


def kur(kaynak_json, ts, yedekle_var=True):
    """Kaynak modeli guvenle yerine kurar. (ozet, yedek_yolu) doner."""
    ozet = dogrula(kaynak_json)
    yeni_agirlik = ozet['agirlik']
    yeni_ad = os.path.basename(yeni_agirlik)
    hedef_agirlik = os.path.join(os.path.dirname(HEDEF_JSON), yeni_ad)

    kucultme_kontrolu(ozet, _mevcut_ozet())

    yedek_yolu = yedekle(HEDEF_JSON, ts) if yedekle_var else None
    if yedekle_var and os.path.exists(HEDEF_JSON) and yedek_yolu is None:
        raise DogrulamaHatasi('yedek alinamadi - kurulum iptal edildi')

    # 1) gecici klasore yaz (hedef dosyalara henuz dokunmadan)
    gecici = os.path.join(os.path.dirname(HEDEF_JSON), '.kur_%s' % ts)
    if os.path.exists(gecici):
        shutil.rmtree(gecici)
    os.makedirs(gecici)
    gecici_json = os.path.join(gecici, os.path.basename(HEDEF_JSON))
    gecici_agirlik = os.path.join(gecici, yeni_ad)
    shutil.copy2(kaynak_json, gecici_json)
    shutil.copy2(yeni_agirlik, gecici_agirlik)
    # agirlik adini hedef adiyla hizala (json'da degistirmiyoruz, dosya adi ayni)
    d = _oku_json(gecici_json)
    d['weights_file'] = yeni_ad
    with io.open(gecici_json, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)

    # 2) atomik degistir
    os.makedirs(os.path.dirname(HEDEF_JSON), exist_ok=True)
    os.replace(gecici_json, HEDEF_JSON)
    os.replace(gecici_agirlik, hedef_agirlik)
    shutil.rmtree(gecici, ignore_errors=True)

    # 3) kurulum sonrasi dogrula: gercekten yukleniyor mu
    try:
        sonra = dogrula(HEDEF_JSON)
        if (sonra['d_model'], sonra['num_blocks'], sonra['V']) != \
           (ozet['d_model'], ozet['num_blocks'], ozet['V']):
            raise DogrulamaHatasi('kurulum sonrasi mimari kaymasi')
        _yukle_dene()
    except Exception as e:
        if yedek_yolu:
            _geri_al(yedek_yolu, HEDEF_JSON)
            raise DogrulamaHatasi('kurulum dogrulamasi basarisiz, yedek '
                                  'geri yuklendi: %s' % e)
        raise
    return sonra, yedek_yolu


def _yukle_dene():
    """model/llm_model.json gercekten yuklenip kisa bir uretim yapabiliyor mu."""
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from llm import load_llm
    m = load_llm(HEDEF_JSON)
    if m is None:
        raise DogrulamaHatasi('load_llm() None dondu - model okunamadi')
    from llm import encode_llm
    seq, _smask = encode_llm(m, 'merhaba', 'selam')
    if not len(seq):
        raise DogrulamaHatasi('girdi kodlanamadi (bos dizi)')
    m.forward(np.asarray(seq[:1], dtype=np.int32).reshape(1, -1),
              cache=m.new_cache(), past=0)
    return m


def main():
    ap = argparse.ArgumentParser(description='Guvenli LLM modeli kurulumu '
                                             '(yedekle + dogrula)')
    ap.add_argument('kaynak', help='kurulacak llm_model.json yolu')
    ap.add_argument('--check', action='store_true',
                    help='sadece dogrula, hicbir sey yazma')
    ap.add_argument('--ts', default=None,
                    help='yedek klasoru zaman damgasi (varsayilan: simdi)')
    ap.add_argument('--no-backup', action='store_true',
                    help='YEDEK ALMA. tehlikeli; sadece yedek klasorunun '
                         'zaten ayri bir kopya oldugundan emin isen kullan')
    args = ap.parse_args()

    ts = args.ts or time.strftime('%Y%m%d_%H%M%S')
    try:
        ozet = dogrula(args.kaynak)
    except DogrulamaHatasi as e:
        print('DOGRULAMA BASARISIZ: %s' % e)
        print('hicbir dosya degistirilmedi.')
        return 1

    print('kaynak        : %s' % args.kaynak)
    print('mimari        : d_model=%d blok=%d V=%d'
          % (ozet['d_model'], ozet['num_blocks'], ozet['V']))
    print('agirlik       : %s bayt / %s dizi'
          % (format(ozet['agirlik_bayt'], ','), ozet['dizisayisi']))
    print('parametre     : %s' % format(ozet['parametre'], ','))
    eski = _mevcut_ozet()
    if eski:
        print('mevcut model  : d_model=%d blok=%d V=%d  (d=%s, %s bayt)'
              % (eski['d_model'], eski['num_blocks'], eski['V'],
                 os.path.getsize(HEDEF_JSON),
                 format(os.path.getsize(os.path.join(
                     os.path.dirname(HEDEF_JSON),
                     _oku_json(HEDEF_JSON).get('weights_file', ''))), ',')
                 if os.path.exists(os.path.join(
                     os.path.dirname(HEDEF_JSON),
                     _oku_json(HEDEF_JSON).get('weights_file', ''))) else '?'))
    if args.check:
        print('\nKONTROL MODU: dogrulama gecti, hicbir sey yazilmadi.')
        return 0

    try:
        sonra, yedek_yolu = kur(args.kaynak, ts, yedekle_var=not args.no_backup)
    except DogrulamaHatasi as e:
        print('KURULUM BASARISIZ: %s' % e)
        return 2

    if yedek_yolu:
        print('\nyedek         : %s' % os.path.relpath(yedek_yolu, BASE))
    else:
        print('\nyedek         : YOK (yerinde model yoktu veya --no-backup)')
    print('kuruldu       : model/llm_model.json + %s'
          % sonra['arch'].get('weights_file'))
    print('dogrulama     : gecti (load_llm + forward)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

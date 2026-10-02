# -*- coding: utf-8 -*-
"""YEDEK METIN OLCUMU: kapi reddedince ekrana ne cikiyor?

--------------------------------------------------------------------------------
OLCULECEK SORU
--------------------------------------------------------------------------------
Kalite kapisi 3 adayi reddedince `_try_kb_rephrase` **HAM kb metnini**
donuyor (`brain.py:2047`). Bu ham metin:

  * cogu zaman KUCUK HARFLE baslar (corpus parcasi ortadan kesilmis),
  * dogallastirma dolgulari icerir ("devam edelim mi?", "bir bakima",
    "kisa ca", "ayrica", "daha fazla detay ister misin?"),
  * YARIM CUMLE olabilir (kesilmis bilgi parcasi),
  * 300+ karakter olabilir (birden cok olgu, hepsi soruyla ilgili degil).

6.12'de olculdu: kapinin reddettigi metinler bleu_ASCII 0,70-0,87, yedek ise
tanim geregi **1,000** (ham kb'nin kendisi). Yani yedek, reddedilenden daha
cok kopyadir. **Sonuc: esik cekilmemeli.**

Peki yedek NASIL IYILESTIRILIR? Kapinin kuralina dokunmadan, sadece
uretim ayarlariyla daha fazla metin uretmek mumkun:

  * `tries` 3 -> 6      : ayni kapi, daha cok aday -> gecen bulunma sansi artar
  * `temperature` 0,7  -> 0,9 / 1,1 : daha cesur metin -> novel daha yuksek

--------------------------------------------------------------------------------
TASARIM (tek bot yuklemesi, sirali varyantlar)
--------------------------------------------------------------------------------
Varyant sirasi ONEMLI: `tries` degisince uretilen adaylarin SIRASI degisir
(best-of-N ilk geceni degil en uzunu secer), yani ayni soruda varyantlar
birbirinden bagimsiz degildir. Bu yuzden her varyant icin tohum SIFIRDAN
kurulur: `np.random.seed(SEED)` ve `random.seed(SEED)` her (varyant, soru)
ciftinde. Bu sayede varyantlar birbirinden bagimsiz olur ve "fark" olcumune
degil, urunun davranisina aittir.

SADECE OLCUM: `_try_kb_rephrase` parametreli bir KOPYA ile degistirilir
(brain.py:2031-2047 ile satir satir ayni mantik), asil kod dosyada degismez.

--------------------------------------------------------------------------------
YEDEK METNIN NESNEL KALITE ISARETLERI
--------------------------------------------------------------------------------
  kucuk_bas    : ekrana cikan metin kucuk harfle basliyor (ozne de yazilmis)
  yarim        : noktalama ile bitmiyor (kesilmis cumle)
  dolgu        : dogallastirma dolgu ifadesi iceriyor
  karakter     : ekrana cikan metnin uzunlugu

KULLANIM:  python tools/yedek_olc.py [rapor_adi]
"""
import io
import json
import os
import random
import re
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from normalize import ascii_normalize  # noqa: E402

SORU_LISTESI = os.path.join(BASE, 'tools', 'soru_listesi.json')
SEED = 7

# (etiket, tries, temperature)
VARYANTLAR = [
    ('bugun_t3_t07', 3, 0.7),    # uretimdeki deger (referans)
    ('t6_t07', 6, 0.7),          # sadece tries
    ('t3_t09', 3, 0.9),          # sadece sicaklik
    ('t6_t09', 6, 0.9),          # ikisi
]

STOP = set(('ve ile bir bu da de ki en cok daha cok gibi olarak ancak icin '
            'sonra yani ise her ancak veya yoktur degil en fazla cok').split())

# dogallastirma dolgulari (naturalize.py'nin ekledigi baglaclardan olculmus)
DOLGU = [
    'devam edelim mi', 'daha fazla detay ister misin', 'bir bakima',
    'kisa ca', 'ayrica', 'ozellikle', 'sebebi su', 'bunun nedeni',
    'bileseniz', 'detayli bilgi ister misin', 'devam edersek',
]


def icerik_kelimeler(metin):
    t = ascii_normalize((metin or '').lower())
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    return [w for w in t.split() if len(w) > 2 and w not in STOP]


def yedek_isaretleri(metin):
    m = (metin or '').strip()
    kucuk = bool(m) and m[0].islower()
    yarim = bool(m) and not m.rstrip().endswith(('.', '!', '?', '…'))
    d = ascii_normalize(m).lower()
    dolgu = [x for x in DOLGU if x in d]
    return {'kucuk_bas': kucuk, 'yarim': yarim, 'dolgu': dolgu,
            'karakter': len(m)}


with io.open(SORU_LISTESI, encoding='utf-8') as f:
    SORULAR = [str(s).strip() for s in json.load(f)['sorular'] if str(s).strip()]

import numpy as np  # noqa: E402
import app as uygulama  # noqa: E402

t0 = time.time()
uygulama.load_bot()
bot = uygulama.bot
print('bot yuklendi (%.1f sn), knowledge_bias=%s'
      % (time.time() - t0, getattr(bot, 'knowledge_bias', '?')))
print('soru: %d   varyant: %s' % (len(SORULAR), [v[0] for v in VARYANTLAR]))

# ------------------------------------------------ parametreli _try_kb_rephrase
AKTIF = {'varyant': None, 'kayit': None}


def rephrase(query, kb, tries):
    """brain.py:2031-2047 ile ayni mantik; tries/temperature disaridan."""
    v = AKTIF['varyant']
    k = AKTIF['kayit']
    if not query or not kb:
        return kb
    if not bot._ensure_llm():
        return kb
    ext = None
    try:
        ext = bot._external_knowledge(query)
    except Exception:
        ext = None
    try:
        knowledge = (ext + '\n' + kb) if ext else kb
        best, best_len = None, 0
        for _ in range(max(1, int(tries))):
            gen = bot.llm.sample(query, temperature=v['temp'], top_k=10,
                                 knowledge=knowledge[:500],
                                 rep_penalty=0.4,
                                 knowledge_bias=bot.knowledge_bias)
            kabul = bool(bot._accept_kb_rephrase(gen, kb))
            gk = icerik_kelimeler(gen)
            dk = set(icerik_kelimeler(kb))
            k['adaylar'].append({
                'kabul': kabul, 'gen': gen,
                'sadakat': (len([w for w in gk if w in dk]) / float(len(gk))
                            if gk and dk else None)})
            if not kabul:
                continue
            if best is None or len(gen) > best_len:
                best, best_len = gen, len(gen)
        if best:
            k['uretilildi'] = True
            return bot.deasciify(best)
    except Exception as e:
        k['hata'] = str(e)[:120]
    k['uretilildi'] = False
    return kb


def rephrase_sarmal(query=None, kb='', tries=3):
    k = AKTIF['kayit']
    if k is not None:
        k['kb'] = kb
    return rephrase(query, kb, AKTIF['varyant']['tries'])


bot._try_kb_rephrase = rephrase_sarmal

# ------------------------------------------------------------------- kosular
# SURE DAMGASI (01.10.2026): arac once surayi yazdiriyor ama RAPORA yazmiyordu.
# Bu yuzden "tries 3->6 ne kadar yavastirir" sorusu OLCULEMEDEN karar
# verilmisti (DEVAM_PROMPTU.md 6.17). Artik varyant basina sure hem kayda
# hem ozete yaziliyor. NOT: sure OLCUM MAKINESINE bagli; karsilastirma ayni
# makinede yapilirsa gecerlidir, Kaggle'daki mutlak saniye degildir.
VARYANT_SURE = {}
tum = []
for etiket, tries, temp in VARYANTLAR:
    print()
    print('--- varyant %s (tries=%d, temp=%.1f) ---' % (etiket, tries, temp))
    v0 = time.time()
    soru_sureleri = []
    for soru in SORULAR:
        np.random.seed(SEED)
        random.seed(SEED)
        k = {'soru': soru, 'kb': '', 'yanit': '', 'uretilildi': False,
             'adaylar': []}
        AKTIF['varyant'] = {'tries': tries, 'temp': temp}
        AKTIF['kayit'] = k
        s0 = time.time()
        try:
            k['yanit'] = bot.get_response(soru) or ''
        except Exception as e:
            k['hata'] = str(e)[:120]
        k['sure_sn'] = round(time.time() - s0, 3)
        soru_sureleri.append(k['sure_sn'])
        AKTIF['kayit'] = None
        tum.append({'varyant': etiket, 'tries': tries, 'temp': temp,
                    **k})
        bay = 'URETILDI' if k['uretilildi'] else 'YEDEK'
        print('  %-26s %-8s aday=%d  %6.2f sn  %s'
              % (soru[:26], bay, len(k['adaylar']), k['sure_sn'],
                 k['yanit'][:44]))
    VARYANT_SURE[etiket] = round(time.time() - v0, 2)
    print('  varyant suresi: %.1f sn  (soru basina %.2f sn)'
          % (VARYANT_SURE[etiket],
             VARYANT_SURE[etiket] / max(1, len(SORULAR))))

# ------------------------------------------------------------------ analiz
print()
print('=' * 96)
print('YEDEK METIN OLCUMU - %d sabit soru, %d varyant (kapinin kurali DEGISMEDI)'
      % (len(SORULAR), len(VARYANTLAR)))
print('=' * 96)
print('%-14s %11s %11s %10s %10s %9s %9s %9s %11s %10s'
      % ('varyant', 'uretildi', 'aday kabul', 'sadakat', 'k_ucuk_bas',
         'yarim', 'dolgu', 'ort krk', 'soru/sn', 'tabana gore'))
print('-' * 96)
ozet = []
for etiket, tries, temp in VARYANTLAR:
    g = [x for x in tum if x['varyant'] == etiket]
    n_aday = sum(len(x['adaylar']) for x in g)
    n_kabul = sum(1 for x in g for d in x['adaylar'] if d['kabul'])
    uret = [x for x in g if x['uretilildi']]
    # sadakat: SADECE uretilen metinler icin (uretim_olc.py ile ayni)
    sad = []
    for x in uret:
        kabul = [d for d in x['adaylar'] if d['kabul']]
        if not kabul:
            continue
        metin = max(kabul, key=lambda d: len(d['gen']))['gen']
        gk = icerik_kelimeler(bot.deasciify(metin))
        dk = set(icerik_kelimeler(x['kb']))
        if gk and dk:
            sad.append(len([w for w in gk if w in dk]) / float(len(gk)))
    # yedek isaretleri: ekrana giden metin
    isr = [yedek_isaretleri(x['yanit']) for x in g]
    sr = [x.get('sure_sn', 0.0) for x in g]
    soru_sn = (sum(sr) / len(sr)) if sr else 0.0
    ozet.append({
        'varyant': etiket, 'tries': tries, 'temp': temp,
        'uretildi': len(uret), 'soru': len(g),
        'uretilim_orani': len(uret) / float(max(1, len(g))),
        'aday': n_aday, 'kabul': n_kabul,
        'kabul_orani': n_kabul / float(max(1, n_aday)),
        'sadakat': (sum(sad) / len(sad)) if sad else None,
        'sadakat_n': len(sad),
        'kucuk_bas': sum(1 for i in isr if i['kucuk_bas']),
        'yarim': sum(1 for i in isr if i['yarim']),
        'dolgu': sum(1 for i in isr if i['dolgu']),
        'ort_karakter': sum(i['karakter'] for i in isr) / float(len(isr)),
        'sure_sn_toplam': VARYANT_SURE.get(etiket),
        'soru_basina_sn': round(soru_sn, 3),
        'uretilen_sn': round(
            (sum(x.get('sure_sn', 0.0) for x in uret) / len(uret)), 3)
        if uret else None,
    })
    o = ozet[-1]
    taban_sn = ozet[0]['soru_basina_sn'] if ozet else 0.0
    carpan = (soru_sn / taban_sn) if taban_sn else 1.0
    print('%-14s %6d/%-4d %6d/%-4d %9s %9d %9d %9d %9.0f %11.2f %9.2fx'
          % (etiket, o['uretildi'], o['soru'], o['kabul'], o['aday'],
             ('%3.0f%%' % (100 * o['sadakat'])) if o['sadakat'] is not None
             else '-',
             o['kucuk_bas'], o['yarim'], o['dolgu'], o['ort_karakter'],
             soru_sn, carpan))
print()
print('uretildi = modelin cumlesi ekrana cikti (kapidan gecti).')
print('k_ucuk_bas / yarim / dolgu = EKRANA CIKAN metnin isaretleri (uretildi +')
print('   yedek birlikte; yani bunlar butun kullaniciya giden metinler).')
print()

# yedek metinlerin kendi kalitesi: sadece kapinin reddettikleri
print('=' * 96)
print('KAPI REDDETTI, EKRANA HAM kb CIKTI - bu metinlerin isaretleri')
print('=' * 96)
for etiket, tries, temp in VARYANTLAR:
    g = [x for x in tum if x['varyant'] == etiket and not x['uretilildi']]
    isr = [yedek_isaretleri(x['yanit']) for x in g]
    if not g:
        continue
    print('%-14s n=%-3d kucuk_bas=%d  yarim=%d  dolgu=%d  ort krk=%.0f'
          % (etiket, len(g),
             sum(1 for i in isr if i['kucuk_bas']),
             sum(1 for i in isr if i['yarim']),
             sum(1 for i in isr if i['dolgu']),
             sum(i['karakter'] for i in isr) / float(len(isr))))
print()
g0 = [x for x in tum if x['varyant'] == 'bugun_t3_t07' and not x['uretilildi']]
print('ORNEK YEDEK METINLER (bugun_t3_t07, kapiden gecemeyenler):')
for x in g0[:6]:
    i = yedek_isaretleri(x['yanit'])
    bay = []
    if i['kucuk_bas']:
        bay.append('KUCUK BAS')
    if i['yarim']:
        bay.append('YARIM CUMLE')
    if i['dolgu']:
        bay.append('DOLGU: ' + ','.join(i['dolgu']))
    print('  [%s] %s' % (', '.join(bay) or 'temiz', x['soru'][:26]))
    print('      %s' % x['yanit'][:110])

# ------------------------------------------------------------- GECIKME (1. EKSIK)
# "tries 3->6 ne kadar yavastirir" sorusu 01.10'da OLCULEMEDEN karar
# verilmisti. Artik olculuyor. DIKKAT: sure OLCUM MAKINESINE baglidir;
# karsilastirma (carpan) ayni makinede anlamlidir, mutlak saniye Kaggle
# icin gecerli degildir.
print()
print('=' * 96)
print('GECIKME (tries/temp bedeli) - ayni makinede olculdu')
print('=' * 96)
print('%-14s %10s %12s %12s %10s'
      % ('varyant', 'soru/sn', 'uretlenen/sn', 'toplam sn', 'tabana gore'))
print('-' * 96)
_tab = ozet[0]['soru_basina_sn'] or 1.0
for o in ozet:
    us = o['uretilen_sn']
    print('%-14s %10.2f %12s %12s %9.2fx'
          % (o['varyant'], o['soru_basina_sn'],
             ('%.2f' % us) if us is not None else '-',
             ('%.1f' % o['sure_sn_toplam'])
             if o['sure_sn_toplam'] is not None else '-',
             o['soru_basina_sn'] / _tab))
print()
print('KAZANIM / BEDEL (taban = ilk varyant):')
for o in ozet[1:]:
    b = ozet[0]
    print('  %-12s ekran orani %+.1f puan, soru basina sure %+.2f kat, '
          'sadakat %s'
          % (o['varyant'],
             100.0 * (o['uretilim_orani'] - b['uretilim_orani']),
             o['soru_basina_sn'] / max(0.001, b['soru_basina_sn']),
             ('%+.1f puan' % (100.0 * (o['sadakat'] - b['sadakat'])))
             if (o['sadakat'] is not None and b['sadakat'] is not None)
             else '-'))
print('  (sure = kullaniciya gorulen yanit suresi; ekran orani = kapidan')
print('   gecmeyen soru orani, yani modelin metninin gorunme sansi)')

ad = sys.argv[1] if len(sys.argv) > 1 else 'kapali'
cikti = os.path.join(BASE, 'olcum_raporlari', 'yedek_olc_%s.json' % ad)
os.makedirs(os.path.dirname(cikti), exist_ok=True)
with io.open(cikti, 'w', encoding='utf-8') as f:
    json.dump({'seed': SEED, 'varyantlar': VARYANTLAR,
               'soru_sayisi': len(SORULAR), 'ozet': ozet, 'kayitlar': tum},
              f, ensure_ascii=False, indent=1)
print()
print('rapor yazildi:', cikti)

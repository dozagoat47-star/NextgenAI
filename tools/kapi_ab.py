# -*- coding: utf-8 -*-
"""KALITE KAPISI: OZGUNLUK ESIGI TARAMASI (%15 -> %0), PAIRED.

SORU: kapinin `ozgunluk >= %15` kuralini (%15 -> %5-8) cekmek ne getirir?

MOTIVASYON (01.10.2026 olcumu, `olcum_raporlari/uretim_duzeltilmis_0110.json`):
kapinin ret/kabul karari sadakatle TERS iliskili.

    karar                        n     sadakat
    KABUL EDILEN                71      %59,2
    RET "ozgunluk yok"           37      %97,4   <-- EN SADIK olanlar
    RET "konu kelimesi yok"      27       %9,2
    RET "konu bilesimi dusuk"    15      %18,3

Yani kapi, bilgi metnini en cok koruyan uretimleri "kopyaliyor" diye
cevapmiyor. Bu dosya esiği gercekten cekmenin bedelini OLÇER.

--------------------------------------------------------------------------------
NEDEN TEK KOSU YETERLI (bu dosyanin en onemli tasarim karari)
--------------------------------------------------------------------------------
Uretim yolu 3 adayi HER ZAMAN uretir ve gecenler arasindan EN UZUN olani secer
(`brain.py:2033-2042` kb yolu, `:1933-1940` sohbet yolu). Yani:

  * aday metinleri esikten BAGIMSIZDIR (uretim esige bakmaz),
  * secim kurali esikten BAGIMSIZDIR ("gecenler arasinda en uzun").

Tek kosuda her adayi kaydedip, sonra esik degistirerek "bu aday bu esikte
kabul edilir miydi?" diye **yeniden hesaplamak** tam olarak ayni seydir. 8 esik
x 8 kosu yerine 1 kosu: uretim maliyeti 8'de 1, sonuc birebir ayni.
Bu yuzden asil kapi **OLDUGU GIBI** birakilir; sarmalama yalnizca olcer ve
asil karari dondurur.

--------------------------------------------------------------------------------
DUZENLILIK
--------------------------------------------------------------------------------
`np.random.seed` **ve** `random.seed` her sorudan once kurulur; ikisi de sart
(`brain.py:1467/1469` `random.choice(responses)` NumPy tohumundan bagimsizdir;
01.10'da olculdu: tohumlanmadan ayni soru 11 sorunun 6'sinda farkli kb verdi).

--------------------------------------------------------------------------------
KARSILASTIRILABILIRLIK
--------------------------------------------------------------------------------
Sorular `tools/soru_listesi.json`'dan okunur (uretim_olc.py ile AYNI sabit liste,
aksi halde bu tarama uretim_olc.py ciktisiyla kiyaslanamaz).

--------------------------------------------------------------------------------
NOT (eski surum)
--------------------------------------------------------------------------------
Onceki surum harf cesitliligi kuralini ('eski' kural / 'yeni' kural) kiyasiyordu.
O kural 25.09'da matematiksel olarak imkansiz bulundu ve `min(12, ...)` ile
DUZELTILDI (bkz. `brain.py:2081-2098`); sorunu kapanmistir. Bu surum o kurali
kapsamaz, onun yerine ozgunluk esigini tarar.

KULLANIM:  python tools/kapi_ab.py [rapor_adi]
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
REFERANS = 0.15          # uretimdeki ozgunluk esigi (brain.py:2126, :2163)
ESIKLER = [0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.00]

STOP = set(('ve ile bir bu da de ki en cok daha cok gibi olarak ancak icin '
            'sonra yani ise her ancak veya yoktur degil en fazla cok').split())

_TRUNCATION_MIN_CHARS = 70  # brain.py:2077 ile ayni


def icerik_kelimeler(metin):
    """uretim_olc.py icerigiyle BIREBIR ayni (sadakat olcumu buradan gelir)."""
    t = ascii_normalize((metin or '').lower())
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    return [w for w in t.split() if len(w) > 2 and w not in STOP]


with io.open(SORU_LISTESI, encoding='utf-8') as f:
    SORULAR = [str(s).strip() for s in json.load(f)['sorular'] if str(s).strip()]

import numpy as np  # noqa: E402
import app as uygulama  # noqa: E402
import brain  # noqa: E402  (STOPWORDS tek kaynak)

t0 = time.time()
uygulama.load_bot()
bot = uygulama.bot
print('bot yuklendi (%.1f sn), knowledge_bias=%s'
      % (time.time() - t0, getattr(bot, 'knowledge_bias', '?')))
print('soru: %d (sabit liste), seed=%d, esikler=%s'
      % (len(SORULAR), SEED, ESIKLER))

# ------------------------------------------------------------------ olcum
# Her kapı cagrisinda ESIKTEN BAGIMSIZ butun sartlari AYRI hesapliyoruz:
#   oncesi -> novelty kontrolunden ONCE butun sartlar gecti mi
#   novel -> novel/|gen_set| orani
#   dup   -> yapiskan tekrar sartindan gecti mi (kontrol novelty SONRASINDA)
# kabul(t) = oncesi AND dup AND novel >= t
# Not: asil kapida `dup` kontrolu ancak novelty gecerse calisir; burada
# bagimsiz hesapliyoruz ki "novelty gecse butun sartlar gecti mi" sorusu
# tek satir cikabilsin.
AKTIF = {'kayit': None, 'cag': 0}


def _harf_kurali(gen):
    letters = [c for c in gen.lower() if c.isalpha()]
    if len(letters) < 6 or len(set(letters)) < min(12, int(len(letters) * 0.30)):
        return False
    return True


def _tekrar(cand):
    if len(cand) < 4:
        return True
    dup = sum(1 for a, b in zip(cand, cand[1:]) if a == b)
    return dup / float(len(cand) - 1) <= 0.5


def _bos(neden):
    return {'oncesi': False, 'novel': 0.0, 'dup': False, 'sebep': neden,
            'sadakat': None}


def olc_kb(gen, kb):
    """brain._accept_kb_rephrase'in esikten bagimsiz parcalari."""
    if not gen or len(gen) < 12 or len(gen) > 260:
        return _bos('uzunluk')
    if not _harf_kurali(gen):
        return _bos('harf/repeat')
    kb_set = set(bot.tokenize(kb))
    if not kb_set:
        return _bos('kb bos')
    cand = bot.tokenize(gen)
    if len(cand) < 3:
        return _bos('az token')
    gen_set = set(cand)
    inter = gen_set & kb_set
    if len(inter) < 2:
        return _bos('konu kelimesi yok (%d)' % len(inter))
    if len(inter) / float(len(gen_set)) < 0.20:
        return _bos('konu bilesimi dusuk')
    gk = icerik_kelimeler(gen)
    dk = set(icerik_kelimeler(kb))
    return {'oncesi': True,
            'novel': len(gen_set - kb_set) / float(len(gen_set)),
            'dup': _tekrar(cand),
            'sebep': '',
            'sadakat': (len([w for w in gk if w in dk]) / float(len(gk))
                        if gk and dk else None)}


def olc_gen(gen, tag, query):
    """brain._accept_generated'in esikten bagimsiz parcalari."""
    if not gen or len(gen) < 12 or len(gen) > 260:
        return _bos('uzunluk')
    if len(gen) >= _TRUNCATION_MIN_CHARS and not gen.rstrip().endswith(
            ('.', '!', '?', '…', '"', ')')):
        return _bos('yarim cumle')
    if not _harf_kurali(gen):
        return _bos('harf/repeat')
    canned = set()
    for t in (bot.intents.get(tag) or []):
        canned |= set(bot.tokenize(t))
    kws = bot.intent_kws.get(tag) or set()
    cand = bot.tokenize(gen)
    if len(cand) < 3:
        return _bos('az token')
    gen_set = set(cand)
    inter = gen_set & (canned | kws)
    if len(inter) < 2:
        return _bos('konu kelimesi yok (%d)' % len(inter))
    if len(inter) / float(len(gen_set)) < 0.25:
        return _bos('konu bilesimi dusuk')
    if query is not None:
        qkws = {x for x in bot.tokenize(query) if x not in brain.STOPWORDS}
        if len(qkws) >= 3 and not (gen_set & qkws):
            return _bos('sorgu konusu yok')
    return {'oncesi': True,
            'novel': len(gen_set - canned) / float(len(gen_set)),
            'dup': _tekrar(cand),
            'sebep': '',
            'sadakat': None}   # sohbet yolunda referans metin yok


# ------------------------------------------------------- olcum katmani
ASIL_KB = bot._accept_kb_rephrase
ASIL_GEN = bot._accept_generated
ASIL_REPHRASE = bot._try_kb_rephrase


def kb_sarmal(gen, kb):
    k = AKTIF['kayit']
    if k is not None:
        r = olc_kb(gen, kb)
        r['gen'] = gen
        r['cag'] = AKTIF['cag']
        k['adaylar'].append(r)
    return ASIL_KB(gen, kb)


def gen_sarmal(gen, tag, query=None):
    k = AKTIF['kayit']
    if k is not None:
        r = olc_gen(gen, tag, query)
        r['gen'] = gen
        r['cag'] = -1        # sohbet yolu
        k['adaylar'].append(r)
    return ASIL_GEN(gen, tag, query=query)


def rephrase_sarmal(query=None, kb='', tries=3):
    k = AKTIF['kayit']
    if k is not None:
        AKTIF['cag'] += 1
        k['turlar'].append({'no': AKTIF['cag'], 'kb': kb})
    return ASIL_REPHRASE(query, kb, tries)


bot._accept_kb_rephrase = kb_sarmal
bot._accept_generated = gen_sarmal
bot._try_kb_rephrase = rephrase_sarmal

kayitlar = []
for soru in SORULAR:
    np.random.seed(SEED)
    random.seed(SEED)
    k = {'soru': soru, 'turlar': [], 'adaylar': [], 'yanit': ''}
    AKTIF['kayit'] = k
    AKTIF['cag'] = 0
    try:
        k['yanit'] = bot.get_response(soru) or ''
    except Exception as e:  # pragma: no cover
        k['hata'] = str(e)[:120]
    AKTIF['kayit'] = None
    kayitlar.append(k)
    print('  %-26s aday=%2d tur=%d  %s'
          % (soru[:26], len(k['adaylar']), len(k['turlar']),
             (k['yanit'] or '')[:58]))

print()
print('kosu bitti (%.1f sn) - esikler simulasyonla taraniyor' % (time.time() - t0))

# ------------------------------------------------------------- simulasyon
def kabul_mi(r, t):
    return bool(r['oncesi'] and r['dup'] and r['novel'] >= t)


def sadakat(metin, kb):
    gk = icerik_kelimeler(metin)
    dk = set(icerik_kelimeler(kb))
    if not gk or not dk:
        return None
    return len([w for w in gk if w in dk]) / float(len(gk))


sonuclar = []
for t in ESIKLER:
    n_aday = n_kabul = ekrana = sohbet = 0
    sad_gen = []
    ref_sad = []
    yeni_aday = []
    yeni_sad = []
    for k in kayitlar:
        n_aday += len(k['adaylar'])
        gecen = [r for r in k['adaylar'] if kabul_mi(r, t)]
        n_kabul += len(gecen)
        # son _try_kb_rephrase cagrisi = ekrana giden yol (uretim_olc.py ile ayni)
        if not k['turlar']:
            sohbet += 1
            continue
        son = k['turlar'][-1]
        tur = [r for r in k['adaylar'] if r['cag'] == son['no']]
        gecen_tur = [r for r in tur if kabul_mi(r, t)]
        if gecen_tur:
            ekrana += 1
            metin = max(gecen_tur, key=lambda r: len(r['gen']))['gen']
            s = sadakat(bot.deasciify(metin), son['kb'])
            if s is not None:
                sad_gen.append(s)
        # 0,15 referansina gore "yeni kabul" (yalnizca t < 0,15 anlamli)
        for r in tur:
            if kabul_mi(r, t) and not kabul_mi(r, REFERANS):
                yeni_aday.append(r)
                if r.get('sadakat') is not None:
                    yeni_sad.append(r['sadakat'])
        for r in tur:
            if kabul_mi(r, REFERANS) and r.get('sadakat') is not None:
                ref_sad.append(r['sadakat'])
    sonuclar.append({
        'esik': t,
        'kabul': n_kabul, 'toplam_aday': n_aday,
        'kabul_orani': n_kabul / float(max(1, n_aday)),
        'ekrana_uretim': ekrana, 'soru': len(kayitlar), 'sohbet_yolu': sohbet,
        'sadakat_n': len(sad_gen),
        'sadakat': (sum(sad_gen) / len(sad_gen)) if sad_gen else None,
        'yeni_kabul': len(yeni_aday), 'yeni_sadakat_n': len(yeni_sad),
        'yeni_sadakat': (sum(yeni_sad) / len(yeni_sad)) if yeni_sad else None,
        'referans_sadakat': (sum(ref_sad) / len(ref_sad)) if ref_sad else None,
    })

# ------------------------------------------------------------------ rapor
print()
print('=' * 96)
print('OZGUNLUK ESIGI TARAMASI - %d sabit soru, TEK kosu (adaylar esikten bagimsiz)'
      % len(kayitlar))
print('=' * 96)
print('%6s %12s %17s %15s %10s %10s'
      % ('esik', 'aday kabul', 'ekrana uretim', 'sadakat', 'yeni', 'yeni sad'))
print('-' * 96)
for s in sonuclar:
    sad = ('%%%4.0f (n=%d)' % (100 * s['sadakat'], s['sadakat_n'])
           if s['sadakat'] is not None else '-')
    yeni = ('%d  %%%.0f' % (s['yeni_kabul'], 100 * s['yeni_sadakat'])
            if s['yeni_kabul'] and s['yeni_sadakat'] is not None
            else ('%d  -' % s['yeni_kabul']))
    print('%5.2f  %4d/%-5d %5d/%-3d %4.0f%% %13s %10s'
          % (s['esik'], s['kabul'], s['toplam_aday'],
             s['ekrana_uretim'], s['soru'],
             100.0 * s['ekrana_uretim'] / max(1, s['soru']), sad, yeni))
print()
print('SADAKAT = ekrana cikan uretilen metnin icerik kelimelerinden bilgi')
print('parcasinda bulunanlarin orani (uretim_olc.py ile ayni tanim).')
print('"yeni" = %%%.2f esiginde REDDEDILIP bu esikte KABUL edilen adaylar'
      % REFERANS)
print('ve onlarin sadakati (yeni sad). Bu sayi dusukse esik cekmek kaliteyi')
print('dusurur; yuksekse kapi gercekten bosuna reddediyor demektir.')
print()
print('sohbet yoluna düşen soru: %d (kb cagrisi yok -> ekrana metni esikten '
      'bagimsiz)' % sonuclar[0]['sohbet_yolu'])

ad = sys.argv[1] if len(sys.argv) > 1 else 'kapi_ozgunluk'
cikti = os.path.join(BASE, 'olcum_raporlari', 'kapi_ozgunluk_%s.json' % ad)
os.makedirs(os.path.dirname(cikti), exist_ok=True)
with io.open(cikti, 'w', encoding='utf-8') as f:
    json.dump({'esikler': ESIKLER, 'referans': REFERANS, 'seed': SEED,
               'soru_sayisi': len(kayitlar), 'sonuclar': sonuclar,
               'kayitlar': kayitlar}, f, ensure_ascii=False, indent=1)
print('rapor yazildi:', cikti)

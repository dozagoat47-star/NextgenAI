# -*- coding: utf-8 -*-
"""SOHBET YOLU KALITE OLCUMU - sohbet siniflarinin TABAN olcumu.

NEDEN BU SCRIPT: uretim_olc.py bilgi yolunu olcuyor
(brain._try_kb_rephrase -> sadakat). Sohbet tamamen farkli bir yol:
  brain.get_response -> _classify -> chat sinifi -> _try_seq_rephrase
  -> best-of-LLM -> kabul kapisi -> ekrana.
Olculen metrikler de farkli: burada sadakat degil SINIFLANDIRMA ISABETI
ve uretim orani olcumek icin. Sohbet hic olculmemisti: sabit 50 sorunun
50'si de bilgi sorusuydu ("X nedir/anlat"), tek sohbet sorusu yoktu.
Bu script o boslugu kapatir.

KULLANIM:  python tools/sohbet_olc.py <etiket> [cikti.json]
  np.random.seed(7) VE random.seed(7) her sorudan once. IKISI DE SART:
  NumPy tohumu LLM orneklemesini, Python `random` tohumu bilgi/yanit
  secimini kontrol ediyor (brain.py:1467/1469 `random.choice`; uretim_olc.py
  dosya basindaki not, 01.10 olcumu). Ikisi de tohumlanmazsa iki kosu
  birebir ayni olmaz ve A/B karsilastirmasi gurultulu olur.

SINIF TASI OLCUMU (bu aracin en degerli kisimi): beklenen tag'i siniflandiriciya
GIRMEMIS intent'ler icin (ornegin kavram_tanimi, tavsiye_isteme,
gelecek_planlari - hepsi 6 desenli, brain.py:1047 `len(patterns) > 6` yuzunden
eleniyor) beklenen tag ASLA yakalanamaz. Arac bunu sessizce gecmek yerine
`sinifta_mi: false` olarak raporlar, boylece sizinti olcumlenir.

KAPSAM: bu bir OLUM aracidir. Uretim koduna, modele, veri dosyalarina
(intents.json, knowledge_map.jsonl, corpus.jsonl, corpus_ids.jsonl)
DOKUNMAZ; yalnizca rapor yazar (olcum_raporlari/, .gitignore'da).
"""
import io
import json
import os
import random
import re
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from normalize import ascii_normalize  # noqa: E402  (sys.path gerekli)

ETIKET = sys.argv[1] if len(sys.argv) > 1 else 'bilinmeyen'
RAPOR_DIZIN = os.path.join(BASE, 'olcum_raporlari')
CIKTI = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    RAPOR_DIZIN, 'sohbet_%s.json' % ETIKET)

SEED = 7

SORU_LISTESI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'soru_listesi_sohbet.json')

# --------------------------------------------------- damga (rapor izi)
# uretim_olc.py ile ayni sozlesme: raporun hangi VERI ve MODEL ile
# uretildigi JSON'a yazilir, sonradan sorulmaz.


def _damga(yol):
    try:
        st = os.stat(yol)
        return [st.st_size, int(st.st_mtime)]
    except OSError:
        return None


def veri_damgasi():
    d = {}
    for ad in ('intents.json', 'knowledge_map.jsonl', 'corpus.jsonl',
               'corpus_ids.jsonl'):
        d[ad] = _damga(os.path.join(BASE, ad))
    try:
        for ad in sorted(os.listdir(BASE)):
            if ad.startswith('chatgrow_') and ad.endswith('.jsonl'):
                d[ad] = _damga(os.path.join(BASE, ad))
    except OSError:
        pass
    return d


def model_damgasi():
    d = {}
    for ad in ('llm_model.json', 'llm_model_weights.npz'):
        d[ad] = _damga(os.path.join(BASE, 'model', ad))
    return d


# ---------------------------------------------------------------- sorular
# Soru listesi AYNEN kullanilir; elle degistirilirse yeni rapor eskisiyle
# kiyaslanamaz (uretim_olc.py dosya basindaki not, 01.10 olcumu).
if not os.path.exists(SORU_LISTESI):
    raise SystemExit('SORU LISTESI YOK: %s' % SORU_LISTESI)
with io.open(SORU_LISTESI, encoding='utf-8') as f:
    _L = json.load(f)
SORULAR = _L['sorular']
BEKLENEN_ADET = _L.get('soru_sayisi', len(SORULAR))
if BEKLENEN_ADET != len(SORULAR):
    raise SystemExit('soru_sayisi beyani %s ama dosyada %d soru var - '
                     'liste elle degistirilmis, kiyas bozulur'
                     % (BEKLENEN_ADET, len(SORULAR)))


def soru_listesi_damgasi():
    return {'dosya': _damga(SORU_LISTESI), 'soru_sayisi': len(SORULAR),
            'ilk': SORULAR[0]['soru'], 'son': SORULAR[-1]['soru']}


# ------------------------------------------------------- canned yanit kumesi
# 'canned_mi' olcumu icin: donen metin intents.json'daki hazir bir yanitin
# birebir kopyasi mi? intents.json SALT OKUNUR.
with io.open(os.path.join(BASE, 'intents.json'), encoding='utf-8') as f:
    _Y = json.load(f)
CANNED = {}
TUM_TAGLAR = set()
BILGI_TAGLAR = set()
for _it in _Y.get('intents', []):
    _tag = _it.get('tag')
    TUM_TAGLAR.add(_tag)
    if len(_it.get('patterns') or []) <= 6:
        BILGI_TAGLAR.add(_tag)
    for _r in _it.get('responses', []):
        CANNED.setdefault(re.sub(r'\s+', ' ', (_r or '').strip()), set()).add(
            _tag)


def _duz(m):
    return re.sub(r'\s+', ' ', (m or '').strip())


def canned_mi(metin):
    """Metin hazir bir yanitin birebir kopyasi mi? (bos kova dahil DEGIL)"""
    if not metin:
        return False
    return _duz(metin) in CANNED


# ---------------------------------------------------------------- kurulum
import app as uygulama  # noqa: E402

t0 = time.time()
uygulama.load_bot()
bot = uygulama.bot
SINIFLAR = set(getattr(bot, 'intent_tags', []))
print('[%s] bot yuklendi (%.1f sn) | sohbet sinifi %d'
      % (ETIKET, time.time() - t0, len(SINIFLAR)))
print('[%s] %d soru: %s ... %s' % (ETIKET, len(SORULAR),
                                   SORULAR[0]['soru'], SORULAR[-1]['soru']))

STOP = set(('ve ile bir bu da de ki en cok daha cok gibi olarak ancak icin '
            'sonra yani ise her veya yoktur degil en fazla cok ve ben sen '
            'bana senin icin ile de da').split())


def icerik_kelimeler(metin):
    """uretim_olc.py ile BIREBIR ayni. Turkce -> ASCII (ascii_normalize),
    sonra kirp. Neden ortak: farkli katlama sadakati yapay olarak dusurur
    (uretim_olc.py 01.10 notu: %33,3 -> %57,1)."""
    t = ascii_normalize((metin or '').lower())
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    return [w for w in t.split() if len(w) > 2 and w not in STOP]


# ------------------------------------------------- olcum katmani (sarmalama)
# _try_seq_rephrase: sohbetin LLM kapisi. Cagrildi mi, ne dondu, kabul edildi
# mi - ucusu de disaridan GORULEMEYEN sey, bu yuzden sarilir (uretim_olc.py
# ile ayni teknik).
kayitlar = []
asil_seq = bot._try_seq_rephrase
aktif = {'kayit': None}


def seq_sarmal(tag, query=None):
    r = asil_seq(tag, query)
    k = aktif['kayit']
    if k is not None:
        k['yol_acildi'] = True
        k['yol_dondu'] = bool(r)
        k['yol_metin'] = r
    return r


bot._try_seq_rephrase = seq_sarmal

# ---------------------------------------------------------------- kosum
# 1) kayitlari kur (olcum YAPILMAZ; siralama burada)
for s in SORULAR:
    bek = s['beklenen']
    kayitlar.append({
        'soru': s['soru'], 'beklenen': bek, 'grup': s.get('grup', ''),
        'not': s.get('not', ''), 'kabul': s.get('kabul', 'tam_dogruluk'),
        'tahmin': '', 'guven': 0.0,
        'isabet': False, 'beklenen_sinifta': bek in SINIFLAR,
        'tahmin_etiket_mi': False, 'tahmin_sinifta': False,
        'yol_acildi': False, 'yol_dondu': False,
        'yanit': '', 'canned_mi': False, 'uretilildi': False,
        'tekrar': False, 'konu_uyumu': None})
# KAYIT SAYISI KILIDI: 02.10'da iki gecis ayrimi yapilirken artik kod
# kalmis ve her kayit iki kez eklenmisti (sessizce 90 kayit). Bu kilit
# olcumu kirleten bir hatayi yakalar.
assert len(kayitlar) == len(SORULAR), \
    'kayit sayisi soru sayisiyla esit degil: %d != %d' % (len(kayitlar),
                                                         len(SORULAR))

# ------------------------------------------------------------- GECIS 1/2
# ONCE siniflandirma, HICBIR get_response cagrisi yapmadan.
#
# 02.10.2026 DUZELTME: ilk yazim tek gecisti - her soru icin once
# predict() sonra get_response(). Bu KIRLETILMIS bir olcumdu: bir onceki
# sorunun get_response'i sonraki sorunun predict() sonucunu degistiriyordu.
# Olculmus etki: "izleyecek bir sey ariyorum" sorusu kirli kosuda
# 'ari hjelm', temiz kosuda 'tavsiye_isteme' donuyor.
# Dogrulama (olcumu kirletmeden): predict() iki kez arka arkaya cagrilirsa
# 45/45 ayni sonuc -> predict KENDI BASINA deterministik; kirlilik yalnizca
# get_response -> predict yonunde. Cozum: iki ayri gecis.
for k, s in zip(kayitlar, SORULAR):
    soru = k['soru']
    bek = k['beklenen']
    np.random.seed(SEED)
    random.seed(SEED)
    try:
        tag, guven = bot.predict(soru)
    except Exception as exc:
        tag, guven = 'HATA:%s' % type(exc).__name__, 0.0
        k['hata_tahmin'] = str(exc)[:120]
    k['tahmin'] = tag
    k['guven'] = guven
    # 'Anlayamadim' bir sentinel'dir, intents.json'da etiket DEGILDIR.
    # Bunu "bozuk etiket" saymak yanlis olurdu.
    k['tahmin_etiket_mi'] = (tag in TUM_TAGLAR) or (tag == 'Anlayamadim')
    k['tahmin_sinifta'] = tag in SINIFLAR
    # kabul kurali: kapsam disi sorularda dogru cevap SOHBET SINIFI OLMAYAMAZ
    # (bilgi yolu veya Anlayamadim kabul). Bu, modelin ciktisina gore
    # ayarlanmis bir esik DEGIL: sohbet sinifi disi olmak nesnel bir
    # kuraldir.
    if k['kabul'] == 'sohbet_sinifi_DEGIL':
        k['isabet'] = not k['tahmin_sinifta']
    elif bek == 'Anlayamadim':
        k['isabet'] = (tag == 'Anlayamadim')
    else:
        k['isabet'] = (tag == bek)

    # ------------------------------------------------------------- GECIS 2/2
# SONRA uretim. Siniflandirma olculdu; burada yalnizca ekrana cikan metin,
# canned orani, tekrar ve konu uyumu olculuyor. get_response cagrilari
# GECIS 1'i ETKILEMEZ cunku gecis 1 tamamlanmis oluyor.
for k in kayitlar:
    soru = k['soru']
    np.random.seed(SEED)
    random.seed(SEED)

    aktif['kayit'] = k
    try:
        yanit = bot.get_response(soru) or ''
    except Exception as exc:
        yanit = ''
        k['hata_uretim'] = str(exc)[:120]
    aktif['kayit'] = None

    k['yanit'] = yanit
    k['canned_mi'] = canned_mi(yanit)
    # uretilildi = canned degilse LLM yolu (veya baska bir uretici) acik
    k['uretilildi'] = bool(yanit) and not k['canned_mi']

    # KONU UYUMU (VEKIL olcum). Icerik kelimelerin kesisimi. Tek basina
    # guvenilir DEGIL - sadece 'yanit sorudan tamamen koptu mu' sorusuna
    # cevap verir. Dosyanin "olcu_sinirlari" notuyla ayni.
    gk = set(icerik_kelimeler(yanit))
    sk = set(icerik_kelimeler(soru))
    if gk and sk:
        k['konu_uyumu'] = round(len(gk & sk) / float(len(gk)), 3)

    # TEKRAR - onceki yanitlardan biriyle neredeyse ayni mi? Her soru
    # farkli sinif oldugu icin ayni metnin tekrari 'hangi soruya da
    # ayni cevabi veriyor' demektir: gercek bir kalite hatasi.
    for o in kayitlar:
        if o is k:
            break
        ok = o.get('_kelimeler')
        if not ok or not gk:
            continue
        birlesim = len(gk | ok)
        if birlesim and len(gk & ok) / float(birlesim) >= 0.85:
            k['tekrar'] = True
            k['tekrar_ilki'] = o['soru'][:34]
            break
    k['_kelimeler'] = gk

for k in kayitlar:
    isaret = ''
    if not k['tahmin_etiket_mi']:
        isaret = '  ETIKET DEGIL'
    elif k['beklenen'] != 'Anlayamadim' and not k['beklenen_sinifta']:
        isaret = '  beklenen sinif YOK'
    elif k['isabet'] and not k['tahmin_sinifta']:
        isaret = '  (tahmin sinif disi)'
    print('  %-34s -> %-24s %s%s'
          % (k['soru'][:34], k['tahmin'][:24],
             'ISABET' if k['isabet'] else '-     ', isaret))

# ---------------------------------------------------------------- ozet
d = len(kayitlar)
# 'Anlayamadim' beklenenler gercek bir sinif degil: yanlis-pozitif tespiti
# icin AYRI sayilir, isabet ortalamasina karismaz.
d_is = [k for k in kayitlar if k['beklenen'] != 'Anlayamadim']
isabet = sum(1 for k in d_is if k['isabet'])
uretilen = sum(1 for k in kayitlar if k['uretilildi'])
canned = sum(1 for k in kayitlar if k['canned_mi'])
bos = sum(1 for k in kayitlar if not k['yanit'].strip())
tekr = sum(1 for k in kayitlar if k['tekrar'])
yol = sum(1 for k in kayitlar if k['yol_acildi'])
yol_dondu = sum(1 for k in kayitlar if k['yol_dondu'])
uyum = [k['konu_uyumu'] for k in kayitlar if k['konu_uyumu'] is not None]
# SINIF TASI: beklenen etiket VAR ama siniflandirici sinifinda DEGIL -> bu
# soruda dogru tahmin edilse bile sohbet yolu ACILMAZ (brain.py:1746
# `chosen_tag in self.intent_tags`). Yani bu sorular yapilandirilmis
# olarak KAZANILAMAZ; sizinti olcumudur.
# DIKKAT: burada TAHMIN edilen etikete degil BEKLENEN etikete bakilir.
# 02.10'da ilk yazimda tahmine bakiliyordu ve 6 copp tahmin yanlislikla
# 'sinif tasi' sayildi; oysa o sorularda beklenen sinif mevcuttu.
sizinti = [k for k in d_is if not k['beklenen_sinifta']]
# kullanilabilir isabet: dogru tahmin EDILDI ve bu sinif gercekten var
# (kullaniciya sohbet yolu acilir). Olcumun ana sayisi bu olmali.
kullanilir = [k for k in d_is if k['isabet'] and k['tahmin_sinifta']]
# tahmin edilen deger ne de etiket degilse (brain.py'dan gelen bozuk metin)
bozuk = [k for k in kayitlar if not k['tahmin_etiket_mi']]
# anlayamadim / kapsam disi sorulardan kacagi: sohbet sinifina GIRDILERI
kacak = [k for k in kayitlar
         if k['kabul'] == 'sohbet_sinifi_DEGIL' and not k['isabet']]

print()
print('=' * 74)
print('SOHBET YOLU OLCUMU  [%s]' % ETIKET)
print('=' * 74)
print('  soru sayisi                 : %d' % d)
print('  ANA OLCU - KULLANILABILIR ISABET: %d/%d = %%%0.0f'
      % (len(kullanilir), len(d_is), 100 * len(kullanilir) / max(1, len(d_is))))
print('    (dogru tahmin EDILDI ve sinif gercekten var -> sohbet yolu acilir)')
print('  ham tahmin isabeti          : %d/%d = %%%0.0f'
      % (isabet, len(d_is), 100 * isabet / max(1, len(d_is))))
print('    ... yanlis sinif           : %d'
      % sum(1 for k in d_is if not k['isabet'] and k['tahmin_sinifta']))
print('    ... ANLAYAMADIM            : %d'
      % sum(1 for k in d_is if not k['isabet'] and not k['tahmin_sinifta']))
print('    ... ETIKET OLMAYAN donus   : %d  (brain.py etiket disi metin dondu)'
      % sum(1 for k in d_is if not k['isabet'] and not k['tahmin_etiket_mi']))
print('  kapsam disi (sohbet sinifina GIRMEMELI) : %d/%d'
      % (sum(1 for k in kayitlar if k['kabul'] == 'sohbet_sinifi_DEGIL'
             and k['isabet']),
         sum(1 for k in kayitlar if k['kabul'] == 'sohbet_sinifi_DEGIL')))
print('  SINIF TASI (beklenen var, sinif YOK): %d soru' % len(sizinti))
print('    -> bu sorularda tahmin dogru olsa bile sohbet yolu ACILMAZ')
print('  ETIKET OLMAYAN TAHMIN        : %d soru' % len(bozuk))
print('  EKRANA URETIM (canned degil) : %d/%d = %%%0.0f'
      % (uretilen, d, 100 * uretilen / max(1, d)))
print('  canned (intents.json kopyasi): %d/%d = %%%0.0f'
      % (canned, d, 100 * canned / max(1, d)))
print('  bos yanit                   : %d' % bos)
print('  LLM yolu acildi             : %d/%d  (dondu: %d)'
      % (yol, d, yol_dondu))
print('  TEKRARLI YANIT              : %d/%d = %%%0.0f'
      % (tekr, d, 100 * tekr / max(1, d)))
print('  konu uyumu (VEKIL, ort.)    : %%%0.0f  (n=%d)'
      % (100 * sum(uyum) / max(1, len(uyum)), len(uyum)))

print()
print('GRUP BAZINDA KULLANILABILIR ISABET:')
grup = {}
for k in kayitlar:
    g = grup.setdefault(k['grup'], [0, 0])
    g[0] += 1
    g[1] += 1 if (k['isabet'] and (k['tahmin_sinifta']
                                   or k['beklenen'] == 'Anlayamadim')) else 0
for ad, (n, t) in sorted(grup.items()):
    print('  %-16s %2d/%2d = %%%3.0f' % (ad, t, n, 100 * t / max(1, n)))

if sizinti:
    print()
    print('SINIF TASI SORULARI (bu intent\'ler siniflandiriciya GIREMIYOR):')
    for k in sizinti:
        print('  %-34s beklenen=%-20s tahmin=%s'
              % (k['soru'][:34], k['beklenen'], k['tahmin'][:20]))
if bozuk:
    print()
    print('ETIKET OLMAYAN TAHMINLER (intents.json\'da boyle bir etiket yok):')
    for k in bozuk:
        print('  %-34s -> %s' % (k['soru'][:34], k['tahmin'][:30]))
if kacak:
    print()
    print('KAPSAM DISI KACAKLARI (sohbet sinifina yanlis gitti):')
    for k in kacak:
        print('  %-34s -> %s' % (k['soru'][:34], k['tahmin'][:24]))
if tekr:
    print()
    print('TEKRARLI YANITLAR:')
    for k in kayitlar:
        if k['tekrar']:
            print('  %-30s ilk: %s' % (k['soru'][:30], k.get('tekrar_ilki', '')))

os.makedirs(RAPOR_DIZIN, exist_ok=True)
with io.open(CIKTI, 'w', encoding='utf-8') as f:
    for k in kayitlar:
        k.pop('_kelimeler', None)
        k.pop('yol_metin', None)
    f.write(json.dumps({
        'etiket': ETIKET, 'seed': SEED, 'kayitlar': kayitlar,
        'ozet': {
            'soru': d,
            'sinif_bekleyen': len(d_is),
            # ANA OLCU: dogru tahmin + sinif gercekten var
            'kullanilabilir_isabet': len(kullanilir),
            'kullanilabilir_isabet_orani':
                round(100.0 * len(kullanilir) / max(1, len(d_is)), 2),
            'isabet': isabet,
            'isabet_orani': round(100.0 * isabet / max(1, len(d_is)), 2),
            'yanlis_sinif': sum(1 for k in d_is if not k['isabet']
                                 and k['tahmin_sinifta']),
            'anlayamadim_donus': sum(1 for k in d_is if not k['isabet']
                                     and not k['tahmin_sinifta']),
            'etiket_olmayan_donus': len(bozuk),
            'uretilildi': uretilen,
            'uretim_orani': round(100.0 * uretilen / max(1, d), 2),
            'canned': canned,
            'bos': bos,
            'tekrarli': tekr,
            'yol_acildi': yol,
            'yol_dondu': yol_dondu,
            'sinif_tasi': len(sizinti),
            'kapsam_disi_kacak': len(kacak),
            'konu_uyumu': (round(100.0 * sum(uyum) / max(1, len(uyum)), 2)
                           if uyum else None),
        },
        'sinif_sayisi': len(SINIFLAR),
        'veri_damgasi': veri_damgasi(),
        'model_damgasi': model_damgasi(),
        'soru_listesi_damgasi': soru_listesi_damgasi(),
    }, ensure_ascii=False, indent=1))
print()
print('rapor yazildi: %s' % CIKTI)
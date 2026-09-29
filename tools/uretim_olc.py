# -*- coding: utf-8 -*-
"""URETIM YOLU KALITE OLCUMU - iki modeli ayni sorularla karsilastirir.

NEDEN BU SCRIPT: onceki kapi olcumleri elle sabitlenmis metinlerden
hesapliyordu (sadakat.py) ve 15 soruyla sinirliydi (uretim_kabul.py).
Ikisi de iki modeli A/B karsilastirmaya yaramaz. Burada:
  * sorular CORPUS.jsonl'den deterministik secilir (veri dosyasi SALT
    OKUNUR, hicbir sey yazilmaz),
  * uretim GERCEK yoldan olur: app.load_bot() -> bot.get_response(),
  * her uretim denemesi (best-of-3) kaydedilir: kabul/red + sebep,
  * sadakat = uretilen metnin icerik kelimelerinden bilgi parcasinda
    GERCEKTEN bulunanlarin orani,
  * cikti JSON -> paired karsilastirma yapilabilir.

KULLANIM:  python uretim_olc.py <etiket> [cikti.json]
  np.random.seed(7) her kosudan once -> iki model AYNI rastgelelik
  dizisini gorur, fark yalnizca modelden gelir.
"""
import io
import json
import os
import re
import sys
import time

import numpy as np

# Repo koku. SABIT YOL KULLANILMAZ: GitHub Actions'ta o yol yok ve
# tests/test_no_hardcoded_paths.py de reddeder. Bu dosya tools/ altinda,
# kok bir ust dizindir.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

ETIKET = sys.argv[1] if len(sys.argv) > 1 else 'bilinmeyen'
# Cikti repo-relative: .gitignore'da (olcum_raporlari/). Onceki hali
# %TEMP%\opencode idi - o dizin gecici oldugu icin olcum raporlari
# kayboluyordu ve repoda izi kalmiyordu.
RAPOR_DIZIN = os.path.join(BASE, 'olcum_raporlari')
CIKTI = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    RAPOR_DIZIN, 'uretim_%s.json' % ETIKET)

SEED = 7
N_SORU = 50

# ---------------------------------------------------------------- sorular
# KAYNAK: knowledge_map.jsonl. NEDEN corpus.jsonl degil: 29.08'te konulari
# corpus id'lerinden sectim ("pachnaeus nedir", "vaughanella nedir") ve
# 50 sorunun 48'i "bilgim yok" ile dondu, yani URETIM YOLU HIC ACILMADI
# (50 soruda 6 deneme). Retrieval ancak kb-map'te karsiligi olan
# sorgularda tetikleniyor; kb-map'in ctx sutunu tam olarak
# "<konu> nedir / ne demek / hakkinda bilgi ver" sorularindan olusuyor.
# Bu yuzden sorulari ORADAN almak garanti ediyor.
#
# Satirlar konu basina ~3 varyantta gruplu, bu yuzden her 600. satir
# farkli bir konu veriyor (deterministik, ekşi komsular yok).
sorular = []
with io.open('knowledge_map.jsonl', encoding='utf-8') as f:
    for i, satir in enumerate(f):
        if i % 600:
            continue
        satir = satir.strip()
        if not satir:
            continue
        try:
            d = json.loads(satir)
        except Exception:
            continue
        c = (d.get('ctx') or '').strip()
        if len(c) >= 8:
            sorular.append(c)
        if len(sorular) >= N_SORU:
            break
SORULAR = sorular
secilen = SORULAR

# ---------------------------------------------------------------- kurulum
import app as uygulama

t0 = time.time()
uygulama.load_bot()
bot = uygulama.bot
print('[%s] bot yuklendi (%.1f sn), knowledge_bias=%s'
      % (ETIKET, time.time() - t0, getattr(bot, 'knowledge_bias', '?')))
print('[%s] %d soru: %s ... %s' % (ETIKET, len(SORULAR),
                                    secilen[0], secilen[-1]))

STOP = set(('ve ile bir bu da de ki en cok daha cok gibi olarak ancak icin '
            'sonra yani ise her ancak veya yoktur degil en fazla cok').split())


def icerik_kelimeler(metin):
    t = re.sub(r'[^a-z0-9 ]+', ' ', (metin or '').lower())
    return [w for w in t.split() if len(w) > 2 and w not in STOP]


# ------------------------------------------------- olcum katmani (sarmalama)
kayitlar = []
asil_kabul = bot._accept_kb_rephrase
asil_rephrase = bot._try_kb_rephrase
aktif = {'kayit': None}


def _sebep(gen, kb):
    """Kapinin hangi sarti ihlal ettigini bulur (beyan kodunun AYNISI).

    29.09: onceki kural `int(len(letters) * 0.30)` idi ve beyan kodundaki
    duzeltmeyi YANSITMIYORDU. Sebep etiketleri boyle yanlis cikiyordu
    ("bozulmus karakter" diyordu ama kural artik min(12, ...) ile
    calisiyor). Kod ile burasi BIREBIR ayni olmali.
    """
    if not gen or len(gen) < 12 or len(gen) > 260:
        return 'cok kisa/uzun (%d)' % len(gen or '')
    letters = [c for c in gen.lower() if c.isalpha()]
    if len(letters) < 6 or len(set(letters)) < min(12, int(len(letters) * 0.30)):
        return 'tekrar/bozulmus karakter'
    kb_set = set(bot.tokenize(kb))
    if not kb_set:
        return 'bilgi bos'
    cand = bot.tokenize(gen)
    if len(cand) < 3:
        return 'cok az token'
    gen_set = set(cand)
    kb_inter = gen_set & kb_set
    if len(kb_inter) < 2:
        return 'konu kelimesi yok (%d tanidik)' % len(kb_inter)
    if kb_inter and len(kb_inter) / float(len(gen_set)) < 0.20:
        return 'konu bilesimi cok dusuk'
    novel = gen_set - kb_set
    if len(novel) / float(len(gen_set)) < 0.15:
        return 'ozgunluk yok (%%%0.0f novel)' % (100 * len(novel)
                                                / max(1, len(gen_set)))
    if len(cand) >= 4:
        dup = sum(1 for a, b in zip(cand, cand[1:]) if a == b)
        if dup / float(len(cand) - 1) > 0.5:
            return 'yapisan tekrar'
    return 'diger'


def kabul_sarmal(gen, kb):
    ok = asil_kabul(gen, kb)
    k = aktif['kayit']
    if k is not None:
        k['denemeler'].append({'gen': gen, 'kabul': bool(ok),
                               'sebep': '' if ok else _sebep(gen, kb)})
    return ok


def rephrase_sarmal(query, kb, tries=3):
    r = asil_rephrase(query, kb, tries)
    k = aktif['kayit']
    if k is not None:
        k['kb'] = kb
        k['yanit'] = r
        # uretilildi mi: ham kb degilse modelin cumlesi ekrana cikti
        k['uretilildi'] = bool(r) and (r.strip() != (kb or '').strip())
    return r


bot._accept_kb_rephrase = kabul_sarmal
bot._try_kb_rephrase = rephrase_sarmal

BILGISIZ = ('bilgim yok', 'uydurmak istemem', 'bilmiyorum')

for soru in SORULAR:
    np.random.seed(SEED)          # her soruda ayni tohum: modele gore degismez
    # konu etiketi: soru bicimine gore kirp (soru[:-6] sadece "X nedir" icin
    # dogruydu, "X ne demek" / "X hakkinda bilgi ver" bozulurdu)
    konu = soru
    for son in (' hakkinda bilgi ver', ' nedir', ' ne demek', ' hakkinda '
                'bilgi', ' nedir?', ' kimdir'):
        if konu.endswith(son):
            konu = konu[:-len(son)]
            break
    kayit = {'soru': soru, 'konu': konu.strip(), 'denemeler': [],
             'kb': '', 'yanit': '', 'uretilildi': False, 'sadakat': None}
    aktif['kayit'] = kayit
    try:
        yanit = bot.get_response(soru) or ''
    except Exception as e:
        yanit = ''
        kayit['hata'] = str(e)[:120]
    aktif['kayit'] = None

    kayit['yanit'] = yanit
    kayit['bos_kova'] = any(t in yanit.lower() for t in BILGISIZ)
    kabul = [d for d in kayit['denemeler'] if d['kabul']]
    if kabul and kayit['uretilildi']:
        # ekrana cikan metni sadakat olc (best-of-3'te en uzun kabul edilen)
        metin = max(kabul, key=lambda d: len(d['gen']))['gen']
        gk = icerik_kelimeler(metin)
        dk = set(icerik_kelimeler(kayit['kb']))
        if gk and dk:
            kayit['sadakat'] = len([w for w in gk if w in dk]) / float(len(gk))
    kayitlar.append(kayit)
    bay = 'BILGI YOK' if kayit['bos_kova'] else (
        'URETILDI' if kayit['uretilildi'] else 'canned')
    print('  %-26s %-9s deneme=%d kabul=%d sadakat=%s'
          % (kayit['konu'][:26], bay, len(kayit['denemeler']), len(kabul),
             ('%3.0f%%' % (100 * kayit['sadakat']))
             if kayit['sadakat'] is not None else '-'))

# ---------------------------------------------------------------- ozet
d = len(kayitlar)
toplam_deneme = sum(len(k['denemeler']) for k in kayitlar)
toplam_kabul = sum(1 for k in kayitlar for d2 in k['denemeler'] if d2['kabul'])
uretilen = sum(1 for k in kayitlar if k['uretilildi'])
bos = sum(1 for k in kayitlar if k['bos_kova'])
sad = [k['sadakat'] for k in kayitlar if k['sadakat'] is not None]

seb = {}
for k in kayitlar:
    for d2 in k['denemeler']:
        if not d2['kabul']:
            seb[d2['sebep'].split('(')[0]] = seb.get(d2['sebep'].split('(')[0], 0) + 1

print()
print('=' * 74)
print('URETIM YOLU OLCUMU  [%s]' % ETIKET)
print('=' * 74)
print('  soru sayisi              : %d' % d)
print('  bos kova ("bilgim yok") : %d  (%%%0.0f)' % (bos, 100 * bos / max(1, d)))
print('  uretim denemesi          : %d' % toplam_deneme)
print('  DENEME KABUL ORANI      : %d/%d = %%%0.0f'
      % (toplam_kabul, toplam_deneme, 100 * toplam_kabul / max(1, toplam_deneme)))
print('  EKRANA URETIM GECEN SORU : %d/%d = %%%0.0f'
      % (uretilen, d, 100 * uretilen / max(1, d)))
print('  sadakat (ortalama)       : %%%0.0f  (n=%d)'
      % (100 * sum(sad) / max(1, len(sad)), len(sad)))
print('  sadakat < %%34          : %d / %d'
      % (sum(1 for s in sad if s < 0.34), len(sad)))
if seb:
    print('  RED NEDENLERI:')
    for ad, adet in sorted(seb.items(), key=lambda x: -x[1])[:8]:
        print('    %-40s %d' % (ad[:40], adet))

os.makedirs(os.path.dirname(CIKTI), exist_ok=True)
with io.open(CIKTI, 'w', encoding='utf-8') as f:
    f.write(json.dumps({'etiket': ETIKET, 'seed': SEED, 'kayitlar': kayitlar},
                       ensure_ascii=False, indent=1))
print()
print('rapor yazildi: %s' % CIKTI)

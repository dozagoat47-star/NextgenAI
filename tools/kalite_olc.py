# -*- coding: utf-8 -*-
"""KALITE OLCUMU - gercek sunucu yolu (app.load_bot).

Dort seyiniciyi sayiya cevirir:
  1. gecikme        -> kac sn, hangi asamada
  2. yapistirma     -> model canned yanitini kopyaliyor mu (copy_bleu ayri)
  3. yarim kelime   -> uretim max_len'de kesiliyor mu (OLCULEBILIR: token
                       sayisi == max_len ise kesilmis demektir)
  4. retrieval      -> "X nedir" sorularinda kova bos kaliyor mu ("bilgim yok")
"""
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

import app as uygulama

uygulama.load_bot()
bot = uygulama.bot

# --- uretimi olc: max_len'e dayanip dayanmadigini olc ---
asil_sample = bot.llm.sample
uretim = {'n': 0, 'kesildi': 0, 'token': [], 'max_len': None}


def sample_olc(*a, **k):
    # max_len konumunu bul (ya konumsal ya adli)
    ml = k.get('max_len')
    if ml is None:
        for i, v in enumerate(a):
            if isinstance(v, int) and i > 0 and not isinstance(v, bool):
                # sonraki pozisyonlarda string yoksa bu max_len olabilir
                if i + 1 >= len(a) or isinstance(a[i + 1], str):
                    ml = v
    out = asil_sample(*a, **k)
    n = len(out) if isinstance(out, str) else len(out[0])
    uretim['n'] += 1
    uretim['token'].append(n)
    if ml:
        uretim['max_len'] = ml
    return out


bot.llm.sample = sample_olc

# --- soru kumesi ---
BILGI = ['futbol nedir', 'python nedir', 'fizik nedir', 'ekonomi nedir',
         'turkiye nedir', 'siber guvenlik nedir', 'foto elektrik nedir',
         'kuantum bilgisayarlar nedir', 'islam nedir', 'kadin nedir']

SOHBET = ['merhaba', 'nasilsin', 'teşekkür ederim', 'seni seviyorum',
          'bana bir şarkı söyle', 'komik bir hikaye anlat', 'ne yapıyorsun',
          'günaydın', 'iyi geceler', 'görüşürüz']

TALEP = ['bana bir film oner', 'bana bir müzik oner', 'tatil için nereye gidilir',
         'yemek tarifi ver', 'manken nasıl yazılır', 'telefon nasıl sarj edilir',
         'kediler neden miawlar', 'köpeğim neden havlıyor',
         'istanbulda yagmurlu mu', 'yarın hava nasıl']

BILGI_KONTROL = ['bilgi tabani ne ise yarar', 'veri maddeciligi nedir',
                 'bulut sistemleri neden onemli', 'api nasil calisir']


def olc(sorular, ad):
    sonuclar = []
    for s in sorular:
        t0 = time.perf_counter()
        try:
            y = bot.get_response(s) or ''
        except Exception as e:
            y = ''
        dt = time.perf_counter() - t0
        sonuclar.append((s, dt, y))
    return sonuclar


BILGI_SONUC = olc(BILGI, 'bilgi')
SOHBET_SONUC = olc(SOHBET, 'sohbet')
TALEP_SONUC = olc(TALEP, 'talep')
KONTROL_SONUC = olc(BILGI_KONTROL, 'kontrol')

BILGISIZ_IPUCU = ('bilgim yok', 'uydurmak istemem', 'bilmiyorum')


def bos_kova(y):
    d = y.lower()
    return any(t in d for t in BILGISIZ_IPUCU)


def noktali_bitiyor(y):
    d = y.strip()
    return bool(d) and d[-1] in '.!?…"\')'


def kesik_mi(y):
    """Yarim kelime kesilmesi: noktalama olmadan bitiyor + bos degil."""
    d = y.strip()
    return bool(d) and not noktali_bitiyor(d) and d[-1].isalpha()


print('=' * 72)
print('KALITE OLCUMU  (uretim max_len = %s)' % uretim['max_len'])
print('=' * 72)

toplam = (BILGI_SONUC + SOHBET_SONUC + TALEP_SONUC + KONTROL_SONUC)
print('toplam soru        : %d' % len(toplam))
print('ortalama sure      : %.3f sn' % np.mean([d for _s, d, _y in toplam]))
print('medyan sure        : %.3f sn' % np.median([d for _s, d, _y in toplam]))

bos = [r for r in toplam if bos_kova(r[2])]
kes = [r for r in toplam if kesik_mi(r[2])]
nok = [r for r in toplam if noktali_bitiyor(r[2])]

print()
print('--- 3) YARIM KELIME / KESILMIS YANIT ---')
print('  noktali biten     : %d / %d  (%%%d)'
      % (len(nok), len(toplam), 100.0 * len(nok) / len(toplam)))
print('  kesik biten       : %d / %d  (%%%d)'
      % (len(kes), len(toplam), 100.0 * len(kes) / len(toplam)))
print('  uretim max_len    : %s' % uretim['max_len'])
if uretim['token']:
    print('  uretilen uzunluk : ort %d  en cok %d'
          % (int(np.mean(uretim['token'])), max(uretim['token'])))

print()
print('--- 4) RETRIEVAL BOS KOVASI ("bilgim yok" donusu) ---')
print('%-34s %s' % ('kume', 'bos/total'))
print('-' * 72)
for ad, sonuc in (('BILGI ("X nedir")', BILGI_SONUC),
                  ('KONTROL (bilgi kumesi)', KONTROL_SONUC),
                  ('SOHBET', SOHBET_SONUC),
                  ('TALEP', TALEP_SONUC)):
    b = sum(1 for r in sonuc if bos_kova(r[2]))
    print('%-34s %d/%d  = %%%d' % (ad, b, len(sonuc),
                                   100.0 * b / len(sonuc)))
b = sum(1 for r in toplam if bos_kova(r[2]))
print('%-34s %d/%d  = %%%d' % ('TOPLAM', b, len(toplam),
                               100.0 * b / len(toplam)))

print()
print('--- 3b) KESILEN YANITLARIN ORNEKLERI ---')
for s, d, y in kes[:6]:
    print('  K: %s' % s)
    print('  B: ...%s' % y.strip()[-70:])
    print()

print('--- 4b) BOS KOVAYA DUSEN "X nedir" ORNEKLERI ---')
for s, d, y in BILGI_SONUC:
    if bos_kova(y):
        print('  K: %s' % s)
        print('  B: %s' % y[:110].replace('\n', ' '))
        print()

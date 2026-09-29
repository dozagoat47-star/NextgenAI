# -*- coding: utf-8 -*-
"""MODEL KENDI CUMLESINI KURUYOR MU - olcerek.

Yanit iki kaynaktan gelebilir:
  (a) canned  = _select_response -> intent'in hazir metni (YAPISTIRMA)
  (b) uretilmis = _try_kb_rephrase / _try_seq_rephrase -> LLM/SeqGen

Olcum: her yanit icin (a) yolu calisti mi, (b) yolu calisti mi, ve cikan
metin canned metne kac kadar benziyor.
"""
import io
import os
import re
import sys
import time

# Repo koku. SABIT YOL KULLANILMAZ: GitHub Actions'ta o yol yok ve
# tests/test_no_hardcoded_paths.py de reddeder. Bu dosya tools/ altinda,
# kok bir ust dizindir.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import app as uygulama

iz = {}


def sar(nesne, ad, etiket):
    asil = getattr(nesne, ad)

    def sarmal(*a, **k):
        r = asil(*a, **k)
        d = iz.setdefault(etiket, 0)
        iz[etiket] = d + 1
        iz[etiket + '_son'] = r
        return r
    setattr(nesne, ad, sarmal)


uygulama.load_bot()
bot = uygulama.bot

for ad, et in (('_select_response', 'CANNED_SECILDI'),
               ('_try_kb_rephrase', 'LLM_KB_REPHRASE'),
               ('_try_seq_rephrase', 'SEQ_REPHRASE'),
               ('_unknown_reply', 'BILGI_YOK')):
    if hasattr(bot, ad):
        sar(bot, ad, et)


def normalize(t):
    t = re.sub(r'[^a-z0-9 ]+', ' ', (t or '').lower())
    return [w for w in t.split() if w]


def kesisim(a, b):
    wa, wb = normalize(a), normalize(b)
    if not wa:
        return 0.0
    return len(set(wa) & set(wb)) / len(set(wa))


SORULAR = ['futbol nedir', 'python nedir', 'turkiye nedir', 'fizik nedir',
           'basketbol nedir', 'kadinlar yol bisikleti nedir', 'muzik nedir',
           'spor nedir', 'yemek nedir', 'film nedir']

print('=' * 100)
print('%-28s %-9s %-11s %-11s %7s' % ('soru', 'yol', 'canned?', 'LLM?', 'kap. %'))
print('=' * 100)

yol_sayac = {}
kapilar = []
for s in SORULAR:
    iz.clear()
    yanit = bot.get_response(s) or ''

    canned_secildi = 'CANNED_SECILDI' in iz
    llm = 'LLM_KB_REPHRASE' in iz or 'SEQ_REPHRASE' in iz
    bilgi_yok = 'BILGI_YOK' in iz

    # canned metni: _select_response'in dondurdugu ya da secilen tag'in yaniti
    tag, prob, unc, rw = bot._classify(s)
    canned_list = bot.intents.get(tag) or []
    canned = iz.get('CANNED_SECILDI_son')
    if isinstance(canned, (list, tuple)) and canned:
        canned = canned[0]
    if not canned and canned_list:
        canned = str(canned_list[0])

    kap = kesisim(yanit, canned) if canned else 0.0
    kapilar.append(kap)

    if bilgi_yok:
        yol = 'BILGI_YOK'
    elif llm:
        yol = 'URETILMIS'
    elif canned_secildi:
        yol = 'CANNED'
    else:
        yol = 'DOGRUDAN'

    yol_sayac[yol] = yol_sayac.get(yol, 0) + 1
    isaret = '  <BILGI YOK>' if bilgi_yok else ''
    print('%-28s %-9s %-11s %-11s %6.0f%%%s'
          % (s[:28], yol, 'evet' if canned_secildi else '-',
             'evet' if llm else '-', 100 * kap, isaret))

print()
print('  yol dagilimi        : %s' % yol_sayac)
print('  ortalama kap. %%      : %%%.0f' % (100 * sum(kapilar) / len(kapilar)))
yuksek = [k for k in kapilar if k > 0.6]
print('  %%60 uzeri kopyalama : %d / %d soru' % (len(yuksek), len(kapilar)))

print()
print('=' * 100)
print('ORNEK: canned metin -> bot yaniti')
print('=' * 100)
for s in SORULAR[:5]:
    iz.clear()
    yanit = bot.get_response(s) or ''
    tag, prob, unc, rw = bot._classify(s)
    canned_list = bot.intents.get(tag) or []
    print()
    print('  SORU    : %s   (tag=%s)' % (s, tag))
    print('  CANNED  : %s' % (str(canned_list[0])[:120] if canned_list else '-'))
    print('  BOT     : %s' % yanit[:120].replace('\n', ' '))
    a, b = normalize(yanit), normalize(canned_list[0] if canned_list else '')
    ort = 'AYNI' if (a and a == b[:len(a)]) else 'FARKLI'
    print('  -> %s' % ort)

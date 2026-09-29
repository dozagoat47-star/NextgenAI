# -*- coding: utf-8 -*-
"""HARF CESITLILIGI KAPISI: eski kural vs yeni kural, AYNI sorularda.

SORU: eski kural `farkli_harf >= 0.30 * harf` idi. Turkce alfabe 29 harf
oldugu icin esik 29/0.30 = 96 karakterde tavanda yapiyor; yani 97-260
karakter arasindaki HER uretim matematiksel olarak reddediliyordu. Bu
betik iki kurali da ayni soru listesine uygular ve GERCEKTE kabul edilen
uretimleri sayar (kapinin dondugu deger, ekranda ne gorundugunun kaynagi).

Cikti: basinda uretilip KABUL edilen -> ekranda modelin cumlesi,
       uretilip REDDEDILEN          -> ekranda canned metin.
"""
import os
import sys
import time

# Repo koku. SABIT YOL KULLANILMAZ: GitHub Actions'ta o yol yok ve
# tests/test_no_hardcoded_paths.py de reddeder. Bu dosya tools/ altinda,
# kok bir ust dizindir.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import app as uygulama

ASILI = {}


def _kapi_uret(kural):
    """Kapinin tam kopyasi; tek fark harf cesitliligi esigi."""
    tokenize = ASILI['tokenize']

    def kapi(gen, kb):
        if not gen or len(gen) < 12 or len(gen) > 260:
            return False
        letters = [c for c in gen.lower() if c.isalpha()]
        esik = (int(len(letters) * 0.30) if kural == 'eski'
                else min(12, int(len(letters) * 0.30)))
        if len(letters) < 6 or len(set(letters)) < esik:
            return False
        kb_set = set(tokenize(kb))
        if not kb_set:
            return False
        cand = tokenize(gen)
        if len(cand) < 3:
            return False
        gen_set = set(cand)
        kb_inter = gen_set & kb_set
        if len(kb_inter) < 2:
            return False
        if len(kb_inter) / float(len(gen_set)) < 0.20:
            return False
        novel = gen_set - kb_set
        if len(novel) / float(len(gen_set)) < 0.15:
            return False
        if len(cand) >= 4:
            dup = sum(1 for a, b in zip(cand, cand[1:]) if a == b)
            if dup / float(len(cand) - 1) > 0.5:
                return False
        return True
    return kapi


SORULAR = [
    'futbol nedir', 'basketbol nedir', 'spor nedir', 'yemek nedir',
    'muzik nedir', 'sinema nedir', 'film nedir', 'kitap nedir',
    'python nedir', 'yazilim nedir', 'bilgisayar nedir', 'internet nedir',
    'ekonomi nedir', 'demokrasi nedir', 'fizik nedir', 'kimya nedir',
    'biyoloji nedir', 'matematik nedir', 'tarih nedir', 'cografya nedir',
    'turkiye nedir', 'istanbul nedir', 'ankara nedir', 'roma nedir',
    'islam nedir', 'kristiyanlik nedir', 'gunes nedir', 'ay nedir',
    'yildiz nedir', 'uzay nedir', 'hava nedir', 'su nedir',
    'ates nedir', 'toprak nedir', 'kalp nedir', 'beyin nedir',
    'kedi nedir', 'kopek nedir', 'aslan nedir', 'balik nedir',
]

uygulama.load_bot()
bot = uygulama.bot
ASILI['tokenize'] = bot.tokenize
bot._accept_kb_rephrase = _kapi_uret('yeni')   # ilk etiket

print('bot yuklendi, %d soru' % len(SORULAR))
print()

sonuc = {}
for kural in ('eski', 'yeni'):
    say = {'denendi': 0, 'kabul': 0, 'red': 0, 'bos': 0, 'cevap': 0}
    kabul_ornek, red_ornek = [], []
    gercek = _kapi_uret(kural)

    def sayacli(gen, kb, _g=gercek, _s=say, _ko=kabul_ornek,
                _ro=red_ornek):
        _s['denendi'] += 1
        ok = _g(gen, kb)
        if ok:
            _s['kabul'] += 1
            _ko.append((gen, kb))
        else:
            _s['red'] += 1
            _ro.append((gen, kb))
        return ok

    bot._accept_kb_rephrase = sayacli
    t0 = time.time()
    for s in SORULAR:
        y = bot.get_response(s) or ''
        d = y.lower()
        if any(t in d for t in ('bilgim yok', 'uydurmak istemem')):
            say['bos'] += 1
        else:
            say['cevap'] += 1
    sonuc[kural] = (say, kabul_ornek, red_ornek, time.time() - t0)

print('=' * 84)
print('KARSILASTIRMA - ayni sorular, ayni model')
print('=' * 84)
print('%-8s %9s %8s %8s %8s %8s' % ('kural', 'cevap', 'bos', 'uretim',
                                    'KABUL', 'RED'))
print('-' * 84)
for kural in ('eski', 'yeni'):
    say, ko, ro, sure = sonuc[kural]
    d = say['denendi']
    print('%-8s %9d %8d %8d %7d(%3.0f%%) %5d(%3.0f%%)  %.0f sn'
          % (kural, say['cevap'], say['bos'], d, say['kabul'],
             100 * say['kabul'] / max(1, d), say['red'],
             100 * say['red'] / max(1, d), sure))
print()
eski = sonuc['eski'][0]
yeni = sonuc['yeni'][0]
print('  kabul orani : %%%.0f -> %%%.0f   (kat %sx)'
      % (100 * eski['kabul'] / max(1, eski['denendi']),
         100 * yeni['kabul'] / max(1, yeni['denendi']),
         (yeni['kabul'] / max(1, eski['kabul']))))
print()
print('  YENI KURALLA KABUL EDILEN (modelin kendi cumleleri):')
for gen, kb in sonuc['yeni'][1][:6]:
    print('    + %s' % gen[:96])
print()
print('  ESKI KURALLA REDDEDILEN, yeniyle KABUL EDILEN ornekleri:')
yeni_kabul = {g for g, _ in sonuc['yeni'][1]}
k = 0
for gen, kb in sonuc['eski'][2]:
    if gen in yeni_kabul and k < 4:
        print('    - eski: RED    -> yeni: KABUL  "%s"' % gen[:86])
        k += 1

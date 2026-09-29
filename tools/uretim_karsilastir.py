# -*- coding: utf-8 -*-
"""Iki uretim yolu raporunu PAIRED karsilastirir.

Ayni sorular, ayni tohum -> soru bazinda eslesme. Olculecekler:
  1) ekrana uretim gecen soru orani (ikili) -> McNemar + t
  2) sadakat (yalnizca ikisinde de uretilen sorularda) -> paired t
  3) deneme basi kabul orani
  4) ret nedenleri farki (hangi sebep azaldi/artti)

YON KURALI: fark = A - B yazilir (eski - yeni), eval_llm.py ile ayni.
Bu yuzden ISARET: negatif fark, B'nin (yeni) DAHA YUKSEK oldugu metrikte
daha iyi oldugunu gosterir.
"""
import io
import json
import math
import os
import sys

tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'olcum_raporlari')
A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tmp, 'uretim_eski.json')
B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(tmp, 'uretim_yeni.json')


def yukle(yol):
    with io.open(yol, encoding='utf-8') as f:
        return json.load(f)


ra, rb = yukle(A), yukle(B)
ka = {k['soru']: k for k in ra['kayitlar']}
kb = {k['soru']: k for k in rb['kayitlar']}
ortak = sorted(set(ka) & set(kb))
print('A = %s   (%d soru)' % (ra['etiket'], len(ka)))
print('B = %s   (%d soru)' % (rb['etiket'], len(kb)))
print('ortak soru: %d' % len(ortak))
if not ortak:
    raise SystemExit('ortak soru yok: karsilastirilamaz')
if list(ka) != list(kb):
    print('  (sira farkli ama eslesme soru bazinda yapildi)')


def ttest(farklar):
    n = len(farklar)
    if n < 2:
        return None
    m = sum(farklar) / n
    var = sum((x - m) ** 2 for x in farklar) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    if se == 0:
        return n, m, 0.0, (float('inf') if m else 0.0)
    return n, m, se, m / se


def yon(y, iyi_yuksek):
    if abs(y) < 1e-12:
        return 'ayni'
    artti = y > 0
    return ('artti' if artti else 'azaldi') + (
        ' (Iyi)' if artti == iyi_yuksek else ' (KOTU)')


print()
print('=' * 76)
print('1) EKRANA URETIM GECEN SORU (ikili: 1 = modelin cumlesi kullanicinya gitti)')
print('=' * 76)
ia = [1 if ka[s]['uretilildi'] else 0 for s in ortak]
ib = [1 if kb[s]['uretilildi'] else 0 for s in ortak]
print('  A: %d/%d = %%%.0f' % (sum(ia), len(ia), 100 * sum(ia) / len(ia)))
print('  B: %d/%d = %%%.0f' % (sum(ib), len(ib), 100 * sum(ib) / len(ib)))
fark = [x - y for x, y in zip(ia, ib)]
n, m, se, t = ttest(fark)
print('  fark (A-B): %+.4f +- %.4f  |  t = %+.2f' % (m, se, t))
print('  -> %s' % yon(m, True))
# McNemar: yalnizca ayrik (discordant) ciftler bilgi verir
b01 = sum(1 for x, y in zip(ia, ib) if x == 0 and y == 1)   # sadece B uretti
b10 = sum(1 for x, y in zip(ia, ib) if x == 1 and y == 0)   # sadece A uretti
print('  ayrik ciftler: sadece B uretti = %d | sadece A uretti = %d'
      % (b01, b10))
if b01 + b10:
    # McNemar (normal yaklasim)
    z = (abs(b01 - b10) - 1) / math.sqrt(b01 + b10)
    print('  McNamar z = %+.2f  (|z| >= 1,96 -> %%95 anlamli)' % z)

print()
print('=' * 76)
print('2) SADAKAT (bilgi parcasinda gercekten bulunan icerik kelimesi orani)')
print('=' * 76)
cift = [s for s in ortak
        if ka[s]['sadakat'] is not None and kb[s]['sadakat'] is not None]
print('  olculebilen ortak soru: %d / %d  (ikisinde de uretim gecmis olmali)'
      % (len(cift), len(ortak)))
if cift:
    sa = [ka[s]['sadakat'] for s in cift]
    sb = [kb[s]['sadakat'] for s in cift]
    print('  A: %%%.0f' % (100 * sum(sa) / len(sa)))
    print('  B: %%%.0f' % (100 * sum(sb) / len(sb)))
    fark2 = [x - y for x, y in zip(sa, sb)]
    n2, m2, se2, t2 = ttest(fark2)
    print('  fark (A-B): %+.4f +- %.4f  |  t = %+.2f' % (m2, se2, t2))
    print('  -> %s' % yon(m2, True))
    print('  B tarafinda sadakat <%%34: %d / %d'
          % (sum(1 for x in sb if x < 0.34), len(sb)))
else:
    print('  HIC ORTUM YOK: iki model de ayni sorularda uretmedi, karsilastirma '
          'anlamli degil')

print()
print('=' * 76)
print('3) DENEME BASI KABUL ORANI')
print('=' * 76)
for rap, ad in ((ra, 'A'), (rb, 'B')):
    td = sum(len(k['denemeler']) for k in rap['kayitlar'])
    tk = sum(1 for k in rap['kayitlar'] for d2 in k['denemeler'] if d2['kabul'])
    print('  %s: %d/%d = %%%.0f' % (ad, tk, td, 100 * tk / max(1, td)))

print()
print('=' * 76)
print('4) RET NEDENLERI (tum denemeler uzerinden)')
print('=' * 76)
seb = {}
for rap in (ra, rb):
    c = {}
    for k in rap['kayitlar']:
        for d2 in k['denemeler']:
            if not d2['kabul']:
                a = d2['sebep'].split('(')[0].strip()
                c[a] = c.get(a, 0) + 1
    seb[rap['etiket']] = c
adet = sorted(set(seb[ra['etiket']]) | set(seb[rb['etiket']]))
print('  %-36s %8s %8s %8s' % ('sebep', ra['etiket'], rb['etiket'], 'fark'))
for a in adet:
    x = seb[ra['etiket']].get(a, 0)
    y = seb[rb['etiket']].get(a, 0)
    print('  %-36s %8d %8d %+8d' % (a[:36], x, y, y - x))

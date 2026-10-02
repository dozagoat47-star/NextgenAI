# -*- coding: utf-8 -*-
"""sure_ve_hesapla'nin TAVANINI gercek kosu logundan olcer.

NEDEN VAR (01.10.2026):
  tests/test_autogrow_kapi.py:356-360 `sure_ve_hesapla`nin SABITLERDEN
  turetildigini dogruluyor: beklenen = kalan/epochs * 60000 / MS_PER_PAIR.
  Bu DONGUSEL bir test: formulu ayni sabitlerle yeniden hesapliyor.
  SABITLERIN dogru oldugunu hicbir test olcumuyor.

  01.10 23:26 kosusundan olculdu:
    * sure_ve_hesapla(540 dk, 12 epoch) = 1.251.733 cift dedi
    * gercek kosuda 288.802 HAM cift kullanildi ve oturumun %86,5'i
      tuketildi (427,0 dk epoch + 40,3 dk encode = 467,3 dk / 540 dk)
    Yani tavan gercek sinirdan ~5 KAT uzakta.

KOK NEDEN (olculerek gosteriliyor, tahmin degil):
  `ms_per_pair` bir EGTIM cifti (post-expansion) maliyetidir:
      1.7217 ms = 2135,0 sn / 1.240.102 train cift
  Ama `sure_ve_hesapla`nin sonucu `coz_max_pairs`'a gider ve orada
  HAM cift birimiyle kullanilir (MAX_PAIRS = ham cift, bak
  kaggle_train.txt "[butce] ham 288.802 cift").
  Aradaki expansion olcumu: 1.377.895 / 288.802 = 4,7711
  -> birim hatasi tam olarak bu carpan kadar.

DUZELTME UYGULANDI (01.10.2026): MS_PER_HAM_CIFT_EPOCH eklendi ve
  sure_ve_hesapla onu kullaniyor. Tavan 1.251.733 -> 256.330 ham cift.
  Aracin "DURUM:" blogu duzeltmenin uygulanip uygulanmadigini
  KENDISI sorar; elle bakilmaz.

IKINCI HATA, AYNI YONDE (01.10.2026, olcum aracinda bulundu):
  MS_HAM hesaplanirken encode de icine katiliyordu:
      (EPOCH_SN + ENCODE_SN) / HAM = 97,09 ms
  Ama sure_ve_hesapla encode'u AYRICA dusuyor:
      kalan_dk = oturum_dk * bosluk - encode_dk
  Encode iki kez sayiliyordu -> sabit %9,45 YUKSEK. Dogru deger
  encode HARIC: 88,7127 ms (12 epoch) = 7,3927 ms/ham/epoch.

Kullanim:
    python tools/sure_olc.py [kaggle_train.txt]
"""
from __future__ import print_function

import io
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import train_llm as T

LOG = os.path.join(BASE, 'kaggle_train.txt')
if len(sys.argv) > 1:
    LOG = sys.argv[1]
if not os.path.exists(LOG):
    raise SystemExit('log yok: %s' % LOG)

ls = io.open(LOG, encoding='utf-8').read().split('\n')


def ara(nisan):
    for l in ls:
        if nisan in l:
            return l
    return None


def n(t):
    return int(re.sub(r'[^0-9]', '', t))


def zorundu(nisan, ne):
    l = ara(nisan)
    if l is None:
        raise SystemExit('HATA: logda "%s" (%s) bulunamadi; log degismis olabilir'
                         % (nisan, ne))
    return l


OTURUM_DK = 540

print('=' * 74)
print('SURE TAVANI OLCUMU  [%s]' % os.path.basename(LOG))
print('=' * 74)
print()

# ------------------------------------------------------------- 1) veri
BUDCE = n(zorundu('MAX_PAIRS=', 'butce').split('MAX_PAIRS=')[1].split(' ')[0])
HAM = n(zorundu('[butce] ham ', 'ham cift').split('ham ')[1].split(' cift')[0])
EGITIM = n(zorundu('egitim cifti (sorgu, yanit): ', 'egitim').split(': ')[1])
_m = re.search(r'train ([\d,]+) \| val ([\d,]+)',
               zorundu('split (ctx-grup bazli)', 'split'))
TRAIN, VAL = n(_m.group(1)), n(_m.group(2))
# log'da TRAIN binlik ayracsiz yazilir: "BPE-encode 1240102 cift"
TOKEN = float(re.search(r'\(([\d.]+)M token',
                        zorundu('BPE-encode %d cift' % TRAIN, 'token')
                        ).group(1)) * 1e6

# ------------------------------------------------------------- 2) sure
ep = []
for l in ls:
    if 'epoch ' not in l or '| lr ' not in l:
        continue
    a = re.search(r'epoch\s+(\d+)/\d+', l)
    b = re.search(r'\|\s*([\d.]+)s\s*\|\s*lr', l)
    if a and b:
        ep.append((int(a.group(1)), float(b.group(1))))
ep.sort()
EPOCH_SN = sum(t for _, t in ep)
ORT_EPOCH_SN = EPOCH_SN / len(ep)
ENCODE_SN = sum(float(x) for x in
                re.findall(r'CANLI OLcum\) 4 cekirdekle (\d+)s', ''.join(ls)))

GENISLEME = float(EGITIM) / HAM          # ham cift -> egitim cifti
MS_EGITIM = ORT_EPOCH_SN * 1000.0 / TRAIN  # ms / egitim cifti
# ms / HAM cift / epoch. ENCODE DAHIL DEGIL: sure_ve_hesapla encode'u
# kalan_dk'dan AYRICA dusuyor (kalan_dk = oturum_dk*bosluk - encode_dk).
# 01.10'da buraya encode eklendiydi -> encode iki kez sayildi -> sabit
# %9,45 yuksek -> tavan gereksiz dar (234.207 yerine dogru deger 256.335).
MS_HAM = EPOCH_SN * 1000.0 / HAM
MS_HAM_ENCODE_DAHIL = MS_HAM + ENCODE_SN * 1000.0 / HAM   # yalnizca rapor

print('VERI')
print('  MAX_PAIRS butcesi          : %s' % format(BUDCE, ','))
print('  ham cift                   : %s' % format(HAM, ','))
print('  egitim cifti               : %s' % format(EGITIM, ','))
print('  train / val                : %s / %s' % (format(TRAIN, ','),
                                                  format(VAL, ',')))
print('  token                      : %.1fM (%.2f tok/cift)'
      % (TOKEN / 1e6, TOKEN / TRAIN))
print()
print('OLCULEN MALIYETLER')
print('  egitim cifti basina        : %.4f ms   (ort epoch %.1f sn / %s train)'
      % (MS_EGITIM, ORT_EPOCH_SN, format(TRAIN, ',')))
print('  HAM cift basina            : %.4f ms   (%.1f sn / %s ham, 12 epoch)'
      % (MS_HAM, EPOCH_SN, format(HAM, ',')))
print('  HAM cift basina + encode   : %.4f ms   (encode AYRI hesaplanir)'
      % MS_HAM_ENCODE_DAHIL)
print('  expansion (ham->egitim)    : %.4f' % GENISLEME)
print()
print('SURE')
print('  epoch sayisi / toplam      : %d / %.1f sn = %.1f dk'
      % (len(ep), EPOCH_SN, EPOCH_SN / 60))
print('  encode                     : %.1f sn = %.1f dk' % (ENCODE_SN,
                                                            ENCODE_SN / 60))
print('  oturum tuketimi (olculen)  : %.1f dk / %d dk = %.1f%%'
      % ((EPOCH_SN + ENCODE_SN) / 60, OTURUM_DK,
         100.0 * (EPOCH_SN + ENCODE_SN) / 60 / OTURUM_DK))
print('  !! veri hazirligi olculmedi -> gercek tuketim DAHA YUKSEK')
print()

# --------------------------------------------------- 3) tavan karsilastirmasi
OTURUM = OTURUM_DK
EPOCHS = len(ep)
kalan = OTURUM * T.VARSAYILAN_BOSLUK - T.ENCODE_DK

# ESKI = 01.10'den onceki formul: MS_PER_PAIR (EGITIM cifti) kullaniyordu.
# Arac bunu KENDI hesaplar, cunku kod artik duzeltilmis olabilir; boylece
# arac "duzeltme uygulanmis mi" sorusunu da kendisi cevaplar.
ESKI = int(kalan / EPOCHS * 60000 / T.MS_PER_PAIR)

# A) sadece MS_PER_PAIR'i olculen EGITIM-cifti degerine cek -> YETERSIZ
A = int(kalan / EPOCHS * 60000 / MS_EGITIM)

# B) ham cift basina olculen sabit, kodun %75 bosluk politikasiyla
B = int(kalan * 60000 / MS_HAM)

# C) ham cift sabiti, bosluk YOK (540 dk'nin tamami)
C = int(OTURUM * 60000 / MS_HAM)

VAZIF = T.sure_ve_hesapla(oturum_dk=OTURUM, epochs=EPOCHS)   # sahadaki kod

print('=' * 74)
print('TAVAN KARSILASTIRMASI  (oturum %d dk, %d epoch, %%75 bosluk, -%s dk '
      'encode)' % (OTURUM, EPOCHS, T.ENCODE_DK))
print('=' * 74)
print('  ESKI formul (MS_PER_PAIR, egitim cifti) : %10s cift   <- BIRIM '
      'HATASI' % format(ESKI, ','))
print('  YENI formul (MS_PER_HAM_CIFT_EPOCH)      : %10s cift   <- sahadaki '
      'kod' % format(VAZIF, ','))
print('  duzeltme uygulanmis mi                   : %s'
      % ('EVET' if abs(VAZIF - B) <= max(2, B * 0.001) else 'HAYIR'))
print()
print('  secenek          tavan        ESKI/bu  yorum')
print('  ' + '-' * 70)
print('  A) MS_PER_PAIR duzeltilmis : %10s   %5.2f   YETERSIZ'
      % (format(A, ','), ESKI / float(A)))
print('  B) ham cift sabiti         : %10s   %5.2f   DOGRU'
      % (format(B, ','), ESKI / float(B)))
print('  C) ham cift, bosluk YOK    : %10s   %5.2f   sinir (%%0 pay)'
      % (format(C, ','), ESKI / float(C)))
print()
print('  HAM cift BASINA OLCULEN DEGER: %.4f ms = %.1f sn / %s ham (12 epoch)'
      % (MS_HAM, EPOCH_SN, format(HAM, ',')))
print('    encode HARIC. sure_ve_hesapla encode suresini ENCODE_DK ile ayrica')
print('    dusuyor; ikisini toplamak encode suresini iki kez sayardi')
print('    (01.10 hatasi: 8,0911 ms yerine dogru deger %.4f ms, %%9,45 fark)'
      % MS_HAM)
print('    (egitim cifti basina %.4f ms, expansion %.4f)'
      % (MS_EGITIM, GENISLEME))
print()
print('  BU KOSUDA KULLANILAN HAM CIFT : %s' % format(HAM, ','))
for ad, t_ in (('A', A), ('B', B), ('C', C)):
    fark = t_ - HAM
    print('  -> %s: ham veriyi %s cift %s (%+.1f%%)'
          % (ad, format(abs(fark), ','),
             'KIRPAR' if fark < 0 else 'KARSILAMAZ (kirpma olmaz)', fark * 100.0 / HAM))
print()
print('YORUM (olcumden cikan, varsayim degil):')
print('  * ESKI formul ESKI/B = %.2f KAT uzakta. Sebebi BAYEM: expansion'
      % (ESKI / float(B)))
print('    %.4f. ESKI sabit ham cift degil EGTIM cifti maliyetidir'
      % GENISLEME)
print('    (%.4f ms), ama donen sayi HAM cift olarak kullaniliyor.'
      % MS_EGITIM)
print('  * A secenegi ESKI tavanini sadece {0:.1f} yuzde dusurur -> birim '
      'hatasi durur, tavan hala ESKI/B = {1:.2f} KAT uzakta. YETERSIZ.'
      .format(100.0 * (1 - A / float(ESKI)), A / float(B)))
print("  * B secenegi expansion'i ve token uzunlugunu icine sindiren TEK")
print('    olculen sabiti kullanir. Dogru yol budur.')
print()
if abs(VAZIF - B) <= max(2, B * 0.001):
    print('DURUM: duzeltme UYGULANMIS. sure_ve_hesapla ham cift sabitini')
    print('       kullaniyor; tavan olculen sinira oturuyor.')
    if VAZIF < HAM:
        print('       Bu kosuda ham veriyi {0} cift ({1:.1f} yuzde) kirpar.'
              .format(format(HAM - VAZIF, ','),
                      100.0 * (HAM - VAZIF) / HAM))
        # Kapsama kaybini VARSAYMA, canli veriden olc. seqgen kirpmasi
        # "her ctx'den en az 1" garantisi veriyor (01.10 duzeltmesi), yani
        # kapsama %100 olmali; olcmuyorsak soylemiyoruz.
        try:
            sys.path.insert(0, BASE)
            from seqgen import load_pairs
            yol = os.path.join(BASE, 'intents.json')
            if os.path.exists(yol):
                import train_llm
                K = dict(use_query=True, ctx_len=train_llm.CTX_CHARS)
                tum = load_pairs(yol, max_pairs=10 ** 9, **K)
                kirp = load_pairs(yol, max_pairs=VAZIF, **K)
                ct = set(p[0] for p in tum)
                ck = set(p[0] for p in kirp)
                print('       CANLI VERIDE OLCULDU (16k intent): cift {0} -> {1} '
                      '(%{2:.1f}) | benzersiz ctx {3} -> {4} = %{5:.2f}'
                      .format(format(len(tum), ','), format(len(kirp), ','),
                              100.0 * len(kirp) / max(1, len(tum)),
                              format(len(ct), ','), format(len(ck), ','),
                              100.0 * len(ck) / max(1, len(ct))))
        except Exception as exc:                       # noqa: BLE001
            print('       (kapsama olculemedi: %s)' % (exc,))
else:
    print('DURUM: duzeltme UYGULANMAMIS. sure_ve_hesapla hala {0} cift '
          'diyor; olculen guvenli sinir B = {1}. Tavan {2:.2f} KAT uzakta.'
          .format(format(VAZIF, ','), format(B, ','), VAZIF / float(B)))
    print('       Duzeltmek icin: MS_PER_HAM_CIFT_EPOCH = {0:.4f}  (bu '
          'aracin olculdugu deger)'.format(MS_HAM / EPOCHS))

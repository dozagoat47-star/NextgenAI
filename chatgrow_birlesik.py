# -*- coding: utf-8 -*-
"""ChatGrow birleştirici: birden çok chatgrow kaynağını tek bir eğitim
dosyasında toplar. Tekrarları temizler (query -> ascii key), kanonik alan
sırasını korur ve batch'i sonradan çoğaltmadan yeni çiftlere öncelik verir.

Amaç: chatgrow.py (Reddit/Discourse canlı çekim) + elle yazılmış seed'i
"chatgrow_birlesik.jsonl"da birleştirip Kaggle tarafında eğitime hazır
tutmak. Canlı kaynaklar her çalıştırmada az sayıda yeni çift ürettiği için
birleştirici "yeni gelen çiftler = öncelik", "sabit seed = korunur" gerektirir.

Kullanım:
    python chatgrow_birlesik.py chatgrow_sohbet.jsonl \
                                chatgrow_discourse_pardus.jsonl \
                                --out chatgrow_birlesik.jsonl
    python chatgrow_birlesik.py A.jsonl B.jsonl C.jsonl --max-putense 1
"""

import argparse
import io
import json
import os
import sys

from chatgrow import clean_line as _clean


def read_jsonl(path):
    """Bir chatgrow kaynağının çiftlerini verir (source/subreddit korunur)."""
    rows = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not row.get('query') or not row.get('answer'):
                continue
            row.setdefault('source', 'seed')
            row.setdefault('subreddit', 'seed')
            row.setdefault('score', 0)
            rows.append(row)
    return rows


def merge(rows_by_source, limit=None):
    """Kaynak listesini (sıralı) tek listeye birleştirir; aynı normalizasyonla
    tekrar eden query'leri düşürür. Önce gelen kaynak önceliklidir."""
    seen = set()
    out = []
    for rows in rows_by_source:
        for row in rows:
            key = _clean(row.get('query') or '')
            if not(key):
                continue
            k = key.lower()  # chatgrow ile aynı key (ascii normalize istemez)
            if k in seen:
                continue
            seen.add(k)
            out.append(row)
            if limit is not None and len(out) >= limit:
                return out
    return out


def write_jsonl(rows, path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='ChatGrow birleştirici: kaynakları tek eğitim dosyasına '
                    'toplar (tekrar: query -> ascii key)')
    ap.add_argument('inputs', nargs='+', metavar='KAYNAK.jsonl',
                    help='birleştirilecek chatgrow dosyaları (sıralı, '
                         'önceki öncelikli)')
    ap.add_argument('--fallback', metavar='SEED.jsonl', default=None,
                    help='YEDEK: canlı kaynaklar 0 çift üretirse (forum '
                         'kapalı/limitli) eğitim boş kalmasın diye bu '
                         'kaynaktan doldurur. Canlı kaynak >= 1 çift '
                         'üretirse BURASI HİÇ KULLANILMAZ.')
    ap.add_argument('--out', default='chatgrow_birlesik.jsonl')
    ap.add_argument('--limit', type=int, default=None,
                    help='en fazla N çift yaz (opsiyonel)')
    args = ap.parse_args(argv)

    # 1) ONCELIKLI: canli kaynaklar (Discourse/Reddit). Seed BURAYA KARISMAZ.
    live = merge([read_jsonl(p) for p in args.inputs], args.limit)

    # 2) YALNIZCA canli 0 cift uretirse (forum kapali/limitli) seed DEvreye girer.
    used_fallback = False
    if not live and args.fallback:
        live = merge([read_jsonl(args.fallback)], args.limit)
        used_fallback = True

    n = write_jsonl(live, args.out)
    live_desc = (f'{args.fallback} (YEDEK-dolgu)' if used_fallback
                 else 'canli kaynak(lar)')

    print(f"[1/2] ANA KAYNAK: {live_desc} = {n} cift")
    if used_fallback:
        print(f"[1/2] NOT: canli kaynaklar 0 cift; egitim bos kalmasin diye "
              f"seed kullanildi")
    print(f"[2/2] {args.out}: {n} cift yazildi")
    return 0 if n else 1


if __name__ == '__main__':
    sys.exit(main())

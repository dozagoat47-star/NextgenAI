#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_book_pairs.py — Wikisource tr kamu-malı metinlerden deterministik
continuation ciftleri uretir:  chatgrow_kitap_*.jsonl

Fikir: cahiz{dusuk-bellek}. Tr Wikisource'ta kamu-mali** makalelerin (masal,
destan, mesnevi onculeri, halk hikayeleri, ornegin "Kategori:Türk masalları")
paragraf/cumle dizisi bir (ctx -> sonraki cumle) continuation akisidir. Model
"okudugu cumlenin devamini uretmeyi" oyrenince (a) genel kultur/uslup
kazanir, (b) uzun kitaplardaki ezber yerine icine gore akis uretir.

Kisitlar (train_llm MAX_SEQ 128 / MAX_CTX 48 token ile uyumlu):
  - ctx   -> ilk cumle(ler), budandiktan SONRA <= MAX_CTX_LEN token (48)
  - resp  -> hemen sonraki 1-2 cumle, refine_resp gibi cumle-sonunda,
             MAX_SEQ-MAX_CTX-... (76 token) icin cok az (40-60).
  - tum metinler clean_chars/fetch_hf_turkish.py ile AYNI normalize'den gecir;
    ASCII disi kirpilarak yerel cozumleme/bench ile birebir.

Uretici mantigi (deterministik, SEED):
  1. Kaynak eser listesi: tr.wikisource kategori taramasi sabit listenin
     alt-kumesi. Kategori ADLARI wikisource'ta kararsizdir; biz SOURCES'I
     (name -> kategori) veririz; kategori bos ya da hatali ise script fallback
     aramasina gecmez (deterministik -> ayni cikti), sadece uyari yazar.
  2. Her eser: api `action=query&prop=revisions&rvprop=content&rvslots=main`
     -> wikitext; basit paragraf+kuruluş çikası (Maten nd metin uretir).
  3. Cumlelere ayir (noktalama), budama sonrasi 8-76 token;
     her uygun cumle -> (ctx=onceki 1-2 cumle, resp=bu cumle)
     Ile (ctx, resp) ciftleri dogar; Jaccard>=0.9 yakın-kopya elenir
     (fetch_hf_turkish.dedupe_pairs ile ayni).
  4. Matematik/formul sayfalari (istersen `--plain` ile at/) ve tumleyen
     isaretleri (örn. "Vikikaynak", "kaynak") ayiklanir.

Kaggle akisi (kaggle_start.sh):
    python build_book_pairs.py --max-pairs 8000 --seed 7 --out chatgrow_kitap.jsonl
Cogunlugunu bu glob zaten yakalar; --fresh ile bench/verify de egitimle ayni
veriyi gorur.

Lisans notu: Wikisource tr`de kamu-mali (ya da tr:PD kategorili) eserler telif
hakki tasimaz; turev/sozluk/aktif-metin elemeleri `--no-subsections` gibi.
Bazi masal/destan metinleri modern anlatilarla karisiktir; `--min-year...`
kisitlamasi yok — esas kamu-mali etiketine gore alinir.
"""

import argparse
import io
import json
import os
import random
import re
import sys
import time

from fetch_hf_turkish import (clean_chars, jaccard, dedupe_pairs,
                              ALLOWED_EXTRAS, MAXSEP, SEED)

# --- kaynak eserler (kategori adi) ------------------------------------------

# Wikisource tr kamu-mali kategori adlari (API ile dogrulandi, 2024-09):
#  Kategori:Türk halk edebiyatında masallar / destanlar / efsaneler /
#  bilmeceler / ninniler / ağıtlar / türküler ...
# ALT NOT: "Türk halk edebiyatında masallar" kategorisinde Orhan Veli gibi
# modern kamu-mali eserler de bulunuyor (telif surecini geçmis) -> script zaten
# icerik/uzunluk filtrelerinden geçtiği icin sorun yok. API adlari zamanla
# degisebilir; script 'resolve_category' ile gercek adi API'den bulur, yoksa
# o kategoriyi deterministik atlar (uyarir, hata degil).
CATEGORIES = [
    'Türk halk edebiyatında masallar',
    'Türk halk edebiyatında destanlar',
    'Türk halk edebiyatında efsaneler',
    'Türk halk edebiyatında bilmeceler',
    'Türk halk edebiyatında ninniler',
    'Türk halk edebiyatında ağıtlar',
    'Türk halk edebiyatında türküler',
    'Türk halk edebiyatında fıkralar',
]

# kategori adlarinin normalize edilmis haliyle aranacak gercek ad (API):
def _norm_cat(c):
    return clean_chars(c, 80).replace('kategori:', '').strip()


# --- API yardimcilar ---------------------------------------------------------

_API = 'https://tr.wikisource.org/w/api.php'


def api(params):
    import urllib.parse, urllib.request
    params = dict(params, format='json')
    url = _API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'moz-5.0'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError('tekrar denemeden sonra api erisilemedi')


def resolve_category(query):
    """`query` icin Wikisource'ta KATEGORI adi bulur; yoksa gercek adlardan
    ilk eslesme. Gercek ad, api allcategories ile sanssız dikdörtgen tarama
    yerine list=allcategories&acprefix ile hizli aranir."""
    q = _norm_cat(query)
    if not q:
        return None
    # tam ad dene
    d = api({'action': 'query', 'list': 'allcategories',
             'acfrom': q, 'aclimit': '40'})
    names = [c['*'] for c in d['query']['allcategories']]
    for name in names:
        if _norm_cat(name) == q:
            return name
    # kapsar-adli prefix eşleşmesi (or. 'turk masallari' icin ...)
    for name in names:
        if _norm_cat(name).startswith(q) or q.startswith(_norm_cat(name)):
            if len(_norm_cat(name)) > 3:
                return name
    return None


def list_category_pages(cat, limit=800):
    """Kategori sayfalarini (kamu-mali eserler) getirir."""
    out = []
    cont = {}
    pages = []
    for _ in range(40):
        p = {'action': 'query', 'list': 'categorymembers',
             'cmtitle': 'Kategori:' + cat, 'cmlimit': '500',
             'cmtype': 'page'}
        p.update(cont)
        d = api(p)
        pages += [m['title'] for m in d['query']['categorymembers']]
        if 'continue' in d:
            cont = {'cmcontinue': d['continue']['cmcontinue']}
        else:
            break
    # kaldir: Konu başlığı gitgide sayfalarında 'Kaynak' kelimesiyle başlayan
    out = [t for t in pages if not re.match(r'^(Kullanıcı|Tartışma|Vikikaynak|Dosya):', t)]
    print(f'[katsayfa] {cat}: {len(out)} eser', flush=True)
    return out[:limit]


def fetch_wikitext(title):
    d = api({'action': 'query', 'prop': 'revisions', 'rvprop': 'content',
             'rvslots': 'main', 'titles': title, 'redirects': '1'})
    for page in d['query']['pages'].values():
        revs = page.get('revisions')
        if revs:
            return revs[0]['slots']['main']['*']
    return ''


_SUBHEAD = re.compile(r'(?m)^[=#*]\s*|^\s*(==|===|{|\{\{|\}|#REDIRECT)')
_TEMPLATE_BLOCK = re.compile(r'\{\{.*?\}\}', re.S)


_POEM_BLOCK = re.compile(r'<poem>(.*?)</poem>', re.S)


def poem_to_verses(wt):
    """wikitext icindeki <poem> bloklarini mısra listesine cevirir."""
    verses = []
    for m in _POEM_BLOCK.finditer(wt or ''):
        for line in m.group(1).splitlines():
            line = clean_chars(line, 400)
            if len(line) > 12:
                verses.append(line)
    return verses


def build_poem_continuation(verses, ctx_max=300, resp_max=300):
    """Mısralar -> (ctx, resp) ciftleri (mısra-yarısı).

    ctx = onceki 1-2 mısra, resp = sonraki mısra. Wikisource tr'de masal,
    destan, ninni, türkü, ağıt carpkakı <poem> mısra bloklarıyla yazılır;
    bu akis "okudugu mısranın devamını getir" özlemini yakalar ve kisa
    kamu-mali metinlerin de devam üretmesini saglar.
    """
    pairs = []
    for i in range(2, len(verses)):
        ctx = ' '.join(verses[max(0, i - 2):i])
        resp = verses[i]
        ctx = clean_chars(ctx, ctx_max)
        resp = clean_chars(resp, resp_max)
        if len(ctx) < 20 or len(resp) < 10:
            continue
        pairs.append((ctx, resp))
    return pairs


def wikitext_to_paragraphs(wt):
    """wikitext -> temiz paragraf listesi (eser duz metin)."""
    wt = _TEMPLATE_BLOCK.sub(' ', wt or '')       # {{...}} sablonlari
    wt = _SUBHEAD.sub(' ', wt)                    # baslik/sablon işaretleri
    par = [clean_chars(p, 2000)
           for p in re.split(r'\n\s*\n|(?:\n\s*)+', wt)
           if clean_chars(p, 2000)]
    return [p for p in par if len(p) > 30]


def sentence_split(para):
    """Paragrafi cumlelerine boler (kisaltmalari korur, turkce)."""
    out, buf = [], ''
    for m in re.finditer(r'[^.!?]+[.!?]+["\'\u2019)]?(?=\s|$)|[^.!?]+$', para):
        s = m.group(0).strip()
        if s:
            out.append(s)
    return [s for s in out if len(clean_chars(s, 400)) > 8]


def build_continuation(wts, ctx_max=300, resp_max=300):
    """Eser paragraflari -> (ctx, resp) ciftleri (cumle-yariyi).

    Her paragrafin cumleleri sirasina gore birlesir; budama sonrası ctx/ resp
    1-3 cumle. Boylece "onceki bir iki cumle -> devami" continuation akisini
    yakalar.
    """
    pairs = []
    for para in wts:
        sents = sentence_split(para)
        if len(sents) < 7:
            continue                       # tek basina deneyimsiz paragraf
        # ctx = ilk yari cumleler, resp = kalan
        half = len(sents) // 2
        ctx = ' '.join(sents[:half])
        resp = ' '.join(sents[half:])
        ctx = clean_chars(ctx, ctx_max)
        resp = clean_chars(resp, resp_max)
        if len(ctx) < 20 or len(resp) < 20:
            continue
        pairs.append((ctx, resp))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='chatgrow_kitap.jsonl')
    ap.add_argument('--cat', action='append', default=[],
                    help='Wikisource tr kategori adi (tekrar edilebilir); '
                         'bos: CATEGORIES listesi')
    ap.add_argument('--max-pairs', type=int, default=8000)
    ap.add_argument('--per-cat', type=int, default=1600,
                    help='kategori basina en fazla eser')
    ap.add_argument('--ctx-len', type=int, default=300, help='ctx_kr (budama)')
    ap.add_argument('--resp-len', type=int, default=300)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args(argv)

    cats = [c for c in args.cat if c] or list(CATEGORIES)
    by_header = []
    resolved = {}
    for c in cats:
        real = resolve_category(c)
        if real:
            resolved[c] = real
            per = max(1, min(args.per_cat, args.max_pairs // max(len(resolved), 1)))
            pages = list_category_pages(real, limit=per)
            by_header.append((real, pages))
        else:
            print(f'!! kategori {"bulunamadi"}: {c}', flush=True)

    rng = random.Random(args.seed)
    all_pairs = []
    for real, pages in by_header:
        for title in pages:
            try:
                wt = fetch_wikitext(title)
            except Exception as e:
                print(f'  !! {title}: {e}', flush=True)
                continue
            # 1) varsa <poem> mısra akisi (masal/destan/ninni/türkü)
            pairs = build_poem_continuation(poem_to_verses(wt),
                                            ctx_max=args.ctx_len,
                                            resp_max=args.resp_len)
            # 2) nesir icin classic continuation (paragraf-yarısı)
            if not pairs:
                pairs = build_continuation(wikitext_to_paragraphs(wt),
                                           ctx_max=args.ctx_len,
                                           resp_max=args.resp_len)
            if pairs:
                print(f'  - {title}: {len(pairs)} cift', flush=True)
                all_pairs.extend(pairs)
            if len(all_pairs) >= args.max_pairs * 2:
                break
        if len(all_pairs) >= args.max_pairs * 2:
            break

    pairs = dedupe_pairs(all_pairs, seed=args.seed)
    rng.shuffle(pairs)
    pairs = pairs[:args.max_pairs]
    with io.open(args.out, 'w', encoding='utf-8') as f:
        for ctx, resp in pairs:
            f.write(json.dumps({'query': ctx, 'answer': [resp]},
                               ensure_ascii=False) + '\n')
    print(f'[kitap] {len(pairs)} cift -> {args.out}', flush=True)
    if pairs:
        for ctx, resp in pairs[:5]:
            print(f'   ornek ctx: {ctx!r}', flush=True)
            print(f'         resp: {resp!r}', flush=True)
    return pairs


if __name__ == '__main__':
    main()

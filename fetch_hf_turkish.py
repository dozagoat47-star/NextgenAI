#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_hf_turkish.py — HF instruction/thinking kaynaklarindan deterministik
(seed, stream, dedup) (ctx, resp) egitim ciftleri uretir.

Uretilen format, chatgrow.py/chatgrow_birlesik.py ciktisiyla AYNI JSONL
semasidir:  {"query": "...", "answer": [...]}
Dosya adi "chatgrow_hf_*.jsonl" -> kaggle_start.sh CGARG glob'u otomatik yakalar.

Kullanim (yerel dogrulama / Kaggle prep hucresi):

    python fetch_hf_turkish.py --out chatgrow_hf_instruction.jsonl \
        --source tascib/turkish-instruction --max-pairs 20000 --seed 7

    python fetch_hf_turkish.py --out chatgrow_hf_thinking.jsonl \
        --source erythropygia/ThinkingData-200K-Turkish \
        --max-pairs 20000 --seed 7 --cot

Streaming okuma (memory dusuk, PC'de rahat calisir); onbellek HF_HOME'dadir.

Alan esleme (HF schema'dan chatgrow'a):
  - instruction-turkish:  {"instruction","input","output","source"}
      -> query = instruction (+ input varsa " | " ile ekle), answer = [output]
  - ThinkingData-200K:    {"answer","reasoning","messages","model"}
      -> query = messages (user) , answer = [answer]
      CoT (--cot) aciksa: reasoning -> "Kisa dusunce: <r>. Oyleyse: <answer>"
      tek bilesik cevap; iki varyant ust urune gider (kotu harman).

--cot icin guvenli filtreler:
  * reasoning/answer temiz + ASCII-normalize (clean_chars) sonrasi BOŞ kalirsa
    cift ATILIR.
  * ctx `ctx_len`, resp `resp_len` karakter budcalari ASILIR; budama CUMLE
    sonundan yapilir (yarim kelime degil) -> ezberi degil akisi korur.
  * yeni (ctx,resp) tekrari + yakın kopya (Jaccard >= 0.9) elenir.
  * ayni resp farkli ctx'ler icin sadece bir kez (kopya uretimi azalsin).
"""

import argparse
import io
import json
import math
import os
import random
import re
import sys
import time

# --- normalize (train_llm.clean_chars / refine_resp ile ayni kurallar) ------
try:
    from normalize import ascii_normalize
except Exception:                      # train dizininden bagimsiz calisma
    def ascii_normalize(s):            # minimal fallback (Turkce -> ASCII)
        ow = {'ş': 's', 'Ş': 'S', 'ğ': 'g', 'Ğ': 'G', 'İ': 'I', 'ı': 'i',
              'ç': 'c', 'Ç': 'C', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U'}
        out = []
        for ch in s:
            out.append(ow.get(ch, ch))
        return ''.join(out)

ALLOWED_EXTRAS = frozenset(".,;:!?…()%’'\"-–/") | frozenset("0123456789")
MAXSEP = 8
SEED = 7


def clean_chars(text, max_len):
    t = ascii_normalize((text or '').replace('\u00a0', ' '))
    out = []
    for ch in t.lower():
        if ch.isascii() and (ch.isalpha() or ch == ' ' or ch in ALLOWED_EXTRAS):
            out.append(ch)
    s = ''.join(out)
    # ardisik bosluk -> tek
    s = re.sub(r'\s+', ' ', s).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s


def refine_resp(r, maxc):
    """Cevabi cumle sonunda budar (yarim kelime ezberi olmaz)."""
    r = (r or '').strip()
    if len(r) < 10:
        return None
    if len(r) > maxc:
        cut = 0
        for i in range(15, min(len(r), maxc + 8)):
            if r[i] in '.?:!':
                cut = i
        r = r[:cut + 1].strip() if cut > 0 else r[:maxc].strip()
    if not r:
        return None
    return r


def _clean_text_list(obj):
    """messages[] gibi listelerden metin toplar (user/asistan rollerine bakar)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        parts = []
        for m in obj:
            if isinstance(m, dict):
                c = m.get('content') or m.get('text') or m.get('value')
                role = (m.get('role') or '').lower()
                if c and role in ('user', 'human', 'soru', 'query'):
                    return str(c)
                if c and not role:                       # role yoksa ilk metin
                    return str(c)
                if c:
                    parts.append(str(c))
            elif isinstance(m, str):
                parts.append(m)
        # user yoksa ilk dolu parcayi sorgu, kalanini cevap sanma
        return parts[0] if parts else ''
    if isinstance(obj, dict):
        for k in ('user', 'query', 'soru', 'input', 'human'):
            if obj.get(k):
                return str(obj[k])
        for k in ('content', 'text'):
            if obj.get(k):
                return str(obj[k])
        return str(obj) if obj else ''
    return str(obj) if obj is not None else ''


def _first_user_text(msgs):
    """messages[] icindeki ILK user/human mesajini dondurur (ctx kismi)."""
    if not isinstance(msgs, list):
        return ''
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get('role') or '').lower()
        if role in ('user', 'human', 'soru', 'query'):
            c = m.get('content') or m.get('text')
            if c:
                return str(c)
    return ''


# --- kaynak tabanli (ctx, resp) donusumleri --------------------------------

def pairs_from_instruction(rec):
    """alpaca-turkish semasi -> (ctx,resp) gencratoru."""
    inst = (rec.get('instruction') or '').strip()
    inp = (rec.get('input') or '').strip()
    out = (rec.get('output') or '').strip()
    if not inst or not out:
        return
    ctx = inst if not inp else f"{inst} | {inp}"
    yield ctx, out


def pairs_from_thinking(rec):
    """ThinkingData-200K semasi -> (ctx,resp) gencratoru (CoT hint'li)."""
    ans = (rec.get('answer') or '').strip()
    if not ans:
        return
    ctx = _first_user_text(rec.get('messages')) or (rec.get('query') or '').strip()
    if not ctx:
        return
    reasoning = (rec.get('reasoning') or '').strip()
    if reasoning:
        yield ctx, f"Kisa dusunce: {reasoning}. Oyleyse: {ans}"
    yield ctx, ans


# --- dedup + yakın-kopya ----------------------------------------------------

def _tokenize_norm(s):
    return set(clean_chars(s, 10_000).split())


def jaccard(a, b):
    a, b = _tokenize_norm(a), _tokenize_norm(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_pairs(pairs, seed=SEED, dup_thr=0.90, keep_log=True):
    """Kesin + yakın-kopya (Jaccard) eleyici.

    - (ctx,resp) kesin ayni -> 1 kere
    - ayni resp beliren ctx'lerden SADECE ilk (az ezber)
    - yeni cift onceden gormedigi ctx/resp ile Jaccard >= dup_thr ise ele
    Karma O(n*sinir); pairs siralidir, deterministik. Soylesin (dahili log).
    """
    seen_ctx = []
    seen_resp = []
    kept = []
    dead_ctx = dead_resp = 0
    resp_map = {}
    for ctx, resp in pairs:
        if resp in resp_map:          # ayni resp daha once vardi -> buda elen
            dead_resp += 1
            continue
        # ctx yakın-kopya kontrol
        dup = False
        for sc in seen_ctx:
            if jaccard(ctx, sc) >= dup_thr:
                dup = True
                dead_ctx += 1
                break
        if dup:
            continue
        seen_ctx.append(ctx)
        seen_resp.append(resp)
        resp_map[resp] = True
        kept.append((ctx, resp))
    if keep_log:
        print(f'[dedup] girdi {len(pairs)} -> cikti {len(kept)} '
              f'(kopya-ctx {dead_ctx}, ayni-resp {dead_resp})', flush=True)
    return kept


# --- ana akis ---------------------------------------------------------------

SOURCES = {
    'tascib/turkish-instruction': pairs_from_instruction,
    'erythropygia/ThinkingData-200K-Turkish': pairs_from_thinking,
}


def stream_hf(source, max_pairs, seed, cot):
    from datasets import load_dataset

    def gen(rows):
        for rec in rows:
            if source.endswith('ThinkingData-200K-Turkish'):
                for ctx, resp in pairs_from_thinking(rec):
                    yield ctx, resp
            else:
                for ctx, resp in pairs_from_instruction(rec):
                    yield ctx, resp

    ds = load_dataset(source, split='train', streaming=True)
    rows = iter(ds)

    # --cot+thinking icin Turkish harman vermek istemiyoruz; ama FETCH'te
    # uzun periode sahip HF set'leri ayiklanacagi icin: once kategoriyi
    # asla kullandirmiyoruz, sadece n-cekitim degrade:
    budget = max_pairs * 2 + 64          # budama kaybı icin fazlalik payi
    if cot:
        budget = max_pairs * 3 + 64

    raw = []
    n = 0
    for rec in rows:
        for ctx, resp in gen([rec]):
            raw.append((ctx, resp))
            n += 1
            if n >= budget:
                break
        if n >= budget:
            break
    print(f'[stream] {source}: {n} ham (ctx,resp) toplandı', flush=True)
    rng = random.Random(seed)
    rng.shuffle(raw)
    return raw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='chatgrow_hf.jsonl')
    ap.add_argument('--source', action='append', default=[],
                    help='HF veri seti (tekrar edilebilir). Bos: tumu')
    ap.add_argument('--max-pairs', type=int, default=20000)
    ap.add_argument('--cot', action='store_true',
                    help='ThinkingData reasoning >= 3 kelime ise '
                         '"Kisa dusunce:..." on-eki (CoT varyanti) olarak ekle')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--ctx-len', type=int, default=48)
    ap.add_argument('--resp-len', type=int, default=140)
    args = ap.parse_args(argv)

    sources = [s for s in args.source if s] or list(SOURCES)
    pairs = []
    for s in sources:
        if s not in SOURCES:
            print(f'!! bilinmeyen kaynak: {s}; {list(SOURCES)} kullanılıyor', flush=True)
            s = None
        t0 = time.time()
        raw = stream_hf(s if s else sources[0], args.max_pairs, args.seed, args.cot)
        kept = []
        for ctx, resp in raw:
            ctx_c = clean_chars(ctx, args.ctx_len)
            resp_c = refine_resp(clean_chars(resp, args.resp_len),
                                 maxc=args.resp_len)
            if not ctx_c or not resp_c:
                continue
            if args.cot and 'Kisa dusunce' in ctx_c:
                continue                    # ctx'e CoT karistirme
            kept.append((ctx_c, resp_c))
        kept = dedupe_pairs(kept, seed=args.seed)
        print(f'[{s}] {len(kept)} cift (budama+kopya sonrasi)', flush=True)
        pairs.extend(kept)

    # uluslararası tekrar-dedupe + karıştır + max-pairs kırp
    out = dedupe_pairs(pairs, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(out)
    if args.max_pairs:
        out = out[:args.max_pairs]
    with io.open(args.out, 'w', encoding='utf-8') as f:
        for ctx, resp in out:
            f.write(json.dumps({'query': ctx, 'answer': [resp]},
                               ensure_ascii=False) + '\n')
    print(f'[yaz] {len(out)} cift -> {args.out}', flush=True)
    if out:
        for ctx, resp in out[:4]:
            print(f'   ornek ctx: {ctx!r}', flush=True)
            print(f'         resp: {resp!r}', flush=True)


if __name__ == '__main__':
    main()

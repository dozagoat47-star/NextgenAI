# -*- coding: utf-8 -*-
"""Nextgen AI - Intent veri zenginlestirme araci.

intents.json'daki HER intent'in yanitlarini dogal (gercek Turkce imla) varyantlarla
genisletir ve bilgi intent'leri icin corpus'tan 'knowledge_map.jsonl' uretir.

KRITIK KURAL: Bilgi intent'lerine (desen sayisi == 6, brain.py:874) PATTERN
EKLENMEZ. Eklenirse >6 desen olur ve intent sohbet sinifina gecer, bilgi/retrieval
davranisini bozar. Bu yuzden script yalnizca YANIT listeleriyle oynar.

Guvenceler:
  - idempotent: ayni yanit metnini tekrar eklemez (bpe.clean_text anahtariyla);
    tekrar calistirinca eklenen yanit sayisi 0 olur.
  - tag Latin-dis harf kuralini, pattern/yanit dolu olma sartini korur/dogrular.
  - knowledge_map: bilgi intent desenleri -> corpus.search('desen') en alakali
    parca; train_llm.py --kb-map ile deterministik RAG verir.

Kullanim:
  python enrich_intents.py [--dry-run] [--k 3] [--out intents.json]
                            [--kb-map knowledge_map.jsonl] [--kb-limit 2000]
"""
import argparse
import io
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, 'intents.json')
KB_MAP_PATH = os.path.join(BASE, 'knowledge_map.jsonl')

from bpe import clean_text as _clean
from naturalize import natural_variants


def _latin(tag):
    return not any(c.isalpha() and ord(c) > 127 for c in tag)


def _load_data(path):
    with io.open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_data(data, path):
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _stable_seed(text):
    h = 5381
    for ch in text:
        h = ((h * 33) + ord(ch)) & 0x7FFFFFFF
    return h


def enrich_responses(intents, k=3):
    """Her intent'in yanitlarini dogal varyantlarla genisletir.

    Bir kez uygulanan k degeri meta envanterde ('_meta.enrich_k') saklanir;
    ayni k ile tekrar calistirilinca yeni varyant uretilmez (kaskat buyume
    engellenir). Daha yuksek k verilirse kalan yanitlardan tekrar genisletilir.
    """
    added = 0
    for it in intents:
        tag = it.get('tag', '')
        resps = it.get('responses', [])
        seen = set()
        for r in resps:
            seen.add(_clean(r))
        for r in list(resps):
            for v in natural_variants(r, k=k, seed=42 + _stable_seed(tag) + _stable_seed(r) * 7):
                key = _clean(v)
                if not key or key in seen:
                    continue
                resps.append(v)
                seen.add(key)
                added += 1
        it['responses'] = resps
    return added


def build_knowledge_map(intents, limit=6000, kb_text_chars=300):
    """Bilgi intent desenleri icin corpus'tan bilgi parcalari eslestirir.

    kb_text_chars=300: bilgi parcasi metni bu karakterle sinirlanir
    (train_llm.KB_TEXT_CHARS ile ayni) -> LLM 192 token sekansinda daha
    zengin bilgi kosullandirmasi gorur."""
    from corpus import Corpus
    corpus = Corpus()
    corpus.load()
    rows, skipped = 0, 0
    out = []
    for it in intents:
        if rows >= limit:
            break
        is_knowledge = len(it.get('patterns', [])) == 6
        pool = it.get('patterns', [])
        if not is_knowledge:
            continue
        for p in pool:
            if rows >= limit:
                break
            clean_ctx = _clean(p)
            if len(clean_ctx) < 6:
                continue
            resp = it.get('responses', [''])
            chunk = corpus.search(clean_ctx)
            if not chunk:
                skipped += 1
                continue
            text = ((chunk.get('title') or '') + '. ' +
                    (chunk.get('text') or ''))[:kb_text_chars]
            if len(text.strip()) < 20:
                skipped += 1
                continue
            out.append({
                'ctx': clean_ctx,
                'resp': _clean(resp[0]) if resp else '',
                'title': chunk.get('title', ''),
                'text': text,
                'score': chunk.get('score', 0.0),
            })
            rows += 1
    return out, skipped


def _validate(data):
    tags = set()
    n_knowledge = n_chat = 0
    for it in data['intents']:
        tag = it.get('tag', '')
        assert len(tag) >= 2, f'kisa tag: {tag!r}'
        assert tag not in tags, f'duplike tag: {tag!r}'
        tags.add(tag)
        assert _latin(tag), f'Latin disi karakter: {tag!r}'
        assert it.get('patterns'), f'{tag}: kalipsiz'
        assert it.get('responses'), f'{tag}: yanitsiz'
        for p in it['patterns']:
            assert p.strip(), f'{tag}: bos kalip'
        for r in it['responses']:
            assert len(r.strip()) >= 2, f'{tag}: cok kisa yanit'
        if len(it['patterns']) == 6:
            n_knowledge += 1
        elif len(it['patterns']) > 6:
            n_chat += 1
        else:
            raise AssertionError(f'{tag}: desen sayisi {len(it["patterns"])} '
                                 f'(6 bilgi / >6 sohbet kuralina uymuyor)')
    print(f'  validate OK | intent: {len(tags)} '
          f'(bilgi: {n_knowledge}, sohbet: {n_chat})')
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--k', type=int, default=3,
                    help='her yanitin dogal varyant sayisi (orijinal dahil)')
    ap.add_argument('--out', default=PATH)
    ap.add_argument('--kb-map', default=KB_MAP_PATH,
                    help='knowledge_map cikis yolu (bos dize = uretme)')
    ap.add_argument('--kb-limit', type=int, default=6000,
                    help='knowledge_map kac desen eslesmesi uretilecek (hiz; '
                         '4400 uzeri tum bilgi desenlerini kapsar)')
    args = ap.parse_args()

    data = _load_data(PATH)
    print('once:', len(data['intents']), 'intent')

    meta = data.get('_meta', {})
    if args.k != meta.get('enrich_k'):
        added = enrich_responses(data['intents'], k=args.k)
        meta['enrich_k'] = args.k
        data['_meta'] = meta
        print('eklenen dogal yanit:', added)
    else:
        added = 0
        print(f'yanit zenginlestirmesi zaten uygulanmis (k={args.k}); '
              'yeni yanit eklenmedi')

    tags = _validate(data)
    if args.dry_run:
        print('dry-run: intents yazilmadi')
    else:
        _save_data(data, args.out)
        print('yazildi:', args.out)

    # Bilgi parcali egitim verisi
    if args.kb_map and not args.dry_run:
        hits, skipped = build_knowledge_map(data['intents'],
                                            limit=args.kb_limit)
        if hits:
            with io.open(args.kb_map, 'w', encoding='utf-8') as f:
                for row in hits:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f'knowledge_map yazildi: {args.kb_map} '
                  f'({len(hits)} eslesme, {skipped} bos/dusuk skor atlandi)')
        else:
            print('knowledge_map bos (eslesme yok), dosya yazilmadi')
    return 0


if __name__ == '__main__':
    sys.exit(main())
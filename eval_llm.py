"""Nextgen AI - LLM uretim kalitesi degerlendirmesi / benchmark (bagimsiz script).

train_llm.py ile ayni veri hattindan (intents.json + knowledge_map.jsonl)
sorgu->gold ciftleri kurar, llm.load_llm() ile yuklu modeli ornekter ve
nesnel metriklerle olcer:

  kopya orani (copy rate)      : BLEU-4 / prec1  --> uretim canned yanitin
                                 ne kadarini aynen kopyaliyor?
  konu dokunusu (topic touch)  : uretilen icerik kelimelerinin sorguya ve
                                 (RAG) bilgi parcasina dokunma orani
  akicilik (fluency)           : 1 - bigram tekrar orani, distinct1,
                                 kelime/uzunluk istatistikleri

Kullanim (yerel, torch gerekmez):
  python eval_llm.py --n 60 --rag
  python eval_llm.py --sweep                 # birden cok decoding konfiguretsi
  python eval_llm.py --out eval_llm_report.json

Metrikler saf/islemci bagimsizdir; ayni seed + model + veri ile birebir ayni
rapor uretir -> model/veri degisikliklerinde regresyon testi gibi calisir.
"""
import argparse
import io
import json
import math
import os
import random
import re
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from llm import load_llm
from train_llm import INTENTS, MAX_PAIRS, refine_resp
from seqgen import load_pairs

KB_MAP = os.path.join(BASE, 'knowledge_map.jsonl')

STOPWORDS = frozenset("""
    bu su o bir ve ile icin kac hangi nasil ne neden niye ama da de daha en var
    yok mi mu ne gibi dahi ya he ben sen biz siz onlar benim senin onun bizim
    sizin beni bana seni sana ona onu lutfen evet hayir tamam olur peki cok
    cok daha sonra once iste ki ancak hem eger yoksa ayni baska sana sana sey
    e mi mis mu sey
""".split())

_WORD_RE = re.compile(r'[a-z\u00e7\u011f\u0131\u00f6\u015f\u00fc0-9]+')


def tokenize(text):
    return _WORD_RE.findall((text or '').lower())


def _ngrams(tokens, n):
    if n <= 1:
        return list(tokens)
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu(ref_tokens, cand_tokens, max_n=4, smooth=False):
    """Modifiye n-gram hassasiyeti tabanli BLEU (Chen&Cherry +1 smoothing).

    Ayni metin -> 1.0, tamamen farkli -> smoothing tabanina yakin dusuk deger.
    Kopya orani icin: ref = canned yanit, cand = uretilen yanit.
    """
    if not cand_tokens:
        return 0.0
    r, c = len(ref_tokens), len(cand_tokens)
    bp = 1.0
    if c < r:
        bp = math.exp(1 - r / max(c, 1))
    precs = []
    for n in range(1, max_n + 1):
        if len(cand_tokens) < n:
            continue
        ref_count = {}
        total = 0
        clipped = 0
        for g in _ngrams(ref_tokens, n):
            ref_count[g] = ref_count.get(g, 0) + 1
        for g in _ngrams(cand_tokens, n):
            total += 1
            if ref_count.get(g, 0) > 0:
                clipped += 1
                ref_count[g] -= 1
        if total == 0:
            continue
        if smooth:
            precs.append((clipped + 1.0) / (total + 1.0))
        elif clipped == 0:
            precs.append(0.0)
        else:
            precs.append(clipped / total)
    if not precs or any(p == 0 for p in precs):
        return 0.0
    return bp * math.exp(sum(math.log(p) for p in precs) / len(precs))


def prec1(ref_tokens, cand_tokens):
    """1-gram kopya hassasiyeti (smoothing'siz clipping; kopya gostergesi)."""
    if not cand_tokens:
        return 0.0
    cnt = {}
    for t in ref_tokens:
        cnt[t] = cnt.get(t, 0) + 1
    hit = 0
    for t in cand_tokens:
        if cnt.get(t, 0) > 0:
            hit += 1
            cnt[t] -= 1
    return hit / len(cand_tokens)


def rep_rate(tokens, n=2):
    """n-gram tekrar orani: 1 - (farkli n-gram / toplam n-gram). 0 = akici."""
    gs = _ngrams(tokens, n)
    if not gs:
        return 0.0
    return 1.0 - len(set(gs)) / len(gs)


def distinct_ratio(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def content_terms(tokens, stopwords):
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def topic_overlap(cand_tokens, ref_tokens, stopwords):
    """Uretilen icerik kelimelerinin referans icerigine dokunma orani."""
    c = content_terms(cand_tokens, stopwords)
    r = set(content_terms(ref_tokens, stopwords))
    if not c:
        return 0.0
    return len(set(c) & r) / len(set(c))


def length_ratio(cand_tokens, ref_tokens):
    if not ref_tokens:
        return 0.0
    return len(cand_tokens) / len(ref_tokens)


def avg_word_len(tokens):
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def _measure(query, gold, generated, knowledge, stopwords):
    cand = tokenize(generated)
    ref = tokenize(gold)
    q = tokenize(query)
    kb = tokenize(knowledge) if knowledge else None
    copy_bleu = bleu(ref, cand)
    copy_prec1 = prec1(ref, cand)
    rep2 = rep_rate(cand, 2)
    distinct1 = distinct_ratio(cand)
    fluency = 0.5 * (1.0 - rep2) + 0.5 * distinct1
    tq = topic_overlap(cand, q, stopwords)
    tk = topic_overlap(cand, kb, stopwords) if kb else None
    tmean = (tq + tk) / 2.0 if tk is not None else tq
    gen_index = 0.5 * (1.0 - copy_bleu) + 0.3 * tmean + 0.2 * fluency
    return dict(
        query=query,
        generated=generated,
        gold=gold,
        has_knowledge=bool(kb),
        n_tokens=len(cand),
        gold_tokens=len(ref),
        copy_bleu=copy_bleu,
        copy_prec1=copy_prec1,
        rep2=rep2,
        distinct1=distinct1,
        fluency=fluency,
        topic_q=tq,
        topic_k=tk,
        topic_mean=tmean,
        length_ratio=length_ratio(cand, ref),
        avg_word_len=avg_word_len(cand),
        gen_index=gen_index,
    )


def sample_report(model, items, temperature=0.7, top_k=10, rep_penalty=0.3,
                  seed=7, stopwords=STOPWORDS, max_len=None,
                  knowledge_bias=0.0):
    """items: (sorgu, gold) ya da (sorgu, gold, bilgi) ucizlileri.

    model.sample(context, temperature=..., top_k=..., knowledge=...,
    rep_penalty=...) imzasini destekleyen herhangi bir nesne olabilir (llm.LLM
    birebir uyumlu). Ayni seed -> birebir ayni rapor (regresyon tekrarlanabilir).
    knowledge_bias>0 ise llm.sample'in bilgi-cekimine (knowledge_bias) beyaz
    aktarilir -> RAG konu dokunusu etkisi olculebilir.
    """
    np.random.seed(seed)
    random.seed(seed)
    rows = []
    for it in items:
        if len(it) == 3:
            query, gold, knowledge = it
        else:
            query, gold = it
            knowledge = None
        try:
            out = model.sample(query, temperature=temperature, top_k=top_k,
                               knowledge=knowledge, rep_penalty=rep_penalty,
                               **({'max_len': max_len} if max_len else {}),
                               **({'knowledge_bias': knowledge_bias}
                                  if knowledge_bias else {}))
        except TypeError:
            out = model.sample(query, temperature=temperature, top_k=top_k,
                               knowledge=knowledge, rep_penalty=rep_penalty)
        except Exception:
            out = ''
        rows.append(_measure(query, gold, (out or '').strip(), knowledge,
                             stopwords))
    return rows


def aggregate_report(rows):
    keys = ['copy_bleu', 'copy_prec1', 'rep2', 'distinct1', 'fluency',
            'topic_q', 'topic_mean', 'length_ratio', 'avg_word_len',
            'gen_index', 'n_tokens', 'gold_tokens']
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows]
        agg[k] = float(np.mean(vals)) if vals else 0.0
        agg[k + '_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
    agg['n_samples'] = len(rows)
    agg['n_empty'] = sum(1 for r in rows if r['n_tokens'] == 0)
    tk = [r['topic_k'] for r in rows if r['topic_k'] is not None]
    agg['topic_k'] = float(np.mean(tk)) if tk else None
    agg['topic_k_std'] = float(np.std(tk)) if len(tk) > 1 else 0.0
    agg['n_knowledge'] = len(tk)
    return agg


def print_report(title, agg):
    print(f'== {title} ==')
    print(f"  kopya orani   (bleu vs canned): {agg['copy_bleu']:.3f} "
          f"(± {agg['copy_bleu_std']:.3f})")
    print(f"  1-gram kopya  (prec1)          : {agg['copy_prec1']:.3f} "
          f"(± {agg['copy_prec1_std']:.3f})")
    print(f"  konu dokunusu (sorgu)          : {agg['topic_q']:.3f} "
          f"(± {agg['topic_q_std']:.3f})")
    if agg.get('topic_k') is not None:
        print(f"  konu dokunusu (bilgi, n={agg['n_knowledge']})"
              f" : {agg['topic_k']:.3f} (± {agg['topic_k_std']:.3f})")
    print(f"  akicilik      (1-bigram tekrar): {agg['fluency']:.3f} "
          f"(± {agg['fluency_std']:.3f})")
    print(f"  zesitlilik    (distinct1)      : {agg['distinct1']:.3f} "
          f"(± {agg['distinct1_std']:.3f})")
    print(f"  uzunluk orani (uretilen/hedef) : {agg['length_ratio']:.2f}")
    print(f"  ort kelime uzunlugu            : {agg['avg_word_len']:.1f}")
    print(f"  genel skor    (0-1)            : {agg['gen_index']:.3f} "
          f"(± {agg['gen_index_std']:.3f})")
    print(f"  ornek: {agg['n_samples']} | bostler: {agg['n_empty']}")
    print()


def _load_items(rag, limit):
    pairs = load_pairs(INTENTS, max_pairs=MAX_PAIRS, use_query=True)
    pairs = [(ctx, rr) for ctx, r in pairs if (rr := refine_resp(r)) is not None]
    kb_pre = {}
    if rag and os.path.exists(KB_MAP):
        with io.open(KB_MAP, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get('ctx') and row.get('text'):
                    kb_pre[row['ctx']] = row['text']
        print(f'kb-map yuklendi: {len(kb_pre)} desen', flush=True)
    items = []
    for ctx, gold in pairs:
        k = kb_pre.get(ctx)
        items.append((ctx, gold, k) if k and rag else (ctx, gold))
        if limit and len(items) >= limit:
            break
    return items


def main():
    ap = argparse.ArgumentParser(description='LLM uretim kalitesi benchmarki')
    ap.add_argument('--n', type=int, default=60, help='orneklenecek cift sayisi')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--rep-penalty', type=float, default=0.4)
    ap.add_argument('--max-len', type=int, default=None,
                    help='uretilecek maksimum token sayisi (varsayilan: model 96)')
    ap.add_argument('--rag', action='store_true',
                    help='knowledge_map.jsonl ile bilgili ciftlere bilgi parcasi ekle')
    ap.add_argument('--sweep', action='store_true',
                    help='birincil decoding konfiguresetlerini karsilastir')
    ap.add_argument('--out', default=None, metavar='PATH',
                    help='rapor JSON dosyasi (varsayilan: yazilmaz)')
    ap.add_argument('--knowledge-bias', type=float, default=0.0,
                    help='llm.sample konu cekimi bonusu (1.2 onerilen; '
                         '0.0 = kapali, egitimli taban cizgisi)')
    args = ap.parse_args()

    model = load_llm()
    if model is None:
        print('model/llm_model.json bulunamadi - once egitilmis model kopyala')
        return 2
    items = _load_items(args.rag, args.n)
    print(f'degerlendirme seti: {len(items)} cift (rag={args.rag})',
          flush=True)
    info = (f'd={model.d_model} blok={model.num_blocks} '
            f'kafa={model.num_heads} ff_mult={getattr(model, "ff_dim", 0) // model.d_model} '
            f'max_ctx={model.max_ctx_len} max_seq={model.max_seq_len} '
            f'{"bpe" if model.tokenizer else "char"}')

    def run(temperature, top_k, rep_penalty, kb):
        rows = sample_report(model, items, temperature=temperature,
                             top_k=top_k, rep_penalty=rep_penalty,
                             seed=args.seed, max_len=args.max_len,
                             knowledge_bias=kb)
        return aggregate_report(rows), rows

    if args.sweep:
        configs = [(0.6, 6, 0.3, 0.0), (0.7, 10, 0.4, 0.0),
                   (0.7, 10, 0.4, 1.2), (0.7, 10, 0.3, 1.2)]
        out_rows = {}
        print(f'MODEL: {info}\n')
        for temperature, top_k, rp, kb in configs:
            agg, rows = run(temperature, top_k, rp, kb)
            label = (f'config t={temperature} k={top_k} rep={rp} kb={kb} '
                     f'score={agg["gen_index"]:.3f} | kop={agg["copy_bleu"]:.3f} '
                     f'| konu={agg["topic_mean"]:.3f} | akic={agg["fluency"]:.3f}')
            print('== ' + label)
            out_rows[label] = agg
    else:
        print(f'MODEL: {info}')
        print(f'CONFIG: temperature={args.temperature} top_k={args.top_k} '
              f'rep_penalty={args.rep_penalty} seed={args.seed} '
              f'knowledge_bias={args.knowledge_bias}\n')
        agg, rows = run(args.temperature, args.top_k, args.rep_penalty,
                        args.knowledge_bias)
        print_report('SONUC', agg)
        worst = sorted(rows, key=lambda r: r['gen_index'])[:5]
        print('-- en dusuk 5 ornek --')
        for r in worst:
            print(f"[{r['gen_index']:.2f}] S: {r['query'][:40]}")
            print(f"     A: {r['generated'][:90]}")
            print(f"     G: {r['gold'][:70]}")
    if args.out:
        with io.open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'model': info, 'n': len(items), 'rag': args.rag,
                       'seed': args.seed, 'config': vars(args),
                       'report': agg}, f, ensure_ascii=False, indent=1)
        print('rapor yazildi:', args.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
# -*- coding: utf-8 -*-
"""TOKEN_PER_PAIR icin daha guvenilir olcum (n=1000, 3 tohum).

Sadece OKUR: hicbir dosyaya yazmaz. Amac, 250 orneklemle secilen 90,6'nin
orneklem gurultusu mu yoksa gercek bir kayma mi oldugunu ayirt etmek.
"""
import os
import random
import sys

import numpy as np

# Repo koku. SABIT YOL KULLANILMAZ: GitHub Actions'ta o yol yok ve
# tests/test_no_hardcoded_paths.py de reddeder. Bu dosya tools/ altinda,
# kok bir ust dizindir.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

os.chdir(BASE)

import train_llm
from llm import LLM, load_tokenizer, encode_llm
from seqgen import load_pairs

tok = load_tokenizer()
if tok is None:
    print('tokenizer yuklenemedi - olcum yapilamaz')
    raise SystemExit(1)

intents = os.path.join(BASE, 'intents.json')
pairs = load_pairs(intents, max_pairs=train_llm.coz_max_pairs(yaz=False),
                   use_query=True, ctx_len=train_llm.CTX_CHARS)
kb = train_llm.build_kb_lut(os.path.join(BASE, 'knowledge_map.jsonl'))

dummy = LLM(None, d_model=4, num_blocks=1, num_heads=1,
            max_ctx_len=train_llm.MAX_CTX_LEN,
            max_seq_len=train_llm.MAX_SEQ_LEN,
            seed=train_llm.SEED, tokenizer=tok)

print('cift sayisi (havuz): %d   kb-map: %d' % (len(pairs), len(kb)))
print('')
print('  tohum   n    ortalama   sapma     min   p50   p95   maks   kb%%')
print('  ' + '-' * 66)

buyuk = []
for tohum in (11, 23, 37):
    ornek = random.Random(tohum).sample(pairs, 1000)
    uzunluklar = []
    kb_varli = 0
    for ctx, resp in ornek:
        rr = train_llm.refine_resp(resp)
        if rr is None:
            continue
        ctx_kb = kb.get(ctx)
        if ctx_kb:
            kb_varli += 1
        seq, _m = encode_llm(dummy, ctx, rr, context=ctx_kb)
        seq = np.asarray(seq)
        L = int(np.argmax(seq == train_llm.PAD)) if (seq == train_llm.PAD).any() \
            else int(seq.shape[0])
        uzunluklar.append(max(1, L))
    a = np.asarray(uzunluklar, dtype=float)
    buyuk.extend(uzunluklar)
    print('  %5d %5d   %8.2f %7.2f %6d %5d %5d %6d   %4.1f%%'
          % (tohum, a.size, a.mean(), a.std(), a.min(),
             int(np.percentile(a, 50)), int(np.percentile(a, 95)),
             int(a.max()), 100.0 * kb_varli / max(1, len(ornek))))

a = np.asarray(buyuk, dtype=float)
se = a.std() / np.sqrt(a.size)
print('')
print('TOPLAM  n=%d  ortalama=%.2f  standart hata=%.2f  (%%95 aralik: %.1f - %.1f)'
      % (a.size, a.mean(), se, a.mean() - 1.96 * se, a.mean() + 1.96 * se))
print('sabit    TOKEN_PER_PAIR simdi = %s' % train_llm.TOKEN_PER_PAIR)
print('fark     canli - sabit = %+.2f (%%%.1f fark)'
      % (a.mean() - train_llm.TOKEN_PER_PAIR,
         100.0 * (a.mean() - train_llm.TOKEN_PER_PAIR) / train_llm.TOKEN_PER_PAIR))

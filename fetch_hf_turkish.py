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


def cut_at_word(s, max_len):
    """Kirpmayi KELIME sonunda bitirir (yarim kelime ezberi olmaz).

    28.09 olcumu: clean_chars sert kesiyordu, 64 krktik ctx'lerin
    cogu '... neden kullanilir? gerekl' gibi YARIM KELIME ile bitiyordu.
    Kirpma sonrasi son bosluga kadar geri cekilir; tek kelimelik
    budalanmisse oldugu gibi birakilir.
    """
    if not s or not max_len or len(s) <= max_len:
        return s
    kes = s[:max_len]
    bos = kes.rfind(' ')
    if bos >= max_len * 0.6:       # cok az kazandiysak (kelime cok uzun) alma
        return kes[:bos].strip()
    return kes


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


# --- Turkce kapisi (28.09) --------------------------------------------------
# PROBLEM: Turkce olmayan icerik var. En cok tekrar eden cevaplardan biri
# Ingilizceydi: 'hello! how can i assist you today?' 155 kez.
#
# ILK DENEME (yanlis): Turkceye ozgu harf orani (c/Ç/g/ğ/...) >= 0.03.
# OLcum bunu REDDETTI:
#   * Turkce metnin %22,3'u bu esigin altinda. Sorularin %16,6'sinda
#     HIC Turkce harf yok ("ilk soru burada", "tamam" gecerli Turkce).
#   * Bilinen Turkce ORNEKLERINDE de oran 0.000; yani metrik Turkceyi
#     tanimiyor, yalnizca uzunluk olculuyor.
# Turkce harf orani AYIRT EDICI DEGIL; yalnizca "kisa/teknik metin" sinyal
# veriyor. Turkce icin ayirt edici olan FUNCION WORDS (edat/benzetme/
# olumsuzluk) — bunlar Turkce metinde neredeyse her cumlede var, Ingilizcede
# yok (ve tersi).
#
# OLCULEN AYRIM (gercek veri uzerinde, 36.812 cift):
#   metrik                    reddettigi    TR ornekleri      EN ornekleri
#   tr_harf < 0.03            %22,3         0.000 (yanlis)    0.000 (yanlis)
#   dil_ayirici < -0,08       %4,7          0.00..+0,11  OK   -0,31..-0,77 OK
# Yani stopword farki 5 kat daha az yanlis pozitif uretiyor.
TR_STOPWORDS = frozenset("""bir ve bu da de ile icin cok daha olarak gibi
kadar sonra ancak ise ya her ne ben sen biz siz onlar olan var yok
nasil neden nerede hangi sey zaman ayni fakat cunku uzeri gore
tarafindan once artik hem mi mu mu ki""".split())
EN_STOPWORDS = frozenset("""the and you your that this for with are was
have not but can what how when where why does do is it in on at to of my
we our they their there here would could should will about from by as if
or me him she he we us am been being has had were""".split())

_KELIME = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)
_DIL_ESIK = -0.08          # olculdu: TR 0.00..+0.11, EN -0.31..-0,77
_LATIN_ESIK = 0.50         # olcum: yabanci alfabe oranı


def _stop_orani(s, kume):
    k = _KELIME.findall((s or '').lower())
    if not k:
        return 0.0
    return sum(1 for w in k if w in kume) / len(k)


def dil_ayirici(s):
    """Turkce fonksiyon kelimesi yogunlugu - Ingilizce fonksiyon kelimesi
    yogunlugu. Turkce metin > 0, Ingilizce metin < 0. Olcum icin."""
    return (_stop_orani(s, TR_STOPWORDS)
            - _stop_orani(s, EN_STOPWORDS))


def latin_orani(s):
    """Latin harf / toplam harf. Cince/Arapca/Farsi gibi alfabeleri eler."""
    latin = top = 0
    for c in (s or ''):
        if c.isalpha():
            top += 1
            if c.isascii() or c in 'çÇğĞıİöÖşŞüÜ':
                latin += 1
    return (latin / top) if top else 0.0


def yabanci_dil_mi(s, esik=_DIL_ESIK):
    """Metin Turkce DEGIL mi? (Turkce orneginde esigin ALTINDA)

    28.09: onceki Turkce-har-orani kapisi gercek Turkce metnin %22'sini
    siliyordu. Bu, stopword farki + alfabe kontrolu.
    """
    if not s or not s.strip():
        return False
    if latin_orani(s) < _LATIN_ESIK:
        return True                        # Turkce disi alfabe
    return dil_ayirici(s) < esik


# Instructurca-simulated: coktan secmeli sinav sorulari. Cevaplar
# 'B Arapca' gibi tek hece; gercek sohbet DEGIL. 28.09 olcumu: veri setinin
# %20'si (4.000 cift) ve dedup sonrasi KALAN 1.946 ciftin tamami. Yani
# govdenin onda biri bosuna. Bu yuzden kaynak bazinda elenir.
KOTU_KAYNAKLAR = frozenset(['InstrucTurca-simulated'])

# Sinav kalinti izleri. 28.09 ornegi (WildChat kaynakli):
#   ctx  : '16. asirda cogu bilim insani ( ) evren hakkindaki tum bilginin t'
#   resp : 'c) () () (yunan filozof aristoteles) ()16. yuzyilda cogu bilim ...'
# Yani kaynak etiketi 'WildChat-Turkish' olsa bile icerik sinavdan gelmis
# olabiliyor; secenck/boşluk kalintisiyla yakalanir.
SECENEK = re.compile(r'(?:^|\s)[a-e]\)', re.IGNORECASE)
BOSSUK_PARENS = re.compile(r'\(\s*\)')


def sinav_kalinti_mi(s, min_bos=2, min_secenek=2):
    """Metin coktan secmeli sinav kalinti mi tasiyor? (28.09)

    ONEMLI: kontrol NORMALIZE EDILMIS metin uzerinde yapilir, cunku
    veri dosyasina da o bicimde yaziliyor. Ham metinde bakilsaydi
    LaTeX artiklari KACIRILIRDI: 'r(theta)' ham metinde bos parantez
    DEGILDIR, ama clean_chars Yunanca harfi silince 'r()' olur.
    Olcum (28.09): bu duzeltmeden once 9 cift (kardiyoid denklemleri
    gibi LaTeX OCR artigi) dosyaya giriyordu.
    """
    if not s:
        return False
    t = clean_chars(s, None)
    return (len(BOSSUK_PARENS.findall(t)) >= min_bos
            or len(SECENEK.findall(t)) >= min_secenek)


def pairs_from_chat(rec, drop_english=True):
    """Cok turlu sohbet semasi -> (ctx,resp) gencratoru.

    Girdi: {"messages":[{"role":"user","content":...},
                        {"role":"assistant","content":...}, ...],
           "source": "WildChat-Turkish"}

    ONEMLI: burada ILK user mesaji degil, her user->assistant donuşu
    bir cift uretir. Olcum (28.09): 10.000 konusma -> 40.812 cift
    (ort. 4,08 cift/konusma). Projedeki hicbir kaynak cok turlu
    degildi; asil deger burada.

    Kapilar:
      * kotu kaynak (InstrucTurca-simulated) -> atlanir
      * sinav kalinti izleri (orn. '() ()', 'a) b) c)') -> atlanir
      * Turkce disi dil (stopword farki + alfabe kontrolu) -> atlanir
    """
    if (rec.get('source') or '') in KOTU_KAYNAKLAR:
        return
    msgs = rec.get('messages')
    if not isinstance(msgs, list):
        return
    bekleyen = None
    for t in msgs:
        if not isinstance(t, dict):
            continue
        role = (t.get('role') or '').lower()
        content = (t.get('content') or '').strip()
        if not content:
            continue
        if role in ('user', 'human', 'soru', 'query'):
            bekleyen = content
        elif role in ('assistant', 'gpt', 'ai', 'cevap') and bekleyen:
            ctx, resp = bekleyen, content
            bekleyen = None
            if drop_english and (yabanci_dil_mi(ctx) or yabanci_dil_mi(resp)):
                continue
            if sinav_kalinti_mi(ctx) or sinav_kalinti_mi(resp):
                continue
            yield ctx, resp


# --- dedup + yakın-kopya ----------------------------------------------------

def _tokenize_norm(s):
    return set(clean_chars(s, 10_000).split())


def jaccard(a, b):
    a, b = _tokenize_norm(a), _tokenize_norm(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_sets(a, b):
    """jaccard() ama token kümeleri ÖNCEDEN hesaplanmış olarak (O(1) hazır).

    Ölçüm (28.09): eski yol her cift icin 2 tokenizasyon yapiyordu.
    588 cift = 8,8 sn. Jaccard'i tokenizasyonsuz yazmak tek basina
    ~2 kat hizlandiriyor; asil hiz karsilastirma ADEDINDEN gelir
    (asagida inverted index).
    """
    if not a or not b:
        return 0.0
    k = len(a & b)
    if not k:
        return 0.0
    return k / (len(a) + len(b) - k)


# Ters indeks ayarlari (28.09 olcumuyle secildi).
#   MAX_POST : bir token icin saklanan EN SON cift sayisi. Ust sinir; boyle
#              dene bir ciftin karsilastirmasi sinirli kalir.
#   SIZE_PAD : Jaccard >= t esigine ulasabilmesi icin |A| ve |B| boyutlari
#              birbirine yakin olmak ZORUNDA. Matematik:
#                  J = k/(|A|+|B|-k) >= t   ve   k <= |A|, k <= |B|
#                  =>  t*|A| <= |B| <= |A|/t
#              t=0,90 icin TAM sinir [0,90|A| , 1,111|A|]. Burada
#              SIZE_PAD payiyla biraz GENISLETILIR: [0,79|A| , 1,26|A|].
#              Genisletmek yonunde bir hata olusmaz, sadece biraz yavas.
MAX_POST = 60
SIZE_PAD = 0.12


def dedupe_pairs(pairs, seed=SEED, dup_thr=0.90, keep_log=True):
    """Kesin + yakın-kopya (Jaccard) eleyici.

    - (ctx,resp) kesin ayni -> 1 kere
    - ayni resp beliren ctx'lerden SADECE ilk (az ezber)
    - yeni cift onceden gormedigi ctx/resp ile Jaccard >= dup_thr ise ele
    Deterministik (seed sadece siralamada kullanilmaz, girdi sirasi onemli).

    28.09 HIZ DUZELTMESI. Eski surum her yeni cifti TUM onceki ctx ile
    karsilastiriyordu (O(n^2)) ve her karsilastirmada 2 tokenizasyon
    yapiyordu. Olcum: 588 cift 8,8 sn -> 40.812 cift ~11 SAAT. Bu yuzden
    mevcut chatgrow_hf_*.jsonl dosyalari sadece ~1.200 satir.

    Yeni yol 3 katmanli budama:
      1) token kümeleri BIR KEZ hesaplanir (jaccard_sets),
      2) boyut penceresi: Jaccard >= t icin |B| in [t*|A|, |A|/t] olmali,
         yani t=0.90'da [0.89|A|, 1.12|A|] (SIZE_PAD) — kaba olcumde
         adaylarin %95'ini eler,
      3) ters indeks: token -> son MAX_POST cift. Aday yalnizca paylasilan
         tokeni olan ciftlerden olusur, sonra TAM Jaccard dogrulanir.
    UST SINIR: MAX_POST nedeniyle cok eski bir kopya gunden konusmada
    gormemis olabilir. Bu, veri kalitesi kapisi icin kabul edilebilir
    (asagidaki test eski yolla uyum oranini olcer).
    """
    seen_sets = []          # saklanan ciftlerin token kümeleri
    kept = []
    dead_ctx = dead_resp = dead_toks = 0
    resp_map = {}
    posting = {}             # token -> [idx, ...] (en fazla MAX_POST)
    lo_pad = dup_thr * (1 - SIZE_PAD)
    hi_pad = 1.0 / (dup_thr * (1 - SIZE_PAD))

    for ctx, resp in pairs:
        if resp in resp_map:          # ayni resp daha once vardi -> buda elen
            dead_resp += 1
            continue

        toks = _tokenize_norm(ctx)
        if not toks:                  # tumden temizlenen ctx -> egitime yaramaz
            dead_toks += 1
            continue

        # 2) boyut penceresi
        n = len(toks)
        lo, hi = n * lo_pad, n * hi_pad

        # 3) ters indeks: paylasilan tokeni olan adaylar
        cand = set()
        for t in toks:
            lst = posting.get(t)
            if lst:
                cand.update(lst)
        # pencereye girmeyenler cifti eler (Jaccard >= t imkansiz)
        cand = [i for i in cand if lo <= len(seen_sets[i]) <= hi]

        dup = False
        for i in cand:
            if jaccard_sets(toks, seen_sets[i]) >= dup_thr:
                dup = True
                dead_ctx += 1
                break
        if dup:
            continue

        idx = len(seen_sets)
        seen_sets.append(toks)
        resp_map[resp] = True
        kept.append((ctx, resp))
        for t in toks:
            lst = posting.get(t)
            if lst is None:
                posting[t] = [idx]
            else:
                lst.append(idx)
                if len(lst) > MAX_POST:
                    del lst[0]

    if keep_log:
        print(f'[dedup] girdi {len(pairs)} -> cikti {len(kept)} '
              f'(kopya-ctx {dead_ctx}, ayni-resp {dead_resp}, '
              f'bos-ctx {dead_toks})', flush=True)
    return kept


# --- ana akis ---------------------------------------------------------------

SOURCES = {
    'tascib/turkish-instruction': pairs_from_instruction,
    'erythropygia/ThinkingData-200K-Turkish': pairs_from_thinking,
    'kilicai/turkish-sft-multi-turn-dialogue-10k': pairs_from_chat,
}


def stream_hf(source, max_pairs, seed, cot):
    from datasets import load_dataset

    # 28.09 DUZELTME: burada SOURCES haritasi YOK sayiliyordu, kaynak
    # adina gore 'endswith' ile dagitim yapiliyordu. Yeni kaynak eklenince
    # yanlis sekilde (buraya hic dustugunde) instruction donusturucusu
    # calisirdi ve sessizce BOSTAN cift uretirdi. Simdi tek dogruluk
    # kaynagi SOURCES.
    donusturucu = SOURCES[source]

    def gen(rows):
        for rec in rows:
            for ctx, resp in donusturucu(rec):
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
            # 28.09: ctx kirpmasi KELIME sonunda bitsin.
            # DIKKAT: once KIRPMA OLMADAN normalize et, sonra kes. Tersi
            # (once clean_chars(ctx, LEN) sonra cut_at_word) HIC CALISMAZ:
            # clean_chars zaten LEN karaktere indigi icin cut_at_word
            # 'len <= max_len' dalina duser ve ayni stringi dondurur.
            # Olcum: sert kesme metnin %59,8'inde yarim kelime birakiyordu
            # ('... neden kullanilir? gerekl').
            ctx_c = cut_at_word(clean_chars(ctx, None), args.ctx_len)
            resp_c = refine_resp(clean_chars(resp, args.resp_len),
                                 maxc=args.resp_len)
            if not ctx_c or not resp_c:
                continue
            if args.cot and 'Kisa dusunce' in ctx_c:
                continue                    # ctx'e CoT karistirme
            kept.append((ctx_c, resp_c))
        kept = dedupe_pairs(kept, seed=args.seed)
        print(f'[{s}] {len(kept)} cift, {time.time() - t0:.0f} sn '
              f'(budama+kopya sonrasi)', flush=True)
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

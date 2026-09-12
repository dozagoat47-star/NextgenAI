"""
Nextgen AI - Generative Katman (RAG-Lite)

Dışarıdan hicbir hazir kutupname/API kullanmadan (yalnizca Python stdlib)
retrieval ciktisini alip kendi cumlelerini kuran metin uretici.

Yaklasim:
  - Bilgi yuklu cekirdekler (ozel isimler, sayilar, tarihler, baslik
    kelimeleri) 'anchor' olarak SABITLENIR; gercekler bozulmaz.
- Kalan yapi (dolgu kelimeler, baglaclar, ek duzenleyiciler) yerel
    (retrieved metin) ve genel (corpus + intents) kelime gecis
    olasiliklarindan (bigram Markov + unigram geri donus, duzeltilmis
    pürüzsüzlestirme) uretilir.
  - 3'ten uzun olan HER bilgi cumlesi uretimden gecirilir; yalnizca mikro
    yanitlar (3 kelime ve alti) ve tablo-yogun (sayi anchor) paragraflar
    guvenli gecis yapar (cikti kalitesi bilgiden once gelmez).

Uretilen cumle retrieved metnin aynisi DEGILDIR; cekirdek faktler korunarak
yeni dizilimler olusturulur.
"""

import os
import io
import json
import random
import re

from collections import defaultdict

from brain import STOPWORDS


def ascii_normalize(text):
    """Turkce ozel karakterler ASCII karsiligina cevrilir (brain ile ayni tablo)."""
    table = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'â': 'a', 'î': 'i', 'û': 'u', 'i': 'i', 'o': 'o', 'u': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'I': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
        'Â': 'A', 'Î': 'I', 'Û': 'U',
        '\u0307': '',
    }
    return text.translate(str.maketrans(table))


# Soz dizimi baglayicilari; uretimde ara bosluklari doldurmak icin olasilikla
# secilir. Korpus basi-token dagilimi bu listenin onune gecer.
DISC_CONNECTORS = [
    'Ayrıca', 'Buna göre', 'Bu bağlamda', 'Sonuç olarak', 'Öte yandan',
    'Bununla birlikte', 'Bu doğrultuda', 'Genel olarak',
]

# Ara duzenleyici sozcukler: bigram modelde gecmezse geri donus havuzu.
GLUE_FALLBACK = ['olarak', 'ise', 'gibi', 'açısından', 'başlığında',
                 'konusunda', 'doğrultusunda']


class TextGenerator:
    """Bigram gecis olasilik matrisi uzerinden anchor-sabit yeni cumle uretici."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.trans = defaultdict(lambda: defaultdict(float))   # head -> {next: cnt}
        self.unigram = defaultdict(float)
        self.starters = defaultdict(float)                     # cumle basi kelimeler
        self.built = False

    # ------------------------------------------------------------- YAPI KURMA
    def _clean_words(self, text):
        """Ascii-normalize edilmis, yalnizca [a-z0-9] tokenlari."""
        t = ascii_normalize((text or '').replace('\u00a0', ' '))
        return re.findall(r'[a-z0-9]+', t.lower())

    def _split_sentences(self, text):
        protected = re.sub(r'(\d)\.(\d)',
                           lambda m: m.group(1) + '\uFF0E' + m.group(2), text)
        sents = re.split(r'[.!?…]+\s*', protected)
        return [s.replace('\uFF0E', '.').strip() for s in sents if s.strip()]

    def build_from_files(self, corpus_path, intents_path,
                         max_chunks=3000, chunk_chars=700):
        """Genel gecis modelini corpus.jsonl (orneklem) + intents.json ile kurar."""
        try:
            texts = []
            if corpus_path and os.path.exists(corpus_path):
                with io.open(corpus_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= max_chunks:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            c = json.loads(line)
                        except Exception:
                            continue
                        txt = (c.get('text') or '')[:chunk_chars]
                        if txt:
                            texts.append(txt)
                        pat = c.get('patterns') or ''
                        if len(pat) >= 12:
                            texts.append(pat[:300])
            if intents_path and os.path.exists(intents_path):
                try:
                    data = json.load(io.open(intents_path, 'r', encoding='utf-8'))
                    for it in data.get('intents', []):
                        for p in it.get('patterns', [])[:8]:
                            texts.append(p)
                        for r in it.get('responses', [])[:6]:
                            texts.append(r)
                except Exception:
                    pass
            self.build_texts(texts)
            self.built = True
            print(f"[generator] Genel gecis modeli kuruldu: "
                  f"{len(texts)} metin, {len(self.unigram)} kelime, "
                  f"{len(self.trans)} baglam")
        except Exception as e:
            self.built = False
            print(f"[generator] build hatasi (gozardi): {e}")

    def build_texts(self, texts):
        for txt in texts:
            for s in self._split_sentences(txt):
                words = self._clean_words(s)
                if not words:
                    continue
                self.starters[words[0]] += 1.0
                prev = None
                for w in words:
                    self.unigram[w] += 1.0
                    if prev is not None:
                        self.trans[prev][w] += 1.0
                    prev = w
        # guvenilmez tekil olaylari buda (noise azaltir)
        self.unigram = {w: c for w, c in self.unigram.items() if c >= 2}
        for head in list(self.trans):
            row = self.trans[head]
            if sum(row.values()) < 3:
                del self.trans[head]
                continue
            for nxt in list(row):
                if row[nxt] < 2:
                    del row[nxt]

    # -------------------------------------------------------- ANCHOR TESPITI
    def _is_anchor_orig(self, orig, norm):
        if not norm:
            return False
        if any(ch.isdigit() for ch in norm):     # 1980, 5, 3.14
            return True
        # Orjinal metinde buyuk harfle baslayan ozel isim/terim
        if orig and orig[0].isupper() and len(norm) >= 3 and norm not in STOPWORDS:
            return True
        return False

    def _extract_anchors(self, text, title=None):
        """Sirali benzersiz anchor kelimeleri (gorunen yazilislariyla) dondurur."""
        anchors, seen = [], set()
        for tok in text.split():
            norm = ascii_normalize(tok.lower()).strip('.,;:!?()[]"\'“”‘’·-')
            if not norm:
                continue
            if self._is_anchor_orig(tok, norm) and norm not in STOPWORDS:
                if norm not in seen:
                    seen.add(norm)
                    anchors.append(tok)
        if title:
            for tok in ascii_normalize(title).replace('_', ' ').split():
                t = tok.strip('.,;:')
                key = t.lower()
                if len(key) >= 3 and key not in STOPWORDS and key not in seen:
                    seen.add(key)
                    anchors.append(t)
        return anchors[:14]

    def _local_model(self, sents):
        """Retrieved metnin yerel bigram/unigram modeli (guclu agirlik alir)."""
        local = defaultdict(dict)
        lgram = defaultdict(float)
        for s in sents:
            words = self._clean_words(s)
            prev = None
            for w in words:
                lgram[w] += 1.0
                if prev is not None:
                    row = local[prev]
                    row[w] = row.get(w, 0.0) + 1.0
                prev = w
        return local, lgram

    def _split_phrases(self, s):
        """Cumleyi yalnizca virgul/noktalivirgul sinirlarindan parcalara boler.

        've/ile' uzerinden bolme, numaralama listelerini ('yildiz, gaz ve toz')
        parcalayip tek kelimelik parcalar uretiyordu; o yuzden sadece tümce
        sirasindaki virguller tümce parcalari olarak alinir (ic duzeni korur).
        """
        parts = re.split(r'\s*,\s*|\s*;\s*', s)
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------ URETICILER
    def _sample_connector(self):
        # Olasilikla genel korpus basi-tokenlerinden, yoksa guvendikleri listeden
        if self.starters and self.rng.random() < 0.5:
            total = sum(self.starters.values())
            r = self.rng.random() * total
            acc = 0.0
            for w, c in self.starters.items():
                acc += c
                if acc >= r:
                    return ascii_normalize(w).capitalize()
        return self.rng.choice(DISC_CONNECTORS)

    def _sample_after(self, prev_norm, local, lgram):
        """prev'den bir sonraki kelimeyi yerel+genel karisimdan uretir."""
        cand = {}
        for w, c in local.get(prev_norm, {}).items():
            cand[w] = cand.get(w, 0.0) + 6.0 * c
        for w, c in self.trans.get(prev_norm, {}).items():
            cand[w] = cand.get(w, 0.0) + 0.8 * c
        if not cand:
            # unigram geri donus: yerel konu kelimelerine agirlik ver
            for w, c in lgram.items():
                if len(w) >= 4 and w not in STOPWORDS:
                    cand[w] = cand.get(w, 0.0) + 1.0
            if not cand:
                return None
        items, weights = zip(*cand.items())
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for w, wt in zip(items, weights):
            acc += wt
            if acc >= r:
                return w
        return items[-1] if items else None

    def _synonym_neighbor(self, word, local, lgram):
        """Ayirt edici olmayan bir sozcugu, benzer baglamdaki bir komsuyla degistirir
        (baglam kosinüs benzerligi). Anchor/ozel isimler asla degistirilmez."""
        ctx = local.get(word, {})
        if not ctx:
            ctx = self.trans.get(word, {})
        if not ctx:
            return None
        best, best_score = None, 0.35
        pool = [w for w in lgram if w != word and len(w) >= 4
                and w not in STOPWORDS and len(set(ctx) & set(local.get(w, {}))) > 0]
        for nxt in self.rng.sample(pool, min(8, len(pool))) if pool else []:
            ctx2 = local.get(nxt, {})
            if not ctx2:
                ctx2 = self.trans.get(nxt, {})
            if not ctx2:
                continue
            inter = set(ctx) & set(ctx2)
            if not inter:
                continue
            na = sum(ctx.get(w, 0.0) for w in inter)
            nb = sum(ctx2.get(w, 0.0) for w in inter)
            if na * nb > 0:
                sim = min(1.0, (na + nb) / (len(inter) + 1.0))
                if sim > best_score:
                    best, best_score = nxt, sim
        return best

    def _crop(self, text, limit=150, scan=120):
        """Kisa/yoğun metin icin guvenli tam-cumle kirpma (kopya passthrough)."""
        text = re.sub(r'\s+', ' ', (text or '')).strip()
        if not text:
            return ''
        m = re.search(r'[.!?]', text[:limit])
        if m:
            return text[:m.end()].strip()
        m2 = re.search(r'[.!?]', text[limit:limit + scan])
        if m2 and len(text[:limit + m2.end()]) <= limit + scan:
            return text[:limit + m2.end()].strip()
        if len(text) <= limit:
            return text.strip()
        return text[:limit].rstrip() + '...'

    # ------------------------------------------------------------ PUBLIC API
    def generate_response(self, raw_text, title=None, max_chars=140):
        """Retrieved ham metinden anchor-sabit yeni cumle uretir.

        Kisa metin / asiri bilgi yogunlugu durumunda gercekleri koruyarak
        dogrudan gecirir (crop). Hata durumunda da guvenli gecis yapar.
        """
        try:
            text = re.sub(r'\s+', ' ', (raw_text or '')).strip()
            if not text:
                return ''
            # Kelime sayisi <= 3 olan mikro yanitlar ('rica ederim', 'evet')
            # yerinde durur; 3'ten uzun HER bilgi cumlesi uretimden gecer.
            if len(self._clean_words(text)) <= 3:
                return self._crop(text)

            sents = self._split_sentences(text)
            if not sents:
                return self._crop(text)

            local, lgram = self._local_model(sents)
            anchors = self._extract_anchors(text, title)
            nwords = max(1, len(self._clean_words(text)))
            # Tablo/seviye gibi sayi-anchor yogun metinler yeniden dizilimde
            # gercekleri bozar; o zaman dogrudan ver. Normal adli metinler
            # (otuken isimler) bu esige takilmaz.
            digit_anchors = [a for a in anchors
                             if any(ch.isdigit() for ch in ascii_normalize(a))]
            if digit_anchors and len(digit_anchors) / float(nwords) > 0.15:
                return self._crop(text)

            chosen = sents[:3]
            out = []
            for idx, s in enumerate(chosen):
                phrases = self._split_phrases(s)
                if not phrases:
                    continue
                restitched = self._restitch(phrases, local, lgram, anchors, idx == 0)
                if restitched:
                    out.append(restitched)
                if sum(len(x) for x in out) > max_chars * 2:
                    break

            result = ' '.join(o.strip() for o in out if o.strip())
            if len(result) < 20:
                return self._crop(text)
            if result and result[-1] not in '.!?…':
                result += '.'
            return result
        except Exception:
            return self._crop(raw_text)

    def _restitch(self, phrases, local, lgram, anchors, is_first):
        # 1) Dolgu/durak kelimeleri olasilikla hafiflet (ic duzen korunur)
        anchor_norms = {ascii_normalize(a.lower()).strip('.,;:') for a in anchors}
        clean = []
        for p in phrases:
            toks = [t for t in p.split() if t]
            kept = []
            for i, t in enumerate(toks):
                norm = ascii_normalize(t.lower()).strip('.,;:')
                drop = (i > 0 and norm in STOPWORDS and len(norm) >= 3
                        and norm not in anchor_norms and self.rng.random() < 0.15)
                if drop:
                    continue
                kept.append(t)
            if kept:
                clean.append(' '.join(kept))

        if not clean:
            return ''

        # 2) Cok parcali cumlelerde son parcalari hafif karistir; ilk parca sabit.
        #    Tek kelimelik parcalar (liste ogesi gibi) sonda kalmasin.
        if (len(clean) >= 3 and self.rng.random() < 0.35
                and all(len(p.split()) >= 3 for p in clean[-2:])):
            last = clean[-1]
            clean[-1] = clean[-2]
            clean[-2] = last

        # 3) Parcalari baglac ile bulanikla; olasilikla gecis sozcugu uret
        parts = []
        for i, p in enumerate(clean):
            parts.append(p)
            if i < len(clean) - 1:
                if self.rng.random() < 0.35:
                    parts.append(self._sample_connector())
                elif self.rng.random() < 0.15:
                    last = parts[-2].split()[-1] if parts[-2] else None
                    if last:
                        nxt = self._sample_after(
                            ascii_normalize(last.lower()).strip('.,;:'), local, lgram)
                        if nxt and len(nxt) >= 4:
                            parts.append(nxt)

        joined = ' '.join(parts)
        joined = joined.strip(' ,;')

        # 4) Anchor kelimelerinin uretimde oldugunu dogrula; yoksa orijinallere
        #    geri don (guvenlik agi: faktler hiç kaybolmasin)
        missing = [a for a in anchors
                   if ascii_normalize(a.lower()).strip('.,;:') not in
                   ascii_normalize(joined).lower()]
        if missing and len(clean) >= 1 and all(m not in joined for m in missing):
            return ' '.join(clean) + ('.' if not clean[-1].endswith('.') else '')

        # 5) Zayif sinononi ile degistirme (opsiyonel, faktsiz sozcuklerde)
        if self.rng.random() < 0.25:
            words = joined.split()
            idx_pool = [i for i, w in enumerate(words)
                        if len(w) >= 4 and w.lower() not in STOPWORDS
                        and ascii_normalize(w.lower()).strip('.,;:') not in anchor_norms]
            if idx_pool:
                i = self.rng.choice(idx_pool)
                replacement = self._synonym_neighbor(
                    ascii_normalize(words[i].lower()).strip('.,;:'), local, lgram)
                if replacement:
                    words[i] = ascii_normalize(replacement).capitalize() \
                        if words[i][0].isupper() else replacement
                    joined = ' '.join(words)
        return joined
"""
Nextgen AI - BPE Subword Tokenizer (sifirdan, bagimliliksiz)
=============================================================
Turkce subword tokenizer. Hazir kutupahne YOK; saf Python (stdlib yeterli).
Dogal Turkce imla korunur: o/u/s/g/c normal ASCII'ye cevilmez, 'abcçdefgğhı'
gibi gercek Turkce harfler vocab'ta yer alir.

Sekans konvansiyonu (llm.py ile hizali):
    PAD=0, BOS=1, SEP=2, EOS=3  (<pad> <bos> <sep> <eos>)

Egitim (minbpe tarzi, kelime-bazli sureklilik):
    - Metinler en cok tekrar eden kelime istatistikleriyle agirliklanir.
    - En sik gecen bitisik sembol cifti tek tokende birlestirilir
      (adam asmaci BPE), vocab_size'a ulasilana ya da esik dusene dek.

Kullanim:
    from bpe import BPETokenizer, train_bpe, clean_text
    tok = train_bpe(texts, vocab_size=8000)
    ids = tok.encode('nasilsin, iyiyim sen nasilsin?')
    text = tok.decode(ids)
    tok.save('tokenizer/bpe.json')
    tok2 = BPETokenizer.load('tokenizer/bpe.json')

Dosya formati (tokenizer/bpe.json):
    {"specials": [...], "chars": [...], "merges": [[a,b], ...], "freq": [..]? yok}
    chars   : temel karakterler (sorted, ' ' dahil), id = 4 + sira
    merges  : sirali ciftler; N'inci birlesme id = len(specials)+len(chars)+N
    decode  : birlesme sirasini canlandirarak token->parca dortgenini kurar
"""

import json
import os
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

SPECIALS = ['<pad>', '<bos>', '<sep>', '<eos>']
PAD, BOS, SEP, EOS = 0, 1, 2, 3

# Turkce kucuk harf kumesi (dogal imla). Buyuk harfler asagiya cekilir.
TURKISH_LETTERS = 'abcçdefgğhıijklmnoöprsştuüvyz'
ASCII_DIGITS = '0123456789'
PUNCT = ".,;:!?…()'\"%*-/&+=#@"
# ' ' tekrar edenleri teklesir; kasitli bosluk korunur.
ALLOWED_CHARS = set(TURKISH_LETTERS + ASCII_DIGITS + PUNCT + ' ')

# Esik: bu esigin altinda kalan ciftler birlesmez (gurultu tokenlari engeller).
MIN_MERGE_FREQ = 2
# Birlesme sayisi tavanini kontrol eder: vocab_size - temel karakter sayisi
# asilirsa geriye kalan tek cift onemsenir mi? (docusalar verimi artirir)
_MAX_ITERATIONS = 1_000_000


# ---------------------------------------------------------------------------
# Normalizasyon
# ---------------------------------------------------------------------------

# Turkce buyuk-kucuk harf duyarliligi:
#   'İ' -> 'i'   (noktali I, Python 'i\u0307' karisikligindan kurtarir)
#   'I' -> 'ı'   (noktasiz I, Turkce imla)
_TURKISH_CASE = str.maketrans({'İ': 'i', 'I': 'ı'})


def turkish_lower(text):
    """Turkce dogru kucuk harf donusumu (İ/I/i/ı ayrımı korunur)."""
    if not text:
        return ''
    return text.translate(_TURKISH_CASE).lower()


def clean_text(text, keep_spaces=True, max_len=0):
    """Tekrar eden bosluklari tekleyerek, izinli karakterleri koruyarak temizler.

    Hedef: tokenizer egitimi ve encode icin tek tutarli dogal-metin uzayi.
    İzinsiz karakterler (emoji, Arapca, civi yazisi vb.) atilir; kalan
    Turkce/ASCII imla oldugu gibi kalir. Satir sonlari bosluga indirgenir.
    """
    if not text:
        return ''
    t = turkish_lower(text.replace('\u00a0', ' ').replace('\u200b', ''))
    out = []
    for ch in t:
        if ch == ' ':
            # Tekrar eden bosluk teklenir, yalnizca amacli ayrintilar korunur
            if out and out[-1] != ' ':
                out.append(' ')
        elif ch in ALLOWED_CHARS:
            out.append(ch)
        elif ch.isspace():
            if out and out[-1] != ' ':
                out.append(' ')
        else:
            continue
    if not keep_spaces:
        out = [c for c in out if c != ' ']
    s = ''.join(out).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s


# ---------------------------------------------------------------------------
# BPE egitimi
# ---------------------------------------------------------------------------

def _word_stats(texts, min_word_freq=2, max_texts=None):
    """Metinleri kelime frekans haritasina indirger (deterministik islem hizi).

    Kelime-bazli BPE: her benzersiz kelimenin kac kez gectigi sayilir; egitme
    dongusu kelime sayisi kadar degil benzersiz kelime sayisi kadar skan eder.
    min_word_freq: tek kelime tekrar mini esik; altinda kalanlar atilir.
    """
    stats = Counter()
    for i, text in enumerate(texts):
        if max_texts and i >= max_texts:
            break
        for w in clean_text(text).split():
            if w:
                stats[w] += 1
    return {w: c for w, c in stats.items() if c >= min_word_freq}


def train_bpe(texts, vocab_size=8000, min_freq=MIN_MERGE_FREQ,
              min_word_freq=2, max_texts=None, allow_specials=True,
              progress_cb=None):
    """Turkce subword vocab'tan BPE tokenizer egitir (verimli, artikimli).

    Algoritma (Karpathy minbpe oncesi, klasik "fast BPE"):
      - Kelimeler benzersizlestirilir ve frekanslariyla agirliklanir.
      - Her bitisik cift icin kelime indeks seti (pair_words) tutulur.
      - En sik cift lazy max-heap ile secilir; her birlesmede yalnizca bu
        cifti ICEREN kelimeler yeniden taranip yerel cift sayimlari guncellenir.
      - Calisma: O(merge x ilgili kelime), butun korpusu her iterasyonda
        taramaksizin.

    Tie-break: En yuksek sayiya sahip birden fazla cift varsa, en kucuk
    (a,b) sembol cifti secilir (engele koyma sirasindan bagimsiz,
    deterministik ve eszamanli egitim icin kararli).

    Args:
        texts: Metin listesi/iterable (corpus + intents ham metinleri).
        vocab_size: Hedef token sayisi (specials + temel karakterler + birlesmeler).
        min_freq: Birlesmeye girecek ciftin minimum gecis sayisi.
        min_word_freq: Kelime istatistigine girecek min tekrar (dedup).
        max_texts: Egitime alinacak metin tavan sayisi (hiz/memory).
        allow_specials: <pad> <bos> <sep> <eos> ilk 4 token olsun mu?
        progress_cb: callable(merge_sayaci, max_merges) - uzun egitimlerde
            ilerleme bildirimi (istege bagli).

    Returns:
        BPETokenizer
    """
    stats = _word_stats(texts, min_word_freq=min_word_freq, max_texts=max_texts)
    if not stats:
        raise ValueError('Egitim metni bos; tokenizer kurulamadi.')

    # Temel karakter kumesi: TUM izinli karakterler (gorulenler degil).
    # Boylece clean_text'in uretebilecegi her karakterin token'i vardir
    # ve encode/decode round-trip'i garantidir.
    chars = sorted(ALLOWED_CHARS)
    c2i = dict(zip(chars, range(len(SPECIALS), len(SPECIALS) + len(chars))))

    import heapq

    # Kelimeler: sembol id dizileri + agirlik; her kelime bir indeks
    words = []          # diziler
    word_cnt = []       # agirliklar (indeks hizali)
    for w, cnt in stats.items():
        words.append([c2i[ch] for ch in w])
        word_cnt.append(cnt)

    pair_words = {}          # (a,b) -> set of word_indices (artikimli)
    pair_counts = Counter()  # (a,b) -> toplam (artikimli)
    heap = []                # lazy max-heap: (-count, pair)

    def _dec(pair, cnt):
        c = pair_counts.get(pair)
        if c is None:
            return
        nc = c - cnt
        if nc <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = nc
        heapq.heappush(heap, (-nc, pair))

    def _inc(pair, cnt, wi):
        nc = pair_counts.get(pair, 0) + cnt
        pair_counts[pair] = nc
        heapq.heappush(heap, (-nc, pair))
        pair_words.setdefault(pair, set()).add(wi)

    for wi, seq in enumerate(words):
        for j in range(len(seq) - 1):
            _inc((seq[j], seq[j + 1]), word_cnt[wi], wi)

    max_merges = max(0, vocab_size - len(SPECIALS) - len(chars))
    merges = []
    next_id = len(SPECIALS) + len(chars)

    for _ in range(max_merges):
        if not heap:
            break
        if progress_cb is not None:
            progress_cb(len(merges), max_merges)
        best_pair = None
        best_freq = 0
        while heap:
            neg, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair)
            if cur is None or cur != -neg:
                continue  # eski/cesit kaydi
            best_pair, best_freq = pair, cur
            heapq.heappush(heap, (neg, pair))  # yeniden koy, kullanmak icin
            break
        if best_pair is None:
            break
        if best_freq < min_freq:
            break

        pair_counts.pop(best_pair, None)
        a, b = best_pair
        z = next_id
        next_id += 1
        merges.append((a, b))

        affected = pair_words.pop(best_pair, None)
        if affected:
            for wi in affected:
                seq_old = words[wi]
                wi_cnt = word_cnt[wi]
                # Eski ciftleri dus
                for j in range(len(seq_old) - 1):
                    _dec((seq_old[j], seq_old[j + 1]), wi_cnt)
                # (a,b) -> z birles
                nseq = []
                i = 0
                n = len(seq_old)
                while i < n:
                    if i + 1 < n and seq_old[i] == a and seq_old[i + 1] == b:
                        nseq.append(z)
                        i += 2
                    else:
                        nseq.append(seq_old[i])
                        i += 1
                words[wi] = nseq
                # Yeni ciftleri artir
                for j in range(len(nseq) - 1):
                    _inc((nseq[j], nseq[j + 1]), wi_cnt, wi)

        if not pair_counts and not heap:
            break
        # -- loop

    return BPETokenizer(chars=chars, merges=merges,
                        specials=list(SPECIALS) if allow_specials else None)


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """Sabit parcalarla (pieces) token<->parca eslemesi kuran tokenizer."""

    def __init__(self, chars=None, merges=None, specials=None):
        self.specials = list(specials) if specials is not None else list(SPECIALS)
        self.chars = list(chars or [])
        self.merges = [tuple(m) for m in (merges or [])]
        self._build()

    # ----------------------------------------------------------- YAPILANDIR
    def _build(self):
        """Parca dortgenini (id->metin) ve birlesme tablolarini kurar."""
        pieces = list(self.specials) + list(self.chars)
        # Kodlamada kullanilacak birlesme tablolari
        rank = {}
        merged_set = set()
        next_id = len(pieces)
        for a, b in self.merges:
            merged_set.add((a, b))
            rank[(a, b)] = len(rank)
            # Yeni token parcasini olustur (soldaki+sagidaki parcalarin birlestirmesi)
            pa = pieces[a] if a < len(pieces) else None
            pb = pieces[b] if b < len(pieces) else None
            if pa is None or pb is None:
                # Eski vector sistemi icin guvenli doldurma (olmamali)
                pa = pa or ''
                pb = pb or ''
            pieces.append(pa + pb)
            next_id += 1

        self.pieces = pieces
        self.merged_set = merged_set
        self.rank = rank
        self.c2i = {ch: i for i, ch in enumerate(self.specials)}
        for j, ch in enumerate(self.chars):
            self.c2i[ch] = len(self.specials) + j
        self.i2c = {i: ch for ch, i in self.c2i.items()}

    def __len__(self):
        return len(self.pieces)

    @property
    def vocab_size(self):
        return len(self.pieces)

    def vocab(self):
        """Token id -> parca sözlüğü (kayit/tanimlama icin)."""
        return {i: p for i, p in enumerate(self.pieces)}

    # ----------------------------------------------------------- ENCODE
    def encode(self, text, add_specials=None):
        """Metni token id dizisine cevirir.

        add_specials: eklenecek on-arka ozel token listesi (llm icin).
        ('add' oncesi temizlik yapilir; arama modeli kodu kendi sekans
        mantigini kullanir -- bu parametre yalnizca klavuzdur.)
        """
        clean = clean_text(text)
        if not clean:
            return [self.c2i['<pad>']]
        seq = [self.c2i.get(ch, self.c2i.get(' ', PAD)) for ch in clean]
        # Ac gozlü birlesme: en dusuk sirali mevcut cifti birles, durum dongusu
        while True:
            best = None
            best_rank = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                r = self.rank.get(pair)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best = i
                if best_rank == 0:
                    break
            if best is None:
                break
            a, b = seq[best], seq[best + 1]
            merged = self.rank[(a, b)] + len(self.specials) + len(self.chars)
            seq[best:best + 2] = [merged]
        return seq

    def encode_batch(self, texts, add_specials=None):
        return [self.encode(t, add_specials=add_specials) for t in texts]

    # ----------------------------------------------------------- DECODE
    def decode(self, ids, skip_specials=True):
        """Token id listesini metne cevirir."""
        out = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.pieces):
                continue
            p = self.pieces[i]
            if skip_specials and i < len(self.specials):
                p = ''
            out.append(p)
        s = ''.join(out).strip()
        # Durak/noktalama arasi sadelesen bosluklar
        s = re.sub(r'\s+([.,;:!?…)])', r'\1', s)
        s = re.sub(r'([.,;:!?…])(?=[a-zçğıöşü0-9])', r'\1 ', s)
        return s

    # ----------------------------------------------------------- KAYIT
    def save(self, path):
        data = {
            'specials': self.specials,
            'chars': self.chars,
            'merges': [list(m) for m in self.merges],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[bpe] Tokenizer kaydedildi: {path} (vocab={len(self)})")

    @staticmethod
    def load(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return BPETokenizer(chars=data.get('chars', []),
                            merges=[tuple(m) for m in data.get('merges', [])],
                            specials=data.get('specials', SPECIALS))

    # ----------------------------------------------------------- ARAYUZ
    def tokenize(self, text, add_specials=None):
        """Token id + metin parca listesi (debug/egitim icin)."""
        ids = self.encode(text, add_specials=add_specials)
        return [(i, self.pieces[i]) for i in ids]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import io
    import sys
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    corpus = [
        'merhaba nasılsın iyiyim sen nasılsın',
        'bugün hava çok güzel yürüyecek misin',
        'nerelisin istanbuldan geliyorum ve orada yaşıyorum',
        'yapay zeka geleceğin teknolojisidir öğrenmek önemli',
        'galaksiler milyonlarca yıldızdan oluşur evren devasız',
        'öğle yemeğinde ne yedin çok açım şu anda',
        'bilgisayar programlama dili öğrenmek zordur ama değerlidir',
        'kitap okumak insanın ufkunu açar geçmişi anlamak için',
        'deprem anında yapman gereken en önemli şey sakin kalmaktır',
        'okyanuslar dünyanın büyük bölümünü kaplar ve iklimi etkiler',
    ]
    tok = train_bpe(corpus, vocab_size=200, min_freq=1, min_word_freq=1)
    print(f'Vocab size: {len(tok)}')
    s = 'merhaba nasılsın, öğle yemeğinde ne yedin'
    ids = tok.encode(s)
    print('ids :', ids)
    print('toks:', [p for _, p in tok.tokenize(s)])
    print('text:', tok.decode(ids))
    assert tok.decode(ids) == clean_text(s), 'round-trip bozuldu'

    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), 'ng_bpe_test.json')
    tok.save(tmp)
    tok2 = BPETokenizer.load(tmp)
    assert tok2.encode(s) == ids
    assert tok2.decode(ids) == tok.decode(ids)
    print('\nround-trip OK:', tok2.decode(ids))
"""
Nextgen AI - Corpus (RAG-lite)

Autogrow'un topladigi ham bilgiyi 'corpus.jsonl' dosyasinda tutar ve dogal dil
sorularinda IDF+cosine ile YEREL baglam aramasi yapar. ChromaDB/FAISS gerekmez;
saf numpy + Python dict'lerle cok hizli calisir.

Dosya formati (satir basina bir JSON nesnesi):
    {"id": "<sluggified title>", "title": "...", "text": "...", "added_at": "..."}

Model egitiminin DEGISMEZ: corpus yalnizca yanit aninda baglam cekmek icin
kullanilir (gercek zamanli RAG). Internete gitmeden once buradan bakilir.
"""

import os
import json
import math
import pickle
import re
import datetime
import hashlib

import numpy as np

from brain import ChatBot, STOPWORDS

CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus.jsonl')
CORPUS_MIN_SCORE = 0.25
TITLE_BOOST = 2.0
SNIPPET_MAX_LEN = 600
IDX_CACHE_V = 2  # agirlikli index onbelleginin format surumu

class CsrIndex:
    """Term -> (doc ndarray, val ndarray) ters indeksi (kompakt CSR depo).

    Bircok dokumanin her terimi icin ayri dikt/ndarray tutmak yerine sirali
    terim listesi + terim baslangic ofsetleri + duz (doc, val) dizileri
    kullanir: bellek ve (onbellegi acarkenki) pickle suresi acisindan cok
    daha hafiftir. API'si eski 'sozluk' ile aynidir: get(w) -> (idx, vals)
    veya None; 'in' ile varlik sorgulanabilir.
    """
    __slots__ = ('terms', 'start', 'doc', 'val')

    def __init__(self, terms=(), start=None, doc=None, val=None):
        self.terms = terms
        self.start = start if start is not None else np.array([0], np.int64)
        self.doc = doc if doc is not None else np.empty(0, np.int64)
        self.val = val if val is not None else np.empty(0, np.float32)

    def _pos(self, w):
        terms = self.terms
        lo, hi = 0, len(terms)
        while lo < hi:
            mid = (lo + hi) >> 1
            if terms[mid] < w:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def get(self, w):
        lo = self._pos(w)
        if lo < len(self.terms) and self.terms[lo] == w:
            s = int(self.start[lo])
            e = int(self.start[lo + 1])
            return self.doc[s:e], self.val[s:e]
        return None

    def __contains__(self, w):
        lo = self._pos(w)
        return lo < len(self.terms) and self.terms[lo] == w

    def __len__(self):
        return len(self.terms)

# EMBEDDING VEKTOR DEPOSU (numpy-only LSA/SVD)
# Onbellek yollari oz-ornek bazinda (self._emb_file/_emb_meta) tutulur;
# varsayilan konum 'corpus_embedding.npz' / 'corpus_embedding_meta.json'dir.
EMB_K = 128          # gizli boyut (latent) sayisi
EMB_MIN_TERM = 3     # term minimum uzunlugu (Turkce kokler kisa: ked, mar, su)
EMB_MIN_DF = 5       # en az 5 dokumanda gecen termler (nadir term gurultusunu keser)

# BM25 (lexical) parametreleri
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_BM25_WEIGHT = 2.0    # baslik kelimelerinin BM25 frekans agirligi

# PPMI+SVD kelime vektor boyutu (terim-co-occurrence uzayinda)
PPMI_K = 100
PPMI_MIN_DF = 3
PPMI_MAX_TERMS = 4000   # co-occurrence matrisini (bellek/sure) kontrol eder

# Char-trigram re-rank (yazim hatasi / donusal eslesme bonusu)
TRI_BONUS = 0.35
TRI_MIN_LEN = 6    # sorgu uzunlugu en az bu kadarsa trigram yeniden siralamasi aktif
TRI_CONTAIN = 0.45 # bu icerme oraninin uzerindeki dokumanlara ciddi bonus
TRIGRAM_CAP_K = 500  # trigram re-rank yalnizca siradaki ilk K adayda (hiz)

EMB_CACHE_V = 2    # onbellegin surum anahtari (yeni vektor turleri eklenince arttir)


def _chunk_content(c):
    """Bir parcanin aranabilir icerik imzası (id/title/text/patterns)."""
    return (c.get('id'), c.get('title'), c.get('text'), c.get('patterns'))


def _content_same(a, b):
    return _chunk_content(a) == _chunk_content(b)


def _corpus_ids_path(path):
    """Korpus id kumesinin yan indeks yolu ('corpus.jsonl' -> 'corpus_ids.jsonl')."""
    return os.path.splitext(path)[0] + '_ids.jsonl'


def _id_encode(cid):
    """Id'yi 'unicode_escape' ile satir-guvenli ASCII'ye kodlar (newline guvenli)."""
    return cid.encode('unicode_escape').decode('ascii')


def _id_decode(s):
    return s.encode('ascii').decode('unicode_escape')


def _read_id_set(path):
    """Korpus id kumesini yan indeksten okur.

    Index yalnizca korpus dosyasindan DAHA GENC ise gecerlidir (dosya
    append_many disinda degismisse bayat sayilir). Index yok/bayat ise
    None doner; cagiran taraf tam taramaya dusmelidir.
    """
    ids_path = _corpus_ids_path(path)
    try:
        if not os.path.exists(path) or not os.path.exists(ids_path):
            return None
        if os.path.getmtime(path) > os.path.getmtime(ids_path):
            return None
        ids = set()
        with open(ids_path, 'r', encoding='ascii') as f:
            for line in f:
                s = line.rstrip('\n')
                if s:
                    ids.add(_id_decode(s))
        if not ids:
            # Bos index guvenilmez: korpus bos degilse (yeni seed vb.) tam taramaya dus.
            with open(path, 'r', encoding='utf-8') as f:
                if bool(f.readline().strip()):
                    return None
        return ids
    except OSError:
        return None


def _scan_ids(path):
    """Korpus dosyasini tek gecisle tarayip id kumesini cikarir (yedek yol)."""
    ids = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cid = json.loads(line).get('id')
                except json.JSONDecodeError:
                    continue
                if cid:
                    ids.add(cid)
    except OSError:
        pass
    return ids


def _write_id_set(path, ids):
    """Id indeksini tam yeniden yazar (guncelleme/yedek tarama sonrasi)."""
    with open(_corpus_ids_path(path), 'w', encoding='ascii') as f:
        for cid in sorted(ids):
            f.write(_id_encode(cid) + '\n')


def _append_id_set(path, ids):
    """Yeni id'leri indekse ekler (saf append hizli yolunda)."""
    with open(_corpus_ids_path(path), 'a', encoding='ascii') as f:
        for cid in sorted(ids):
            f.write(_id_encode(cid) + '\n')


class Corpus:
    """JSONL tabanli, bellek ici IDF+cosine vektör deposu (RAG-lite)."""

    def __init__(self, path=CORPUS_FILE):
        self.path = path
        self.tokenizer = ChatBot()
        self.chunks = []
        self.idf = {}
        self._df = {}
        self.vectors = []
        self.loaded = False
        self.min_score = CORPUS_MIN_SCORE
        # Embedding onbellek yollari örnek bazinda (testler gercek cache'i ezmesin):
        # 'corpus.jsonl' -> 'corpus_embedding.npz' / 'corpus_embedding_meta.json'
        stem = os.path.splitext(path)[0]
        self._emb_file = stem + '_embedding.npz'
        self._emb_meta = stem + '_embedding_meta.json'
        self._idx_cache = stem + '_index.pkl'  # agirlikli index onbellegi
        # Embedding depolari (None = kullanilmiyor, eski IDF yoluna dusulur)
        self._doc_counts = []
        self._lex_cols = CsrIndex()      # term -> (doc_idx[], val[]) CSR ters indeks
        self._slug_idx = {}      # slug -> dokuman indeksi (hizli baslik eslesme)
        self._slug_tri_idx = {}  # trigram -> dokuman indeks kumesi
        self.emb = None          # {'docvecs': (d x k) float32, 'V': (k x v), ...}
        self.emb_terms = None    # {term: terim_indeksi}
        self.emb_idf = None      # {term: idf}
        # BM25 lexical altyapisi
        self._doc_tf = []        # her dokumanin term->ham frekans sozlugu
        self._doc_len = []       # her dokumanin frekans toplami (baslik 2x)
        self._doc_len_arr = None # numpy float32 (dokuman basina uzunluk)
        self._lex_tf = CsrIndex()        # term -> (doc_idx[], tf[]) CSR ters indeks
        self._bm25_idf = {}      # term -> BM25 idf
        self._avgdl = 1.0
        # PPMI+SVD kelime vektor deposu
        self.ppmi = None         # {'docvecs', 'W', 'terms', 'idf'}
        # Char-trigram indexi (yazim hatasi / takilma eslemesi)
        self._doc_tri = []       # her dokumanin ascii-normalize trigram kumesi
        # Patterns icin hizli tam-kelime esleme indeksi (search() icin)
        self._pat_idx = {}       # kelime -> chunk indeks kumesi
        self._pat_sets = []      # her chunk icin pattern token kumesi

    def _slug(self, title):
        return self.tokenizer.ascii_normalize(title.strip().lower()).replace(' ', '_')

    def _row_hash(self, c):
        """Bir parcanin icerik kimligi (id+baslik+metin+kalip)."""
        h = hashlib.md5()
        for part in (str(c.get('id', '')), str(c.get('title', '')),
                     str(c.get('text', '')), str(c.get('patterns', ''))):
            h.update(part.encode('utf-8', 'ignore'))
            h.update(b'\x1f')
        return h.hexdigest()

    def _row_hash_all(self):
        """Tum parcalarin icerik kimlikleri (chunks sirasiyla)."""
        return [self._row_hash(c) for c in self.chunks]

    def _tokens(self, text):
        return [t for t in self.tokenizer.tokenize(text) if t not in STOPWORDS]

    def _trigrams(self, text):
        """Ascii-normalize metnin bosluksuz 3-gram kumesi (yazim hatasi toleransi)."""
        s = self.tokenizer.ascii_normalize(text.lower())
        s = re.sub(r'[^a-z0-9]', '', s)
        if len(s) < 3:
            return frozenset()
        return frozenset(s[i:i + 3] for i in range(len(s) - 2))

    @staticmethod
    def _build_csr(counts_list):
        """Dokuman basina {term: agirlik} listesini CsrIndex'e cevirir.

        Terimler alfabetik siralanir; her terimin dokuman ve agirlik degeri
        duz dizilerde ard arda tutulur (per-term ndarray objesi yok).
        """
        pairs = []
        for i, cnts in enumerate(counts_list):
            if cnts:
                for w, v in cnts.items():
                    pairs.append((w, i, v))
        if not pairs:
            return CsrIndex()
        pairs.sort(key=lambda t: t[0])
        terms = []
        starts = []
        docs = np.empty(len(pairs), np.int64)
        vals = np.empty(len(pairs), np.float32)
        k = 0
        last = None
        for w, i, v in pairs:
            if w != last:
                terms.append(w)
                starts.append(k)
                last = w
            docs[k] = i
            vals[k] = v
            k += 1
        starts.append(len(pairs))
        return CsrIndex(terms, np.array(starts, np.int64), docs, vals)

    def _idx_digest(self):
        """Korpus dosyasinin hizli icerik imzasi (mtime + boyut + blake2b-8)."""
        if not os.path.exists(self.path):
            return None
        st = os.stat(self.path)
        h = hashlib.blake2b(digest_size=8)
        with open(self.path, 'rb') as f:
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                h.update(buf)
        return {'mtime': st.st_mtime_ns, 'size': st.st_size,
                'b2': h.hexdigest()}

    def _load_index_cache(self):
        """Onbellegi yukler; dosya yok/guncel degilse False (tam yuklemeye dusulur).

        State, load() ile uretilebilecek tum agirlikli index yapisini tasir;
        boylece korpus icerigi degismedigi surece tokenizasyon + trigram +
        ters index derlemesi (dakikalarca suren kisim) atlanir.
        """
        if not os.path.exists(self._idx_cache):
            return False
        sig = self._idx_digest()
        if sig is None:
            return False
        try:
            with open(self._idx_cache, 'rb') as f:
                state = pickle.load(f)
        except Exception:
            return False
        if state.get('v') != IDX_CACHE_V or state.get('sig') != sig:
            return False
        st = state.get('st')
        if not isinstance(st, dict):
            return False
        for k, v in st.items():
            setattr(self, k, v)
        self.loaded = True
        # _doc_tf, _doc_counts'un ayni listesidir (cache'te tek kopya).
        self._doc_tf = self._doc_counts
        return True

    def _save_index_cache(self):
        """Mevcut index yapisini onbellege yazar (sonraki yukleme hizli)."""
        sig = self._idx_digest()
        if sig is None:
            return
        st = {
            'chunks': self.chunks,
            'idf': self.idf,
            '_df': self._df,
            'vectors': self.vectors,
            '_doc_counts': self._doc_counts,
            '_lex_cols': self._lex_cols,
            '_lex_tf': self._lex_tf,
            '_doc_len': self._doc_len,
            '_doc_len_arr': self._doc_len_arr,
            '_bm25_idf': self._bm25_idf,
            '_avgdl': self._avgdl,
            '_slugs': self._slugs,
            '_slug_idx': self._slug_idx,
            '_slug_tri_idx': self._slug_tri_idx,
            '_doc_tri': self._doc_tri,
            '_pat_idx': self._pat_idx,
            '_pat_sets': self._pat_sets,
        }
        try:
            with open(self._idx_cache, 'wb') as f:
                pickle.dump({'v': IDX_CACHE_V, 'sig': sig, 'st': st},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
        except (OSError, TypeError, pickle.PickleError):
            pass

    def load(self):
        """corpus.jsonl dosyasini yukler ve vektörleri insa eder.

        Agirlikli index (df/idf/BM25/lex/vektor/trigram/kalip) ilk kurulumda
        '_index.pkl' onbellegine yazilir; corpus icerigi degismezse sonraki
        yuklemelerde yeniden derlenmez (dakikalar -> saniyeler).
        """
        if self._load_index_cache():
            self._ensure_embeddings()
            print(f"[CORPUS] {len(self.chunks)} bilgi parcasi (onbelleg) yuklendi "
                  f"({len(self.idf)} kelime).")
            return self.chunks
        self.chunks = []
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        df = {}
        title_tokens = []
        text_tokens = []
        for c in self.chunks:
            tt = self._tokens(c.get('title', ''))
            xt = self._tokens(c.get('text', '') + ' ' + c.get('patterns', ''))
            title_tokens.append(tt)
            text_tokens.append(xt)
            for w in set(tt + xt):
                df[w] = df.get(w, 0) + 1

        self._df = df
        n = len(self.chunks)
        self.idf = {w: math.log(n / (1.0 + cnt)) for w, cnt in df.items()}
        self._bm25_idf = {w: math.log(1.0 + (n - cnt + 0.5) / (cnt + 0.5))
                          for w, cnt in df.items()}

        # BM25 frekans sayaclari (baslik kelimeleri 2x agirlikli).
        # _doc_tf ve _doc_counts ayni sozluktur (TITLE_BM25_WEIGHT ==
        # TITLE_BOOST == 2.0); iki kopya bellek tutmasin diye tek liste
        # kullanilir.
        self._doc_tf = []
        self._doc_len = []
        for i in range(n):
            counts = {}
            for w in title_tokens[i]:
                counts[w] = counts.get(w, 0) + TITLE_BM25_WEIGHT
            for w in text_tokens[i]:
                counts[w] = counts.get(w, 0) + 1.0
            doc_i = sum(counts.values()) or 1.0
            self._doc_tf.append(counts)
            self._doc_len.append(doc_i)
        self._doc_counts = self._doc_tf
        self._doc_len_arr = np.array(self._doc_len, np.float32)
        self._avgdl = max(1.0, float(np.mean(self._doc_len_arr))) if n else 1.0

        self._slugs = []
        self._slug_idx = {}     # slug -> dokuman indeksi (tam eslesme bonusu)
        self._slug_tri_idx = {} # trigram -> dokuman indeks kumesi (kismi eslesme)
        self.vectors = []
        for i, c in enumerate(self.chunks):
            slug = self._slug(c.get('title', ''))
            self._slugs.append(slug)
            if slug:
                self._slug_idx.setdefault(slug, i)
                for tr in self._trigrams(slug):
                    self._slug_tri_idx.setdefault(tr, set()).add(i)
            counts = self._doc_tf[i]

            vec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {w: v / norm for w, v in vec.items()}
            self.vectors.append(vec if norm > 0 else {})

        # Inverted indexler (karma -> duz CSR): lexic/BM25 taramalarinda
        # term bazli O(kelime) arama icin.
        self._lex_tf = Corpus._build_csr(self._doc_tf)
        self._lex_cols = Corpus._build_csr(self.vectors)

        # Char-trigram kumeleri (normalize metin uzerinden, bosluksuz)
        self._doc_tri = []
        for c in self.chunks:
            raw = (c.get('title', '') + ' ' + c.get('text', '')
                   + ' ' + c.get('patterns', ''))
            self._doc_tri.append(self._trigrams(raw))

        # Patterns tam-kelime indeksi: search() icin O(kelime) hizli yol
        self._pat_idx = {}
        self._pat_sets = []
        for i, c in enumerate(self.chunks):
            pat = c.get('patterns', '')
            if pat:
                ptoks = set(self._tokens(pat))
            else:
                ptoks = set()
            self._pat_sets.append(ptoks)
            for w in ptoks:
                self._pat_idx.setdefault(w, set()).add(i)

        self.loaded = True
        self._ensure_embeddings()
        self._save_index_cache()
        print(f"[CORPUS] {len(self.chunks)} bilgi parcasi yuklendi ({len(self.idf)} kelime).")
        return self.chunks

    # ------------------------------------------------------------------
    # EMBEDDING VEKTOR DEPOSU (numpy-only, LSA/SVD)
    # doc_emb = Q @ U_B   (normalize)   |   query_emb = qtfidf @ V_B.T  (normalize)
    # cosine benzerligi = docvecs @ query_emb
    # ------------------------------------------------------------------

    def _ensure_embeddings(self):
        """Onbellegi kullan veya (gerekirse) embedding deposunu yeniden uret.

        Onbellek dogru surumde ise dogrudan okunur. Geçersizse (corpus
        degismis) tam SVD'nin dakikalarca surmesi yerine ONAYLI temel
        uzayina () yeni/dogrulanmis dokumalar fold-in edilir (append durumu).
        Coklu, cikarilan ya da sifirdan kurulan corpus'ta tam yeniden kuruluma
        dusulur.
        """
        try:
            import numpy as np
        except ImportError:
            return
        if len(self.chunks) < 2:
            self.emb = None
            return

        mtime = os.path.getmtime(self.path) if os.path.exists(self.path) else -1
        cached = False
        if os.path.exists(self._emb_file) and os.path.exists(self._emb_meta):
            try:
                with open(self._emb_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if (meta.get('mtime') == mtime and meta.get('v') == EMB_CACHE_V
                        and 0 < meta.get('k', 0) <= EMB_K):
                    cached = True
            except (json.JSONDecodeError, OSError):
                cached = False
        if cached:
            try:
                data = np.load(self._emb_file)
                self.emb = {'docvecs': data['docvecs'], 'V': data['V']}
                with open(self._emb_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                self.emb_terms = meta.get('terms', {})
                self.emb_idf = meta.get('idf', {})
                self.ppmi = {
                    'docvecs': data['ppmi_docvecs'],
                    'W': data['ppmi_W'],
                    'terms': meta.get('ppmi_terms', {}),
                    'idf': meta.get('ppmi_idf', {}),
                    'k': int(data['ppmi_W'].shape[1]),
                }
                return
            except Exception:
                self.emb = None
                self.ppmi = None
        # Geçersiz onbellek: mevcut temel uzayi varsa fold-in dene
        if os.path.exists(self._emb_file) and os.path.exists(self._emb_meta):
            try:
                with np.load(self._emb_file) as data:
                    with open(self._emb_meta, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if self._incremental_embeddings(data, meta, mtime):
                        return
            except Exception:
                pass
            self.emb = None
            self.ppmi = None
        self._build_embeddings(mtime)

    def _incremental_embeddings(self, data, meta, mtime):
        """Mevcut LSA/PPMI temel uzayina DEGISEN/YENI dokumalari izdusurur.

        Onbellekteki temel (V / ppmi-W) aynen korunur; yalnizca icerigi
        degisen ya da yeni eklenen satirlar fold-in ile yeniden izdusurulur,
        diger vektorler onbellekten aynen kalir (SVD/PPMI yeniden
        kurulmaz). Satirlarin yaridan fazlasi degistiyse False -> arayan
        tam kuruluma duser (temel uzay cogunlukla eski/ilgisiz kalmis).
        """
        import numpy as np
        n = len(self.chunks)
        old_n = len(meta.get('slugs', []))
        if old_n <= 0 or old_n != data['docvecs'].shape[0] or n < old_n:
            return False

        old_ids = meta['slugs']
        cur_ids = [c.get('id', '') for c in self.chunks]
        old_hash = meta.get('hashes')

        # Degisen satir kumesi.
        # Sik durum (append): yeni parcalar sona eklendiyse on-ek eslesmesi,
        # tum dokuman hashlerini yeniden hesap ETMEDEN hizli yol (O(n) kimlik
        # karsilastirmasi, hash yok). Sadece normal gorunumdeki icerik
        # degisimlerinde (n==old_n) pahali ama nadir tum-hash taramasi yapilir.
        if n > old_n and cur_ids[:old_n] == old_ids:
            if not old_hash:
                return False
            changed = list(range(old_n, n))
            cur_hash = list(old_hash) + [
                self._row_hash(self.chunks[i]) for i in range(old_n, n)]
        else:
            cur_hash = self._row_hash_all() if old_hash else []
            changed = []
            for i in range(min(old_n, n)):
                if (old_ids[i] != cur_ids[i]
                        or (old_hash and old_hash[i] != cur_hash[i])):
                    changed.append(i)
            if n > old_n:
                changed.extend(range(old_n, n))
        if not changed:
            return False
        if len(changed) > max(4, int(0.5 * n)):
            return False

        V = np.asarray(data['V'], np.float32)          # (k, vocab_n)
        t2i = meta.get('terms', {})
        vidf = meta.get('idf', {})
        W = np.asarray(data['ppmi_W'], np.float32)     # (v, kp)
        pt2i = meta.get('ppmi_terms', {})
        pidf = meta.get('ppmi_idf', {})

        old_doc = np.asarray(data['docvecs'], np.float32)      # (old_n, k)
        old_ppmi = np.asarray(data['ppmi_docvecs'], np.float32)  # (old_n, kp)
        k, kp = V.shape[0], W.shape[1]
        docvecs = np.concatenate(
            [old_doc, np.zeros((max(0, n - old_n), k), np.float32)], axis=0)
        ppmi_docvecs = np.concatenate(
            [old_ppmi, np.zeros((max(0, n - old_n), kp), np.float32)], axis=0)

        for i in changed:
            acc = np.zeros(k, np.float32)
            for w, cnt in self._doc_counts[i].items():
                idx = t2i.get(w)
                if idx is None:
                    continue
                val = float((1.0 + math.log(max(1.0, cnt))) * vidf.get(w, 0.0))
                if val:
                    acc += val * V[:, idx]
            nrm = float(np.sqrt(np.dot(acc, acc)))
            docvecs[i] = acc / nrm if nrm > 1e-9 else acc

            pacc = np.zeros(kp, np.float32)
            for w, cnt in self._doc_counts[i].items():
                jj = pt2i.get(w)
                if jj is None:
                    continue
                pacc += W[jj] * float(0.5 + cnt) * pidf.get(w, 0.0)
            pnrm = float(np.sqrt(np.dot(pacc, pacc)))
            ppmi_docvecs[i] = pacc / pnrm if pnrm > 1e-9 else pacc

        self.emb = {'docvecs': docvecs, 'V': V}
        self.emb_terms = t2i
        self.emb_idf = vidf
        self.ppmi = {'docvecs': ppmi_docvecs, 'W': W,
                     'terms': pt2i, 'idf': pidf, 'k': kp}
        try:
            np.savez(self._emb_file, docvecs=docvecs, V=V,
                     ppmi_docvecs=ppmi_docvecs, ppmi_W=W)
            meta2 = dict(meta)
            meta2.update({'mtime': mtime, 'k': k, 'v': EMB_CACHE_V,
                          'slugs': cur_ids, 'hashes': cur_hash})
            self._save_emb_meta(meta2)
        except OSError:
            pass
        print(f"[EMB] {len(changed)} parca fold-in ({n} toplam, {k} boyut)")
        return True

    def _save_emb_meta(self, meta):
        with open(self._emb_meta, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)

    def _build_embeddings(self, mtime):
        """Randomized SVD (Halko et al.) ile TF-IDF matrisini gizli boyuta dusur."""
        import numpy as np
        n = len(self.chunks)

        # Term sozlugu: df >= EMB_MIN_DF ve uzunluk >= EMB_MIN_TERM
        df = {}
        for counts in self._doc_counts:
            for w in counts:
                df[w] = df.get(w, 0) + 1
        terms = sorted({
            w for w, c in df.items()
            if c >= EMB_MIN_DF and len(w) >= EMB_MIN_TERM
        })
        if not terms:
            self.emb = None
            return
        t2i = {w: j for j, w in enumerate(terms)}
        vidf = {w: math.log(n / (1.0 + cnt)) for w, cnt in df.items()}
        vocab_n = len(terms)
        k = min(EMB_K, n, vocab_n)

        # Omurga (per-doc term-count listesi -> float32 matris arcigeleri)
        doc_rows = []
        for counts in self._doc_counts:
            cols = []
            vals = []
            for w, cnt in counts.items():
                j = t2i.get(w)
                if j is None:
                    continue
                cols.append(j)
                vals.append(float((1.0 + math.log(max(1.0, cnt))) * vidf[w]))
            if cols:
                doc_rows.append((np.array(cols, np.int64), np.array(vals, np.float32)))
            else:
                doc_rows.append((np.array([], np.int64), np.array([], np.float32)))

        rng = np.random.default_rng(42)
        Omega = rng.standard_normal((vocab_n, k)).astype(np.float32)

        # Y = M @ Omega  (dokuman satirlarinda topla)
        Y = np.zeros((n, k), np.float32)
        for i, (cols, vals) in enumerate(doc_rows):
            Y[i] = np.sum(vals[:, None] * Omega[cols], axis=0)

        # Q = QR(Y)  (modifye Gram-Schmidt)
        Q = np.zeros_like(Y)
        for j in range(k):
            q = Y[:, j].copy()
            for l in range(j):
                q -= np.dot(Q[:, l], Y[:, j]) * Q[:, l]
            norm = np.sqrt(np.dot(q, q))
            if norm > 1e-9:
                Q[:, j] = q / norm

        # B = Q.T @ M
        B = np.zeros((k, vocab_n), np.float32)
        for i, (cols, vals) in enumerate(doc_rows):
            B[:, cols] += Q[i, :, None] * vals[None, :]

        UB, S, VB = np.linalg.svd(B, full_matrices=False)

        # Doc embeddings = Q @ UB ; satir normalize (cosine icin)
        docvecs = (Q @ UB.astype(np.float32)).astype(np.float32)
        norms = np.sqrt(np.sum(docvecs * docvecs, axis=1, keepdims=True))
        norms[norms < 1e-9] = 1.0
        docvecs = docvecs / norms

        self.emb = {'docvecs': docvecs, 'V': VB.astype(np.float32)}
        self.emb_terms = t2i
        self.emb_idf = vidf

        self._build_ppmi(terms, df, n)

        try:
            np.savez_compressed(self._emb_file, docvecs=docvecs, V=VB.astype(np.float32),
                                ppmi_docvecs=self.ppmi['docvecs'],
                                ppmi_W=self.ppmi['W'])
            with open(self._emb_meta, 'w', encoding='utf-8') as f:
                json.dump({'mtime': mtime, 'k': k, 'v': EMB_CACHE_V,
                           'terms': t2i, 'idf': vidf,
                           'ppmi_terms': self.ppmi['terms'],
                           'ppmi_idf': self.ppmi['idf'],
                           'slugs': [c.get('id', '') for c in self.chunks],
                           'hashes': self._row_hash_all()}, f)
        except OSError:
            pass
        print(f"[EMB] {n} parca -> {k} boyutlu vektor deposu insa edildi ({vocab_n} term).")

    def _build_ppmi(self, terms, df, n):
        """PPMI+SVD kelime vektorleri (co-occurrence uzerinden, word2vec benzeri).

        Term co-occurrence matrisi : C[i,j] = dokuman sayisi (i ve j birlikte),
        pozitif PMI ile agirliklanir, SVD ile k -boyutlu kelime vektorlerine
        indirgenir. Dokuman vektoru = idf-agirlikli kelime vektor toplami.
        Boylece ayni anlami tasiyan farkli kelimelerle yapilan sorular
        (ornek: "iklim isinması" ~ "kuresel isinma") semantik olarak tutar.

        Bellek disiplini: co-occurrence matrisi float32 ve en sik gecen
        PPMI_MAX_TERMS termiyle sinirli tutulur (SVD bellek/sure acisindan
        makul kalsin diye).
        """
        import numpy as np
        if n < 2:
            self.ppmi = None
            return

        cand = [w for w, c in df.items() if c >= PPMI_MIN_DF and len(w) >= EMB_MIN_TERM]
        cand.sort(key=lambda w: -df[w])
        cand = cand[:PPMI_MAX_TERMS]
        pt2i = {w: i for i, w in enumerate(cand)}
        v = len(pt2i)
        if v == 0:
            self.ppmi = None
            return
        kidx = min(PPMI_K, v, n)

        # Co-occurrence sayaci (float32, sadece farkli term kombinasyonlari)
        cnt = np.zeros(v, np.float32)
        C = np.zeros((v, v), np.float32)
        for counts in self._doc_counts:
            inds = np.array(sorted({
                pt2i[w] for w in counts if w in pt2i
            }), np.int64)
            if inds.size == 0:
                continue
            cnt[inds] += 1.0
            C[np.ix_(inds, inds)] += 1.0
        # Diyagonal (oz-co-occurrence) PMI hesabini sasirtiyor -> sifirla
        np.fill_diagonal(C, 0.0)

        # PPMI: max(0, log( C[i,j] * n / (cnt[i]*cnt[j]) ))   (float32)
        outer = np.outer(cnt, cnt)
        div = np.zeros_like(outer)
        safe = outer > 0
        with np.errstate(divide='ignore'):
            div[safe] = n / outer[safe]
        P = C * div
        P[safe] = np.maximum(P[safe], 0.0)
        np.log(P, out=P, where=P > 0)
        P[P < 0] = 0.0

        # SVD -> kelime vektorleri (W^{T} W ~ PPMI matrisi)
        # Bellek duzeyinde tutarli: full_matrices=False ve float32 bu boyutta
        # okulur; gorsel olarak kucuk k boyutlu W zaten dokumanda kullanilir.
        U, S, _ = np.linalg.svd(P, full_matrices=False)
        W = (U[:, :kidx] * S[:kidx]).astype(np.float32)

        # Dokuman vektorleri: term vektorlerinin idf agirlikli toplami
        pn = float(n)
        vidf = {w: math.log(pn / (1.0 + df[w])) for w in cand}
        docvecs = np.zeros((n, kidx), np.float32)
        for i, counts in enumerate(self._doc_counts):
            acc = np.zeros(kidx, np.float32)
            for w, raw in counts.items():
                j = pt2i.get(w)
                if j is None:
                    continue
                acc += W[j] * (0.5 + raw) * vidf.get(w, 0.0)
            docvecs[i] = acc
        norms = np.sqrt(np.sum(docvecs * docvecs, axis=1, keepdims=True))
        norms[norms < 1e-9] = 1.0
        docvecs = docvecs / norms

        self.ppmi = {'docvecs': docvecs, 'W': W,
                     'terms': pt2i, 'idf': vidf, 'k': kidx}
        print(f"[PPMI] {v} term -> {kidx} boyutlu kelime vektoru ({n} parca).")

    def _query_embedding(self, qtoks):
        """Soru kelimelerini ayni kuzey uzayinda bir vektore izdusurur."""
        import numpy as np
        if not self.emb or not qtoks:
            return None
        q = np.zeros(len(self.emb_terms), np.float32)
        qcounts = {}
        for w in qtoks:
            qcounts[w] = qcounts.get(w, 0) + 1.0
        for w, cnt in qcounts.items():
            j = self.emb_terms.get(w)
            if j is None:
                continue
            q[j] = float((1.0 + math.log(cnt)) * self.emb_idf.get(w, 0.0))
        qemb = q @ self.emb['V'].T
        norm = np.sqrt(np.dot(qemb, qemb))
        if norm <= 1e-9:
            return None
        return qemb / norm

    def _search_embedding(self, qtoks, k=8):
        """Embedding deposu ile en iyi k aday parcanin indekslerini ve skorlarini dondurur."""
        import numpy as np
        qemb = self._query_embedding(qtoks)
        if qemb is None or self.emb is None:
            return None
        dots = self.emb['docvecs'] @ qemb
        n = min(k, len(dots))
        if dots.size == 0:
            return None
        idx = np.argpartition(dots, -n)[-n:]
        idx = idx[np.argsort(dots[idx])[::-1]]
        return [(int(i), float(dots[i])) for i in idx if dots[i] > 0.0]

    def _bm25_scores(self, qcounts):
        """Vektorize BM25 skorlari (tum dokumanlar). [0,1]'e normalize."""
        import numpy as np
        n = len(self.chunks)
        lex = np.zeros(n, np.float32)
        for w, qcnt in qcounts.items():
            row = self._lex_tf.get(w)
            if row is None:
                continue
            idx, tf = row
            idf = self._bm25_idf.get(w, 0.0)
            denom = tf + BM25_K1 * (1.0 - BM25_B
                                    + BM25_B * self._doc_len_arr[idx] / self._avgdl)
            scores = idf * (tf * (BM25_K1 + 1.0)) / denom
            lex[idx] += qcnt * scores
        nz = float(lex.max()) if lex.size else 0.0
        if nz > 0:
            lex /= nz
        return lex

    def _ppmi_expand(self, qtoks, top=2, min_sim=0.55):
        """PPMI kelime vektorleri ile sorgu genisletme.

        Sorgu teriminin vektor uzayinda en yakin 1-2 PPMI komsusunu bulur ve
        benzerlik gucunde geri verir. Boylece ayni konudaki farkli kelimeler
        ("iklim isinmasi" sorulunca "kuresel isinma") arama terimi olur.
        Dusuk benzerlikli komsular (cok anlamli kelimelerin gurultusu) min_sim
        esigini gecmez. Unut cagirmak icin degil, HASIL terim dagarciginda
        sinyali genisletmek icin kullanilir.
        """
        if not self.ppmi:
            return {}
        W = self.ppmi['W']                       # (v,k)
        terms = self.ppmi['terms']
        inv = self.ppmi.get('_inv')
        if inv is None:
            inv = {j: w for w, j in terms.items()}
            self.ppmi['_inv'] = inv
        norms = np.sqrt(np.sum(W * W, axis=1)) + 1e-9
        Wn = W / norms[:, None]
        out = {}
        for w in qtoks:
            j = terms.get(w)
            if j is None:
                continue
            sims = Wn @ Wn[j]
            sims[j] = -1.0
            topn = min(top, len(sims))
            inds = np.argpartition(sims, -topn)[-topn:]
            inds = inds[np.argsort(sims[inds])[::-1]]
            for t in inds.tolist():
                s = float(sims[t])
                if s < min_sim:
                    continue
                tw = inv.get(t)
                if tw and tw != w:
                    out[tw] = max(out.get(tw, 0.0), s)
        return out

    def _trigram_containment(self, qtri):
        """Sorgu 3-gramlarinin dokumanlarca karsilanma orani (0..1)."""
        if not qtri:
            return None
        out = np.zeros(len(self.chunks), np.float32)
        ql = len(qtri)
        for i, dt in enumerate(self._doc_tri):
            if dt:
                out[i] = len(dt & qtri) / ql
        return out

    def search(self, query, k=2, min_score=CORPUS_MIN_SCORE):
        """Soru icin en alakali bilgi parcasini dondurur yoksa None.

        Returns:
            dict veya None: {'title', 'text', 'score'}
        """
        if not self.loaded or not self.chunks:
            return None

        qtoks = self._tokens(query)
        if not qtoks:
            return None

        # 1) KALIP KISA YOLU: ogrenilmis parcalarda soru kalibi kayitlidir.
        #    Sorunun anlamli kelimelerinin TAMAMI o kalip icinde geciyorsa
        #    uzun metin kosinusunun sinyali seyreltip kaybetmesi onlenir
        #    ("akropol nerede" -> 400 karaklik Akropolis metnine ragmen tutar).
        distinct = len(qtoks) >= 2 or max(len(w) for w in qtoks) >= 6
        if distinct and self._pat_idx:
            # Inverted index: her qtok icin aday kumeleri kes ve kesis
            candidates = None
            for w in qtoks:
                idx_set = self._pat_idx.get(w)
                if idx_set is None:
                    candidates = set()
                    break
                if candidates is None:
                    candidates = set(idx_set)
                else:
                    candidates &= idx_set
            for ci in candidates:
                if set(qtoks) <= self._pat_sets[ci]:
                    return {'title': self.chunks[ci].get('title', ''),
                            'text': self.chunks[ci].get('text', ''),
                            'score': 0.5}
        elif distinct:
            for c in self.chunks:
                pat = c.get('patterns', '')
                if not pat:
                    continue
                pat_norm = self.tokenizer.ascii_normalize(pat.lower())
                if all(w in pat_norm for w in qtoks):
                    return {'title': c.get('title', ''),
                            'text': c.get('text', ''),
                            'score': 0.5}

        qnorm = self.tokenizer.ascii_normalize(query.lower())
        qcounts = {}
        for w in qtoks:
            qcounts[w] = qcounts.get(w, 0) + 1.0

        # 2) PPMI SORGU GENISLETME: kelime vektoru komsulari da lex'te aransin
        #    ("iklim isinmasi" -> "kuresel isinma" da eslenir). Cok anlamli
        #    kelimelerde yanlis genisletmeyi min_sim esigi keser.
        for tw, sim in self._ppmi_expand(qtoks).items():
            qcounts[tw] = qcounts.get(tw, 0.0) + sim

        # 3) LEXIC SKOR (idf-agirlikli cosine, per-doc normalize):
        #    Mutlak idf agirligi nadir/kesin terimi ('klorofil') onde tutar.
        qvec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in qcounts.items()}
        qnorm_v = math.sqrt(sum(v * v for v in qvec.values()))
        if qnorm_v <= 0:
            return None
        qvec = {w: v / qnorm_v for w, v in qvec.items()}

        lex = np.zeros(len(self.chunks), np.float32)
        for w, val in qvec.items():
            col = self._lex_cols.get(w)
            if col:
                lex[col[0]] += val * col[1]

        # Baslik bonusu: tam slug eslesmesi indeksten, kismi/yazim-varianti
        # eslesme (triagram ortusen adaylarda ucuz substring+regex).
        for w in qtoks:
            si = self._slug_idx.get(w)
            if si is not None:
                lex[si] += 0.30
        qtri = self._trigrams(query)
        if qtri:
            cand = set()
            for tr in qtri:
                cand.update(self._slug_tri_idx.get(tr, ()))
            for i in cand:
                tslug = self._slugs[i]
                if not tslug or tslug in qtoks:
                    continue
                if tslug in qnorm and re.search(
                        r'(?<![a-z0-9])' + re.escape(tslug) + r'(?![a-z0-9])', qnorm):
                    lex[i] += 0.30

        # 4) SEMANTIK SKOR (LSA dokuman vektoru)
        dots = None
        if self.emb:
            qemb = self._query_embedding(qtoks)
            if qemb is not None:
                dots = (self.emb['docvecs'] @ qemb).astype(np.float32)

        # 5) BIRLESTIRME: kanitlanmis formul = semantic + sinirli lexic
        if dots is not None and len(dots) == len(lex):
            final = dots + 1.5 * np.minimum(lex, 0.6)
        else:
            final = lex

        # 6) CHAR-TRIGRAM YENIDEN SIRALAMA: yazim hatasi / donusal eslesme.
        #    Yalnizca taban skoru > 0 olan adaylari gueclendirir; sogum taban
        #    skorsuz (ilgisiz) dokumanlarin trigrama dayanip on plana cikmasi
        #    engellenir ("akropol nerede" -> alakasiz makale hatali donmesin).
        #    Hiz: 43K dokumanin tamaminda set islemi yerine yalnizca ontaki
        #    adaylarda hesaplanir (siralamanin degismeyecegi esik gorev gorur).
        if len(qnorm) >= TRI_MIN_LEN and qtri:
            tri = np.zeros(len(final), np.float32)
            active = np.flatnonzero(final > 0)
            if len(active) > TRIGRAM_CAP_K:
                active = active[np.argsort(final[active])[-TRIGRAM_CAP_K:]]
            if len(active):
                tri[active] = np.array(
                    [len(self._doc_tri[i] & qtri) for i in active], np.float32) / len(qtri)
            final = final + tri * TRI_BONUS

        # 7) BM25 NADIR-TERIM KURTARMA: semantik/lex sicakligi dusukse ve BM25
        #    tek bir dokumanda net once cikiyorsa onu onde tut (kesin terimli
        #    ve embedding vocab'ina girmemis sorular icin emniyet kemeri).
        if float(np.max(final)) < 0.8:
            bm = self._bm25_scores({w: 1.0 for w in qtoks})
            if bm is not None and bm.size and float(bm.max()) >= 0.6:
                final = final + 0.6 * bm

        i = int(np.argmax(final))
        score = float(final[i])
        if score <= 0.0 or score < min_score:
            return None
        entry = self.chunks[i]
        return {'title': entry.get('title', ''),
                'text': entry.get('text', ''),
                'score': score}

    @staticmethod
    def snippet(text, max_len=SNIPPET_MAX_LEN):
        """Bilgi parcasini ilk cumle sinirinda kisaltir."""
        if len(text) <= max_len:
            return text
        cut = text[:max_len]
        for sep in ('. ', '! ', '? ', '\n'):
            idx = cut.rfind(sep)
            if idx > max_len // 2:
                return cut[:idx + 1]
        return cut.rsplit(' ', 1)[0] + '...'

    @staticmethod
    def seed_from_intents():
        """corpus.jsonl yoksa mevcut intents.json'dan baslangic corpus'u uretir.

        Autogrow daha zengin tam metinler ekleyene kadar 507 konu icin
        aninda yerel baglam saglar. Ayni id'li yeni kayit eskisini gunceller.
        """
        if os.path.exists(CORPUS_FILE):
            return False
        intents_file = os.path.join(os.path.dirname(__file__), 'intents.json')
        if not os.path.exists(intents_file):
            return False

        with open(intents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        now = datetime.datetime.utcnow().isoformat()
        lines = []
        for it in data['intents']:
            tag = it.get('tag', '')
            if not tag:
                continue
            text = ' '.join(it.get('responses', []))
            if len(text) < 40:
                continue
            # Kaliplar konu kelimelerini tasidigindan tohum metnine eklenir.
            # Boylece "galaksi hakkinda bilgi ver" gibi sorular dogru parcaya dokunur.
            patterns = ' '.join(it.get('patterns', []))
            slug = tag.strip().lower().replace(' ', '_')
            lines.append(json.dumps({
                'id': slug, 'title': tag, 'text': text,
                'patterns': patterns, 'source': 'intents-seed', 'added_at': now
            }, ensure_ascii=False))

        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"[CORPUS] intents.json'dan {len(lines)} baslangic parcasi uretildi.")
        return True

    @staticmethod
    def append_many(chunks, path=CORPUS_FILE):
        """Bilgi parcalarini corpus.jsonl'a yazar (ayni id guncellenir).

        Yeni id'ler dosya SONUNA eklenir (61 MB'lık dosya yeniden yazilmaz);
        mevcut id'ler guncelleniyorsa yalnizca o satirlar degistirilir, onunun
        byte'lari oldugu gibi korunur. Id kumesi '<dizin>_ids.jsonl' yan
        indeksinden hizli okunur; index bayatsa dosya tek gecisle taranir.
        """
        updated = {}
        for ch in chunks:
            cid = ch.get('id')
            if not cid:
                continue
            updated[cid] = {
                'id': cid,
                'title': ch.get('title', ''),
                'text': ch.get('text', ''),
                'patterns': ch.get('patterns', ''),
                'source': ch.get('source', 'autogrow'),
                'added_at': datetime.datetime.utcnow().isoformat(),
            }
        if not updated:
            return 0

        exist = set()
        index_ok = False
        if os.path.exists(path):
            index_ids = _read_id_set(path)
            if index_ids is not None:
                exist, index_ok = index_ids, True
            else:
                exist = _scan_ids(path)

        new_ids = [cid for cid in updated if cid not in exist]
        updates = {cid: rec for cid, rec in updated.items() if cid in exist}

        # Dosya sonuna yazilacaksa satir sonuyla bitip bitmedigini kontrol et.
        file_nl = True
        if os.path.exists(path):
            with open(path, 'rb') as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    file_nl = f.read(1) == b'\n'

        if not updates:
            # HIZLI YOL (sik gecen ogrenme akisi): yalnizca sona ekle.
            with open(path, 'a', encoding='utf-8') as f:
                if not file_nl:
                    f.write('\n')
                for cid in new_ids:
                    f.write(json.dumps(updated[cid], ensure_ascii=False) + '\n')
            if index_ok:
                # Index dogruydu: yeni id'leri satir sonuna eklemek yeterli.
                if new_ids:
                    _append_id_set(path, new_ids)
            else:
                # Index bayat/eksikti: tam taramadan gelen kumeyi yeniden yaz.
                _write_id_set(path, exist | set(new_ids))
        else:
            # NADIR YOL (mevcut parca yeniden ogrenildi): degisen satirlari
            # yerinde guncelle, digerlerini oldugu gibi koru.
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(path, 'w', encoding='utf-8') as f:
                for line in lines:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        cid = json.loads(s).get('id')
                    except json.JSONDecodeError:
                        f.write(line)
                        continue
                    if cid in updates:
                        f.write(json.dumps(updates[cid], ensure_ascii=False) + '\n')
                    else:
                        f.write(line)
            if new_ids:
                _write_id_set(path, exist | set(new_ids))
            else:
                _write_id_set(path, exist)

        total = len(exist) + len(new_ids)
        print(f"[CORPUS] corpus.jsonl guncellendi: {total} parca.")
        return total

    def refresh(self):
        """append_many sonrasi bellekteki indexi gunceller.

        Dosyaya yalnizca YENI parcalar eklendiyse (sik gecen ogrenme akisi)
        tum indexler artimli olarak guncellenir (saniyeler); eklenenlerle
        birlikte icerik degisen/ciikarilan parca da varsa guvenli tam
        yeniden yuklemeye dusulur (embedding fold-in ile hizlanir).
        """
        if not self.loaded or not os.path.exists(self.path):
            if os.path.exists(self.path):
                self.load()
            return self.loaded

        new_chunks = []
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        new_chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return self.loaded

        old_n = len(self.chunks)
        new_n = len(new_chunks)
        old_ids = [c.get('id', '') for c in self.chunks]
        new_ids = [c.get('id', '') for c in new_chunks]
        if new_n > old_n and new_ids[:old_n] == old_ids:
            # HIZLI YOL yalnizca onceki satirlarin ICERIGi de degismediyse
            # guvenlidir: append_many ayni id'li parcanin text/patterns'ini
            # guncelleyebilir; o durumda lexical index (df/lex_tf/vectors/
            # pattern) bayat kalir ve yeni icerik aramalarda bulunamaz.
            if all(_content_same(new_chunks[i], self.chunks[i])
                   for i in range(old_n)):
                self._append_chunks(new_chunks[old_n:], new_chunks)
                print(f"[CORPUS] Artimli guncelleme: +{new_n - old_n} parca "
                      f"({new_n} toplam).")
                return True

        # Icerik degisimi / cikarma / siralama -> guvenli tam yukleme.
        self.load()
        return self.loaded

    def _append_chunks(self, added, full_chunks):
        """Yalnizca sondan eklenen parcalar icin indexleri artimli gunceller."""
        import numpy as np
        old_n = len(self.chunks)
        m = len(added)

        # 1) Yeni parcalarin tokenlari + df guncellemesi
        add_toks = []
        for c in added:
            tt = self._tokens(c.get('title', ''))
            xt = self._tokens(c.get('text', '') + ' ' + c.get('patterns', ''))
            add_toks.append((tt, xt))

        df = self._df
        for i in range(m):
            for w in set(add_toks[i][0] + add_toks[i][1]):
                df[w] = df.get(w, 0) + 1
        n = len(full_chunks)
        self.idf = {w: math.log(n / (1.0 + cnt)) for w, cnt in df.items()}
        self._bm25_idf = {w: math.log(1.0 + (n - cnt + 0.5) / (cnt + 0.5))
                          for w, cnt in df.items()}

        # 2) BM25 counter + cosine vektor + slug/trigram/pattern indexleri
        for j in range(m):
            i = old_n + j
            c = full_chunks[i]
            tt, xt = add_toks[j]
            counts = {}
            for w in tt:
                counts[w] = counts.get(w, 0) + TITLE_BM25_WEIGHT
            for w in xt:
                counts[w] = counts.get(w, 0) + 1.0
            doc_i = sum(counts.values()) or 1.0

            self._doc_tf.append(counts)   # _doc_counts ile ayni liste (alias)
            self._doc_len.append(doc_i)

            vec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {w: v / norm for w, v in vec.items()}
            self.vectors.append(vec if norm > 0 else {})

            slug = self._slug(c.get('title', ''))
            self._slugs.append(slug)
            if slug:
                self._slug_idx.setdefault(slug, i)
                for tr in self._trigrams(slug):
                    self._slug_tri_idx.setdefault(tr, set()).add(i)

            raw = (c.get('title', '') + ' ' + c.get('text', '')
                   + ' ' + c.get('patterns', ''))
            self._doc_tri.append(self._trigrams(raw))

            pat = c.get('patterns', '')
            ptoks = set(self._tokens(pat)) if pat else set()
            self._pat_sets.append(ptoks)
            for w in ptoks:
                self._pat_idx.setdefault(w, set()).add(i)

        self.chunks = full_chunks
        self._doc_len_arr = np.array(self._doc_len, np.float32)
        self._avgdl = max(1.0, float(np.mean(self._doc_len_arr))) if n else 1.0
        self._doc_counts = self._doc_tf
        self._lex_tf = Corpus._build_csr(self._doc_tf)
        self._lex_cols = Corpus._build_csr(self.vectors)

        # 3) Embedding deposu: degisen/yeni satirlari fold-in ile guncelle
        self._ensure_embeddings()
        self._save_index_cache()
        return self.loaded
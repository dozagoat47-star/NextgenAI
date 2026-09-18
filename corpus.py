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
import re
import datetime

import numpy as np

from brain import ChatBot, STOPWORDS

CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus.jsonl')
CORPUS_MIN_SCORE = 0.25
TITLE_BOOST = 2.0
SNIPPET_MAX_LEN = 600

# EMBEDDING VEKTOR DEPOSU (numpy-only LSA/SVD)
EMB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus_embedding.npz')
EMB_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus_embedding_meta.json')
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


class Corpus:
    """JSONL tabanli, bellek ici IDF+cosine vektör deposu (RAG-lite)."""

    def __init__(self, path=CORPUS_FILE):
        self.path = path
        self.tokenizer = ChatBot()
        self.chunks = []
        self.idf = {}
        self.vectors = []
        self.loaded = False
        self.min_score = CORPUS_MIN_SCORE
        # Embedding depolari (None = kullanilmiyor, eski IDF yoluna dusulur)
        self._doc_counts = []
        self._lex_cols = {}      # term -> (numpy doc_idx[], numpy val[]) inverted index
        self._slug_idx = {}      # slug -> dokuman indeksi (hizli baslik eslesme)
        self._slug_tri_idx = {}  # trigram -> dokuman indeks kumesi
        self.emb = None          # {'docvecs': (d x k) float32, 'V': (k x v), ...}
        self.emb_terms = None    # {term: terim_indeksi}
        self.emb_idf = None      # {term: idf}
        # BM25 lexical altyapisi
        self._doc_tf = []        # her dokumanin term->ham frekans sozlugu
        self._doc_len = []       # her dokumanin frekans toplami (baslik 2x)
        self._doc_len_arr = None # numpy float32 (dokuman basina uzunluk)
        self._lex_tf = {}        # term -> (doc_idx[], tf[]) inverted index
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

    def _tokens(self, text):
        return [t for t in self.tokenizer.tokenize(text) if t not in STOPWORDS]

    def _trigrams(self, text):
        """Ascii-normalize metnin bosluksuz 3-gram kumesi (yazim hatasi toleransi)."""
        s = self.tokenizer.ascii_normalize(text.lower())
        s = re.sub(r'[^a-z0-9]', '', s)
        if len(s) < 3:
            return frozenset()
        return frozenset(s[i:i + 3] for i in range(len(s) - 2))

    def load(self):
        """corpus.jsonl dosyasini yukler ve vektörleri insa eder."""
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

        n = len(self.chunks)
        self.idf = {w: math.log(n / (1.0 + cnt)) for w, cnt in df.items()}
        self._bm25_idf = {w: math.log(1.0 + (n - cnt + 0.5) / (cnt + 0.5))
                          for w, cnt in df.items()}

        # BM25 frekans sayaclari (baslik kelimeleri 2x agirlikli)
        self._doc_tf = []
        self._doc_len = []
        self._lex_tf = {}
        tf_pairs = {}
        for i in range(n):
            counts = {}
            for w in title_tokens[i]:
                counts[w] = counts.get(w, 0) + TITLE_BM25_WEIGHT
            for w in text_tokens[i]:
                counts[w] = counts.get(w, 0) + 1.0
            doc_i = sum(counts.values()) or 1.0
            self._doc_tf.append(counts)
            self._doc_len.append(doc_i)
            for w, cnt in counts.items():
                tf_pairs.setdefault(w, []).append((i, cnt))
        for w, pairs in tf_pairs.items():
            idx = np.array([p[0] for p in pairs], np.int64)
            tf = np.array([p[1] for p in pairs], np.float32)
            self._lex_tf[w] = (idx, tf)
        self._doc_len_arr = np.array(self._doc_len, np.float32)
        self._avgdl = max(1.0, float(np.mean(self._doc_len_arr))) if n else 1.0

        self._doc_counts = []
        self._slugs = []
        self._slug_idx = {}     # slug -> dokuman indeksi (tam eslesme bonusu)
        self._slug_tri_idx = {} # trigram -> dokuman indeks kumesi (kismi eslesme)
        self.vectors = []
        col_pairs = {}
        for i, c in enumerate(self.chunks):
            slug = self._slug(c.get('title', ''))
            self._slugs.append(slug)
            if slug:
                self._slug_idx.setdefault(slug, i)
                for tr in self._trigrams(slug):
                    self._slug_tri_idx.setdefault(tr, set()).add(i)
            counts = {}
            for w in title_tokens[i]:
                counts[w] = counts.get(w, 0) + TITLE_BOOST
            for w in text_tokens[i]:
                counts[w] = counts.get(w, 0) + 1.0
            self._doc_counts.append(counts)

            vec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {w: v / norm for w, v in vec.items()}
                for w, val in vec.items():
                    col_pairs.setdefault(w, []).append((i, val))
            self.vectors.append(vec if norm > 0 else {})

        # Inverted index: term sensorünü vektorize lexic tarama icin ac
        for w, pairs in col_pairs.items():
            idx = np.array([p[0] for p in pairs], np.int64)
            vals = np.array([p[1] for p in pairs], np.float32)
            self._lex_cols[w] = (idx, vals)

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
        print(f"[CORPUS] {len(self.chunks)} bilgi parcasi yuklendi ({len(self.idf)} kelime).")
        return self.chunks

    # ------------------------------------------------------------------
    # EMBEDDING VEKTOR DEPOSU (numpy-only, LSA/SVD)
    # doc_emb = Q @ U_B   (normalize)   |   query_emb = qtfidf @ V_B.T  (normalize)
    # cosine benzerligi = docvecs @ query_emb
    # ------------------------------------------------------------------

    def _ensure_embeddings(self):
        """Onbellegi kullan veya (gerekirse) embedding deposunu yeniden uret."""
        try:
            import numpy as np
        except ImportError:
            return
        if len(self.chunks) < 2:
            self.emb = None
            return

        mtime = os.path.getmtime(self.path) if os.path.exists(self.path) else -1
        cached = False
        if os.path.exists(EMB_FILE) and os.path.exists(EMB_META):
            try:
                with open(EMB_META, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if (meta.get('mtime') == mtime and meta.get('k') == EMB_K
                        and meta.get('v') == EMB_CACHE_V):
                    cached = True
            except (json.JSONDecodeError, OSError):
                cached = False
        if cached:
            try:
                data = np.load(EMB_FILE)
                self.emb = {'docvecs': data['docvecs'], 'V': data['V']}
                with open(EMB_META, 'r', encoding='utf-8') as f:
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
        self._build_embeddings(mtime)

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
            np.savez_compressed(EMB_FILE, docvecs=docvecs, V=VB.astype(np.float32),
                                ppmi_docvecs=self.ppmi['docvecs'],
                                ppmi_W=self.ppmi['W'])
            with open(EMB_META, 'w', encoding='utf-8') as f:
                json.dump({'mtime': mtime, 'k': k, 'v': EMB_CACHE_V,
                           'terms': t2i, 'idf': vidf,
                           'ppmi_terms': self.ppmi['terms'],
                           'ppmi_idf': self.ppmi['idf']}, f)
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
    def append_many(chunks):
        """Yeni bilgi parcalarini corpus.jsonl'a ekler (ayni id guncellenir)."""
        records = {}
        if os.path.exists(CORPUS_FILE):
            with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        records[rec.get('id')] = rec
                    except json.JSONDecodeError:
                        continue

        for ch in chunks:
            cid = ch.get('id')
            if not cid:
                continue
            records[cid] = {
                'id': cid,
                'title': ch.get('title', ''),
                'text': ch.get('text', ''),
                'patterns': ch.get('patterns', ''),
                'source': ch.get('source', 'autogrow'),
                'added_at': datetime.datetime.utcnow().isoformat(),
            }

        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            for rec in records.values():
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"[CORPUS] corpus.jsonl guncellendi: {len(records)} parca.")
        return len(records)

    def refresh(self):
        """append_many sonrasi bellekteki indexi gunceller.

        Sunucu calisirken ogrenilen parcalar bir sonraki soruya aninda
        kutuphaneden cevap verebilsin diye dosya yeniden yuklenir.
        """
        if self.loaded and os.path.exists(self.path):
            self.load()
        return self.loaded
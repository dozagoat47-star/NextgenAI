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
        self.emb = None          # {'docvecs': (d x k) float32, 'V': (k x v), ...}
        self.emb_terms = None    # {term: terim_indeksi}
        self.emb_idf = None      # {term: idf}

    def _slug(self, title):
        return self.tokenizer.ascii_normalize(title.strip().lower()).replace(' ', '_')

    def _tokens(self, text):
        return [t for t in self.tokenizer.tokenize(text) if t not in STOPWORDS]

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

        self._doc_counts = []
        self._slugs = []
        self.vectors = []
        col_pairs = {}
        for i, c in enumerate(self.chunks):
            self._slugs.append(self._slug(c.get('title', '')))
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
                if meta.get('mtime') == mtime and meta.get('k') == EMB_K:
                    cached = True
            except (json.JSONDecodeError, OSError):
                cached = False
        if cached:
            try:
                data = np.load(EMB_FILE)
                self.emb = {
                    'docvecs': data['docvecs'],
                    'V': data['V'],
                }
                with open(EMB_META, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                self.emb_terms = meta.get('terms', {})
                self.emb_idf = meta.get('idf', {})
                return
            except Exception:
                self.emb = None
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

        try:
            np.savez_compressed(EMB_FILE, docvecs=docvecs, V=VB.astype(np.float32))
            with open(EMB_META, 'w', encoding='utf-8') as f:
                json.dump({'mtime': mtime, 'k': k, 'terms': t2i, 'idf': vidf}, f)
        except OSError:
            pass
        print(f"[EMB] {n} parca -> {k} boyutlu vektor deposu insa edildi ({vocab_n} term).")

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
        for c in self.chunks:
            pat = c.get('patterns', '')
            if not pat:
                continue
            pat_norm = self.tokenizer.ascii_normalize(pat.lower())
            # Tek kelimelik eslesme yalnizca ayirt edici (en az 6 harf) bir
            # kelimeyle olsun; 'sence' gibi genel kelimeyle kisa-yol
            # tetiklenip ilgisiz cevap cekmesin.
            distinct = len(qtoks) >= 2 or max(len(w) for w in qtoks) >= 6
            if distinct and all(w in pat_norm for w in qtoks):
                return {'title': c.get('title', ''),
                        'text': c.get('text', ''),
                        'score': 0.5}

        qnorm = self.tokenizer.ascii_normalize(query.lower())
        qcounts = {}
        for w in qtoks:
            qcounts[w] = qcounts.get(w, 0) + 1.0
        qvec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in qcounts.items()}
        qnorm_v = math.sqrt(sum(v * v for v in qvec.values()))
        if qnorm_v <= 0:
            return None
        qvec = {w: v / qnorm_v for w, v in qvec.items()}

        # 2) TAM-CORPUS LEXIC SKOR (vektorize): inverted index üzerinden tüm
        #    parcalara mutlak idf-agirlikli benzerlik verilir. Nadir/kesin term
        #    ('klorofil') embedding vocab'ina girmese bile burada kurtarilir.
        lex = np.zeros(len(self.chunks), np.float32)
        for w, val in qvec.items():
            col = self._lex_cols.get(w)
            if col:
                lex[col[0]] += val * col[1]

        # 3) EMBEDDING SKOR: semantik yakinlik (cosine) - mevcut ise
        dots = None
        if self.emb:
            qemb = self._query_embedding(qtoks)
            if qemb is not None:
                dots = (self.emb['docvecs'] @ qemb).astype(np.float32)

        # Baslik bonusu yalnizca gercek kelimeyse (kelime sinirinda eslesme).
        # Hiz: once ucuz set/atlik aramasi, regex yalnizca atlik eslestiginde.
        for i, tslug in enumerate(self._slugs):
            if not tslug:
                continue
            if tslug in qtoks:
                lex[i] += 0.30
            elif tslug in qnorm and re.search(
                    r'(?<![a-z0-9])' + re.escape(tslug) + r'(?![a-z0-9])', qnorm):
                lex[i] += 0.30

        # 4) BIRLESTIRME: semantik + mutlak lexic onayi (nadir term kurtarmasi)
        if dots is not None:
            if len(dots) != len(lex):
                dots = None
        if dots is not None:
            final = dots + 1.5 * np.minimum(lex, 0.6)
        else:
            final = lex

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
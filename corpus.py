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
import datetime

from brain import ChatBot, STOPWORDS

CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus.jsonl')
CORPUS_MIN_SCORE = 0.25
TITLE_BOOST = 2.0
SNIPPET_MAX_LEN = 600


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

        self.vectors = []
        for i, c in enumerate(self.chunks):
            counts = {}
            for w in title_tokens[i]:
                counts[w] = counts.get(w, 0) + TITLE_BOOST
            for w in text_tokens[i]:
                counts[w] = counts.get(w, 0) + 1.0

            vec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                self.vectors.append({w: v / norm for w, v in vec.items()})
            else:
                self.vectors.append({})
        self.loaded = True
        print(f"[CORPUS] {len(self.chunks)} bilgi parcasi yuklendi ({len(self.idf)} kelime).")
        return self.chunks

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

        qnorm = self.tokenizer.ascii_normalize(query.lower())
        qcounts = {}
        for w in qtoks:
            qcounts[w] = qcounts.get(w, 0) + 1.0
        qvec = {w: (0.5 + cnt) * self.idf.get(w, 0.0) for w, cnt in qcounts.items()}
        qnorm_v = math.sqrt(sum(v * v for v in qvec.values()))
        if qnorm_v <= 0:
            return None
        qvec = {w: v / qnorm_v for w, v in qvec.items()}

        best = None
        for i, v in enumerate(self.vectors):
            score = sum(qvec.get(w, 0.0) * dw for w, dw in v.items())
            title = self.chunks[i].get('title', '')
            tslug = self._slug(title)
            if tslug in qnorm:
                score += 0.30
            if score > 0.0:
                entry = self.chunks[i]
                cand = {'title': entry.get('title', ''),
                        'text': entry.get('text', ''),
                        'score': score}
                if best is None or score > best['score']:
                    best = cand

        if best is None or best['score'] < min_score:
            return None
        return best

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
                'source': ch.get('source', 'autogrow'),
                'added_at': datetime.datetime.utcnow().isoformat(),
            }

        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            for rec in records.values():
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"[CORPUS] corpus.jsonl guncellendi: {len(records)} parca.")
        return len(records)
"""
Nextgen AI - SeqGen: Sifirdan LSTM karakter-sirasi koulli uretec
================================================================
Hazir kutupahne/API YOK; yalnizca numpy. Amac: (kullanici sorgusu) ->
(dogal Turkce yanit cumlesi) iliskisini ogrenen, karakter karakter YENI
cumle uretebilen bir RNN.

Yaklasim:
  - Ornek (context, response): pattern kisimlari (sorgu) + cevaplar
    (intents.json + corpus.jsonl 'den).
  - Tek LSTM hucresi: once sorgu karakterleri okunur (kodlayici),
    sonra <BOS> ile cevap karakterleri uretilir (kode). Tum sekans tek
    zincirde BPTT ile egitilir; kayip YALNIZCA cevap pozisyonlarinda.
  - Cikis: softmax karakter dagilimi -> temperature + top-k ornekleme.

Dosyalar (model/ altinda):
  - seq_model.json : sozluk + hiperparametreler + agirliklar
"""

import os
import io
import json
import math
import random
import re
import string
import sys

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'seq_model.json')

PAD, BOS, EOS = 0, 1, 2
ALLOWED_EXTRAS = set(".,;:!?…()%’'\"-–/") | set(string.digits)


def utf8_stdout():
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def ascii_normalize(text):
    """Turkce ozel karakterleri ASCII karsiliklarina cevirir (brain ile ayni tablo)."""
    table = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'â': 'a', 'î': 'i', 'û': 'u', 'i': 'i', 'o': 'o', 'u': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'I': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
        'Â': 'A', 'Î': 'I', 'Û': 'U',
        '\u0307': '',
    }
    return text.translate(str.maketrans(table))


def clean_chars(text, max_len):
    """Kucuk harf, izinli karakterler, kisaltma."""
    t = ascii_normalize((text or '').replace('\u00a0', ' '))
    out = []
    for ch in t.lower():
        if ch.isalpha() or ch == ' ' or ch in ALLOWED_EXTRAS:
            out.append(ch)
    s = ''.join(out).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s


def build_vocab(texts, min_count=2):
    """Veriden karakter sozlugu kurar. 0/1/2 = PAD/BOS/EOS."""
    counts = {}
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1
    vocab = ['<PAD>', '<BOS>', '<EOS>']
    for ch, cnt in sorted(counts.items()):
        if cnt >= min_count:
            vocab.append(ch)
    return vocab


class SeqModel:
    """Tek LSTM hucresi + yumusakmax cikis. BPTT + Adam (numpy)."""

    def __init__(self, vocab, hidden=96, seed=7):
        self.vocab = list(vocab)
        self.c2i = {ch: i for i, ch in enumerate(self.vocab)}
        self.i2c = {i: ch for i, ch in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.H = hidden
        bound = 0.08
        self.Wxh = np.random.uniform(-bound, bound, (self.V, 4 * self.H))
        self.Whh = np.random.uniform(-bound, bound, (self.H, 4 * self.H))
        self.bh = np.zeros((1, 4 * self.H))
        self.Wy = np.random.uniform(-bound, bound, (self.H, self.V))
        self.by = np.zeros((1, self.V))
        self.init_adam()

    # ---------------------------------------------------------- KURULUM
    def init_adam(self, b1=0.9, b2=0.999):
        self.b1, self.b2 = b1, b2
        self.t = 0
        self._m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params().items()}

    def params(self):
        return {'Wxh': self.Wxh, 'Whh': self.Whh, 'bh': self.bh,
                'Wy': self.Wy, 'by': self.by}

    def _onehot(self, idx):
        return np.eye(self.V, dtype=np.float32)[idx]

    @staticmethod
    def _sigmoid(a):
        return 1.0 / (1.0 + np.exp(-a))

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------- ILERI
    def _fwd_step(self, xs, h, c):
        a = xs @ self.Wxh + h @ self.Whh + self.bh
        H = self.H
        i = self._sigmoid(a[:, 0:H])
        f = self._sigmoid(a[:, H:2 * H])
        g = np.tanh(a[:, 2 * H:3 * H])
        o = self._sigmoid(a[:, 3 * H:4 * H])
        cc = f * c + i * g
        hh = o * np.tanh(cc)
        return hh, cc, (h, c, i, f, g, o, cc, hh)

    # ---------------------------------------------------------- GERI
    def _bwd_step(self, dh, dc, cache, xs, dWxh, dWhh, dbh):
        h_prev, c_prev, i, f, g, o, cc, hh = cache
        H = self.H
        dcc = dc + dh * o * (1.0 - np.tanh(cc) ** 2)
        di = dcc * g
        dg = dcc * i
        df = dcc * c_prev
        do = dh * np.tanh(cc)
        di_raw = di * i * (1.0 - i)
        df_raw = df * f * (1.0 - f)
        dg_raw = dg * (1.0 - g * g)
        do_raw = do * o * (1.0 - o)
        dvec = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=1)
        dWxh += xs.T @ dvec
        dWhh += h_prev.T @ dvec
        dbh += dvec.sum(axis=0, keepdims=True)
        dh_prev = dvec @ self.Whh.T
        dc_prev = f * dcc
        return dh_prev, dc_prev

    def forward(self, in_idx, tgt_idx, mask):
        """in_idx/tgt_idx:(B,L) mask:(B,L). loss, acc, caches, X, probs."""
        B, L = in_idx.shape
        X = np.zeros((B, L, self.V))
        for s in range(L):
            X[:, s] = self._onehot(in_idx[:, s])
        h = np.zeros((B, self.H))
        c = np.zeros((B, self.H))
        caches = []
        probs = [None] * L
        tot = cnt = acc = 0
        for s in range(L):
            h, c, cache = self._fwd_step(X[:, s, :], h, c)
            caches.append(cache)
            probs[s] = self._softmax(h @ self.Wy + self.by)
            m = mask[:, s]
            if m.any():
                tgt = tgt_idx[:, s]
                safe = probs[s][np.arange(B), tgt]
                tot += float(np.sum(-np.log(np.clip(safe, 1e-12, 1.0)) * m))
                cnt += int(m.sum())
                acc += int(np.sum((np.argmax(probs[s], axis=1) == tgt) * m))
        return tot / max(1, cnt), acc / max(1, cnt), caches, X, probs

    def backward(self, tgt_idx, mask, caches, X, probs):
        B, L = tgt_idx.shape
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        tgt1 = self._onehot(tgt_idx)
        dh = np.zeros((B, self.H))
        dc = np.zeros((B, self.H))
        for s in range(L - 1, -1, -1):
            m = mask[:, s]
            dout = (probs[s] - tgt1[:, s]) * m[:, None]
            dh_s = dh + dout @ self.Wy.T
            grads['Wy'] += caches[s][7].T @ dout
            grads['by'] += dout.sum(axis=0, keepdims=True)
            dh, dc = self._bwd_step(
                dh_s, dc, caches[s], X[:, s, :],
                grads['Wxh'], grads['Whh'], grads['bh'])
        cnt = max(1, int(mask.sum()))
        for k in grads:
            grads[k] /= cnt
        return grads

    def train_minibatch(self, in_idx, tgt_idx, mask, lr):
        self.t += 1
        loss, acc, caches, X, probs = self.forward(in_idx, tgt_idx, mask)
        grads = self.backward(tgt_idx, mask, caches, X, probs)
        self._apply_grads(grads, lr)
        return loss, acc

    def _apply_grads(self, grads, lr):
        tot = 0.0
        for g in grads.values():
            tot += float(np.sum(g * g))
        norm = math.sqrt(tot)
        scale = 1.0
        if norm > 5.0:
            scale = 5.0 / norm
        for k, g in grads.items():
            if scale != 1.0:
                g = g * scale
            self._m[k] = self.b1 * self._m[k] + (1 - self.b1) * g
            self._v[k] = self.b2 * self._v[k] + (1 - self.b2) * g * g
            mhat = self._m[k] / (1 - self.b1 ** self.t)
            vhat = self._v[k] / (1 - self.b2 ** self.t)
            self.params()[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

    # ---------------------------------------------------------- URETIM
    def sample(self, context, temperature=1.0, top_k=14, max_len=150):
        """Sorgu ver; karakter karakter taze yanit uret."""
        ctx = clean_chars(context, 40)
        chars = [self.c2i[ch] for ch in ctx if ch in self.c2i]
        if ctx and not chars:
            chars = [self.c2i.get(' ', PAD)]
        B = 1
        h = np.zeros((B, self.H))
        c = np.zeros((B, self.H))
        for idx in chars:
            h, c, _ = self._fwd_step(self._onehot([idx]), h, c)

        x = self._onehot([self.c2i['<BOS>']])
        out_chars = []
        banned = {PAD, BOS}
        seen_ngrams = set()
        for _ in range(max_len):
            h, c, _ = self._fwd_step(x, h, c)
            probs = self._softmax(h @ self.Wy + self.by)[0]
            if abs(temperature - 1.0) > 1e-9:
                probs = probs ** (1.0 / temperature)
                probs = probs / probs.sum()
            ids = np.argsort(probs)[::-1]
            pool = [int(i) for i in ids if int(i) not in banned and int(i) != EOS]
            if not pool:
                break
            pool = pool[:top_k]
            p = probs[pool]
            p = p / p.sum()
            idx = int(np.random.choice(pool, p=p))
            out_chars.append(idx)
            x = self._onehot([idx])

            # Tekrar korumasi: 6-gram tekrar ederse uretim donmeye basladi.
            if len(out_chars) >= 6:
                ngram = tuple(out_chars[-6:])
                if ngram in seen_ngrams:
                    # donek cembere girebilecek kuyrukları at; degerlendirmeden kes
                    del out_chars[-6:]
                    break
                seen_ngrams.add(ngram)

        text = self._decode_clean(''.join(self.i2c[i] for i in out_chars))
        return text

    def _decode_clean(self, text):
        text = text.strip()
        if not text:
            return ''
        text = re.sub(r'\s+([,.;:!?…])', r'\1', text)
        text = re.sub(r'([,.;:!?…])(?=[a-zçğıöşü0-9])', r'\1 ', text)
        text = text[:1].upper() + text[1:]
        return text


def load_seq(path=MODEL_PATH):
    """model/seq_model.json'dan SeqModel yukler (yoksa None)."""
    if not os.path.exists(path):
        return None
    with io.open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data:
        return None
    m = SeqModel(data['vocab'], hidden=data.get('H', 96))
    for k in ['Wxh', 'Whh', 'bh', 'Wy', 'by']:
        if k in data:
            setattr(m, k, np.array(data[k]))
    return m


def save_seq(model, path=MODEL_PATH):
    d = {'vocab': model.vocab, 'H': model.H,
         'Wxh': model.Wxh.tolist(), 'Whh': model.Whh.tolist(),
         'bh': model.bh.tolist(), 'Wy': model.Wy.tolist(),
         'by': model.by.tolist()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"[seqgen] Model kaydedildi: {path}")


# ---------------------------------------------------------------- EGITIM
def load_pairs(intents_path, max_pairs=5000, max_per_intent=26):
    """(context, response) ciftleri: context=INTENT TAG'i (dusuk varyans).

    Tag yaniti belirgin sekilde ayirt eder; sorgu karakterleri eklemek
    varyansi artirip ogrenmeyi yavaslatiyordu. Cagri aninda klasifikatorun
    verdigi tag gercek kullanimda da biliniyor.
    """
    pairs = []
    if intents_path and os.path.exists(intents_path):
        data = json.load(io.open(intents_path, 'r', encoding='utf-8'))
        for it in data.get('intents', []):
            tag = clean_chars(it.get('tag', ''), 32)
            resps = [clean_chars(r, 70) for r in it.get('responses', [])]
            resps = [r for r in resps if len(r) >= 6]
            if not tag or not resps:
                continue
            cnt = 0
            for r in resps:
                pairs.append((tag, r))
                cnt += 1
                if cnt >= max_per_intent:
                    break
    rng = random.Random(3)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def encode_pair(m, ctx, resp):
    ci = [m.c2i[ch] for ch in ctx if ch in m.c2i]
    ri = [m.c2i[ch] for ch in resp if ch in m.c2i]
    if not ci:
        ci = [m.c2i.get(' ', PAD)]
    if not ri:
        ri = [m.c2i.get('.', PAD)]
    seq = ci + [m.c2i['<BOS>']] + ri
    start = len(ci)  # ilk tahmin pozisyonu (BOS'tan itibaren)
    return seq, start


def make_batches(m, pairs, B):
    items = sorted(pairs, key=lambda pr: len(pr[0]) + len(pr[1]))
    batches = []
    for i in range(0, len(items), B):
        block = items[i:i + B]
        seqs, starts = [], []
        for ctx, resp in block:
            s, n = encode_pair(m, ctx, resp)
            seqs.append(s)
            starts.append(n)
        L = max(len(s) for s in seqs)
        inp = np.zeros((len(block), L), np.int64)
        tgt = np.zeros((len(block), L), np.int64)
        mask = np.zeros((len(block), L), np.float32)
        for j, s in enumerate(seqs):
            inp[j, :len(s)] = s
            tgt[j, :len(s) - 1] = s[1:]
            mask[j, starts[j]:max(starts[j], len(s) - 1)] = 1.0
        batches.append((inp, tgt, mask))
    return batches


def train(path_model=MODEL_PATH, epochs=26, hidden=128, lr=0.004, batch=64,
          data_limit=5000, val_split=0.1):
    utf8_stdout()
    base = os.path.dirname(os.path.abspath(__file__))
    pairs = load_pairs(os.path.join(base, 'intents.json'),
                       max_pairs=data_limit)
    if not pairs:
        print('[seqgen] Egitim verisi yok; cikis yapiliyor.')
        return None

    all_text = []
    for ctx, resp in pairs:
        all_text.append(ctx)
        all_text.append(resp)
    vocab = build_vocab(all_text)
    print(f"[seqgen] Sozluk: {len(vocab)} karakter | Ornek: {len(pairs)} | batch: {batch} | H: {hidden}")
    m = SeqModel(vocab, hidden=hidden)
    n_train = int(len(pairs) * (1 - val_split))
    tr = make_batches(m, pairs[:n_train], batch)
    va = make_batches(m, pairs[n_train:], batch)
    print(f"[seqgen] Train batch: {len(tr)} | Val batch: {len(va)}")

    best_val = 1e9
    best_state = None
    patience = 6
    bad = 0
    for ep in range(1, epochs + 1):
        t0 = __import__('time').time()
        random.Random(ep).shuffle(tr)
        tot = cnt = 0
        for inp, tgt, mask in tr:
            loss, _ = m.train_minibatch(inp, tgt, mask, lr)
            tot += loss
            cnt += 1
        tv = 0.0
        for inp, tgt, mask in va:
            l, _, _, _, _ = m.forward(inp, tgt, mask)
            tv += l
        tv /= max(1, len(va))
        dt = __import__('time').time() - t0
        print(f"epoch {ep}/{epochs} - train: {tot / max(1, cnt):.4f} - val: {tv:.4f} - {dt:.1f}s")
        if tv < best_val - 1e-4:
            best_val = tv
            best_state = {k: v.copy() for k, v in m.params().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[seqgen] Erken durdurma. Best val: {best_val:.4f}")
                break
    if best_state:
        for k, v in best_state.items():
            setattr(m, k, v)
    save_seq(m, path_model)
    return m


if __name__ == '__main__':
    utf8_stdout()
    model = train()
    if model:
        for q in ['greeting', 'hava_durumu', 'programlama_dilleri',
                  'kedi_bakimi', 'tesekkur', 'egitim']:
            print(f"? {q}\nAI: {model.sample(q, temperature=0.8, top_k=12)}")
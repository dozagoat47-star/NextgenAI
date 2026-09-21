"""
Nextgen AI - LoRA fine-tuning
=============================
model.json (taban transformer, Colab'da GPU ile eğitilmiş) HİÇ DEĞİŞMEZ.
Yeni intent/örnekler burada küçük bir LoRA adaptörüne (rank~8) öğretilir:

  model/lora.json  -> taban ağırlıkların ÜZERİNE binen adaptör (A/B, yeni
                      sözcük embed satırları, yeni sınıf kafa satırları)
  bot_data.json    -> genişletilmiş vocabulary / intent_tags / intents / kws
  intents.json     -> yeni niyet kayıtları kalıcı olarak eklenir

Taban model dosyasına dokunulmaz; iki kez /learn çağrılırsa adaptör büyüyerek
birikimli çalışır (deltalar korunur).

CLI:
  python finetune.py --tag "ornek_niyet" --patterns "pat1" --patterns "pat2" \
      --responses "yanit1"
API:
  from finetune import finetune_add
  finetune_add(model_dir, intents_path, [{'tag':..., 'patterns':[...],
                                          'responses':[...]}])
"""

import argparse
import json
import math
import os

import numpy as np

from brain import ChatBot
from transformer import TransformerNN


def atomic_write_json(path, data, indent=None):
    """JSON'u oncelikli yazar: gecici dosyaya yazip os.replace ile ATOMİK
    degistirir. Yarim/bozuk yazim (kill, cokme, disk dolu) hedef dosyayi asla
    tronke etmez; app.py'nin acilista JSONDecodeError ile dusmesi onlenir.
    """
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

DEFAULT_RANK = 8
DEFAULT_ALPHA = 16.0
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 80
DEFAULT_BATCH = 32
REPLAY_CAP = 96               # eski davranışı koruyan "replay" örneği sayısı
GRAD_CLIP = 5.0
WEIGHT_DECAY = 1e-4


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _seed():
    return int.from_bytes(os.urandom(8), 'little') % (2 ** 31 - 1)


# ---------------------------------------------------------------------------
# Adaptör durumu (taban modele göre birikimli deltalar)
# ---------------------------------------------------------------------------

class Adapter:
    def __init__(self, model, rank=DEFAULT_RANK, alpha=DEFAULT_ALPHA):
        self.model = model
        self.rank = int(rank)
        self.alpha = float(alpha)
        r = self.rank
        d = model.d_model
        ff = model.ff_dim
        rng = np.random.RandomState(0)

        def _newA(shape):
            return _f32(rng.randn(*shape) * 0.01)

        def _newB(shape):
            return np.zeros(shape, np.float32)

        self.A = {}
        self.B = {}
        for bi in range(model.num_blocks):
            for name in ('Wq', 'Wk', 'Wv', 'Wo'):
                self.A[f'b{bi}_{name}'] = _newA((r, d))
                self.B[f'b{bi}_{name}'] = _newB((d, r))
            self.A[f'b{bi}_W1'] = _newA((r, ff))
            self.B[f'b{bi}_W1'] = _newB((d, r))
            self.A[f'b{bi}_W2'] = _newA((r, d))
            self.B[f'b{bi}_W2'] = _newB((ff, r))
        self.embed_extra = np.zeros((0, d), np.float32)
        self.whead_extra = np.zeros((0, d), np.float32)
        self.bhead_extra = np.zeros((0, 1), np.float32)
        self.new_vocab = []
        self.new_tags = []

    def to_dict(self):
        return {
            'arch': 'lora',
            'rank': self.rank,
            'alpha': self.alpha,
            'vocab_added': len(self.new_vocab),
            'head_added': len(self.new_tags),
            'new_vocab': list(self.new_vocab),
            'new_tags': list(self.new_tags),
            'embed_extra': self.embed_extra.tolist(),
            'whead_extra': self.whead_extra.tolist(),
            'bhead_extra': self.bhead_extra.tolist(),
            'deltas': {k: {'A': self.A[k].tolist(), 'B': self.B[k].tolist()}
                       for k in self.A},
        }

    @classmethod
    def from_dict(cls, model, ad):
        a = cls(model, rank=ad.get('rank', DEFAULT_RANK),
                alpha=ad.get('alpha', DEFAULT_ALPHA))
        for k in a.A:
            a.A[k][...] = _f32(ad['deltas'][k]['A']).reshape(a.A[k].shape)
            a.B[k][...] = _f32(ad['deltas'][k]['B']).reshape(a.B[k].shape)
        if ad.get('embed_extra'):
            a.embed_extra = _f32(ad['embed_extra']).reshape(-1, a.model.d_model)
        if ad.get('whead_extra'):
            a.whead_extra = _f32(ad['whead_extra']).reshape(-1, a.model.d_model)
            a.bhead_extra = _f32(ad['bhead_extra']).reshape(-1, 1)
        a.new_vocab = list(ad.get('new_vocab', []))
        a.new_tags = list(ad.get('new_tags', []))
        return a

    def extend(self, new_words, new_tags):
        """Yeni sözcük embed satırları (n,d) ve yeni sınıf kafa satırlarını (n,d) ekler."""
        d = self.model.d_model
        rng = np.random.RandomState(1)
        if new_words:
            n = len(new_words)
            extra = _f32(rng.randn(n, d) * 0.02)     # taban embed std ~0.02 ile aynı
            self.embed_extra = np.concatenate([self.embed_extra, extra], axis=0)
            self.new_vocab.extend(new_words)
        if new_tags:
            n = len(new_tags)
            head = _f32(rng.randn(n, d) * 0.05)
            self.whead_extra = np.concatenate([self.whead_extra, head], axis=0)
            self.bhead_extra = np.concatenate(
                [self.bhead_extra, np.zeros((n, 1), np.float32)], axis=0)
            self.new_tags.extend(new_tags)

    def apply(self):
        self.model.apply_lora(self.to_dict())
        return self.model


# ---------------------------------------------------------------------------
# Yükleme / meta veri genişletme
# ---------------------------------------------------------------------------

def load_model_with_lora(model_dir, rank=DEFAULT_RANK, alpha=DEFAULT_ALPHA):
    """Taban model + (varsa) var olan LoRA adaptörünü yükler.

    Yeni adaptör TALEP edilen rank/alpha ile kurulur; var olan adaptör ise
    KENDİ kayıtlı rank/alpha değerleriyle yüklenir (birikimli büyüme: kayıtlı
    deltalar yeni scale ile yeniden yorumlanmaz).
    """
    mdl = TransformerNN(vocab_size=1, num_intents=1, max_seq_len=1)
    mdl.load(os.path.join(model_dir, 'model.json'))
    lora_path = os.path.join(model_dir, 'lora.json')
    if os.path.exists(lora_path):
        with open(lora_path, 'r', encoding='utf-8') as f:
            ad = json.load(f)
        return mdl, Adapter.from_dict(mdl, ad)
    return mdl, Adapter(mdl, rank=rank, alpha=alpha)


def _extend_metadata(bot_data, additions, bot):
    """bot_data üzerinde yeni sözcükler + yeni etiketler + intent/kws güncelleme.

    Sıralama kritiktir: tabanlar KORUNUR, yeniler SONA eklenir (indeksler kaymaz).
    Döndürür: (new_words, new_tags, bot_data)
    """
    vocab = list(bot_data['vocabulary'])
    tags = list(bot_data['intent_tags'])
    kws = {t: set(s) for t, s in bot_data.get('intent_kws', {}).items()}
    intents = dict(bot_data.get('intents', {}))

    new_words = []
    new_tags = []
    for item in additions:
        tag = item['tag']
        if tag not in tags:
            new_tags.append(tag)
            tags.append(tag)
            kws.setdefault(tag, set())
        patterns = item.get('patterns') or []
        new_resps = [r for r in (item.get('responses') or []) if r]
        if tag in tags:
            # Mevcut intent: yanit havuzu KORUNUR. Tekrarlanan ogrenmede (ayni
            # tag / /learn) havuz komple ustune yazilmaz; yeni yanitlar tekil
            # bicimde sona eklenir (veri kaybi + havuz erimesi onlenir).
            existing = intents.get(tag) or []
            merged = list(existing)
            for r in new_resps:
                if r not in merged:
                    merged.append(r)
            intents[tag] = merged or ['Faydalı bilgiler edindim!']
        else:
            intents[tag] = new_resps or ['Faydalı bilgiler edindim!']
        for pat in patterns:
            for w in bot.tokenize(pat):
                if w not in vocab:
                    vocab.append(w)
                    new_words.append(w)
                kws[tag].add(w)

    bot_data['vocabulary'] = vocab
    bot_data['intent_tags'] = tags
    bot_data['intents'] = intents
    bot_data['intent_kws'] = {t: sorted(s) for t, s in kws.items()}
    return new_words, new_tags


def _build_training_set(bot_data, additions, intents_data):
    """Yeni örnekler + küçük eski hafıza (replay) karışımını döndürür (texts, labels)."""
    rng = np.random.RandomState(_seed())
    tags = bot_data['intent_tags']
    texts = []
    labels = []
    for item in additions:
        if item['tag'] not in tags:
            continue
        idx = tags.index(item['tag'])
        for pat in item.get('patterns') or []:
            texts.append(pat)
            labels.append(idx)

    # Replay: taban intent'lerden seyrek örnekler (hafıza koruması)
    base_by_tag = {i['tag']: i.get('patterns') or [] for i in intents_data.get('intents', [])
                   if i['tag'] in tags}
    replay_pool = []
    for tag, pats in base_by_tag.items():
        for pat in pats[:2]:                      # her intent'ten en fazla 2
            if pat not in texts:
                replay_pool.append((pat, tags.index(tag)))
    if len(replay_pool) > REPLAY_CAP:
        replay_pool = list(rng.permutation(replay_pool))[:REPLAY_CAP]
    for text, idx in replay_pool:
        texts.append(text)
        labels.append(idx)
    return texts, labels


# ---------------------------------------------------------------------------
# LoRA eğitimi
# ---------------------------------------------------------------------------

def _adam_params(adapter):
    """Eğitilecek parametreler: A/B matrisleri + yeni embed/kafa satırları."""
    params = {}
    for k in adapter.A:
        params[k + '_A'] = adapter.A[k]
        params[k + '_B'] = adapter.B[k]
    params['embed_extra'] = adapter.embed_extra
    params['whead_extra'] = adapter.whead_extra
    params['bhead_extra'] = adapter.bhead_extra
    return params


def _clip_and_update(params, grads, m, v, step, lr):
    """Adam + gradyan clipping (A/B'de weight-decay, satırlarda yok)."""
    total_norm = 0.0
    for g in grads.values():
        total_norm += float(np.sum(g * g))
    total_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0
    scale = 1.0 if total_norm <= GRAD_CLIP else GRAD_CLIP / total_norm

    b1, b2, eps = float(0.9), float(0.999), float(1e-8)
    bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
    for key, p in params.items():
        g = grads.get(key)
        if g is None:
            continue
        g = g * scale
        m[key] = b1 * m[key] + (1 - b1) * g
        v[key] = b2 * v[key] + (1 - b2) * g * g
        m_hat = m[key] / bc1
        v_hat = v[key] / bc2
        decay = WEIGHT_DECAY * p if (WEIGHT_DECAY > 0
                                     and key.endswith(('_A', '_B'))) else 0.0
        p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + decay)
    return scale


def train_lora(model, adapter, X, y, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
               batch=DEFAULT_BATCH, verbose=True):
    """LoRA adaptörünü eğitir; taban parametrelere dokunmaz."""
    rng = np.random.RandomState(_seed())
    m = X.shape[0]

    params = _adam_params(adapter)
    opt_m = {k: np.zeros_like(p) for k, p in params.items()}
    opt_v = {k: np.zeros_like(p) for k, p in params.items()}
    step = 0
    best_loss = float('inf')
    best_state = None
    wait = 0
    patience = 12

    def snapshot():
        return {k: v.copy() for k, v in params.items()}

    for epoch in range(epochs):
        idx = rng.permutation(m)
        epoch_loss = 0.0
        nb = 0
        for st in range(0, m, batch):
            step += 1
            Xb = X[idx[st:st + batch]]
            yb = y[idx[st:st + batch]]
            probs = model.forward(Xb, apply_dropout=False)
            loss = model.compute_loss(probs, yb)
            epoch_loss += loss
            nb += 1
            G = model.grads(yb, normalize=True)

            grads = {}                                 # key -> gradyan dizisi
            # True-LoRA gradyanı: etkin delta = (alpha/rank)*(B@A) olduğundan
            # A/B gradyanları da aynı scale ile ölçeklenir (transformer.apply_lora
            # ile tutarlı). scale'sız bırakılan eski hali LR'i ~1/scale küçültüyordu.
            scale = adapter.alpha / max(int(adapter.rank), 1)
            for k in adapter.A:
                dW = G[k]
                grads[k + '_A'] = scale * (adapter.B[k].T @ dW)
                grads[k + '_B'] = scale * (dW @ adapter.A[k].T)
            if adapter.embed_extra.shape[0] > 0:
                lhg = model._cache.get('lora_head_grads')
                if lhg is not None:
                    pooled = lhg['pooled']
                    dhe = lhg['d_head_extra']
                    grads['whead_extra'] = dhe.T @ pooled
                    grads['bhead_extra'] = dhe.sum(axis=0, keepdims=True).T
                leg = model._cache.get('lora_embed_grads')
                if leg is not None:
                    grads['embed_extra'] = leg
            _clip_and_update(params, grads, opt_m, opt_v, step, lr)
            adapter.apply()                            # etkin ağırlıkları tazele

        avg = epoch_loss / max(1, nb)
        if avg < best_loss - 1e-5:
            best_loss = avg
            best_state = snapshot()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"LoRA erken durdurma: epoch {epoch + 1}, loss {best_loss:.4f}")
                for k, vv in best_state.items():
                    params[k][...] = vv
                adapter.apply()
                break
        if verbose and (epoch + 1) % 20 == 0:
            print(f"LoRA Epoch {epoch + 1}/{epochs} - Loss: {avg:.4f}")

    preds = np.argmax(model.forward(X, apply_dropout=False), axis=1)
    acc = float(np.mean(preds == y))
    return avg if nb else 0.0, acc


# ---------------------------------------------------------------------------
# Ana giriş
# ---------------------------------------------------------------------------

def finetune_add(model_dir, intents_path, additions, rank=DEFAULT_RANK,
                 alpha=DEFAULT_ALPHA, epochs=DEFAULT_EPOCHS, verbose=True):
    """Yeni intent kayıtlarını öğretir; lora.json + bot_data.json + intents.json yazar."""
    for it in additions:
        if not it.get('tag') or not it.get('patterns'):
            raise ValueError(f"geçersiz kayıt: {it}")
        if not it.get('responses'):
            it['responses'] = ['Faydalı bilgiler edindim!']

    bot_data_path = os.path.join(model_dir, 'bot_data.json')
    with open(bot_data_path, 'r', encoding='utf-8') as f:
        bot_data = json.load(f)
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)

    bot = ChatBot()
    new_words, new_tags = _extend_metadata(bot_data, additions, bot)
    dataset = _build_training_set(bot_data, additions, intents_data)
    texts, labels = dataset

    model, adapter = load_model_with_lora(model_dir, rank=rank, alpha=alpha)
    adapter.extend(new_words, new_tags)

    # Sözcük -> indeks eşlemesi (genişletilmiş vocabulary üzerinden)
    bot.vocabulary = list(bot_data['vocabulary'])
    bot.vocab_to_idx = {w: i for i, w in enumerate(bot.vocabulary)}
    bot.pad_idx = len(bot.vocabulary)
    bot.max_seq_len = model.max_seq_len
    max_len = model.max_seq_len

    X = np.array([bot.text_to_indices(t, max_len) for t in texts], dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    adapter.apply()
    if verbose:
        print(f"LoRA: {len(texts)} örnek, {len(new_words)} yeni sözcük, "
              f"{len(new_tags)} yeni intent ({adapter.rank} rank)")

    loss, acc = train_lora(model, adapter, X, y, epochs=epochs, verbose=verbose)

    # Kalıcılık (atomik: yarim yazimda dosyalar bozulmaz)
    lora_path = os.path.join(model_dir, 'lora.json')
    atomic_write_json(lora_path, adapter.to_dict())
    atomic_write_json(bot_data_path, bot_data, indent=2)

    existing_tags = {i['tag'] for i in intents_data.get('intents', [])}
    for it in additions:
        if it['tag'] not in existing_tags:
            intents_data['intents'].append(it)
    atomic_write_json(intents_path, intents_data, indent=2)

    if verbose:
        print(f"LoRA kaydedildi: {lora_path} (loss {loss:.4f}, acc {acc:.2%})")
    return {'tags_added': list(new_tags), 'vocab_added': len(new_words),
            'examples': len(texts), 'loss': float(loss), 'acc': float(acc),
            'lora_file': os.path.basename(lora_path)}


def forget_intent(model_dir, intents_path, tag, verbose=True):
    """Öğretilmiş bir LoRA intent'ini geri alır; taban intent'ler silinemez.

    lora.json yeni sınıf/kafa satırlarından ve yalnızca o intent'in kullandığı
    yeni sözcük satırlarından temizler; intents.json + bot_data.json günceller.
    """
    lora_path = os.path.join(model_dir, 'lora.json')
    if not os.path.exists(lora_path):
        raise ValueError(
            f"'{tag}' silinemedi: lora.json yok. Yalnizca LoRA ile ogretilmis "
            "intent'ler silinebilir; taban intent'ler model.json'da sabittir.")

    with open(lora_path, 'r', encoding='utf-8') as f:
        ad = json.load(f)
    new_tags = list(ad.get('new_tags', []))
    if tag not in new_tags:
        raise ValueError(
            f"'{tag}' LoRA-ogretilmis bir intent degil. Mevcutlar: "
            f"{', '.join(new_tags) or 'yok'}")
    head_index = new_tags.index(tag)

    bot_data_path = os.path.join(model_dir, 'bot_data.json')
    with open(bot_data_path, 'r', encoding='utf-8') as f:
        bot_data = json.load(f)
    if tag not in bot_data['intent_tags']:
        raise ValueError(f"'{tag}' bot_data.intent_tags icinde yok")
    bot_data['intent_tags'].remove(tag)
    bot_data['intents'].pop(tag, None)
    bot_data['intent_kws'].pop(tag, None)

    # Kalan intent'lerde HANGI yeni sözcükler hâlâ kullanılıyor?
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    bot = ChatBot()
    used = set()
    for item in intents_data['intents']:
        if item.get('tag') == tag:
            continue
        for pat in item.get('patterns') or []:
            used.update(bot.tokenize(pat))

    new_vocab = list(ad.get('new_vocab', []))
    remove_words = [w for w in new_vocab if w not in used and w in bot_data['vocabulary']]
    keep_vocab = [w for w in new_vocab if w not in remove_words]
    for w in remove_words:
        bot_data['vocabulary'].remove(w)

    # Sınıf kafa satırları + embed satırlarını buda (indeksler hizalı kalır)
    d = int(np.asarray(ad.get('whead_extra')).shape[1])
    whead = _f32(ad.get('whead_extra')).reshape(len(new_tags), d)
    bhead = _f32(ad.get('bhead_extra')).reshape(len(new_tags), 1)
    embed_extra = (_f32(ad.get('embed_extra')).reshape(-1, d)
                   if ad.get('embed_extra') else np.zeros((0, d), np.float32))

    keep_h = [i for i in range(len(new_tags)) if i != head_index]
    keep_w = [i for i in range(len(new_vocab)) if new_vocab[i] not in remove_words]

    ad['new_tags'] = [new_tags[i] for i in keep_h]
    ad['head_added'] = len(ad['new_tags'])
    ad['new_vocab'] = keep_vocab
    ad['vocab_added'] = len(keep_vocab)
    ad['whead_extra'] = whead[keep_h].tolist()
    ad['bhead_extra'] = bhead[keep_h].tolist()
    ad['embed_extra'] = embed_extra[keep_w].tolist()

    atomic_write_json(lora_path, ad)
    atomic_write_json(bot_data_path, bot_data, indent=2)
    intents_data['intents'] = [i for i in intents_data['intents']
                               if i.get('tag') != tag]
    atomic_write_json(intents_path, intents_data, indent=2)

    if verbose:
        print(f"LoRA geri alim: '{tag}' silindi "
              f"(vocab={len(remove_words)}, kalan lora intent = "
              f"{len(ad['new_tags'])})")
    return {'tags_removed': [tag], 'vocab_removed': remove_words,
            'remaining_lora_tags': ad['new_tags']}


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='model')
    ap.add_argument('--intents', default='intents.json')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--patterns', action='append', required=True)
    ap.add_argument('--responses', action='append', default=None)
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    args = ap.parse_args()
    finetune_add(args.model_dir, args.intents,
                 [{'tag': args.tag, 'patterns': args.patterns,
                   'responses': args.responses}], epochs=args.epochs)


if __name__ == '__main__':
    _cli()
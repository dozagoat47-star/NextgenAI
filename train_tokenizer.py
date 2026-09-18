"""
Nextgen AI - Turkce BPE Tokenizer Egitim CLI
==============================================
corpus.jsonl + intents.json uzerinde BPE tokenizer egitir ve
tokenizer/bpe.json ciktisini uretir. Saf Python, harici kutuphane yok.

Kullanim:
    python train_tokenizer.py --vocab-size 16000
    python train_tokenizer.py --vocab-size 8000 --max-texts 20000 --min-freq 2
"""

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402


def load_texts(corpus_path, intents_path, max_texts, verbose=True):
    """Egitim metinlerini corpus + intents'tan toplar (deterministik sira)."""
    texts = []
    seen = set()
    t0 = time.time()

    def _add(t):
        if not t:
            return
        t = bpe.clean_text(t)
        if t and t not in seen:
            seen.add(t)
            texts.append(t)

    # 1) Intents: hem kullanicinin yazabilecekleri (patterns) hem yanitlar
    if intents_path and os.path.exists(intents_path):
        with open(intents_path, encoding='utf-8') as f:
            intents = json.load(f).get('intents', [])
        for it in intents:
            for p in it.get('patterns', []):
                _add(p)
            for r in it.get('responses', []):
                _add(r)
        if verbose:
            print(f'[tokenizer] intents: {len(intents)} intent okundu')
    else:
        print(f'[tokenizer] UYARI: {intents_path} bulunamadi, atlaniyor')

    # 2) Corpus: RAG icerigi (Wikipedia + seed). Ongorulmus metin tavani.
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_texts and i >= max_texts:
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get('text') or ''
                _add(t)
                p = d.get('patterns')
                if p:
                    _add(p if isinstance(p, str) else ' '.join(p))
        if verbose:
            print(f'[tokenizer] corpus: {i + 1} satir tarandi '
                  f'(tavan={max_texts})')
    else:
        print(f'[tokenizer] UYARI: {corpus_path} bulunamadi, atlaniyor')

    if verbose:
        print(f'[tokenizer] benzersiz temiz metin: {len(texts)} '
              f'({time.time() - t0:.1f}s)')
    return texts


def main():
    ap = argparse.ArgumentParser(
        description='Nextgen AI Turkce BPE tokenizer egitimi')
    ap.add_argument('--corpus', default='corpus.jsonl')
    ap.add_argument('--intents', default='intents.json')
    ap.add_argument('--out', default='tokenizer/bpe.json')
    ap.add_argument('--vocab-size', type=int, default=16000,
                    help='Hedef vocab (specials + karakterler + birlesmeler)')
    ap.add_argument('--min-freq', type=int, default=2,
                    help='Birlesme icin minimum cift frekansi')
    ap.add_argument('--min-word-freq', type=int, default=2,
                    help='Kelime istatistine girecek min tekrar (dedup)')
    ap.add_argument('--max-texts', type=int, default=20000,
                    help="Corpus'tan okunacak satir tavani (0 = hepsi)")
    ap.add_argument('--progress', action='store_true',
                    help='Her 500 birlesmede ilerleme yazdir')
    args = ap.parse_args()

    texts = load_texts(args.corpus, args.intents, args.max_texts)
    if not texts:
        print('Egitim metni yok; cikis yapiliyor.', file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f'[tokenizer] BPE egitimi basladi (vocab={args.vocab_size}, '
          f'min_freq={args.min_freq}) ...')

    last = [0]

    def _progress(done, total, _last=last, _t0=t0):
        if args.progress and done - _last[0] >= 500:
            _last[0] = done
            print(f'  birlesme {done}/{total} '
                  f'({time.time() - _t0:.1f}s)', flush=True)

    tok = bpe.train_bpe(texts, vocab_size=args.vocab_size,
                        min_freq=args.min_freq,
                        min_word_freq=args.min_word_freq,
                        progress_cb=_progress)
    print(f'[tokenizer] egitim tamam ({time.time() - t0:.1f}s, '
          f'vocab={len(tok)}, merges={len(tok.merges)})')

    # Dogrulama ornekleri
    s = 'merhaba nasılsın bugün hava çok güzel istanbuldan geliyorum'
    ids = tok.encode(s)
    print(f'[tokenizer] ornek encode: {s!r}')
    print(f'[tokenizer]   tokenlar : {[p for _, p in tok.tokenize(s)]}')
    print(f'[tokenizer]   decode   : {tok.decode(ids)!r}')
    ok = tok.decode(ids) == bpe.clean_text(s)
    print(f'[tokenizer] round-trip : {"OK" if ok else "BOZUK!"}')

    tok.save(args.out)
    print(f'[tokenizer] hazir: {os.path.abspath(args.out)} '
          f'(vocab={len(tok)})')


if __name__ == '__main__':
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
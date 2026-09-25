"""Common Crawl tukrevi (mC4-TR) metinlerini corpus.jsonl'a kademeli ekler.

Otomasyon: .github/workflows/crawl_corpus.yml haftalik kosar. Her kosu
--max-mb ham veri kadar indirip paragraflara bolder ve bilgi parcalari
olarak corpus.jsonl'a ekler (id = sha1 icerik, tekrar gelmez). Ilerleme
--state dosyasinda tutulur; --hedef-mb dolunca kosular biter (yeni
crawllar gelince hedef artirilip dispatch ile tekrar doldurulabilir).

Kullanim:
  python build_crawl_corpus.py [--hedef-mb 50] [--max-mb 500] [--dry-run]
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import urllib.parse

DATASET = 'allenai/c4'
CONFIG = 'tr'
PARQUET_API = 'https://datasets-server.huggingface.co/parquet'
DEFAULT_STATE = 'crawl_state.json'
DEFAULT_KORPUS = 'corpus.jsonl'

_RE_HTML = re.compile(r'<[^>]+>')
_RE_URL = re.compile(r'\bhttps?://\S+', re.IGNORECASE)
_RE_MULTI_WS = re.compile(r'[ \t\xa0]+')
_RE_SPLIT_LINES = re.compile(r'\n\s*\n')


def _normalize(text):
    t = _RE_HTML.sub(' ', text)
    t = _RE_URL.sub(' ', t)
    t = _RE_MULTI_WS.sub(' ', t)
    t = t.replace('\r', ' ')
    return t.strip()


def _paragraphs(text, min_len=40, max_len=1200):
    """Metni paragraflara boler; uzun paragraflari cumle paketlerine ayirir."""
    t = _normalize(text)
    out = []
    for para in _RE_SPLIT_LINES.split(t):
        para = re.sub(r'[ \t]+', ' ', para).strip()
        if len(para) < min_len:
            continue
        if len(para) <= max_len:
            out.append(para)
            continue
        sents = re.split(r'(?<=[.!?])\s+', para)
        pack, used = [], 0
        for s in sents:
            pack.append(s)
            used += len(s) + 1
            if len(s) >= min_len and (used >= 300 or len(' '.join(pack)) > 900):
                joined = ' '.join(pack).strip()
                if len(joined) >= min_len:
                    out.append(joined[:max_len])
                pack, used = [], 0
        if pack:
            joined = ' '.join(pack).strip()
            if len(joined) >= min_len:
                out.append(joined[:max_len])
    return out


def _chunk_id(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


def _fetch_parquet_list():
    url = PARQUET_API + '?' + urllib.parse.urlencode({'dataset': DATASET, 'config': CONFIG})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read().decode('utf-8'))
    except Exception as e:
        print(f"[CRAWL] parquet listesi alinamadi: {e}")
        return None
    files = data.get('parquet_files') or []
    print(f"[CRAWL] {DATASET}/{CONFIG}: {len(files)} parquet shard bulundu.")
    return files


def _download_to(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f, 1024 * 1024)


def _read_existing_ids(ids_path):
    ids = set()
    if not os.path.exists(ids_path):
        return ids
    try:
        with open(ids_path, 'r', encoding='ascii') as f:
            for line in f:
                s = line.rstrip('\n')
                if s:
                    try:
                        ids.add(s.encode('ascii').decode('unicode_escape'))
                    except (ValueError, UnicodeDecodeError):
                        ids.add(s)
    except OSError:
        pass
    return ids


def _shard_key(f):
    return f.get('url') or f.get('filename')


def _load_state(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {'kaynak': DATASET, 'config': CONFIG, 'islenmis': [],
            'byte_eklenen': 0, 'hedef_byte': None, 'kosu': 0}


def _save_state(path, state):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_chunks(corpus_path, ids_path, chunks, existing):
    added = 0
    bytes_added = 0
    new_ids = []
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
    with open(corpus_path, 'a', encoding='utf-8') as f:
        for ch in chunks:
            cid = ch['id']
            if cid in existing:
                continue
            rec = json.dumps({
                'id': cid,
                'title': ch.get('title', ''),
                'text': ch['text'],
                'patterns': '',
                'source': ch.get('source', 'mc4'),
                'added_at': now,
            }, ensure_ascii=False)
            f.write(rec + '\n')
            bytes_added += len(rec.encode('utf-8')) + 1
            existing.add(cid)
            new_ids.append(cid)
            added += 1
    with open(ids_path, 'a', encoding='ascii') as f:
        for cid in sorted(new_ids):
            f.write(cid.encode('unicode_escape').decode('ascii') + '\n')
    return added, bytes_added


def _process_parquet(file_url, file_name, corpus_path, ids_path, existing,
                     byte_budget):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[CRAWL] pyarrow yok; 'pip install pyarrow' gerekli.")
        raise

    tmp = tempfile.mkdtemp(prefix='crawl_')
    dest = os.path.join(tmp, os.path.basename(file_name))
    try:
        _download_to(file_url, dest)
        table = pq.read_table(dest, columns=['text'])
        added = 0
        bytes_added = 0
        chunks = []
        for batch in table.to_batches():
            for row in batch.to_pylist():
                text = row.get('text')
                if not text:
                    continue
                for para in _paragraphs(text):
                    cid = _chunk_id(para)
                    if cid in existing:
                        continue
                    chunks.append({'id': cid,
                                   'title': para[:120].replace('\n', ' ').strip(),
                                   'text': para,
                                   'source': 'mc4'})
            n, b = _append_chunks(corpus_path, ids_path, chunks, existing)
            added += n
            bytes_added += b
            chunks = []
            if byte_budget and bytes_added >= byte_budget:
                return added, bytes_added, True
        return added, bytes_added, False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hedef-mb', type=float, default=50.0,
                    help='corpus.jsonl eklenecek toplam hedef (MB)')
    ap.add_argument('--max-mb', type=float, default=500.0,
                    help='tek kosuda islenecek ham veri ust siniri (MB)')
    ap.add_argument('--state', default=DEFAULT_STATE)
    ap.add_argument('--korpus', default=DEFAULT_KORPUS)
    ap.add_argument('--idset', default=None,
                    help='varsayilan: korpus adindan turetir (_ids.jsonl)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    ids_path = args.idset or os.path.splitext(args.korpus)[0] + '_ids.jsonl'
    state = _load_state(args.state)
    hedef = int(args.hedef_mb * 1024 * 1024)
    state['hedef_byte'] = state.get('hedef_byte') or hedef
    state['kosu'] = state.get('kosu', 0) + 1

    if state.get('byte_eklenen', 0) >= state.get('hedef_byte', hedef):
        print(f"[CRAWL] Hedef dolmus ({state['byte_eklenen']}/{state['hedef_byte']} byte) - is yapilmayacak.")
        return 0

    files = _fetch_parquet_list()
    if files is None:
        return 1
    done = set(state.get('islenmis', []))
    pending = [f for f in files if _shard_key(f) not in done]
    if not pending:
        print("[CRAWL] Tum parquet shardlar islenmis; yeni veri yok.")
        return 0

    remaining = state.get('hedef_byte') - state.get('byte_eklenen', 0)
    budget_remaining = int(args.max_mb * 1024 * 1024)
    processed_this_run = 0

    for f in pending:
        size = int(f.get('size') or 0)
        if budget_remaining <= 0:
            print("[CRAWL] Bu kosunun ham indirme limiti doldu; sonraki kosu devam edecek.")
            break
        url = f.get('url') or f.get('rfilename')
        if not url:
            url = 'https://huggingface.co/datasets/{0}/resolve/main/{1}'.format(
                DATASET, urllib.parse.quote(f['filename']))
        print(f"[CRAWL] Isleniyor: {f['filename']} ({size//(1024*1024)} MB)")
        if args.dry_run:
            print(f"[CRAWL]   dry-run: indirme/islem yapilmayacak.")
            processed_this_run += 1
            continue
        added, bytes_added, hit_limit = _process_parquet(
            url, f['filename'], args.korpus, ids_path,
            _read_existing_ids(ids_path), remaining)
        if added:
            state['byte_eklenen'] += bytes_added
            remaining -= bytes_added
            print(f"[CRAWL]   {added} yeni parca ({bytes_added//(1024*1024)} MB) eklendi.")
        done.add(_shard_key(f))
        processed_this_run += 1
        budget_remaining -= size
        if hit_limit or remaining <= 0:
            break

    if not args.dry_run:
        state['islenmis'] = sorted(done)
        _save_state(args.state, state)
    print(f"[CRAWL] Kosu {state['kosu']} bitti: {processed_this_run} shard islendi; "
          f"toplam {state.get('byte_eklenen', 0)}/{state.get('hedef_byte')} byte eklenmis.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
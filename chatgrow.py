"""
Nextgen AI - ChatGrow: gercek insan sohbetinden sekillenen konusma verisi.

autogrow (Wikipedia -> bilgi intentleri) gibi, bu araclar da sohbet tarafina
odaklanir: insanlarin gercekte nasil konustugunu toplayip (soru -> yanit)
ciftleri olarak kaydeder. Boylece LLM'in "konusma bicimi" canned sablonlarin
otesine gecer.

Kullanim:
    1) Reddit app olustur (https://www.reddit.com/prefs/apps -> "script"):
       - client id (harfler/sayilar) ve secret alinir.
    2) Ortami ayarla (oturumluk veya .bashrc):
         set REDDIT_CLIENT_ID=xxxx
         set REDDIT_CLIENT_SECRET=yyyy
    3) Calistir:
         python chatgrow.py                       # varsayilan: r/Turkey, gunun top'lari
         python chatgrow.py --subreddits r/Turkey r/turkishlearning r/istanbul --limit 15
         python chatgrow.py --out chatgrow_sohbet.jsonl --sample 10

Cikti: chatgrow_sohbet.jsonl  ->  {"query": ..., "answer": ..., "source": "reddit",
                                  "subreddit": ..., "score": ..., "ts": ...}

GIZLILIK/GUVENLIK:
  - Resmi Reddit API (uygulamaya-ozel token), oranli ve bot-farkindalikli.
  - Kullanici adi / alt-reddit andizi / URL / e-posta / telefon maskeli yontemden
    ayiklanir; [deleted] ve Automation bot yorumlari elenir.
  - NSFW/18+ paylasimlar, zararli anahtar kelime iceren satirlar kaydedilmez.
  - Egitsel kullanim; paylasimlar kisri kimlik bilgisi tasimamali, tekil satirlar
    bagimsizdir (thread baglami saklanmaz).

Entegrasyon notu: uretilen ciftler daha sonra enrich/naturalize hatti ile LLM
egitim verisine (train_llm pair kaynagi) veya canned sohbet intent'ine donusturulur.
"""

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time

import requests

from clean_intents import ascii_normalize, is_harmful_tag, strip_foreign_scripts

if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"
USER_AGENT = "NextgenAI/1.0 (educational chatbot; local test) requests/2.0"

DEFAULT_SUBREDDITS = ["r/Turkey"]
MIN_SCORE = 10          # dusuk puanli thread'ler gurultu
MIN_COMMENTS = 8        # tek tarafli/gunlugumsu iletiler qa-reply uretmez
MIN_LINE_LEN = 4
MAX_LINE_LEN = 140
MAX_RESPONSES = 3
LISTING_LIMIT = 25

# Kisisel veri/referans kaliplari (iki katman):
#   HARD: URL / e-posta / telefon -> satir TAMAMEN DUSURULUR (eğitim verisine sizamaz).
#   SOFT: u/kullanici ve r/sub andizlari -> icerikten soyulur (kimlik bilgisi degil).
HARD_PII_PATTERN = re.compile(
    r'[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+|www\.\S+'
    r'|\b0\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b|\b0\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b'
    r'|\+\d[\d\s-]{8,}',
    re.IGNORECASE)
SOFT_PII_PATTERN = re.compile(r'u/[\w-]+|r/[\w-]+', re.IGNORECASE)
MARKDOWN_NOISE = re.compile(r'[*_`#>\[\]~]')

SKIP_AUTHORS = {'automoderator', '[deleted]', 'lss_this', 'gezi_bot'}


def get_token(client_id, client_secret):
    """Uygulamaya-ozel (app-only) erisim token'i: yalnizca herkese acik icerik."""
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {'Authorization': f'Basic {auth}',
               'User-Agent': USER_AGENT}
    data = {'grant_type': 'client_credentials'}
    resp = requests.post(REDDIT_TOKEN_URL, data=data, headers=headers, timeout=20)
    resp.raise_for_status()
    tok = resp.json().get('access_token')
    if not tok:
        raise RuntimeError("Reddit token yaniti geçersiz; client id/secret dogrulayin.")
    return tok


def reddit_get(token, path, params, retries=3):
    """Resmi OAuth istegi; 429/5xx durumunda bekleyip tekrar dener."""
    headers = {'Authorization': f'Bearer {token}',
               'User-Agent': USER_AGENT}
    for attempt in range(retries):
        try:
            resp = requests.get(REDDIT_API + path, params=params,
                                headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print("  [BEKLE] Reddit oran limiti, 20 sn bekleniyor...")
                time.sleep(20)
            else:
                print(f"  [HATA] HTTP {resp.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"  [HATA] {e}")
            time.sleep(5)
    return None


def fetch_posts(token, subreddit, section, limit):
    """Subreddit listesinden istenen turdeki paylasimlari doner."""
    params = {'limit': min(limit, 100)}
    if section in ('top', 'controversial'):
        params['t'] = 'day'
    data = reddit_get(token, f"/r/{subreddit}/{section}", params)
    if not data:
        return []
    return [c['data'] for c in data.get('data', {}).get('children', [])
            if not c.get('data', {}).get('stickied')]


def fetch_comments(token, permalink):
    """Paylasimin yorum agacindan ust seviye yorum govdelerini doner."""
    data = reddit_get(token, permalink.rstrip('/') + '.json',
                      {'limit': 40, 'depth': 2})
    if not data or len(data) < 2:
        return []
    root = data[1]
    bodies = []

    def walk(node):
        children = node.get('data', {}).get('children', [])
        for ch in children:
            k = ch.get('kind')
            d = ch.get('data', {})
            if k == 't1':
                author = str(d.get('author', '')).lower()
                body = d.get('body') or ''
                if author not in SKIP_AUTHORS and body:
                    bodies.append(body)
                replies = d.get('replies')
                if replies:
                    walk(replies)
            elif k == 'more':
                continue

    walk(root)
    return bodies


def clean_line(text):
    """Tek bir satiri gizlilik/kalite kurallarindan gecirir; temizse dondurur."""
    if not text:
        return None
    if HARD_PII_PATTERN.search(text):
        return None
    text = re.sub(MARKDOWN_NOISE, '', text)
    text = SOFT_PII_PATTERN.sub('', text)
    text = re.sub(r'\s{2,}', ' ', text).strip(' .')
    text = strip_foreign_scripts(text)
    if not text:
        return None
    if len(text) < MIN_LINE_LEN or len(text) > MAX_LINE_LEN:
        return None
    if re.search(r'http|www\.|reddit\.com', text.lower()):
        return None
    if is_harmful_tag(text):
        return None
    toks = text.split()
    if len(toks) < 3:
        return None
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4 or len(set(letters)) < int(len(letters) * 0.25):
        return None
    return text.strip()


def build_chats(token, subreddit, section, limit):
    """Subreddit'ten (query -> yanit) ciftleri kurar."""
    chats = []
    seen = set()
    for post in fetch_posts(token, subreddit, section, limit):
        if post.get('over_18'):
            continue
        if post.get('num_comments', 0) < MIN_COMMENTS:
            continue
        if post.get('score', 0) < MIN_SCORE:
            continue
        q = clean_line(post.get('title') or '')
        if not q:
            continue
        key = ascii_normalize(q.lower())
        if key in seen:
            continue
        comments = [clean_line(c) for c in
                    fetch_comments(token, post.get('permalink', ''))]
        answers = [c for c in comments if c]  # ayiklanmis koprusuz gorunum
        if len(answers) < 1:
            continue
        answers = answers[:MAX_RESPONSES]
        seen.add(key)
        chats.append({
            'query': q,
            'answer': answers,
            'source': 'reddit',
            'subreddit': subreddit,
            'score': post.get('score', 0),
        })
    return chats


def main():
    ap = argparse.ArgumentParser(description='ChatGrow: gercek sohbet verisi toplama')
    ap.add_argument('--subreddits', nargs='*', default=DEFAULT_SUBREDDITS,
                    help='kaynak subredditler (varsayilan: r/Turkey)')
    ap.add_argument('--section', default='top',
                    choices=['hot', 'new', 'top', 'rising', 'controversial'])
    ap.add_argument('--limit', type=int, default=LISTING_LIMIT)
    ap.add_argument('--out', default='chatgrow_sohbet.jsonl')
    ap.add_argument('--sample', type=int, default=0,
                    help='insan denetimi icin ilk N cifti ekrana yaz')
    ap.add_argument('--dry-run', action='store_true',
                    help='network yok: yalnizca ortam dogrula ve biter')
    args = ap.parse_args()

    cid = os.environ.get('REDDIT_CLIENT_ID', '')
    csec = os.environ.get('REDDIT_CLIENT_SECRET', '')
    if args.dry_run:
        print('[dry-run] REDDIT_CLIENT_ID=%s  REDDIT_CLIENT_SECRET=%s' %
              ('set' if cid else 'YOK', 'set' if csec else 'YOK'))
        return 0 if cid and csec else 1
    if not cid or not csec:
        print('HATA: REDDIT_CLIENT_ID ve REDDIT_CLIENT_SECRET ortam degiskenleri '
              'gerekli. (Kayit: reddit.com/prefs/apps -> script)')
        return 2

    token = get_token(cid, csec)
    print(f"[1/3] Subredditler: {', '.join(args.subreddits)} ({args.section})")

    all_chats = []
    for sub in args.subreddits:
        print(f"[2/3] {sub} taranıyor...")
        sub = sub.lower().lstrip('r/')
        chats = build_chats(token, sub, args.section, args.limit)
        print(f"      {len(chats)} sohbet cifti bulundu")
        all_chats.extend(chats)
    random.shuffle(all_chats)

    if args.out:
        with io.open(args.out, 'w', encoding='utf-8') as f:
            for c in all_chats:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(f"[3/3] YAZILDI: {args.out} ({len(all_chats)} cift)")

    if args.sample > 0:
        print("\n--- DENETIM ORNEGI (insan gozu) ---")
        for c in all_chats[:args.sample]:
            print(f"? {c['query']}")
            for a in c['answer']:
                print(f"  - {a}")
        if not all_chats:
            print('(cift bulunamadi)')


if __name__ == '__main__':
    main()
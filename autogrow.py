"""
Nextgen AI - Autonomous Knowledge Growth
Wikipedia'dan rastgele konular ceker, intents.json'a ekler.
GitHub Actions'ta zamanlanmis olarak calisir ve modelin kendi kendine buyumesini saglar.

Kullanim:
    python autogrow.py                # 4 rastgele konu ekle
    python autogrow.py --count 6      # 6 rastgele konu
    python autogrow.py --topics "Fizik,Denizel_biyoloji"  # belirli konular
"""

import sys
import io
import os
import json
import argparse
import random
import requests

from scrape_intents import fetch_wiki_summary, split_sentences, merge_intents, INTENTS_FILE

if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_BASE = "https://tr.wikipedia.org/w/api.php"
AUTOGROW_MAX_INTENTS = 250
MAX_PATTERNS = 6
MAX_RESPONSES = 3

QUESTION_TEMPLATES = [
    "{topic} nedir",
    "{topic} hakkinda bilgi ver",
    "{topic} ne demek",
    "{topic} anlat",
    "bana {topic} hakkinda bilgi ver",
    "{topic} hakkinda konus",
    "{topic} hakkinda soru sor",
]


def fetch_random_titles(count):
    """Wikipedia 'random' sayfalarindan rastgele basliklar ceker."""
    titles = []
    params = {
        'action': 'query',
        'list': 'random',
        'rnnamespace': 0,
        'rnlimit': count * 3,
        'format': 'json'
    }
    try:
        resp = requests.get(API_BASE, params=params, timeout=10,
                            headers={'User-Agent': 'NextgenAI/1.0'})
        if resp.status_code == 200:
            pages = resp.json().get('query', {}).get('random', [])
            titles = [p['title'] for p in pages]
    except Exception as e:
        print(f"  [HATA] Rastgele konu cekilemedi: {e}")
    return titles


def is_quality_title(title):
    """Kaliteli konu basligi secimi."""
    if '(' in title:
        return False
    if len(title.split()) > 5:
        return False
    if title.endswith((' (il)', ' (ilçe)', ' (anlam ayrımı)')):
        return False
    return True


def build_intent(title):
    """
    Tek bir Wikipedia makalesinden soru-cevap intent'i uretir.

    Returns:
        dict veya None: {'tag', 'patterns', 'responses'}
    """
    extract, real_title = None, title
    summary = fetch_wiki_summary(title)
    if not summary:
        return None
    extract = summary

    sentences = split_sentences(extract)
    if len(sentences) < 2 or len(extract) < 150:
        return None

    clean = real_title.lower().replace('_', ' ').strip()
    tag = clean
    patterns = [t.format(topic=clean) for t in QUESTION_TEMPLATES[:MAX_PATTERNS]]

    random.shuffle(sentences)
    responses = sentences[:MAX_RESPONSES]

    return {
        'tag': tag,
        'patterns': patterns,
        'responses': responses
    }


def main():
    parser = argparse.ArgumentParser(description='Autonomous knowledge growth')
    parser.add_argument('--count', type=int, default=4, help='kac rastgele konu eklenecek')
    parser.add_argument('--topics', type=str, default='',
                        help='virgulle ayrilmis belirli konular')
    args = parser.parse_args()

    with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
        existing_count = len(json.load(f)['intents'])

    if existing_count >= AUTOGROW_MAX_INTENTS:
        print(f"[DUR] Zaten {existing_count} intent var, cap {AUTOGROW_MAX_INTENTS}. Ekleme yapilmedi.")
        return

    print("=" * 50)
    print("  NEXTGEN AI - AUTONOMOUS KNOWLEDGE GROWTH")
    print("  Wikipedia'dan kendini buyutme")
    print("=" * 50)

    if args.topics:
        candidates = [t.strip() for t in args.topics.split(',') if t.strip()]
    else:
        candidates = fetch_random_titles(args.count)

    new_intents = []
    for title in candidates:
        if not is_quality_title(title):
            continue
        intent = build_intent(title)
        if intent:
            print(f"  [YENI] {intent['tag']} ({len(intent['responses'])} response)")
            new_intents.append(intent)

    if not new_intents:
        print("  Eklenebilecek yeni konu bulunamadi.")
        return

    print("\n  INTENTS DOSYASI GUNCELLENIYOR")
    merged, added, updated = merge_intents(INTENTS_FILE, new_intents)

    with open(INTENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nYeni konular: {added}, Guncellenen: {updated}")
    print(f"Toplam intent sayisi: {len(merged['intents'])}")
    print("Sira: python train.py")


if __name__ == '__main__':
    main()
"""
Nextgen AI - Autonomous Knowledge Growth
Wikipedia'dan konular ceker, intents.json'a ekler.
GitHub Actions'ta zamanlanmis olarak calisir ve modelin kendi kendine buyumesini saglar.

Kullanim:
    python autogrow.py                       # tek tur, seckin maddeler
    python autogrow.py --count 10            # tek turda 10 konu
    python autogrow.py --minutes 20          # 20 dakika boyunca surekli gez (her saat/tur)
    python autogrow.py --source mixed        # seckin + rastgele karisik
    python autogrow.py --topics "Fizik,Denizel_biyoloji"  # belirli konular
"""

import sys
import io
import os
import json
import time
import argparse
import random
import requests

from scrape_intents import split_sentences, merge_intents, INTENTS_FILE

if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_BASE = "https://tr.wikipedia.org/w/api.php"
USER_AGENT = "NextgenAI/1.0 (educational chatbot; local test) requests/2.0"
AUTOGROW_MAX_INTENTS = 800
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

TURKISH_TO_ASCII = {
    'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
    'â': 'a', 'î': 'i', 'û': 'u',
    'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
    '\u0307': '',
}


def tr_ascii(text):
    """Etiket ve kaliplari ASCII Turkce ile tutarli hale getirir."""
    return text.translate(str.maketrans(TURKISH_TO_ASCII))


def api_get(params, retries=3):
    """Wikipedia action API istegi; 429/5xx durumunda bekle ve tekrar dene."""
    for attempt in range(retries):
        try:
            resp = requests.get(API_BASE, params=params, timeout=15,
                                headers={'User-Agent': USER_AGENT})
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print("  [BEKLE] Wikipedia kisa bir sure kisitlama uyguladi, 60 sn bekleniyor...")
                time.sleep(60)
            else:
                print(f"  [HATA] HTTP {resp.status_code}")
                time.sleep(10)
        except Exception as e:
            print(f"  [HATA] {e}")
            time.sleep(10)
    return None


def fetch_random_titles(count):
    """Wikipedia 'random' sayfalarindan rastgele basliklar ceker."""
    params = {
        'action': 'query',
        'list': 'random',
        'rnnamespace': 0,
        'rnlimit': count * 3,
        'format': 'json'
    }
    data = api_get(params)
    if not data:
        return []
    return [p['title'] for p in data.get('query', {}).get('random', [])]


def fetch_category_titles(categories, limit):
    """Secilmis/kaliteli madde kategorilerinden rastgele basliklar ceker."""
    titles = []
    for cat in categories:
        continue_token = None
        for _ in range(6):
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': cat,
                'cmnamespace': 0,
                'cmlimit': 50,
                'format': 'json'
            }
            if continue_token:
                params['cmcontinue'] = continue_token
            data = api_get(params)
            if not data:
                break
            titles += [m['title'] for m in data.get('query', {}).get('categorymembers', [])]
            continue_token = data.get('continue', {}).get('cmcontinue')
            if not continue_token:
                break
    random.shuffle(titles)
    return titles[:limit]


def fetch_batch_extracts(titles):
    """
    Makale basliklarini tek tek degil, toplu isteklerle ceker.

    Returns:
        dict: {baslik: giris paragrafi}
    """
    extracts = {}
    for i in range(0, len(titles), 10):
        batch = titles[i:i + 10]
        params = {
            'action': 'query',
            'prop': 'extracts',
            'exintro': 1,
            'explaintext': 1,
            'redirects': 1,
            'format': 'json',
            'titles': '|'.join(batch)
        }
        data = api_get(params)
        if data:
            for page in data.get('query', {}).get('pages', {}).values():
                if 'extract' in page:
                    extracts[page['title']] = page['extract']
        time.sleep(0.5)
    return extracts


def is_quality_title(title):
    """Kaliteli konu basligi secimi."""
    if '(' in title:
        return False
    if len(title.split()) > 6:
        return False
    return True


def build_intent(title, extract):
    """
    Makale ozetinden soru-cevap intent'i uretir.

    Returns:
        dict veya None: {'tag', 'patterns', 'responses'}
    """
    if not extract or len(extract) < 150:
        return None

    sentences = split_sentences(extract)
    if len(sentences) < 2:
        return None

    clean = tr_ascii(title.lower().replace('_', ' ').strip())
    if len(clean.split()) > 6:
        return None

    patterns = [t.format(topic=clean) for t in QUESTION_TEMPLATES[:MAX_PATTERNS]]

    random.shuffle(sentences)
    responses = sentences[:MAX_RESPONSES]

    return {
        'tag': clean,
        'patterns': patterns,
        'responses': responses
    }


def grow_once(source, count):
    """
    Tek bir buyume turu calistirir.

    Returns:
        (added, updated) veya None (cap'a ulasildi, duz dur)
    """
    with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
        existing_count = len(json.load(f)['intents'])

    if existing_count >= AUTOGROW_MAX_INTENTS:
        print(f"[DUR] Zaten {existing_count} intent var, cap {AUTOGROW_MAX_INTENTS}. Ekleme yapilamaz.")
        return None

    if source == 'featured':
        candidates = fetch_category_titles(
            ['Kategori:Seçkin maddeler', 'Kategori:Kaliteli maddeler'], count * 3)
    elif source == 'mixed':
        candidates = fetch_category_titles(
            ['Kategori:Seçkin maddeler', 'Kategori:Kaliteli maddeler'], count * 3)
        candidates += fetch_random_titles(count * 3)
        random.shuffle(candidates)
    else:
        candidates = fetch_random_titles(count * 3)

    qualified = [t for t in candidates if is_quality_title(t)]
    print(f"  Aday sayisi: {len(candidates)}, kalite filtre: {len(qualified)}")

    extracts = fetch_batch_extracts(qualified)
    if not extracts:
        print("  Ozet alinamadi (muhtemelen Wikipedia kisitlamasi). Birazdan tekrar deneyin.")
        return (0, 0)

    new_intents = []
    for title, extract in extracts.items():
        intent = build_intent(title, extract)
        if intent:
            print(f"  [YENI] {intent['tag']} ({len(intent['responses'])} response)")
            new_intents.append(intent)
        else:
            print(f"  [SKIP] {title} (ozet cok kisa/yetersiz)")

    if not new_intents:
        print("  Eklenebilecek yeni konu bulunamadi.")
        return (0, 0)

    # siniri asmamak icin yalnizca kalan bosluk kadar yeni intent alinir
    room = AUTOGROW_MAX_INTENTS - existing_count
    if room <= 0:
        return None
    if len(new_intents) > room:
        print(f"  Cap kaldi: {room}, {len(new_intents) - room} fazla aday ayiklandi.")
        new_intents = new_intents[:room]

    print("\n  INTENTS DOSYASI GUNCELLENIYOR")
    merged, added, updated = merge_intents(INTENTS_FILE, new_intents)

    with open(INTENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"  Bu tur: yeni {added}, guncellenen {updated} | toplam {len(merged['intents'])} intent")
    return (added, updated)


def main():
    parser = argparse.ArgumentParser(description='Autonomous knowledge growth')
    parser.add_argument('--count', type=int, default=4,
                        help='her turda cekilecek konu sayisi')
    parser.add_argument('--minutes', type=float, default=0,
                        help='kac dakika boyunca gezilsin (0 = tek tur)')
    parser.add_argument('--topics', type=str, default='',
                        help='virgulle ayrilmis belirli konular')
    parser.add_argument('--source', type=str, default='featured',
                        choices=['featured', 'random', 'mixed'],
                        help='hangi kaynaktan konu cekilecek')
    args = parser.parse_args()

    print("=" * 50)
    print("  NEXTGEN AI - AUTONOMOUS KNOWLEDGE GROWTH")
    print("  Wikipedia'dan kendini buyutme")
    print("=" * 50)

    if args.topics:
        candidates = [t.strip() for t in args.topics.split(',') if t.strip()]
        extracts = fetch_batch_extracts([t for t in candidates if is_quality_title(t)])
        new_intents = []
        for title, extract in extracts.items():
            intent = build_intent(title, extract)
            if intent:
                new_intents.append(intent)
        if new_intents:
            merged, added, updated = merge_intents(INTENTS_FILE, new_intents)
            with open(INTENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"Yeni konular: {added}, Guncellenen: {updated}")
            print(f"Toplam intent sayisi: {len(merged['intents'])}")
            print("Sira: python train.py")
        else:
            print("Eklenebilecek yeni konu bulunamadi.")
        return

    start = time.time()
    total_added = total_updated = 0
    round_no = 0
    sep = "=" * 50
    while True:
        round_no += 1
        print("\n" + sep + "\n  TUR " + str(round_no) + "\n" + sep)
        result = grow_once(args.source, args.count)
        if result is None:
            break
        total_added += result[0]
        total_updated += result[1]

        if args.minutes <= 0:
            break

        elapsed = time.time() - start
        print(f"  Gecen sure: {elapsed / 60:.1f} dk / {args.minutes} dk")
        if elapsed >= args.minutes * 60:
            break
        time.sleep(5)

    print(f"\nOZET: toplam yeni {total_added}, guncellenen {total_updated}")
    print(f"Gecen sure: {(time.time() - start) / 60:.1f} dk")
    print("Sira: python train.py")


if __name__ == '__main__':
    main()
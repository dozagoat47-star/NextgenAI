"""
Nextgen AI - Web Scraper for Training Data
Wikipedia API ile Türkçe makaleler cekerek intents olusturur.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
import re
import random
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(SCRIPT_DIR, 'intents.json')
SCRAPED_FILE = os.path.join(SCRIPT_DIR, 'scraped_intents.json')

API_URL = "https://tr.wikipedia.org/api/rest_v1/page/summary/{title}"
SEARCH_URL = "https://tr.wikipedia.org/w/api.php"

TOPICS = {
    'bilim': {
        'pages': ['Bilim', 'Bilimsel_yontem', 'Fen_bilimleri'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} ne demek',
            '{topic} anlat', '{topic} ogret', '{topic} ne icin kullanilir'
        ]
    },
    'tarih': {
        'pages': ['Osmanlı_İmparatorluğu', 'Türkiye_Cumhuriyeti', 'İstanbul', 'Ankara'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} ne zaman kuruldu',
            '{topic} hakkinda anlat', '{topic} kim kurdu', '{topic} hakkinda soyle'
        ]
    },
    'uzay': {
        'pages': ['Güneş_Sistemi', 'Mars', 'Jüpiter', 'Ay', 'Evren'],
        'question_templates': [
            '{topic} hakkinda bilgi ver', '{topic} nedir', '{topic} buyuk mu',
            '{topic} hakkinda ne biliyorsun', '{topic} anlat bana'
        ]
    },
    'hayvanlar': {
        'pages': ['Köpek', 'Kedi', 'Aslan', 'Kartal', 'Kaplan'],
        'question_templates': [
            '{topic} hakkinda bilgi ver', '{topic} nedir', '{topic} ne yer',
            '{topic} nasil bir hayvandir', '{topic} nerede yasar'
        ]
    },
    'teknoloji': {
        'pages': ['Yapay_zeka', 'Bilgisayar', 'Internet', 'Yazılım', 'Python_(programlama_dili)'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} ne ise yarar',
            '{topic} nasil calisir', '{topic} hakkinda anlat'
        ]
    },
    'yemek': {
        'pages': ['Türk_mutfagi', 'Pizza', 'Corba', 'Makarna', 'Pilav'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} nasil yapilir',
            '{topic} hakkinda ne biliyorsun', '{topic} anlat bana'
        ]
    },
    'cografya': {
        'pages': ['Anadolu', 'Avrupa', 'Akdeniz', 'Karadeniz'],
        'question_templates': [
            '{topic} nerededir', '{topic} hakkinda bilgi ver', '{topic} nedir',
            '{topic} buyuk mu', '{topic} hakkinda anlat'
        ]
    },
    'spor': {
        'pages': ['Futbol', 'Basketbol', 'Yuzme', 'Atletizm', 'Voleybol'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} nasil oynanir',
            '{topic} hakkinda ne biliyorsun', '{topic} anlat'
        ]
    },
    'saglik': {
        'pages': ['İnsan_vücut', 'Kalp', 'Beyin', 'Akciger', 'Beslenme'],
        'question_templates': [
            '{topic} hakkinda bilgi ver', '{topic} nedir', '{topic} ne ise yarar',
            '{topic} nasil calisir', '{topic} hakkinda anlat'
        ]
    },
    'musiki': {
        'pages': ['Müzik', 'Keman', 'Gitar', 'Piyano'],
        'question_templates': [
            '{topic} hakkinda bilgi ver', '{topic} nedir', '{topic} nasil calinir',
            '{topic} hakkinda ne biliyorsun', '{topic} anlat'
        ]
    },
    'matematik': {
        'pages': ['Matematik', 'Geometri', 'Cebir', 'Istatistik', 'Sayi'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} zor mu',
            '{topic} ne ise yarar', '{topic} hakkinda anlat'
        ]
    },
    'felsefe': {
        'pages': ['Felsefe', 'Etik', 'Bilgi', 'Düşünce'],
        'question_templates': [
            '{topic} nedir', '{topic} hakkinda bilgi ver', '{topic} ne demek',
            '{topic} hakkinda anlat', '{topic} hakkinda konus'
        ]
    }
}


def fetch_wiki_summary(title):
    """Wikipedia REST API'den makale ozetini cek."""
    url = API_URL.format(title=title)
    try:
        resp = requests.get(url, timeout=10, headers={'User-Agent': 'NextgenAI/1.0'})
        if resp.status_code == 200:
            data = resp.json()
            return data.get('extract', '')
    except Exception as e:
        print(f"  [HATA] {title}: {e}")
    return ''


def fetch_wiki_search(topic, limit=5):
    """Wikipedia Search API ile konuyla ilgili makaleler bul."""
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': topic,
        'srlimit': limit,
        'format': 'json'
    }
    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=10,
                            headers={'User-Agent': 'NextgenAI/1.0'})
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('query', {}).get('search', [])
            return [r['title'].replace(' ', '_') for r in results]
    except Exception as e:
        print(f"  [HATA] Arama hatasi: {e}")
    return []


def split_sentences(text):
    """Metni cumlelere ayir."""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in parts:
        s = s.strip()
        if 25 < len(s) < 250 and not s.startswith(('http', 'www')):
            sentences.append(s)
    return sentences


def generate_patterns(topic_name, template_list):
    """Bir konu icin soru kaliplari uret."""
    patterns = []
    clean = topic_name.lower().replace('_', ' ').replace(' (programlama dili)', '').strip()

    for template in template_list:
        patterns.append(template.format(topic=clean))

    extras = [
        f"bana {clean} hakkinda bilgi ver",
        f"{clean} ogret",
        f"{clean} hakkinda soru",
        f"{clean} ne demek",
        f"{clean} hakkinda her seyi biliyor musun"
    ]
    patterns.extend(random.sample(extras, min(3, len(extras))))
    return list(set(patterns))


def scrape_topic(topic_tag, topic_data):
    """Tek bir konudan veri topla."""
    all_sentences = []

    for page_title in topic_data['pages']:
        summary = fetch_wiki_summary(page_title)
        if summary:
            sents = split_sentences(summary)
            all_sentences.extend(sents)
            print(f"  {page_title}: {len(sents)} cumle")
        else:
            print(f"  {page_title}: bos")

    topic_clean = topic_tag.replace('_', ' ')
    search_titles = fetch_wiki_search(topic_clean, limit=3)
    for title in search_titles[:2]:
        if title not in [p.replace('_', ' ') for p in topic_data['pages']]:
            summary = fetch_wiki_summary(title)
            if summary:
                sents = split_sentences(summary)
                all_sentences.extend(sents)
                print(f"  [arama] {title}: {len(sents)} cumle")

    if not all_sentences:
        return None

    random.shuffle(all_sentences)
    selected = all_sentences[:10]
    patterns = generate_patterns(topic_tag, topic_data['question_templates'])

    return {
        'tag': topic_tag,
        'patterns': patterns,
        'responses': selected
    }


def merge_intents(existing_file, new_intents):
    """Yeni intents'lari mevcut intents.json ile birlestir."""
    with open(existing_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    existing_tags = {i['tag'] for i in existing_data['intents']}
    added = 0
    updated = 0

    for new_intent in new_intents:
        if new_intent['tag'] in existing_tags:
            for intent in existing_data['intents']:
                if intent['tag'] == new_intent['tag']:
                    for p in new_intent['patterns']:
                        if p not in intent['patterns']:
                            intent['patterns'].append(p)
                    for r in new_intent['responses']:
                        if r not in intent['responses']:
                            intent['responses'].append(r)
                    updated += 1
                    print(f"  [GUNCELLE] {new_intent['tag']}")
        else:
            existing_data['intents'].append(new_intent)
            existing_tags.add(new_intent['tag'])
            added += 1
            print(f"  [YENI] {new_intent['tag']}")

    return existing_data, added, updated


def main():
    print("=" * 50)
    print("  NEXTGEN AI - WEB SCRAPER")
    print("  Wikipedia API ile veri toplama")
    print("=" * 50)
    print(f"\nToplam {len(TOPICS)} konu taranacak...\n")

    all_new_intents = []
    total_p = 0
    total_r = 0

    for topic_tag, topic_data in TOPICS.items():
        print(f"\n--- {topic_tag.upper()} ---")
        intent = scrape_topic(topic_tag, topic_data)
        if intent:
            all_new_intents.append(intent)
            total_p += len(intent['patterns'])
            total_r += len(intent['responses'])
            print(f"  => {len(intent['patterns'])} pattern, {len(intent['responses'])} response")

    print("\n" + "=" * 50)
    print("  INTENTS DOSYASI GUNCELLENIYOR")
    print("=" * 50)

    merged, added, updated = merge_intents(INTENTS_FILE, all_new_intents)

    with open(INTENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    with open(SCRAPED_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_new_intents, f, ensure_ascii=False, indent=2)

    print(f"\nYeni konu: {added}, Guncellenen: {updated}")
    print(f"Toplam yeni pattern: {total_p}")
    print(f"Toplam yeni response: {total_r}")
    print(f"Intents: {INTENTS_FILE}")

    print("\nSiradaki adim: python train.py")


if __name__ == '__main__':
    main()

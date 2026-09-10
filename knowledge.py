"""
Nextgen AI - Live Knowledge Retrieval
Model bilmedigi sorular icin Wikipedia'dan anlik bilgi ceker.
"""

import sys
import io
import re
import requests

STOPWORDS = {
    'nedir', 'ne', 'neydi', 'nedi', 'kimdir', 'kim', 'nasil', 'neden',
    'nicin', 'nerede', 'nerede', 'hakkinda', 'bilgi', 'ver', 'soyle',
    'anlat', 'konus', 'ogret', 'yaz', 'bana', 'ben', 'sen', 'bu', 'su',
    'bir', 'mi', 'mu', 'degil', 'acikla', 'detay', 'icer', 'ban', 'beraber',
    'soruyorum', 'merak', 'ediyorum', 'istedim', 'ilet', 'amonle',
    'acayip', 'coo', 'sey', 'hani', 'her', 'seyn', 'bagir', 'mr', 'ly',
    'any', 'velo', 'cam', 'asiri', 'var'
}

API_BASE = "https://tr.wikipedia.org/w/api.php"


def clean_query(text):
    """Soru metninden onemsiz kelimeleri temizler."""
    text = text.lower()
    text = re.sub(r'[^\w\sçğıöşüâîû]', ' ', text, flags=re.UNICODE)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) >= 3]
    return ' '.join(words[:6])


def search_pages(query):
    """Wikipedia arama API'sinden ilk 3 sonucu dondurur."""
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'srlimit': 3,
        'format': 'json'
    }
    try:
        resp = requests.get(API_BASE, params=params, timeout=10,
                            headers={'User-Agent': 'NextgenAI/1.0'})
        if resp.status_code == 200:
            results = resp.json().get('query', {}).get('search', [])
            return [r['title'] for r in results]
    except Exception:
        pass
    return []


def fetch_summary(title):
    """REST API'den makale ozetini ceker."""
    url = f"https://tr.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=10,
                            headers={'User-Agent': 'NextgenAI/1.0'})
        if resp.status_code == 200:
            data = resp.json()
            if 'anlam ayrımı' in (data.get('description') or '').lower():
                return None, data.get('title', '')
            return data.get('extract', ''), data.get('title', '')
    except Exception:
        pass
    return None, title


def fetch_answer(question):
    """
    Kullanici sorusuna Wikipedia'dan kisa bir cevap uretir.

    Returns:
        dict veya None: {'title', 'answer', 'url'}
    """
    query = clean_query(question)
    if not query:
        return None

    titles = search_pages(query)
    for title in titles:
        extract, real_title = fetch_summary(title)
        if not extract:
            continue

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', extract)][:3]
        answer = ' '.join(sentences)
        if len(answer) < 60:
            continue

        if len(answer) > 400:
            answer = answer[:397] + '...'

        return {
            'title': real_title,
            'answer': answer,
            'url': f"https://tr.wikipedia.org/wiki/{title.replace(' ', '_')}"
        }
    return None
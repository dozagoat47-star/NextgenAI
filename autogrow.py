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
import json
import os
import time
import argparse
import random
import re
import requests

from scrape_intents import split_sentences, merge_intents, INTENTS_FILE
from corpus import Corpus
from clean_intents import (ascii_letters_only, ascii_normalize,
                            is_harmful_tag, non_latin_letters,
                            strip_foreign_scripts)
from finetune import atomic_write_json

if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_BASE = "https://tr.wikipedia.org/w/api.php"
USER_AGENT = "NextgenAI/1.0 (educational chatbot; local test) requests/2.0"
# Intent tavani. Bu bir KALITE kapisi degil, kacak koruma siniridir:
# eski deger 800 idi ve doldugu icin AutoGrow intents.json'a HIC yazamiyordu
# (cap_reached) -> yeni bilgi korpusa gidiyor ama intent olmadigi icin LLM'e
# hic girmiyordu.
#
# NEDEN 6.000: olcum. enrich_intents.build_knowledge_map maliyeti 29.1
# ms/desen (4.584 desen = 133 sn) ve corpus.load() 189 sn. 6.000 intent
# (~5.236 bilgi -> 31.416 desen) icin ~15 dk + 3 dk = 18.5 dk. 20.000
# intent 61 dk surerdi, gunluk ise sigmazdi. Artan konular ondalik
# kategorilerden gelir (asagida) ve kalite kapisindan gecer, yani genisleme
# bilgi degil, gurultu degildir.
#
# Deger build_knowledge_map butcesiyle sinirli (asagi olculdu); istenirse
# AUTOGROW_MAX_INTENTS ortam degiskeniyle kapatilabilir.
#
# 28.09 YUKSILTILDI 6.000 -> 20.000. Gerekce olcumu:
#   - 6.000 kapisi DOLUYDU (6.000/6.000): intent uretimi durmus, soru
#     buyumeye devam ediyordu.
#   - buyume hizi 7.790 intent/gun (son 14 saatte 4.598 intent, 2.98 MB).
#   - 842 bayt/intent -> 20.000 intent = 16,1 MB intents.json.
#   - model ETKILENMEZ: bilgi intentleri siniflandirici DEGILDIR,
#     model.json num_intents 40'da sabit kalir. yalnizca arama/LUT
#     maliyeti artar (knowledge_map butcesi).
#   - knowledge_map butcesi ayri bir butce: kbmap.yml --kb-limit 40000
#     kullanir, yani 20.000 intent bu butcede FIRLA yer alir.
# Bu deger build suresiyle degil, veri butcesiyle sinirlandirildi; build
# maliyeti artik kosu basina tavanla (asagida) kisitlanir.
AUTOGROW_MAX_INTENTS = int(os.environ.get('AUTOGROW_MAX_INTENTS', 20000))

# KOSU BASINA TAVAN (28.09). Olcum: autogrow.yml saatlik (24 kosu) +
# autogrow-deep.yml gunluk (1 kosu) = 25 kosu/gun; 7.790/25 = ortalama
# 312 intent/kosu. Tavan 400: normal akis DEGISMER (ort. 312 < 400),
# ama sicak bir kosu (60 dk derin tarama, bol kaliteli baslik) tek
# commit'te binlerce intent dokup gecmis ve dosyayi sismaz.
# 0 = tavansiz (eski davranis).
AUTOGROW_MAX_PER_RUN = int(os.environ.get('AUTOGROW_MAX_PER_RUN', 400))
MAX_PATTERNS = 6
MAX_RESPONSES = 3

# Korpusa girecek metin icin kalite kapisi.
# Olcum (128.555 parcacik): medyan 179 krkt, %52'si 200 krktin altinda,
# %35'i 100 krktin altinda. Nedeni kaynak: uniform rastgele akis tr'inin
# tamamindan ornek cekiyor ve gozlemsiz basliklarin cogunlugunu getiriyor
# (rastgele akistan gelen ozetlerin %53'u TEK cumle, medyan 111 krkt).
# Boyle bir parcacik bir soruya CEVAP olamaz, yalnizca retrieve adaylarini
# kirletir. Iki cumle + 200 krkt: featured akisinin tamamini gecirir
# (medyan 1408) ve stub'lari eler.
MIN_CORPUS_TEXT = 200
MIN_CORPUS_SENTENCES = 2

# Intent etiketi icin asgari uzunluk. Tek/iki harfli basliklar (Wikipedia'nin
# harf maddeleri: L, W, I) konu degildir; 'l nedir' gibi kalipsiz intent
# uretirler. intents.json'a tag='l' girdi ve egitim oncesi kalkan olan
# test_core.TestIntentsSchema (tag >= 2) dusuyordu.
MIN_TAG_CHARS = 3

# Gunluk buyume ondalik kategorilerden gelsin. Uniform rastgele akis
# gozlemsiz basliklari tercih ediyor: botun bilgi tabani bu yuzden
# 'benefse', 'avec', 'minerva mcgonagall' gibi konulardan olustu; kullanici
# sordugu fizik/internet/programlama ise hic yoktu (olcum: 23 klasik
# konudan 9'u, onlarin da cogu yanlis eslesme).
TOPIC_CATEGORIES = [
    'Kategori:Fizik', 'Kategori:Kimya', 'Kategori:Biyoloji',
    'Kategori:Matematik', 'Kategori:Teknoloji', 'Kategori:Ekonomi',
    'Kategori:Coğrafya', 'Kategori:Edebiyat', 'Kategori:Felsefe',
    'Kategori:Tıp', 'Kategori:İslam', 'Kategori:Türkiye',
]
# Wikipedia API'si kota uyguluyor: bir turda tum kategoriler cekilirse
# 429 yiyip tur bos gecirir. Her tur komsu sayida kategori secilir ve
# rotasyonla butun liste birkac gun icinde gezilir.
TOPIC_CATEGORIES_PER_ROUND = 3

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

    clean = ascii_normalize(title.lower().replace('_', ' ').strip())
    if len(clean.split()) > 6:
        return None
    # Cok kisa baslik konu degildir: tek harfli maddeler (L, W, I) bir
    # soruya cevap olamaz ve 'l nedir' gibi kalipsiz intent uretirler.
    # Yakalandi: intents.json'a tag='l' girmisti ve test_core
    # (egitim oncesi kalkan) dusuyordu. test_core tag >= 2 istiyor;
    # burada daha sert: 3.
    if len(clean.strip()) < MIN_TAG_CHARS:
        return None
    # ASCII disi harf kalan basliklar reddedilir. ascii_normalize
    # aksanlari NFD ile atar ama bazi Latin harfleri AYRISMAZ:
    # U+01C1 (tik sesi), U+0111 (Bosna-Hersek d'si), U+02BB (okina).
    # Bunlar gecerek intents.json'a yaziliyor ve
    # test_core.TestIntentsSchema.test_tags_latin'i dusuruyordu.
    duzeltilmis = ascii_letters_only(clean)
    if duzeltilmis is None or len(duzeltilmis) < MIN_TAG_CHARS:
        print(f"  [SKIP] {title} (etiket Latin disi harf iceriyor: "
              f"{non_latin_letters(clean)})")
        return None
    clean = duzeltilmis
    if is_harmful_tag(clean):
        print(f"  [SKIP] {title} (guvenlik/icerik filtresi)")
        return None

    patterns = [t.format(topic=clean) for t in QUESTION_TEMPLATES[:MAX_PATTERNS]]

    # Yanit havuzu hijyeni: Latin disi yazim bloklari + kesik ozetler elenir.
    responses = []
    for s in sentences:
        s = strip_foreign_scripts(s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        if not s or len(s.split()) < 3:
            continue
        if re.search(r'\(\s*d\.?\s*$|\(\s*$', s):
            continue
        responses.append(s)
        if len(responses) >= MAX_RESPONSES:
            break
    if len(responses) < 2:
        return None

    return {
        'tag': clean,
        'patterns': patterns,
        'responses': responses
    }


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _sentence_count(text):
    """Cumle sayisi, uzunluk filtresi olmadan.

    scrape_intents.split_sentences 25-250 krkt disi parcalari ATAR; kalite
    kapisi icin bu yaniltici olurdu ('Bu ikinci cumledir.' 17 krkt olup
    gercekte ikinci cumledir, ama sayilmaz).
    """
    text = re.sub(r'\s+', ' ', text or '').strip()
    return len([s for s in _SENT_SPLIT.split(text) if s.strip()])


def corpus_worthy(extract):
    """Ozet, korpusa yazilmayi hak ediyor mu?

    Tek cumlelik/200 krktin altindaki stub'lar bir soruya cevap olamaz;
    yalnizca corpus.search adaylarini kirletir. Kapi gecisli olce kirletici
    azalir (olcum: 39 soruluk kapida kabul %100 -> %20, tehlikeli kabul 0).
    """
    if not extract or len(extract) < MIN_CORPUS_TEXT:
        return False
    return _sentence_count(extract) >= MIN_CORPUS_SENTENCES


def grow_once(source, count):
    """
    Tek bir buyume turu calistirir.

    Intent cap dolmus olsa bile RAG-lite corpus'u buyumeye devam eder:
    NN eğitim verisi (intents) sinirli kalir, corpus ise sinirsiz birikir.

    Returns:
        (added, updated) ; intent eklenemezse (0, 0) ; hic veri yoksa (0, 0)
    """
    with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
        existing_count = len(json.load(f)['intents'])

    cap_reached = existing_count >= AUTOGROW_MAX_INTENTS
    if cap_reached:
        print(f"[BILGI] Intent cap {AUTOGROW_MAX_INTENTS} doldu ({existing_count}). "
              f"RAG-lite corpus buyumeye devam ediyor.")

    if source == 'featured':
        candidates = fetch_category_titles(
            ['Kategori:Seçkin maddeler', 'Kategori:Kaliteli maddeler'], count * 3)
    elif source == 'mixed':
        cats = random.sample(TOPIC_CATEGORIES, TOPIC_CATEGORIES_PER_ROUND)
        candidates = fetch_category_titles(cats, count * 3)
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
    corpus_chunks = []
    elenen = 0
    for title, extract in extracts.items():
        if not cap_reached:
            intent = build_intent(title, extract)
            if intent:
                print(f"  [YENI] {intent['tag']} ({len(intent['responses'])} response)")
                new_intents.append(intent)
            else:
                print(f"  [SKIP] {title} (ozet cok kisa/yetersiz)")

        # Corpus icin kalite kapisi: 2+ cumle ve 200+ krkt. Eski esik 40
        # krkt idi, bu yuzden korpus tek cumlelik stub'larla doldu.
        if corpus_worthy(extract):
            clean_extract = strip_foreign_scripts(extract)
            clean_extract = re.sub(r'\s{2,}', ' ', clean_extract).strip()
            corpus_chunks.append({
                'id': ascii_normalize(title.strip().lower().replace(' ', '_')).replace(' ', '_'),
                'title': ascii_normalize(title.strip()),
                'text': clean_extract,
                'source': 'autogrow',
            })
        else:
            elenen += 1

    if elenen:
        print(f"  [KALITE] {elenen} ozet elendi (2+ cumle ve 200+ krkt gerekli).")

    if cap_reached:
        Corpus.append_many(corpus_chunks)
        if corpus_chunks:
            print(f"  [CORPUS] {len(corpus_chunks)} konu corpus'a eklendi/guncellendi "
                  f"(intentler {existing_count} olarak sabit).")
        else:
            print("  Eklenebilecek yeni konu bulunamadi.")
        return (0, 0)

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
        corpus_chunks = corpus_chunks[:room]

    print("\n  INTENTS DOSYASI GUNCELLENIYOR")
    merged, added, updated = merge_intents(INTENTS_FILE, new_intents)

    atomic_write_json(INTENTS_FILE, merged, indent=2)

    # Toplanan ham bilgi RAG-lite corpus'unda biriksin (sinir yok: bilgi hic
    # kaybolmasin). Ayni id'li kayit taze metinle guncellenir.
    Corpus.append_many(corpus_chunks)

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
        corpus_chunks = []
        for title, extract in extracts.items():
            intent = build_intent(title, extract)
            if intent:
                new_intents.append(intent)
                corpus_chunks.append({
                    'id': ascii_normalize(title.strip().lower().replace(' ', '_')).replace(' ', '_'),
                    'title': ascii_normalize(title.strip()),
                    'text': strip_foreign_scripts(extract),
                    'source': 'autogrow',
                })
        if new_intents:
            merged, added, updated = merge_intents(INTENTS_FILE, new_intents)
            atomic_write_json(INTENTS_FILE, merged, indent=2)
            Corpus.append_many(corpus_chunks)
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
    # Kosu basina tavan: butce bitince dongu KIRILIR, yeni intent
    # uretilmez. Boylece intents.json'in tek commit'te ne kadar
    # buyudugu sinirli kalir (git maliyeti karesel).
    butce = AUTOGROW_MAX_PER_RUN
    while True:
        if butce > 0 and total_added >= butce:
            print(f"\n[TAVAN] Kosu basi intent sinirina ulasildi "
                  f"({total_added}/{butce}). Dongu durduruluyor.")
            break
        round_no += 1
        print("\n" + sep + "\n  TUR " + str(round_no) + "\n" + sep)
        # Kalan butce kadar cek: son turda 8 yerine 3 iste, gerekmeden
        # uretilen intent'ler havada kalmasin.
        count = args.count
        if butce > 0:
            kalan = butce - total_added
            if kalan <= 0:
                break
            count = max(1, min(args.count, kalan))
        result = grow_once(args.source, count)
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
    if butce > 0 and total_added >= butce:
        print(f"Kosu tavani {butce} ULASTI: intents.json buyumesi "
              f"bir sonraki kosuya birakildi.")
    print(f"Gecen sure: {(time.time() - start) / 60:.1f} dk")
    print("Sira: python train.py")


if __name__ == '__main__':
    main()
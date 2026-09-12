"""
Nextgen AI - Flask Web Server
"""

import sys
import io
import traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import random
import re
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify
from brain import ChatBot
from knowledge import fetch_answer
from corpus import Corpus
from generator import TextGenerator

app = Flask(__name__)

bot = ChatBot()
corpus = Corpus()
generator = TextGenerator()
model_loaded = False
corpus_loaded = False
all_patterns = []
_last_tag = None  # baglam: son yanitin intent'i ('espri' devam istekleri icin)

DEFAULT_UNKNOWN = ("Bu konuda henüz yeterli bilgiye sahip değilim, "
                   "farklı bir şekilde sormak ister misin?")

FEEDBACK_TEMPLATE = ("Özür dilerim, verdiğim bilgi yanlış veya eksik olabilir. "
                     "Doğrusunu öğrenmem için beni yönlendirebilirsin.")

ADVICE_TEMPLATE = ("Ben bir yapay zekayım; kişisel ve sağlıkla ilgili kararlarını "
                   "senin yerine veremem. Konuyla ilgili bir uzmana ya da güvendiğin "
                   "birine danışmanı öneririm.")

# Yetersizlik/yetkinlik eleştirisi ("sen hiçbir şey bilmiyorsun", "çok
# yetersizsin") duygu olarak üzüntü/sarılma şeklinde yorumlanmamalı; model
# kendini bilgi açısından savunup öneri ister.
SYSTEM_FEEDBACK_TEMPLATE = ("Gelişmekte olan bir yapay zekayım, henüz her konuda "
                            "yeterli bilgim olmayabilir. Önerilerini iletebilirsin.")

INADEQUACY_PHRASES = [
    'hicbir sey bilmiyor', 'hicbirsey bilmiyor', 'bir sey bilmiyor', 'birsey bilmiyor',
    'hicbir sey bilemiyor', 'hicbirsey bilemiyor', 'bilmiyorsun', 'yetersizsin',
    'yetersizsiniz', 'ise yaramaz', 'ise yaramiyor', 'beceriksiz', 'cok kotu biliyorsun',
    'salaksin', 'akilli degilsin', 'beyinsiz', 'boylesini bile bilmiyorsun',
]

# Kimlik ve güven soruları veritabanı aramasına düşüp "bilgi yok" dememeli;
# sabit, dürüst bir yanıt bağlanır ("sen kimsin", "sana güvenebilir miyim").
TRUST_TEMPLATE = ("Ben yerel çalışan bir yapay zekayım. Bilgileri veritabanımdan "
                  "kurgularım, kritik konularda teyit etmeni öneririm.")

TRUST_PHRASES = [
    'guvenebilir miyim', 'guvenebilir miyiz', 'guvenilir misin', 'guvenebilir misin',
    'guvenir misin', 'sana guven', 'sen kimsin', 'kimsin', 'kim oldugunu soyle',
    'kendini tanit', 'kendini anlat',
]

# Kullanıcı modelin tepkisini beğenmediğinde (eleştiri/geri bildirim) arama veya
# corpus'a gitmeden DOĞRUDAN özür yanıtı verilir.
FEEDBACK_PHRASES = [
    'yalan soyl', 'yalan at', 'dogru soyl', 'sacmal', 'yanlis biliy',
    'yanlis cevap', 'kandirm', 'dalga gec', 'ciddi ol', 'durst ol',
    'bos konus', 'gerceg soyl', 'inanm',
]

# Kisisel/karar soruları yalnizca HASSAS konu (saglik, tip, diyet, hapis/hukuk)
# icercekse sinir-bilen yanit alsin. 'sence' tek basina yeterli degil; telefon
# tavsiyesi gibi genel sorular normal arama ile yanıtlanabilir.
ADVICE_VERBS = ['sence', 'yapayim mi', 'benim icin', 'adima karar ver']

SENSITIVE_TOPICS = [
    'saglik', 'hasta', 'hastal', 'doktor', 'tedavi', 'ilac', 'ameliyat',
    'teshis', 'diyet', 'kilo', 'kalp', 'tansiyon', 'seker', 'depresyon',
    'anksiyete', 'kaygi', 'kanser', 'sigara', 'alkol', 'hamile', 'gebelik',
    'hapis', 'ceza', 'hukuk', 'avukat', 'dava', 'mahkeme', 'suclu',
]


# Uretime GIRMEYECEK tagler: sabit tek-cümlelik sohbet/kibarlik yanitlari
# ve fikra (espri) — bunlar oldugu gibi verilir; bilgi yanitlari ise
# generator'dan gecerek kopyala-yapistirmadan kurtulur.
GENERATIVE_EXEMPT = {
    'karsilama', 'kendini_tanit', 'tesekkur', 'veda', 'durum',
    'yardim', 'mutluluk', 'uzuntu', 'espri',
}


# SAYILANABILIR LISTE ISLEMLERI: "3 film oner", "2 tane dizi soyle" gibi
# sorularda sorgudaki sayi yakalanir ve ilgili intent'in yanit havuzundan o
# kadar FARKLI oge bir listede birlestirilir ("1. X 2. Y 3. Z ...").
COUNT_WORDS = {'bir': 1, 'iki': 2, 'uc': 3, 'dort': 4, 'bes': 5,
               'alti': 6, 'yedi': 7, 'sekiz': 8, 'dokuz': 9, 'on': 10}
LISTABLE_NOUNS = ['film', 'dizi', 'kitap', 'sarki', 'muzik', 'oyun', 'sehir']
COUNT_TAGS = {'film', 'sarki', 'oyun'}


def extract_count(text):
    """Sorgudaki sayiyi (rakam veya Turkce okunus) dondurur; yoksa None."""
    t = bot.ascii_normalize(text.lower())
    nums = re.findall(r'\b\d{1,2}\b', t)
    if nums:
        n = int(nums[0])
        if 1 <= n <= 10:
            return n
    if re.search(r'\bbir ?kac\b', t):
        return 3
    for w in re.findall(r'[a-z0-9]+', t):
        if w in COUNT_WORDS:
            return COUNT_WORDS[w]
    return None


def build_counted_list(count, items):
    """Intent yanit havuzundan o kadar FARKLI ogeyi "1. X ..." listesine dizer.
    Her oge uretim katmanindan gecirilir (baslik anchor olarak korunur)."""
    seen, uniq = set(), []
    for it in items:
        title = it.split('!')[0].split(',')[0]
        key = re.sub(r'[^a-z0-9]+', '', bot.ascii_normalize(title.lower()))
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    if len(uniq) < 2:
        return None
    random.shuffle(uniq)
    chosen = uniq[:count]
    out = []
    for i, it in enumerate(chosen, 1):
        text = generator.generate_response(it).strip()[:140]
        text = text.rstrip('!.,; ').strip()
        out.append(f"{i}. {text}.")
    return ' '.join(out)


def is_feedback_phrase(text):
    t = bot.ascii_normalize(text.lower())
    return any(p in t for p in FEEDBACK_PHRASES)


def is_inadequacy_feedback(text):
    t = bot.ascii_normalize(text.lower())
    return any(p in t for p in INADEQUACY_PHRASES)


def is_trust_question(text):
    t = bot.ascii_normalize(text.lower())
    return any(p in t for p in TRUST_PHRASES)


# EYLEM KILIDI: eylem fiili ('yap/anlat/soyle') + mizah kelimesi birlikteyse
# sorgu kesinlikle fıkra/şaka niyetine kilitlenir; 'mizah tanimi' gibi tanim
# intent'lerine veya ansiklopedi fallback'ine düşmez.
JOKE_VERBS = ['yap', 'anlat', 'soyle', 'uydur', 'yaz']
HUMOR_STEMS = ['mizah', 'espri', 'fikra', 'saka']


def is_joke_request(text):
    stems = set(bot.tokenize(text))
    if not (stems & set(JOKE_VERBS)):
        return False
    return bool(stems & set(HUMOR_STEMS))


# DOLGU-ONLİ TERCIHTEN YOK: 'bir daha yap', 'baska bir tane daha soyle'
# gibi konusuz devam istekleri. Son yanit fikraydiysa yeni fikra doner.
REPEAT_CUES = {'daha', 'tane', 'baska', 'bir'}


def is_repeat_request(text):
    stems = set(bot.tokenize(text))
    if not (stems & REPEAT_CUES):
        return False
    # Bilgilendirici (konu tasiyan) tek kelime olmamali: tamami dolu/stopword.
    return all(w in bot.keyword_weights and bot.keyword_weights[w] == 0.0
               for w in stems if '_' not in w)


def is_advice_question(text):
    t = bot.ascii_normalize(text.lower())
    if not any(v in t for v in ADVICE_VERBS):
        return False
    # 'sence paris hangi ulkede' gibi gercek bilgi sorusu engellenmesin
    if is_factual_query(text):
        return False
    # yalnizca hassas konu kelimesi varsa sinir-bilen yanit
    return any(s in t for s in SENSITIVE_TOPICS)

FACTUAL_MARKERS = ['nedir', 'ne demek', 'hakkinda', 'kimdir', 'kimlerdir', 'nerede',
                   'ne zaman', 'nasil yapilir', 'kac yil', 'tarihi', 'ozetle', 'acikla',
                   'tanimi', 'tanim', 'anlami',
                   'kaynak', 'wikipedia', 'yapilir misin', 'verebilir misin',
                   'ulkede', 'ulkesinde', 'ulkesi', 'sehirde', 'neresinde', 'ilcesi',
                   'nerede', 'bolgesinde', 'kim kurdu', 'kim yazdi', 'kim buldu']


def is_factual_query(text):
    t = bot.ascii_normalize(text.lower())
    if any(m in t for m in FACTUAL_MARKERS):
        return True
    return t.count('?') > 0 and any(w in t for w in
                                   [' ne ', ' kim ', ' nerede ', ' nasil ', ' kac ', ' hangi ', ' neden '])


def fallback_answer(message):
    """Dataset cevabina guvenilmezse: corpus -> internet (sessizce ogrenerek) -> bilmiyorum.

    Retrieval ciktisi dogrudan basilmaz; TextGenerator uretim katmanindan
    gecerek anchor-sabit yeni cumle olarak verilir.
    """
    chunk = corpus.search(message)
    if chunk and chunk['score'] >= corpus.min_score:
        print(f"[CHAT] Corpus eslesmesi (%.2f): {chunk['title']}" % chunk['score'])
        return generator.generate_response(chunk['text'], title=chunk.get('title', ''))

    if is_factual_query(message):
        knowledge = fetch_answer(message)
        if knowledge:
            # Ogrenme: internetten gelen bilgi corpus'a yazilir; bir dahaki
            # soruya kutuphane aninda cevap verir (yeniden egitim gerekmez).
            try:
                slug = bot.ascii_normalize(knowledge['title'].strip().lower()).replace(' ', '_')
                Corpus.append_many([{'id': slug,
                                     'title': knowledge['title'],
                                     'text': knowledge['answer'],
                                     'patterns': message,
                                     'source': 'learned'}])
                corpus.refresh()
                print("[CHAT] Internet bilgisi corpus'a kaydedildi: " + knowledge['title'])
            except Exception as e:
                print(f"[CHAT] Corpus kaydinda hata: {e}")
            raw = knowledge.get('raw') or knowledge['answer']
            return generator.generate_response(raw, title=knowledge.get('title', ''))

    return DEFAULT_UNKNOWN


def load_bot():
    global bot, model_loaded, all_patterns, corpus, corpus_loaded
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'model')
    intents_file = os.path.join(script_dir, 'intents.json')

    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, 'model.json')):
        bot.load_model(model_dir)
        model_loaded = True
        print("[OK] Model loaded!")
    else:
        print("[ERROR] Model not found!")
        model_loaded = False

    if os.path.exists(intents_file):
        with open(intents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_patterns = []
        for intent in data['intents']:
            for p in intent['patterns']:
                all_patterns.append(bot.ascii_normalize(p.lower()))

    # RAG-lite: ilk acilista corpus yoksa mevcut intents'lardan tohumla
    Corpus.seed_from_intents()
    corpus.load()
    corpus_loaded = True

    # Generative katman: genel gecis modelini corpus + intents uzerinden kur
    generator.build_from_files(os.path.join(script_dir, 'corpus.jsonl'),
                               os.path.join(script_dir, 'intents.json'))


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['GET', 'POST', 'OPTIONS'])
def chat():
    global _last_tag
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = {'message': request.args.get('message', '')}
        user_message = data.get('message', '')
        print(f"[CHAT] Message: {user_message}")
        if not user_message.strip():
            return jsonify({'response': 'Bir seyler yaz!'})
        if not model_loaded:
            return jsonify({'response': 'Model yuklenmedi! once train.py calistir.'})

        if is_feedback_phrase(user_message) or is_inadequacy_feedback(user_message):
            if is_inadequacy_feedback(user_message):
                print("[CHAT] Yetersizlik elestirisi, sistem yaniti veriliyor.")
                response = SYSTEM_FEEDBACK_TEMPLATE
            else:
                print("[CHAT] Feedback algilandi, dogrudan yanit veriliyor.")
                response = FEEDBACK_TEMPLATE
        elif is_joke_request(user_message):
            print("[CHAT] Mizah/eylem istegi, espri niyetine kilitleniyor.")
            response = random.choice(bot.intents.get('espri', ["Aklıma komik bir şey gelmedi şimdi!"]))
            _last_tag = 'espri'
        elif is_trust_question(user_message):
            # Kimlik/güven sorusu: veritabanı araması veya "bilgi yok" degil,
            # sabit ve dürüst bir tanıtım yanıtı.
            print("[CHAT] Kimlik/guven sorusu, sabit yanit veriliyor.")
            response = TRUST_TEMPLATE
        elif is_repeat_request(user_message) and _last_tag == 'espri':
            print("[CHAT] Konusuz devam istegi, son niyet espri -> yeni fikra.")
            response = random.choice(bot.intents.get('espri', ["Aklıma komik bir şey gelmedi şimdi!"]))
        elif is_advice_question(user_message):
            # Hassas konu (saglik/diyet/hukuk) + kisisel yonlendirme istegi:
            # dataset cevaplarini bile asar, sinir-bilen yanit verir.
            print("[CHAT] Kisisel/karar sorusu, tavsiye siniri yaniti.")
            response = ADVICE_TEMPLATE
        elif bot.can_answer(user_message) and not bot.has_unknown_subject(user_message):
            _tag = bot._classify(user_message)[0]
            response = bot.get_response(user_message)
            _last_tag = _tag
            count = extract_count(user_message)
            if (count and count >= 2 and _tag in COUNT_TAGS
                    and _tag in bot.intents):
                listed = build_counted_list(count, bot.intents.get(_tag, []))
                if listed:
                    print(f"[CHAT] Sayili liste istegi ({count}), '{_tag}' havuzundan derlendi.")
                    response = listed
                elif _tag not in GENERATIVE_EXEMPT:
                    response = generator.generate_response(response)
            elif _tag not in GENERATIVE_EXEMPT:
                response = generator.generate_response(response)
        else:
            print(f"[CHAT] Dataset'e guvenilmedi (guven/ornek filteri), fallback deneniyor: {user_message}")
            response = fallback_answer(user_message)
            _last_tag = None

        print(f"[CHAT] Response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        return jsonify({'response': f'Hata: {str(e)}'})


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json(force=True, silent=True)
        text = data.get('text', '').strip().lower() if data else ''
        text = bot.ascii_normalize(text)
        if not text or len(text) < 2:
            return jsonify({'suggestions': []})
        suggestions = [p for p in all_patterns if text in p or p.startswith(text)]
        suggestions = sorted(suggestions, key=lambda x: x.startswith(text), reverse=True)[:6]
        return jsonify({'suggestions': suggestions})
    except Exception:
        return jsonify({'suggestions': []})


@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model_loaded,
        'corpus_loaded': corpus_loaded,
        'corpus_chunks': len(corpus.chunks) if corpus_loaded else 0,
        'vocabulary_size': len(bot.vocabulary) if model_loaded else 0,
        'intent_count': len(bot.intent_tags) if model_loaded else 0
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  NEXTGEN AI - WEB SERVER")
    print("=" * 50)
    load_bot()
    print("Open: http://localhost:5000")
    print("=" * 50)
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, host='0.0.0.0', port=5000)

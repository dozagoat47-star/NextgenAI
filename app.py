"""
Nextgen AI - Flask Web Server
"""

import sys
import io
import traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify
from brain import ChatBot
from knowledge import fetch_answer
from corpus import Corpus

app = Flask(__name__)

bot = ChatBot()
corpus = Corpus()
model_loaded = False
corpus_loaded = False
all_patterns = []

DEFAULT_UNKNOWN = ("Bu konuda henüz yeterli bilgiye sahip değilim, "
                   "farklı bir şekilde sormak ister misin?")

FACTUAL_MARKERS = ['nedir', 'ne demek', 'hakkinda', 'kimdir', 'kimlerdir', 'nerede',
                   'ne zaman', 'nasil yapilir', 'kac yil', 'tarihi', 'ozetle', 'acikla',
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
    """Dataset cevabina guvenilmezse: corpus -> internet (ogrenerek) -> bilmiyorum."""
    chunk = corpus.search(message)
    if chunk and chunk['score'] >= corpus.min_score:
        print(f"[CHAT] Corpus eslesmesi (%.2f): {chunk['title']}" % chunk['score'])
        return (f"Kütüphanemden buldum: {Corpus.snippet(chunk['text'])}\n"
                f"(Konu: {chunk['title']})")

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
            return (f"Bu konuyu araştırıp hafızama ekliyorum! "
                    f"{knowledge['answer']}\n(Kaynak: {knowledge['title']})")

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

        if bot.can_answer(user_message) and not bot.has_unknown_subject(user_message):
            response = bot.get_response(user_message)
        else:
            print(f"[CHAT] Dataset'e guvenilmedi (guven/ornek filteri), fallback deneniyor: {user_message}")
            response = fallback_answer(user_message)

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

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

app = Flask(__name__)

bot = ChatBot()
model_loaded = False
all_patterns = []

FALLBACK_THRESHOLD = 0.25

FACTUAL_MARKERS = ['nedir', 'ne demek', 'hakkinda', 'kimdir', 'kimlerdir', 'nerede',
                   'ne zaman', 'nasil yapilir', 'kac yil', 'tarihi', 'ozetle', 'acikla',
                   'kaynak', 'wikipedia', 'yapilir misin', 'verebilir misin']


def is_factual_query(text):
    t = bot.ascii_normalize(text.lower())
    if any(m in t for m in FACTUAL_MARKERS):
        return True
    return t.count('?') > 0 and any(w in t for w in
                                   [' ne ', ' kim ', ' nerede ', ' nasil ', ' kac ', ' hangi ', ' neden '])


def load_bot():
    global bot, model_loaded, all_patterns
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

        _, probability = bot.get_probability(user_message)

        if is_factual_query(user_message) and (
                probability < FALLBACK_THRESHOLD
                or (probability < 0.7 and bot.keyword_strength(user_message) < 1.5)):
            print(f"[CHAT] Bilgi sorusu, dusuk guven (%.2f), internetten araniyor..." % probability)
            knowledge = fetch_answer(user_message)
            if knowledge:
                response = f"İnternette buldum: {knowledge['answer']}\n(Kaynak: {knowledge['title']})"
            else:
                response = bot.get_response(user_message)
        else:
            response = bot.get_response(user_message)

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

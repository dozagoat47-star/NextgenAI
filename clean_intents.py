"""
Nextgen AI - Veri Hijyeni Aracı
===============================
autogrow'un Wikipedia'dan topladigi ham verideki kalite/guvenlik sorunlarini
temizler ve yeniden girmesini engellemek icin autogrow'a yeniden kullanilabilir
yardimci fonksiyonlar saglar.

Duzenlenen sorunlar:
  1. Tag'lar ASCII'ye normalize edilmez (Turkce karakter + apostrof karisimi,
     birlesik nokta U+0307) -> ayni konu farkli tag'lara bolunuyor.
  2. Yanitlara Latin disi yazim bloklari karisir (Yunanca/Rusca/Arapca/
     Japonca ozgun adlar) -> chat'te sacma goruntu.
  3. "d." / "(" gibi kesik ozet cumleleri (truncated stub) yanit olur.
  4. [skip ci] ile calisan harmfull/bozuk tagler ("seri katil" biyografileri vs.)
  5. corpus.jsonl'de ayni kirli metinler birikir.

Kullanim:
    python clean_intents.py            # intents.json + corpus.jsonl temizler
    python clean_intents.py --no-corpus  # yalnizca intents
"""

import os
import io
import re
import sys
import json
import argparse
import datetime
import unicodedata


# Latin (Turkce dahil) ozel harflerin ASCII karsiliklari. Ayristirilamayan
# harfler (eszett, ligatur vb.) dogrudan eslenir; kalan aksanlar NFD ile
# sokulur. Boylesi autogrow'un ekledigi yabanci adlar ('prevert') da ASCII
# olur ve 'prévert' yazilsa dahi eslesir.
_LATIN_TO_ASCII = {
    'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
    'â': 'a', 'î': 'i', 'û': 'u', 'i': 'i', 'o': 'o', 'u': 'u',
    'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
    'Â': 'a', 'Î': 'i', 'Û': 'u',
    'ß': 'ss', 'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE',
    'ð': 'd', 'Ð': 'D', 'ø': 'o', 'Ø': 'O', 'ł': 'l', 'Ł': 'L',
    'þ': 'th', 'Þ': 'TH',
    '\u0307': '',
}
_LATIN_TRANSLATE = str.maketrans(_LATIN_TO_ASCII)

if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(SCRIPT_DIR, 'intents.json')
CORPUS_FILE = os.path.join(SCRIPT_DIR, 'corpus.jsonl')

# Latin disi yazim bloklari (ozgu n adlar icin 2+ karakterlik bloklar).
FOREIGN_SCRIPT = re.compile(
    '[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0400-\u04FF\u0500-\u052F'
    '\u0370-\u03FF\u1F00-\u1FFF\u4E00-\u9FFF\u3040-\u30FF'
    '\uAC00-\uD7AF\u10A0-\u10FF\u0590-\u05FF]+')

# "Bilal Mazhar (Arapca: ...) (d." gibi kesik/yarim kalan biyografi ozetleri.
TRUNCATED_TAIL = re.compile(r'\(\s*d\.?\s*$|\(\s*$')

# Guvenlik: konusu dogrudan zarar verici/bozuk icerik olan tagler (tam eslesme
# degil, alt dize aramasi). Bot bu konularda bilgi vermeyi reddetmesin butun
# olarak; yalnizca biyografi/kronoloji turu vasati scrap iclerin temizlenmesi.
HARMFUL_TAG = [
    'seri katil', 'suikast', 'katliam', 'olumu', 'oldurulmes', 'cinayet',
    'tecavuz', 'intihar',
]

MIN_TOKENS = 3


def ascii_normalize(text):
    """Turkce ozel karakterler ve birlesik nokta ASCII'ye cevrilir; ayrica
    yabanci aksanlar (e-acute, o-macron, eszett...) da sokulur."""
    t = text.translate(_LATIN_TRANSLATE)
    if any(ord(c) > 127 for c in t):
        t = ''.join(c for c in unicodedata.normalize('NFD', t)
                    if not unicodedata.combining(c))
    return t


def strip_foreign_scripts(text):
    """Latin disi yazim bloklarini metinden temizler, dolguyu birlestirir."""
    if not text:
        return text
    t = FOREIGN_SCRIPT.sub(' ', text)
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t


def is_truncated(text):
    """Kesik/yarim kalan ozet cumlesi mi? (ornek: '(d.', '(', '135' )"""
    t = text.strip()
    if not t or len(t) < 5:
        return True
    if TRUNCATED_TAIL.search(t):
        return True
    # Yalnizca tek/tarih parcalarindan olusan kisa kirpintilar
    if len(t.split()) <= 1:
        return True
    return False


def is_harmful_tag(tag):
    norm = ascii_normalize((tag or '').lower())
    return any(k in norm for k in HARMFUL_TAG)


def clean_responses(responses):
    """Yanit havuzunu kirpinti/yabanci-yazim/zararli icerikten arindirir."""
    seen = []
    kept = []
    for r in responses or []:
        r = (r or '').strip()
        r = strip_foreign_scripts(r)
        r = re.sub(r'\s{2,}', ' ', r).strip()
        if is_truncated(r):
            continue
        if len(r.split()) < MIN_TOKENS:
            continue
        key = ascii_normalize(r.lower())
        if key in seen:
            continue
        seen.append(key)
        kept.append(r)
    return kept


def clean_intent(intent):
    """Tek intent'i tag normalize + yanit temizligiyle yeniden uretir."""
    tag = ascii_normalize((intent.get('tag') or '').strip().lower()).strip()
    patterns = [ascii_normalize(p.strip()) for p in (intent.get('patterns') or [])]
    patterns = [p for p in patterns if p]
    responses = clean_responses(intent.get('responses'))
    return {
        'tag': tag,
        'patterns': list(dict.fromkeys(patterns)),
        'responses': responses,
        '_source': intent.get('source', intent.get('tag', '')),
    }


def merge_by_tag(intents):
    """Normalize edilmis tag'a gore birlesitr (duplike konular tek olur)."""
    merged = {}
    for it in intents:
        key = it['tag']
        if not key:
            continue
        if key not in merged:
            merged[key] = {'tag': key, 'patterns': list(it['patterns']),
                           'responses': list(it['responses'])}
        else:
            for p in it['patterns']:
                if p not in merged[key]['patterns']:
                    merged[key]['patterns'].append(p)
            for r in it['responses']:
                if r not in merged[key]['responses']:
                    merged[key]['responses'].append(r)
    return list(merged.values())


def clean_intents_file(filepath):
    data = json.load(io.open(filepath, 'r', encoding='utf-8'))
    intents = data['intents']

    before = len(intents)
    harmful = [it['tag'] for it in intents if is_harmful_tag(ascii_normalize((it.get('tag') or '').strip().lower()))]
    intents = [it for it in intents if not is_harmful_tag(ascii_normalize((it.get('tag') or '').strip().lower()))]

    cleaned = [clean_intent(it) for it in intents]
    before_resp = sum(len(it.get('responses') or []) for it in intents)

    cleaned = merge_by_tag(cleaned)
    cleaned = [it for it in cleaned if it['patterns'] and it['responses']]

    after_resp = sum(len(it.get('responses') or []) for it in cleaned)
    dropped = before - len(cleaned)

    data['intents'] = cleaned
    with io.open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INTENTS] {before} -> {len(cleaned)} intent ({dropped} dusen)")
    print(f"[INTENTS] Yanit havuzu: {before_resp} -> {after_resp}")
    if harmful:
        print(f"[INTENTS] Zararli icerik tagleri temizlendi: {len(harmful)}")
        for h in harmful[:10]:
            print(f"   - {h}")
    return len(cleaned)


def clean_corpus_file(filepath):
    if not os.path.exists(filepath):
        print("[CORPUS] dosya yok, atlaniyor.")
        return
    lines = []
    before = 0
    kept = 0
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            before += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = strip_foreign_scripts(rec.get('text', ''))
            text = re.sub(r'\s{2,}', ' ', text).strip()
            title = ascii_normalize(rec.get('title', '') or '').strip()
            if is_truncated(text) or len(text.split()) < 10:
                continue
            rec['title'] = title
            rec['text'] = text
            rec['id'] = ascii_normalize((rec.get('id') or title.lower())).replace(' ', '_')
            lines.append(rec)
            kept += 1
    with io.open(filepath, 'w', encoding='utf-8') as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"[CORPUS] {before} -> {kept} parca")


def main():
    parser = argparse.ArgumentParser(description='Nextgen AI veri temizligi')
    parser.add_argument('--no-corpus', action='store_true',
                        help='corpus.jsonl temizligini atla')
    parser.add_argument('--intents', type=str, default=INTENTS_FILE,
                        help='temizlenecek intents dosyasi (default: intents.json)')
    args = parser.parse_args()

    print("=" * 50)
    print("  NEXTGEN AI - VERI TEMIZLIGI")
    print("=" * 50)

    clean_intents_file(args.intents)
    if not args.no_corpus:
        clean_corpus_file(CORPUS_FILE)

    print("=" * 50)
    print("  SIRA: python train.py --epochs 3000")
    print("=" * 50)


if __name__ == '__main__':
    main()
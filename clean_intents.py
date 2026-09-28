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

from normalize import ascii_normalize

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

# Etiket icin asgari karakter sayisi. Tek/iki harfli basliklar
# (Wikipedia'nin harf maddeleri: L, W, I) konu degildir. autogrow da
# ayni esigi uygular; test_core.TestIntentsSchema en az 2 istiyor.
MIN_TAG_CHARS = 3


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


# ASCII OLMAYAN HARF tespiti. test_core.TestIntentsSchema.test_tags_latin'in
# birebir kurali: c.isalpha() and ord(c) > 127. Rakam, tire, nokta,
# alt cizgi ve bosluk serbest (tag'lar 'call of duty 4: modern
# warfare' gibi olabiliyor).
#
# NEDEN GEREKIYOR: ascii_normalize once _LATIN_TRANSLATE tablosunu
# uygular, sonra NFD ayristirmasiyla aksanlari atar. Ama bazi Latin
# HARFLERI ayristirilamaz, dolayisiyla gecirilirler:
#     U+01C1 (tik sesi, 'ǁkaras bolgesi')
#     U+0111 (Bosna-Hersek d'si, 'đakovo')
#     U+02BB (okina, 'liliʻuokalani')
# Bunlar Wikipedia basliklarindan geldi ve intents.json'a yazildi;
# test_core o intent'i elendi diye egitim oncesi kalkan dustu.
def non_latin_letters(tag):
    """Etiketteki ASCII disi harfler (bos liste = temiz)."""
    return sorted({c for c in (tag or '')
                   if c.isalpha() and ord(c) > 127})


def is_latin_tag(tag):
    """Etiket yalnizca ASCII harf iceriyor mu (test_core'in istedigi)."""
    return not non_latin_letters(tag)


def ascii_letters_only(tag):
    """Etiketten ASCII disi harfleri at, sonra bosluklari topla.

    Etiketi OLDUGU GIBI birakmak yerine duzeltmek tercih edilir:
    Wikipedia'nin 'Dakovо' maddesi 'dakovo' olarak kazanilir. Yine de
    sonuc bos veya Latin disi harf iceriyorsa None doner - kapidan
    gecmemis olur.
    """
    if tag is None:
        return None
    t = ''.join(c for c in tag if not (c.isalpha() and ord(c) > 127))
    t = re.sub(r'\s{2,}', ' ', t).strip()
    if not t or not is_latin_tag(t):
        return None
    return t


def normalize_tag_latin(intents):
    """Intent listesini Latin olmayan harfli etiketten arindirir.

    Etiket duzeltilebiliyorsa DUZELTILIR ve korunur; duzeltilemiyorsa
    dusurulur. Boylece uretici kendi ciktisini kendi kendine
    iyilestirir: intents.json'a elle mudahale gerekmez.

    Returns:
        (temizlenmis_liste, duzeltilen_etiketler, dusurulen_etiketler)
    """
    temiz = []
    duzeltilen = []
    dusurulen = []
    for it in intents:
        eski = it.get('tag', '')
        yeni = eski if is_latin_tag(eski) else ascii_letters_only(eski)
        if yeni is None or len(yeni) < MIN_TAG_CHARS:
            if not is_latin_tag(eski):
                dusurulen.append((eski, non_latin_letters(eski)))
            continue
        if not is_latin_tag(eski):
            duzeltilen.append((eski, yeni))
        # Duzeltilmis ad zaten kullanimda olabilir: AutoGrow ayni konuyu
        # yeni isimle uretmis olabilir ya da iki kotu etiket ayni konuya
        # (ornegin 'đakovo' ve 'akovo') donusebilir. Iki yolda da tek
        # intent kalmali.
        var = next((x for x in temiz if x.get('tag') == yeni), None)
        if var is None:
            if yeni == eski:
                temiz.append(it)
            else:
                kopya = dict(it)
                kopya['tag'] = yeni
                temiz.append(kopya)
        else:
            for p in it.get('patterns') or []:
                if p not in var['patterns']:
                    var['patterns'].append(p)
            for r in it.get('responses') or []:
                if r not in var['responses']:
                    var['responses'].append(r)
    return temiz, duzeltilen, dusurulen


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
    """Tek intent'i tag normalize + yanit temizligiyle yeniden uretir.

    Etiket iki adimdan gecer:
      1) ascii_normalize  -> Turkce/yabanci aksan (NFD)
      2) ascii_letters_only -> AYRISAMAYAN Latin harfler (U+01C1, U+0111,
         U+02BB ...). 1. adim bunlari gecirir; 2. adim olmazsa kapidan
         sonra duzer ve konu KAYBOLUR. Etiket tamamen harf disi bir seye
         duserse bos string doner ve merge_by_tag eler; o durumda zaten
         kullanilabilir bir konu adi kalmamistir.
    """
    ham = ascii_normalize((intent.get('tag') or '').strip().lower()).strip()
    tag = ascii_letters_only(ham) or ''
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
    """Normalize edilmis tag'a gore birlesitr (duplike konular tek olur).

    Cok kisa etiketler elenir: Wikipedia'nin harf maddeleri (L, W, I)
    konu degildir ve 'l nedir' gibi kalipsiz uretir. intents.json'a
    tag='l' girmisti; egitim oncesi kalkan olan
    test_core.TestIntentsSchema (tag >= 2) bunu yakaladi.
    """
    merged = {}
    for it in intents:
        key = it['tag']
        if not key or len(key.strip()) < MIN_TAG_CHARS:
            continue
        if not is_latin_tag(key):
            # Latin disi harfli etiket girmez (test_core kalkani).
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


def clean_corpus_file(filepath, max_drop=0.05, force=False):
    """Korpusu temizler; TEHLIKELI kayipta dosyaya dokunmadan durur.

   Uyari sebebi: 10 kelimeden kisa metin elenir ve korpusun %27'si bu
    esigin altinda. Bu script elle calistirildiginda 129.577 -> 94.662
    yazarak 34.915 parca SILDI. Kayip sessizdi; sadece sayilar
    yaziliyordu.

    Simdi esik asilirsa (varsayilan %5) hicbir sey yazilmaz, komut
    basarisiz biter ve --force ile bilerek gecilebilir.
    """
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

    if before and not force:
        kayip = (before - kept) / before
        if kayip > max_drop:
            pct = 100 * kayip
            print(f"[CORPUS] UYARI: %{pct:.1f} kayip ({before} -> {kept} parca). "
                  f"Esik %{100 * max_drop:.0f}. Dosya DEGISTIRILMEDI.")
            print("[CORPUS] Bu genelde beklenmedik bir esiktir; once "
                  "nedenini anla, sonra --force ile bilerek calistir.")
            raise SystemExit(
                f'corpus temizligi iptal: korpusun %{pct:.1f} kaybi '
                f'({100 * max_drop:.0f} esigini) asiyor')

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
    parser.add_argument('--max-corpus-drop', type=float, default=0.05,
                        help='bu kayiptan fazlasi olursa corpus YAZILMAZ '
                             '(varsayilan 0.05 = %%5). 0.0 = esik yok.')
    parser.add_argument('--force', action='store_true',
                        help='tehlikeli kayip esigini gec (bilerek)')
    args = parser.parse_args()

    print("=" * 50)
    print("  NEXTGEN AI - VERI TEMIZLIGI")
    print("=" * 50)

    clean_intents_file(args.intents)
    if not args.no_corpus:
        clean_corpus_file(CORPUS_FILE, max_drop=args.max_corpus_drop,
                          force=args.force)

    print("=" * 50)
    print("  SIRA: python train.py --epochs 3000")
    print("=" * 50)


if __name__ == '__main__':
    main()
"""
Nextgen AI - Dogal Turkce Yanit Parafrazci
==========================================
Canned (ezber) Turkce cevaplari dogal sohbet diline donusturur.
Her cevap icin birden fazla dogal varyant ureterek egitim verisi
cesitliligini ve dolayisiyla modelin dogal konusma yetenegini artirir.

Kullanim:
    from naturalize import natural_variants, naturalize_pairs
    variants = natural_variants("merhaba ben nextgen asistaniyim", k=3)

Tum metin dogal (gercek) Turkce imla ile uretilir: o/u/s/g/c gibi karakterler
dosyalanir; kucuk harf donusumu Turkce kurallaridir (I->i degil I->u). Bu uzay
bpe.py'nin (subword tokenizer) metin uzayiyla birebir aynidir.
"""

import random
import re

# ---------------------------------------------------------------------------
# Dogal Turkce yapi taslari (ASCII-only)
# ---------------------------------------------------------------------------

# Cumle acicilari (baslangic eklemeleri)
OPENERS = [
    "", "", "", "", "", "", "", "", "",       # cogunlukla acici yok
    "bak, ",
    "şöyle: ",
    "aslında, ",
    "yani, ",
    "hmm, ",
    "kısaca söyleyeyim: ",
    "doğrudan söyleyeyim: ",
    "bu konuda söyleyebilecegim sey şu: ",
    "sana şöyle anlatayım: ",
    "şu şekilde düşünebilirsin: ",
    "güzel bir soru. ",
    "kesinlikle. ",
    "tabii, ",
]

# Cumle icinde kullanilabilir doldurucular (aslinda bunlari sonda kullaniyoruz)
FILLERS = [
    "", "", "", "", "", "", "",                # cogunlukla doldurucu yok
    "yani ",
    "şöyle ki ",
    "bir bakıma ",
    "kısaca ",
]

# Ikinci cumle basina baglama kelimeleri (dogal okunur olanlar)
CONNECTORS = [
    "", "",                                     # cogunlukla baglayici yok
    "ayrıca, ",
    "üstelik, ",
    "diğer taraftan, ",
    "buna ek olarak, ",
    "kısacası, ",
    "özellikle, ",
]

# Soru-sonrasi kapanislar
CLOSERS_Q = [
    "", "", "",
    " ne dersin?",
    "değil mi?",
    " biliyor muydun?",
    " hiç düşündün mü?",
]

# Bilgi/onermeler icin kapanislar
CLOSERS_STMT = [
    "", "", "", "",
    " umarım yardımcı olmuştur.",
    " başka bir şey merak ediyor musun?",
    " daha fazla detay ister misin?",
    " istersen üstüne konuşalım.",
    " devam edelim mi?",
]

# Kelime esanlamli varyantlari (yuksek dogallik, dusuk risk)
WORD_ALT = {
    "büyük": ["geniş", "önemli"],
    "güzel": ["harika", "muhteşem"],
    "kötü": ["hoş olmayan", "olumsuz"],
    "var": ["mevcut", "bulunmaktadır"],
    "yok": ["mevcut değil", "bulunmamaktadır"],
    "gerekli": ["lazım", "şart"],
    "çok": ["oldukça", "son derece"],
    "sonra": ["ardından", "daha sonra"],
}


def _stable_hash(text):
    """Pythondaki rastgelelesirilmis hash() yerine deterministik hash."""
    h = 5381
    for ch in text:
        h = ((h * 33) + ord(ch)) & 0x7FFFFFFF
    return h


def _seeded_random(seed, salt):
    """Deterministik sifreleme: ayni girdi her zaman ayni rng."""
    return random.Random(seed * 100000 + salt)


def _split_sentences(text):
    """Metni cumlelere ayir (nokta/ünlem/soru isareti bazli)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _join_sentences(sents):
    """Cumleleri tekrar birlestir (aralarina nokta + bosluk)."""
    if not sents:
        return ""
    return ' '.join(s.rstrip('.!?') + '.' for s in sents if s)


def _maybe_fill(sentence, rng, prob):
    """Cumle basina dogal bir doldurucu ekle."""
    if not sentence or rng.random() > prob or not FILLERS:
        return sentence
    f = rng.choice(FILLERS)
    if not f:
        return sentence
    return f + sentence[0].lower() + sentence[1:]


def _maybe_connector(sents, rng, prob):
    """Ikinci cumle basina baglama ekle."""
    if len(sents) < 2 or rng.random() > prob or not CONNECTORS:
        return sents
    c = rng.choice(CONNECTORS)
    if not c:
        return sents
    sents[1] = c + sents[1][0].lower() + sents[1][1:]
    return sents


def _maybe_swap(sentence, rng, prob):
    """Tek kelimelik esanlamli varyant uygula.

    Ilk kelimeye (i==0) dokunulmaz: modelin ilk-yanit-token'i kosullu
    ogrenebilmesi icin varyantlarin ilk token'i orijinalle ayni kalmalidir.
    """
    if not sentence or rng.random() > prob:
        return sentence
    words = sentence.split()
    if len(words) < 2:
        return sentence
    for i in range(1, len(words)):
        w = words[i]
        base = w.rstrip('.,!?;:')
        suf = w[len(base):]
        if base in WORD_ALT and rng.random() < 0.5:
            words[i] = rng.choice(WORD_ALT[base]) + suf
            return ' '.join(words)
    return sentence


_MID_FILLERS = [
    ", yani ",
    ", kısaca ",
    ", şöyle ki ",
    ", bir bakıma ",
    "; üstelik ",
    "; ayrıca ",
    ", aynı zamanda ",
]


def _maybe_insert_mid(sentence, rng, prob=0.35):
    """Uzun cumlenin ortasina dogal bir doldurucu/arasoz ekle.

    Kisa cumlelere dokunmaz (dogal aksani bozar). Uzun bilgi cumlelerini
    iceriden degistirerek gercek bir varyant yaratir; ekleme noktasi dogal
    bir virgul/hizlandirici siniridir ve cumlenin %30-85'i arasina dusturulur
    (cumle basina takilmis '; ustelik' gibi garip okumalar engellenir).
    """
    if len(sentence) < 60 or rng.random() > prob:
        return sentence
    lower = sentence.lower()
    candidates = [m.start() for m in re.finditer(r',\s', lower)]
    lo = int(len(sentence) * 0.30)
    hi = int(len(sentence) * 0.85)
    candidates = [p for p in candidates if lo <= p <= hi]
    if not candidates:
        # virgul yoksa: kelime bazli orta bolgeyi sinirla (SLICE index'i ile)
        words = sentence.split()
        if len(words) < 8:
            return sentence
        cut_from = max(2, len(words) // 3)
        cut_to = max(3, len(words) // 2)
        if cut_from >= cut_to or cut_from >= len(words):
            return sentence
        b4 = ' '.join(words[:cut_from])   # slice konumu, kelime icerigi degil
        pos = len(b4)                     # burada bir bosluk karakteri var
    else:
        pos = rng.choice(candidates)
    if not (0 < pos < len(sentence) - 1):
        return sentence
    f = rng.choice(_MID_FILLERS)
    # Komadan sonra / kelime sinirinda dogal sekilde birakir; karakter yemez.
    return sentence[:pos] + f.rstrip() + ' ' + sentence[pos + 1:]


def _maybe_open(sentence, rng, prob):
    """Cumle basina acici ekle."""
    if not sentence or rng.random() > prob or not OPENERS:
        return sentence
    o = rng.choice(OPENERS)
    if not o:
        return sentence
    if o.endswith(' '):
        return o + sentence
    return o + sentence


def _maybe_close(text, is_question, rng, prob):
    """Metin sonuna kapanis ekle."""
    pool = CLOSERS_Q if is_question else CLOSERS_STMT
    if rng.random() > prob or not pool:
        return text
    c = rng.choice(pool)
    if not c:
        return text
    return text.rstrip() + c


def natural_variants(response, k=3, seed=42):
    """Canned Turkce cevabin k dogal varyantini uretir.

    Her varyant ayni icerigi farkli yuzey formuyla ifade eder ve modelin
    egitiminde gordugu formata uyar (ASCII, kucuk harf). Orijinal cevap da
    ilk varyant olarak korunur (model ezberi de gorsun).

    ILK TOKEN KURALI: Tum varyantlar orijinalin ilk token'iyla baslar
    (cumle basi acici/doldurucu ile 2. cumle ve sonrasinda oynanir). Bu,
    modelin 'soru -> ilk yanit kelimesi' iliskisini ogrenmesini saglar;
    baslangici rastgele degismis yanitlar ezbere dolgu uretimine yol aciyordu.

    Args:
        response: Ham (ya da temizlenmis) cevap metni.
        k: Uretilecek varyant sayisi (orijinal dahil).
        seed: Deterministik uretim tohumu.

    Returns:
        list[str]: Dogal varyantlar (1..k adet).
    """
    if not response or not response.strip():
        return []

    from bpe import clean_text as _bpe_clean
    text = _bpe_clean(response)
    if len(text) < 6:
        return [text]

    base_hash = _stable_hash(text)
    sentences = _split_sentences(text) or [text]
    is_question = text.rstrip().endswith('?')
    first_word = text.split()[0].strip('.,!?;:') if text.split() else ''

    variants = [text]
    for vi in range(1, k):
        # Ayni varyant tumayi demolar icin farkli tuz ile yeniden dene
        body = None
        for attempt in range(6):
            rng = _seeded_random(seed, base_hash + vi * 7 + attempt * 13)

            sents = list(sentences)
            # Ilk token ORIGINALIN ilk token'i ile ayni kalir: acici/doldurucu/
            # esanlamli degisim SADECE 2. cumle ve sonrasina uygulanir. Boylece
            # model 'soru -> ilk yanit kelimesi' kosullu dagilimini ogrenebilir
            # (cok cesitli baslangic, ezbere filier uretimine yol aciyordu).
            sents = [_maybe_fill(s, rng, 0.12) if i > 0 else s
                     for i, s in enumerate(sents)]
            sents = _maybe_connector(sents, rng, 0.25)
            sents = [s if i == 0 else _maybe_open(s, rng, 0.40)
                     for i, s in enumerate(sents)]
            wp = 0.40 if len(sents) <= 2 else 0.20
            sents = [_maybe_swap(s, rng, wp) for s in sents]
            sents = [_maybe_insert_mid(s, rng, 0.40) for s in sents]

            body = _join_sentences(sents)
            body = _maybe_close(body, is_question, rng, 0.30)
            body = ' '.join(body.strip().split())
            head = body.split()[0].strip('.,!?;:') if body.split() else ''
            if body and head == first_word and body not in variants:
                break
            body = None
        if body:
            variants.append(body)

    return variants[:k]


def naturalize_pairs(pairs, k=2, seed=42):
    """(context, response) ciftlerini dogal varyantlarla genisletir.

    Her (ctx, resp) cifti, resp'nin dogal varyantlariyla k cift uretir
    (orijinal de dahil). Ayni sorgu icin cok cesitli dogal yanitlarla
    kosullu egitim saglanir.

    Args:
        pairs: [(context, response), ...]
        k: Varyant sayisi (orijinal dahil).
        seed: Deterministik tohum.

    Returns:
        list[(context, response)]: Genisletilmis cift listesi.
    """
    out = []
    for ci, (ctx, resp) in enumerate(pairs):
        for v in natural_variants(resp, k=k, seed=seed + ci):
            out.append((ctx, v))
    return out


# ---------------------------------------------------------------------------
# Hizli test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import io
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    examples = [
        "merhaba, ben nextgen asistaniyim. size nasil yardimci olabilirim?",
        "istanbul turkiye'nin en buyuk sehridir ve nufusu yaklasik 16 milyondur.",
        "hayir, bu konuda bilgim yok.",
        "evet, tabii ki yardimci olurum!",
    ]
    for ex in examples:
        vs = natural_variants(ex, k=4)
        print(f"\nORIJINAL: {ex}")
        for i, v in enumerate(vs):
            print(f"  V-{i}: {v}")

    # Deterministik kontrol (ayni seed -> ayni cikti)
    a = natural_variants("istanbul turkiye'nin en buyuk sehridir.", k=3)
    b = natural_variants("istanbul turkiye'nin en buyuk sehridir.", k=3)
    print('\ndeterministik:', a == b)
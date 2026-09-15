"""
Nextgen AI - Karakter normalizasyonu (TEK KAYNAK).

Turkce ozel harfler + yabanci aksanlar ASCII karsiligina cevrilir.
Boylece 'ogle' / 'öğle' / 'OĞLE' / 'jacques prevert' / 'jacques prévert'
hepsi ayni torbaya duser. Bu modul hicbir baska module bagimli degildir;
brain, generator, seqgen, clean_intents buradan kullanir.

Latin disi karakterler (eszett, ligaturler, nordic harfleri vb.) NFD ile
ayristirilamadigindan dogrudan eslenir: 'ß' -> 'ss', 'æ' -> 'ae'.
"""

import unicodedata

_LATIN_TO_ASCII = {
    'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
    'â': 'a', 'î': 'i', 'û': 'u', 'i': 'i', 'o': 'o', 'u': 'u',
    'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
    'Â': 'a', 'Î': 'i', 'Û': 'u',
    'ß': 'ss', 'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE',
    'ð': 'd', 'Ð': 'D', 'ø': 'o', 'Ø': 'O', 'ł': 'l', 'Ł': 'L',
    'þ': 'th', 'Þ': 'TH',
    '\u0307': '',  # Python'in 'İ'.lower() ciktisindaki kombinasyon noktasi
}
_LATIN_TRANSLATE = str.maketrans(_LATIN_TO_ASCII)


def ascii_normalize(text):
    """Turkce/yabanci aksani ASCII karsiligina cevirir (buyuk/kucuk korunur)."""
    t = text.translate(_LATIN_TRANSLATE)
    if any(ord(c) > 127 for c in t):
        t = ''.join(c for c in unicodedata.normalize('NFD', t)
                    if not unicodedata.combining(c))
    return t
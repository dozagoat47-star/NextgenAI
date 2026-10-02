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
    'â': 'a', 'î': 'i', 'û': 'u',
    'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'I': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
    'Â': 'A', 'Î': 'I', 'Û': 'U',
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
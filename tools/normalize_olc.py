#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""normalize_olc.py — normalize.ascii_normalize büyük harf bozma etkisini ölçer.

Ölçümler:
1. Korpus token'larında büyük harf bozulması oranı
2. Retrieval tutarlılığı (sorgu + corpus aynı fonksiyondan geçiyor mu?)
3. deasciify sözlüğü eşleşme kaybı
4. Kullanıcıya giden metinlerdeki görsel bozulma
"""

import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from normalize import ascii_normalize


def load_corpus_tokens(corpus_path, limit=None):
    """corpus.jsonl'den token'ları çıkar."""
    tokens = []
    count = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            try:
                rec = json.loads(line)
                text = rec.get('text', '')
                # Basit tokenizasyon
                words = re.findall(r'\b\w+\b', text)
                tokens.extend(words)
                count += 1
            except json.JSONDecodeError:
                continue
    return tokens


def analyze_uppercase_damage(tokens):
    """Token'lardaki büyük harf bozulmasını analiz et."""
    damaged = 0
    total_non_ascii = 0
    damage_details = Counter()

    for token in tokens:
        normalized = ascii_normalize(token)
        # Sadece büyük harf içeren tokenları kontrol et
        has_upper = any(c.isupper() for c in token)
        has_non_ascii = any(ord(c) > 127 for c in token)

        if has_upper and has_non_ascii:
            total_non_ascii += 1
            # Büyük harf kaybı var mı?
            if normalized != normalized.upper() and normalized != normalized.lower():
                # Karışık durum - bazı harfler bozulmuş
                pass

            # Eğer orijinalde büyük harf varsa ama normalize sonrası hepsi küçükse
            if token != token.lower() and normalized == normalized.lower():
                damaged += 1
                damage_details[token] += 1

    return damaged, total_non_ascii, damage_details


def test_retrieval_consistency():
    """Sorgu ve corpus aynı normalize fonksiyonundan geçtiği için tutarlı mı?"""
    test_cases = [
        "İstanbul",
        "İsparta",
        "Irak",
        "FIFA",
        "COVID",
        "NASA",
        "TÜRKİYE",
        "Ankara",
        "ÇESMA",
        "Ğabror",
        "Öğle",
        "Şehir",
        "Üniversite",
    ]

    print("\n=== Retrieval Tutarlılık Testi ===")
    for case in test_cases:
        norm = ascii_normalize(case)
        # Her ikisi de aynı fonksiyondan geçecekleri için
        # sorgu=case, corpus=case -> her ikisi de norm olacak -> EŞLEŞİR
        # sorgu=case.lower(), corpus=case -> her ikisi de norm olacak -> EŞLEŞİR
        print(f"  {case:15s} -> {norm}")

    # Ama deasciify için sorun:
    print("\n=== deasciify Eşleşme Kaybı ===")
    # deasciify sözlüğü "Isparta" -> "İsparta" eşleşmesi bekler
    # ama normalize("İsparta") = "isparta" (küçük)
    # deasciify["isparta"] yok, deasciify["Isparta"] var
    deasciify_examples = {
        "isparta": "İsparta",
        "istanbul": "İstanbul",
        "irak": "Irak",
        "fifa": "FIFA",
    }
    for key, val in deasciify_examples.items():
        norm_val = ascii_normalize(val)
        match = "EVET" if norm_val == key else "HAYIR"
        print(f"  deasciify['{key}'] = '{val}' -> normalize = '{norm_val}' -> Eşleşme: {match}")


def measure_corpus_impact(corpus_path, sample_size=50000):
    """Korpus üzerindeki etkiyi ölç."""
    print(f"\n=== Korpus Etkisi Ölçümü (ilk {sample_size:,} kayıt) ===")

    tokens = load_corpus_tokens(corpus_path, sample_size)
    print(f"Toplam token: {len(tokens):,}")

    damaged, total_non_ascii, damage_details = analyze_uppercase_damage(tokens)

    print(f"Büyük harf + non-ASCII içeren token: {total_non_ascii:,}")
    print(f"Büyük harf tamamen kaybolan (küçülmüş): {damaged:,}")
    if total_non_ascii > 0:
        print(f"Oran: {damaged/total_non_ascii*100:.2f}%")

    print("\nEn çok bozulan token'lar (ilk 20):")
    for token, count in damage_details.most_common(20):
        norm = ascii_normalize(token)
        print(f"  {token:20s} -> {norm:20s} (x{count})")

    # Corpus metinlerindeki görsel etki
    print("\n=== Corpus Metin Örnekleri (normalize sonrası) ===")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            try:
                rec = json.loads(line)
                text = rec.get('text', '')[:200]
                norm_text = ascii_normalize(text)
                print(f"  ORIJINAL: {text}")
                print(f"  NORMALIZE: {norm_text}")
                print()
            except json.JSONDecodeError:
                continue


def main():
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpus.jsonl')

    if not os.path.exists(corpus_path):
        print(f"corpus.jsonl bulunamadı: {corpus_path}")
        return

    test_retrieval_consistency()
    measure_corpus_impact(corpus_path)


if __name__ == '__main__':
    main()
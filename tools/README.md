# Olcum araclari

Bunlar karar araclari, uretim kodu degil. Hepsi **olcer**, hicbiri veri
dosyalarina yazmaz. `eval_llm.py` ve `train_llm.py` modelin kalitesini
olcer; buradakiler **uretim yolunun** ne yaptigini olcer.

Tum raporlar `olcum_raporlari/` altina yazilir (`.gitignore`'da).

## Iki sey ayri tutulur

`eval_llm.py` **capraz dogrulama** setiyle olcer: modelin egitimde hic
gormedigi sorular, altin cevapla karsilastirilir. Guclu ama yapay.

Buradaki araclar **uretim yolunu** olcer: gercek `app.load_bot()` +
`bot.get_response()`, gercek kalite kapisi, gercek best-of-3 ornekleme.

29.09'da ikisi zit sonuc verdi: eval yeni modeli alakada belirgin
iyi bulurken, uretim yolunda yeni model daha az metin geciriyordu.
Nedeni kapinin harf-cesitliligi hatasiydi; duzeltilince fark ortustu.
Ders: **uretim yolunu sadece eval ile olcmek yaniltir.**

## uretim_olc.py - uretim yolunu olcer (ana arac)

```
python tools/uretim_olc.py <etiket> [cikti.json]
```

50 soruyu `knowledge_map.jsonl`'den deterministik secer, her soru icin
uretim denemelerini (best-of-3) kaydeder:

- **deneme kabul orani** - kapidan gecen uretim / toplam deneme
- **ekrana uretim gecen soru** - kullanicinin gercekten modelin cumlesini
  gordugu oran (kalanlar canned'a duser)
- **sadakat** - uretilen metnin icerik kelimelerinden bilgi parcasinda
  gercekten bulunanlarin orani (uydurma olcumu)
- **ret nedenleri** - hangi kapi sarti ihlal edildi, dagilimiyla

`np.random.seed(7)` her soruda tekrar kurulur ve `_external_knowledge`
yereldir (ag yok), yani **iki kosu birebir ayni** sonucu verir. Iki modeli
karsilastirmak icin bu belirleyicilik sarttir.

## uretim_karsilastir.py - iki uretim raporunu paired karsilastirir

```
python tools/uretim_karsilastir.py <A.json> <B.json>
```

Fark **A - B** yazilir (eval_llm.py ile ayni yon). Dort eksen: ekrana
uretim gecen soru (ikili + McNemar), sadakat (paired t), deneme basi
kabul, ret nedenleri farki.

## kapi_ab.py - kalite kapisini iki kuralla karsilastirir

Harf-cesitliligi kuralinin eski (`0.30 * harf`) ve yeni
(`min(12, 0.30 * harf)`) halini AYNI sorularda uygular. Bu arac
29.09'daki kapı hatasini buldu: eski kural Turkce alfabe 29 harf oldugu
icin 96 karakterden sonra matematiksel olarak imkansizdi, yani 97-260
karakter araliginin tamami reddediliyordu.

## kendi_cumlesi.py - model kendi cumlesini mi kuriyor

Her yanit icin yolu ayirt eder: canned (yapistirma) mi, uretilmis mi?
Asil soru buydu; `uretim_olc.py` bunun daha genis hali.

## kalite_olc.py - gecikme, yarim kelime, yapistirma

Dort seyi sayiya cevirir: yanit suresi ve asamasi, uretimin `max_len`'de
kesilip kesilmedigi (token sayisi == max_len ise kesilmis demektir),
yapistirma orani.

## token_olc.py - TOKEN_PER_PAIR'i bagimsiz olcer

`train_llm.py` icindeki TOKEN_PER_PAIR sabitinin gercekten ortalama
token sayisi olup olmadigini dogrular. Bu arac 29.09'daki dongusel
olcum hatasini yakaladi: log `n * TOKEN_PER_PAIR` yaziyordu, yani
"olcum" kendi varsayimini tekrarliyordu.

## Kurallar

- Bu araclar **veri dosyalarina dokunmaz** (`intents.json`,
  `knowledge_map.jsonl`, `corpus.jsonl`, `corpus_ids.jsonl`).
  `app.load_bot()` corpusu yukler ama yazmaz; yazan yol
  `fallback_answer`'in ogrenme yoludur ve bu araclar onu **cagirmaz**.
- `eval_llm.py` `train_llm.py`'yi import eder: olcum surerken
  `train_llm.py`'ye dokunma, A/B'yi kirletir.
- Karsi lastirmada `--compare-key` sadece t-testinin olculdugu metrigi
  secer; farkli decoding ayari olan raporlar uyariyla reddedilir.

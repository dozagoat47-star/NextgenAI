# Devam Promptu — Nextgen AI (Nextgen_API)

> **Bu dosya bir yapay zekâya devam promptu olarak verilir.**
> Aşağıdaki "KISA PROMPT" bölümünü yeni sohbete yapıştır, sonra bu dosyayı
> okumasını söyle. Geri kalanı senin için.

---

# KISA PROMPT (yeni sohbete yapıştır)

```
Bu repo için devam ediyoruz. Önce DEVAM_PROMPTU.md dosyasının TAMAMINI oku
(bash: cat DEVAM_PROMPTU.md). İçindeki kurallara, ölçülen değerlere ve
kısıtlara uy. Dosyada yazılı olan her şeyi kanıt kabul et, kendi tahminini
ondan önce koyma. Başladığında ilk işi dosyanın "Sıradaki adım" bölümünden al.
```

---

# 1. PROJE

**Nextgen AI** — sıfırdan yazılmış, **NumPy-only** (hazır ML kütüphanesi yok)
Türkçe sohbet asistanı. Web arayüzü Flask (`app.py`, port 5000).

| | |
|---|---|
| Çalışma dizini | `C:\Users\cxc\Desktop\Nextgen_API` |
| Git remote | `https://github.com/dozagoat47-star/NextgenAI.git` |
| GitHub hesabı | **dozagoat47-star** |
| Branch | `main` |
| HEAD = origin/main | `11bac35` (çalışma ağacı temiz) |
| Son 5 commit | `11bac35`, `af0bd23`, `4cf4d12`, `7f2d110`, `bd63543` |
| Bağımlılıklar | `requirements.txt`: numpy>=1.24, flask>=2.3, requests>=2.31, openpyxl>=3.1 |
| Disk | 2,1 gigabayt → **1,47 gigabayt** (temizlik sonrası); `model/` 329 megabayt |
| Veri büyüyor (CI) | 29.09: 11.149 intent · 29.09 sonrası: **12.730 intent**, kb-map **39.983** desen, eğitim çifti ~1,09 milyon |
| **Çalışma tarzı** | Kodu elle düzelt, tahminle atlama; iddiasının sayısı olsun. Karar vermeden önce ölç, ölçtüğünü payla. Belirsizlikte sor. |

## Mimari (README'den)

| Katman | Dosya | Görev |
|---|---|---|
| Sınıflandırıcı | `transformer.py`, `brain.py` | Transformer encoder (NumPy) — sohbet niyetlerini sınıflandırır |
| Bilgi arama | `corpus.py`, `knowledge.py` | RAG-lite: LSA/SVD + PPMI + BM25; bilinmeyen soru Wikipedia'dan |
| Üretim | `generator.py`, `seqgen.py`, `seq2seq.py` | Anchor-sabit parafraz + koşullu LSTM üreteç + Seq2Seq transformer |
| Öğrenme | `finetune.py` | LoRA adaptörüyle anlık intent ekleme/çıkarma (`/learn`, `/forget`) |
| Sunucu | `app.py` | Flask web arayüzü, yönlendirme kuralları |

İki katmanlı çalışma prensibi:
- **Sohbet niyetleri** (deseni >6 olan intentler) transformer ile sınıflandırılır.
- **Bilgi niyetleri** (Wikipedia şablonlu 750+) IDF anahtar-kelime retrieval ile.
- Model güven veremezse `corpus` → internet (Wikipedia) fallback'i; internetten gelen
  bilgi `corpus.jsonl`'a kaydedilir.

Bu projedeki asıl dil modeli **`train_llm.py` + `llm.py`** (BPE + encoder-decoder
transformer, yine NumPy). `kaggle_start.sh` bunu eğitir.

## API uçları (`app.py`)

```
POST /chat    {"message": "..."}              -> {"response": "..."}
POST /learn   {"entries": [{tag,patterns,responses}]}   LoRA ile öğretir
POST /forget  {"tag": "..."}                   LoRA intentini geri alır
GET  /status  model/corpus durumu
```

---

# 2. EN ÖNEMLİ KURAL — TAHMİN YAPMA, ÖLÇ

Bu kullanıcı için kural numara 1. Her iddianın arkasında **sayı** olmalı.

- "Kötü", "iyi", "daha iyi oldu", "muhtemelen", "sanırım" deme. Say, ölç, karşılaştır.
- Bir değişikliğin işe yarayıp yaramadığını **o değişiklikten bağımsız** bir ölçümle tespit et.
- Kod okuyarak sonuç çıkarma. Kod oku → **çalıştır ve ölç** → sonra yorumla.
- Nedenini bildiğini sanma. Nedenini bilmiyorsan önce ölçüm kur.
- Ölçtüğün şeyi **paydayla birlikte** yaz. Yanlış payda oranı kat kat düşürür.
- Test etmediysen "düzelttim" deme. Çalıştır, çıktıyı oku, sonra bildir.
- Ölçüm sana ters düşerse **ölçümü kabul et**, tahmini değil. 29.09'da iki kez oldu:
  - "Kitap hattı iyi veriyi israf ediyor" dedim → byte-aynı çıktı, ölçüm beni düşürdü.
  - "RAG %2,7, veri yok" dedim → gerçek **%52,8**, ölçüm hatasıydı.
    İkisi de tahmindi, ikisi de ölçümle çürütüldü.
- Rastgele tahmin yaptığını fark edersen **açıkça söyle** ve düzelt. Suçlayıcı ton
  kullanma; ölçüme dön.

## Karar kuralı (sonuçlara bakmadan ÖNCE sabit — 29.09'da belirlendi)

Varsayılan eşik: akıcılık **düşmez** + altın kapsama en fazla **0,015** düşer +
fark paired t-testinde anlamlı (**|t| ≥ 2**). Bu, karşılaştırmaları tutarlı
tutmak için — tek başına yasak değil, ölçülen şeyi nasıl değerlendireceğimiz.

**Ama ölçüm sana başka bir şey gösteriyorsa kuralı değiştirebilirsin** — yeter
ki değişikliği gerekçelendir, güncelle ve dosyaya yaz. 29.09'daki `%2,7`
ölçüm hatası, sabit bir eşiğin bile yanlış sayıyla uygulanabileceğini
gösterdi.

---

# 3. ÇALIŞMA KURALLARI

**Tek zorunlu kural: git geçmişi.** Force push kullanma. CI günde 3-4 kez
`main`'e push yapıyor, force push o commitleri siler. Reddedilirse:
`git fetch` → `git rebase origin/main` → normal push.

Geri kalanı — serbest, gerekçesi varsa sebeplendir:

- **Veri dosyaları** (`intents.json`, `knowledge_map.jsonl`, `corpus.jsonl`,
  `corpus_ids.jsonl`, `chatgrow_*.jsonl`): normalde üreten scriptler
  (`build_book_pairs.py`, `autogrow.py`, `chatgrow.py`) ve CI günceller —
  çünkü elle düzeltilen veri bir sonraki CI koşusunda ezilir. Ama elle
  düzenlemek de yasak **değil**: geçici olarak düzeltip test etmek, kötü bir
  kaydı temizlemek, küçük bir düzeltmeyi hızlıca denemek için yapılabilir.
  Yaparsan neyi neden yaptığını yaz ve üretici scriptten de geçir.
- **Model mimarisi**: `d_model`/blok küçültülebilir, ama ölçümle gerekçelendirilir
  (bkz. §2 karar kuralı). `--untie-embeddings` tam da böyle bir ölçülmüş seçenek.
- **`model/` klasörü** git'e girmez. Üzerine yazmadan önce yedek al
  (`model_kur.py` `--check` yapar, küçültmeyi engeller, `model/yedek/<ts>/`
  altına rollback noktası yazar). Zorunluluk değil güvenlik yolu.
- **Testler** veri dosyalarını okuyabilir; yazmamalı. Yazarsa CI'da üretici
  koşusunun üstüne biner.
- Kullanıcı **kısa cevap istiyor.** Belirsizlikte tahmin etme, sor.
- İki iş isteniyorsa ikisini birden yap (kullanıcı açıkça böyle istedi).

---

# 4. EĞİTİM: KAGGLE'DE NASIL YAPILIYOR

Eğitim **Kaggle'da** yapılır, GPU ile. Yerel NumPy eğitimi saatler sürer.

## Donanım / kota
- **Accelerator: GPU P100** (veya **T4x2**).
- Kaggle haftada **30 saat** ücretsiz GPU.
- Oturum süresi pratikte **9 saat**; `kaggle_start.sh` bunu `OTURUM_DK=540` olarak varsayar.
- 29.09 ölçümü (d=384/6 blok, T4x2, max_seq 256): 12 epoch ≈ 279 dakika ≈ 4,7 saat.

## Adımlar (Kaggle.com → New Notebook)

1. **Ayarlar** (yukarı sağ): **Internet: ON** | **Accelerator: GPU P100**
2. **1. hücre** — repo klonu + bağımlılık:
   ```
   !git clone https://github.com/dozagoat47-star/NextgenAI.git
   %cd NextgenAI
   !python -m pip install --quiet numpy
   ```
3. **2. hücre** — eğitim:
   ```
   !bash kaggle_start.sh train
   ```
4. **Save (Version)** → **Output** sekmesi → **Download All**
5. İndirilen `nde-irma.zip` içindeki **iki dosya birlikte** `model/` klasörüne kopyalanır:
   `llm_model.json` + `llm_model_weights.npz`

> Veri/intents'i değiştirdiysen repo'ya push ettikten sonra **sadece 1. adım (clone)**
> yeterli — tüm dosyalar taze gelir.

## `kaggle_start.sh` modları

| komut | ne yapar | süre |
|---|---|---|
| `!bash kaggle_start.sh verify` | dry-run doğrulama, GPU gerekmez, **TAM encode yapılmaz** | ~1-2 dakika |
| `!bash kaggle_start.sh bench` | 1 epoch zamanlama (cache/encode + 1 epoch birlikte ölçülür) | ~10 dakika + 1 epoch |
| `!bash kaggle_start.sh train` | asıl eğitim | ~4,7 saat |

## Gerçek komut (train modu)

```
python train_llm.py --rag --kb-map knowledge_map.jsonl --natural 5 $CGARG \
  --epochs 12 --patience 4 --batch-size 128 --val-every 2 \
  --d-model 384 --num-blocks 6 --max-seq-len 256 \
  --weight-decay 0.01 --dropout 0.10 \
  --max-pairs-cap <sure_ve_hesapla()>  | tee kaggle_train.log
```

`$CGARG` otomatik: `chatgrow_*.jsonl` varsa `--chatgrow ...` eklenir.

## Env değişkenleriyle kapasite/ayar

| değişken | varsayılan | not |
|---|---|---|
| `LLM_EPOCHS` | 12 | üst sınır **ve** `lr_horizon = min(EPOCHS, patience+20)` |
| `LLM_PATIENCE` | 4 | **epoch cinsinden** (eski sürümde "kötü val ölçümü" sayıyordu, düzeltildi) |
| `LLM_OTURUM_DK` | 540 | oturum süresi → `MAX_PAIRS` tavanını hesaplar |
| `LLM_CAP` / `LLM_BLOCKS` | 384 / 6 | ~22,9 milyon parametre |
| `LLM_SEQ` | 256 | `MAX_SEQ_LEN` ile **AYNI olmalı** (tek kaynak) |
| `LLM_TIE` | 1 | 0 → `--untie-embeddings` (23,0 milyon → 16,8 milyon, dosya 87,6 → 64,1 megabayt) |
| `LLM_WD` | 0.01 | AdamW, yalnız ağırlık matrisleri |
| `LLM_DROPOUT` | 0.10 | |
| `LLM_NATURAL` | 5 | doğal çoğaltma çarpanı |
| `LLM_CAP=256 LLM_BLOCKS=4` | — | küçük deney örneği |

## Ölçülmüş zaman sabitleri (`train_llm.py`)

| sabit | değer | nasıl ölçüldü |
|---|---|---|
| `ENC_CIFT_SN` | 664,5 çift/saniye | 1.024.172 çift, encode 1541 saniye (921.748 train + 102.424 val) |
| `MS_PER_PAIR` | 1,5139 milisaniye/çift | 921.748 çift, ortalama 1395 saniye/epoch (val olan epoch'lar ~1435, val atlananlar ~1338 → **tek epoch seçmek %7 yanıltırdı**, ortalama alındı) |
| `TOKEN_PER_PAIR` | **100,0** token | 5.000 çift, canlı `encode_llm` → PAD budanmış gerçek uzunluk (eğitim popülasyonu; ham popülasyon 95,97 ± 0,97). 29.09'daki 89,3 veri büyüdüğü için %12 bayatlamıştı |
| `TOKEN_PER_PAIR` (29.09) | 89,3 | 3 tohum × 1.000 ham çift, o dönemin 11.149 intent'lik verisi |

**Dikkat (28.09 hatası):** MS_PER_PAIR küçük saymak bütçeyi **BÜYÜTÜYOR** (12 epoch
396 dakika yerine gerçekte 431 dakika ister → oturum kesilirdi). Zaman bütçesi pratikte bağlayıcı
değil; **veri bütçesi (`MAX_PAIRS`) asıl kısıt**. Tavan formülü
`train_llm.sure_ve_hesapla()`'da — bash'ta değil, test edilebilir olsun diye.

## `max_seq 256` neden ölçülmüş özel bir sayı

`kb_budget = max_seq - max_ctx - 8`; küçük değerde yanıta yer kalmaz ve RAG yanıtları
kırpılır. `knowledge_map.jsonl`, 1000 RAG örneği + 68.794 çiftin gerçek karışımı:

| max_seq | ort uzunluk | işlem | RAG yanıtı kırpıldı |
|---|---|---|---|
| 128 | 110 | 1,00× | **%87** ← eski (bu hatayı yarattı) |
| 192 | 137 | 1,24× | %37 |
| 256 | 143 | 1,30× | **%3** ← seçilen |

Artan işlem yalnız %30; uzunluk-kırpımlı batch'ler (`_pack_encoded`) RAG'sız
68.794 çiftin ortalama uzunluğunu değiştirmiyor.

## 29.09 eğitim çıktısı (yetkili kaynak: `model/yeni_2909/kaggle_train.log`)

```
veri bütçesi        : 210.159 çift (intents %93,9) + chatgrow 13.656 (%6,1)
  ↳ kitap hattı          609  (%0,27)
doğallaştırma ×5    : 1.024.172 çift
RAG bağlamlı çift   : %52,9   (log "%2,7" demişti — HATALI, bakınız §6.2)
BPE vocab 16.000 · 100,3 milyon token · 8 epoch (patience ile erken kesildi)
en iyi val 0,3959 @ 8. epoch · acc 0,913
```

## Google Colab alternatifi
`colab/nextgen_llm_colab.ipynb` — Colab'da `SAVE_DIR=/content/drive/MyDrive/NextgenAI_llm`,
çıktıyı zip'leyip indir, iki dosyayı `model/`'e kopyala. `--max-seq-len 256`
`train_llm.py MAX_SEQ_LEN` ile aynı olmalı. (Colab artık **ikincil** yol; Kaggle birincil.)

## CI (GitHub Actions) — veri hattı otomatik

`.github/workflows/`: `autogrow.yml`, `autogrow-deep.yml`, `chatgrow.yml`,
`crawl_corpus.yml`, `kbmap.yml`, `ci.yml`, `opencode.yml`, `opencode-scheduled.yml`

Bunlar veri dosyalarını **kendi üreticileriyle** üretip **doğrudan `main`'e push eder** →
`git pull`/`rebase` sırasında çakışma normaldir, force push yasak.
Yerelde `scripts/sync_chatgrow.ps1` aynı işi yapar.

---

# 5. ANA HEDEF VE MEVURT ÖLÇÜMLER

Hedef: modelin kalitesini artırmak. İki şikâyet ölçülmüş olarak teyitli:

| ölçüm | değer | araç |
|---|---|---|
| kabul (bilgi cevabı verme) | **%39** | `tools/kapi_ab.py` |
| ekrana çıkan metin | **%58** | `tools/uretim_olc.py` |
| sadakat / fidelity | **%22–31** | `tools/uretim_olc.py` |
| kopyalama (`copy_bleu`) | yüksek | `eval_llm.py` |
| fluency | düşük (yanıtlar kısa) | `eval_llm.py` |
| "bilgim yok" cevabı | **~%40** | `tools/uretim_olc.py` |

`~%40` "bilgim yok" veri eksiğidir — **kodla çözülemez** (AutoGrow'un o konularda tanım
toplaması gerekir; veri dosyasına dokunulmadan yapılamaz).

---

# 6. YAPILMIŞ VE KANITLANMIŞ İŞLER (tekrar etme)

## 6.1 Kopyalama decoding ile çözülmez (kanıtlandı)
6 decoding ayarı tarandı (n=400), paired t-testi: **hiçbiri geçmedi**. Aşırı ayar kontrol
olarak t=+2,13 anlamlı çıktı → ölçüm gücü doğrulandı.
`brain.py:2035` decoding ayarlarına **dokunulmadı** (kanıt yok).

Tek anlamlı değişim (t=1,0 / k=40 / kb=0) `copy_bleu`'yu 0,215→0,175 düşürürken
`gold_recall`'ı da 0,322→0,280 düşürdü → kopyalama ile kapsama zıt eksenler.

**`copy_bleu` ve `gold_recall` aynı ölçünün iki görünümü** (altın metinle n-gram
örtüşmesi) — ikisi bağımsız kanıt değildir.

Metrik gerilimi: `qa_score` içinde `copy_bleu` **+0,15** ağırlıkla var → kopyalamayı
düşürmek `qa_score`'u da düşürür. Metrik std: `copy_bleu` ±0,36, `gold_recall` ±0,42
→ **paired test şart**.

## 6.2 Ölçüm hataları düzeltildi

**Eval varsayılanı tutarsızlığı** — eval `kb=0.0`, üretim `1.2` (`brain.py:764`).
`compare_reports` artık 8 decoding anahtarını karşılaştırıp farklıysa **uyarıyor** ve
sonucu "model seçimi" diye sunmuyor (`7f2d110`).

**RAG kapsam ölçümü** (`11bac35`, en son) — log `27.732/1.024.172 = %2,7` diyordu,
oysa sayaç **benzersiz ctx** sayıp paydayı **toplam çift** ile bölüyordu.
Doğal varyantlar ctx'i değiştirmiyor + `load_pairs` aynı desen için birden çok çift
üretiyor → bir ctx ortalama 5,8–13,8 kez geçiyor.

| | oran |
|---|---|
| ESKİ (tek_ctx/toplam) | %9,3 (tam bütçede %2,7) |
| YENI (çift bazlı) | **%52,9** |
| çarpan | 5,7× |

Düzeltme `rag_context_stats()` fonksiyonunda, **yalnızca yazdırır** — `ctx_map` içeriği,
önbellek parmak izi ve eğitim girdisi bit bit aynı, model çıktısı değişmiyor.
4 test eklendi.

## 6.3 Kitap hattı ÖLÇÜLDÜ ve KAPALI — kaynak tükenmiş
`--max-pairs 8000` hedefi **ulaşılamaz**:
- Tam kapasite koşusu (`--max-pairs 3000 --per-cat 500 --seed 7`) ürettiği dosya
  depodaki 609 çiftlik dosyayla **byte-aynı** (sha256 `71251cb0ac3d6965`).
  Sonuç ayardan değil **kaynaktan** geliyor.
- 8 kategoride toplam **128 sayfa**; `ninniler` 56 + `ağıtlar` 20 = sayfaların %59'u.
- Türk Wikisource'ta **nesir zengini başka kategori yok**: 16 aday tarandı
  (roman, öykü, deneme, edebiyat, cumhuriyet dönemi…), en zengini 3 sayfa.
- Yanıt medyanı **kategori karışımından**: destanlar (nesir) 29 kelime, masallar 5,
  ninniler 4, ağıtlar 5 kelime; `efsaneler`+`bilmeceler` **0 çift**.
- Şi+nesir **birleştirmesi denendi → 0 çift ekledi** (byte-aynı sonuç). İki yol ayrık:
  şi olan sayfada nesir boş, tersi. Kazanmayan değişiklik **geri alındı**; sadece ölçüm
  `build_book_pairs.py` docstring'ine yazıldı.
- 609 çift / 223.815 taban = **%0,27**. Bu hat akıcılık kaynağı olamaz.

`build_book_pairs.py` **öğretmiyor ki öğretmez**: "devam cümlesi" (continuation)
üretiyor, bilgi değil. Yanıt 6,2 kelime → niyet yanıtlarıyla aynı uzunlukta.

## 6.4 Disk temizliği — 676 megabayt, kanıtlayarak silindi
2 Kaggle zip'i (içerik diskte doğrulandı), byte-aynı yedek kopyalar, 0 referanslı model
dosyaları, 4 `llm_data_*.npz` (gitignore'lu, Kaggle klonunda var olamaz).
`model/` 941 → 329 megabayt. Tek rollback noktası: `model/yedek/20260929_031340`.
İki `kaggle_train.log` korundu.

## 6.5 Korpus önbellekleri (1.038 megabayt) SİLİNMEDİ — ölçüldü
Önbellekli yükleme **18,4 saniye** / önbelleksiz **273,8 saniye**. Çöp değil, her açılışta
4,3 dakika kazandıran hız önbelleği. Karar `corpus.py` `load()` docstring'ine yazıldı.

## 6.6 40 sohbet sınıfı kasıtlı
`brain.py:1047` kuralı: `len(patterns) > 6` → sohbet, `≤6` → bilgi. AutoGrow bilgi
intent'lerini tam 6 desenli Wikipedia şablonundan üretiyor → `num_intents: 40` sabit
(`autogrow.py:57` yorumu bunu doğruluyor). Büyüme retrieval katmanına gidiyor.

**DİKKAT — bu sayı SABİT DEĞİLDİR.** AutoGrow CI'da her koşuda `intents.json`'u
büyütür, yani aşağıdaki değer **29.09 ölçümüdür ve sonrasında değişmiştir.**
29.09: 11.557 intent (40 sohbet + 11.517 bilgi). Sonraki ölçüm: bilgi intent
**11.888**, toplam 11.928. Numarayı kullanmadan önce **yeniden ölç**. 40 sohbet sınıfı
sabittir; büyüyen kısım yalnızca bilgi intent'leridir (tam 6 desenli şablon).
`corpus.jsonl`: 137.032 parça, örneklemde %12,3 uzun metin / %87,7 kısa tanım.

## 6.7 Ölçüm araçları repoya taşındı
`tools/`: `uretim_olc.py`, `uretim_karsilastir.py`, `kapi_ab.py`, `kendi_cumlesi.py`,
`token_olc.py`, `kalite_olc.py` + `tools/README.md` + `tests/test_olcum_araclari.py`
(7 test). Sabit yol 5 scriptten kaldırıldı, `__file__`'dan türetiliyor.
Çıktı `%TEMP%` → `olcum_raporlari/` (gitignore'lu). Repodan gerçekten koşuldu
(50 soru: kabul %34, ekrana %52, sadakat %31).

## 6.8 Kapı distinct-letter hatası düzeltildi (`7f2d110` öncesi)
`brain.py:2098/2145` → kabul %14→%39, ekrana çıkan metin %24→%58.

## 6.9 `app.py` veri bozma hatası düzeltildi (`bd63543`)
`app.py:374` → 11 test eklendi.

---

# 7. ANA ÇIKARIM VE 29.09 ÖLÇÜM SONUÇLARI

**29.09'daki "bilgi verisi yok" tezi ölçümle çürütülmüştü.** Eğitim çiftlerinin
**yarısı (%52,9 → güncel veride %48,4) zaten bilgi bağlamı taşıyor.**

**29.09 sonrası yapılan ölçüm:**

| soru | cevap | kanıt |
|---|---|---|
| `KB_TEXT_CHARS=300` bağlamı kırpıyor mu? | **HAYIR, %0,0** | kb metni maks 192 token < 200 bütçe; 0/39.983 desen 300 karakteri aşıyor (zaten üretimde 300'de kesiliyor) |
| Kopyalama bilgi-bağlamlı çiftlerde yoğunlaşıyor mu? | **HAYIR** | gruplar arası: bilgili 0,185 / bilgisiz 0,217 (t=−1,04, anlamsız). Karıştırıcısız paired (aynı 250 soru, bağlam açık/kapalı): kopya +0,095 (t=3,42) ama **altın içerik +0,150 (t=4,54)** → kopyalama artışı faydalı |
| Model bağlamı kullanıyor mu? | **EVET, güçlü** | öğretmen koşullu argmax %68,2 → **%86,7** (NLL 2,717 → 1,329; t=+8,06) |
| Üretimdeki bağlam cevabı taşıyor mu? | **EVET** | 250 bilgi sorusunun **%60,0**'ında altının tüm içerik kelimeleri `Corpus.search` metninde (eşit 300 karakter bütçede eğitim bağlamından +0,029, t=−2,72) |
| Kaybedilen içerik neden kaybediliyor? | **Üretim biçimi** | altın kelimelerinin %30,8'i geçiyor; kalanın **%59,4'ü bağlamda VAR**, %40,6'sı hiç yok |
| Decoding tükendi mi? | **EVET, 4 eksende** | aşağıdaki tablo |

**Decoding taramaları** (hepsi n=250, aynı sorular, paired t-testi; karar kuralı
geçmediği için hiçbiri uygulanmadı — `brain.py:2035` ve `brain.py:764` aynen
duruyor, ayar kendisi değil ölçüm sonucu değiştirebilir):

| eksen | değerler | `gold_recall` | t |
|---|---|---|---|
| `knowledge_bias` | 0,0 / 1,2 / 3,0 | 0,277 / 0,301 / 0,302 | −0,96 / +0,04 |
| sıcaklık | 0,7 / 0,3 / 0,05 | 0,293 / 0,266 / 0,280 | −1,24 / −0,60 |
| `rep_penalty` | 0,4 / 0,2 / 0,0 | 0,295 / 0,303 / 0,301 | +0,36 / +0,28 |
| top-k / rep / sıcaklık | §6.1'deki 6 ayar | — | hiçbiri geçmedi |

**Oracle (tavan) probu KULLANILAMAZ — 3 denemede de tutarsız:** çıplak altın
0,033 · başlık+altın 0,140 · NLL'de 2,577 (üretim 1,329'**DAN KÖTÜ**). Nedeni:
altın yanıtlar 204 karakterde kesilmiş, ASCII'ye bozulmuş, tekrar eden
Wikipedia parçaları; decoder'ın 6-gram tekrar kesmesi bunları yarım kesiyor.
**Bağlam biçimi dağılım dışı olunca model sohbet kalıbına düşüyor** (6 token,
"devam edelim mi?"). `enrich_intents.py:132`'deki `"<Başlık>. <metin>"` biçimi
modelin bağımlı olduğu bir kısıt — ölçümü bozacak, `brain.py:1047` ve
`enrich_intents.py:132` birlikte çalışmalı.

## 7.1 Refüt edilenler (tekrar etme, hepsi ölçüldü)

| iddia | sonuç |
|---|---|
| "greedy altını üretir" (öğretmen koşullu %86,7'den) | **YANLIŞ** — t=0,05: `gold_recall` −0,013 (t=−0,60). Öğretmen koşullu ölçüm altın *ön eki* verir; exposure bias |
| "altın yanıtlar bozuk" | **YANLIŞ** — rep2 0,004 / distinct1 0,965 (sohbet 0,001 / 0,978) |
| "doğallaştırma içeriği kısaltıyor" | **YANLIŞ** — varyantlar %97,6 içerik koruyor, %12,6 **uzatıyor** |
| "bağlaç enjeksiyonu sadakati düşürüyor" | **ÖLÇÜLEMEDİ** — işaretli/isaretsiz `gold_recall` farkı üç kümede de \|t\|<1,4 ve **işaret yönü değişiyor** |

## 7.2 Ölçülmüş kusurlar

- **`naturalize.py` metin bozulması (ölçüldü, henüz düzeltilmedi):** varyantların **%1,06**'sı
  hatalı (bitişik `x.y` %0,64, yapıştırılmış işaret %0,42, çift enjeksiyon
  %0,00). Ham veride zaten %0,42 `x.y` var → büyük kısmı kaynaktan. Kaynak:
  `_maybe_insert_mid` virgülsüz yolunda kesme konumu
  `len(' '.join(words[:k]))` ile **string uzunluğundan** hesaplanıyor
  (`naturalize.py:186-205`); metin tek boşlukla kurulmamışsa konum kayıyor
  (`"kasalidir.sismik"`, `"bir; üstelik bakima"`). Değiştirilmedi: kalite etkisi
  kanıtlanmadı (§2 kuralı), doğrulaması 4,7 saatlik Kaggle koşusu ister.
- **`TOKEN_PER_PAIR` bayatlamıştı (DÜZELTİLDİ):** 89,3 → **100,0** (eğitim
  popülasyonunda canlı encode 100,02; ham popülasyonda 95,97 ± 0,97). Sapma
  koddan değil veriden: intents 11.149 → 12.730, kb-map 29.970 → 39.983. Bütçe
  etkisi yok (`sure_ve_hesapla` bu sabiti kullanmaz).
- **Eğitim/üretim bağlam uyuşmazlığı:** aynı 250 sorunun **0/250**'sinde
  eğitimdeki `knowledge_map` metni ile üretimdeki `Corpus.search` metni aynı.
- **Çalışan VS Code debug sunucusu** (PID 7064, port 5000): **ÇÖZÜLDÜ** —
  29.09 sonrası ölçüldü, PID yok ve port 5000 boş.

## 7.4 Bilinmeyen (ölçülmedi, tahmin de edilmedi)

`MS_PER_PAIR = 1,5139` milisaniye/çift, 29.09'da **89,3 token/çift** verisiyle ölçüldü.
Çift başına token %12 arttı (100,0), yani epoch süresi de artmış olmalı; GPU
olmadan ölçülemez. Yerel olarak `kaggle_start.sh bench` ile ölçülmeli. Veri
bütçesi (`MAX_PAIRS`) daha önce bağlayıcıydı, o bozulmadı.

## 7.3 Sıradaki adım

Ölçülebilir kalan yer **veri üretimi**: `autogrow.py` / `enrich_intents.py`
ürettikleri hedefler. Ölçülmüş adaylar:
1. Doğallaştırma varlık adı bozuyor (`alyson hannigan` → *"aleis denisof"*);
   %1,6 varyantta içerik kapsamı <%50.
2. Aynı ctx için farklı yanıtlar karışıyor (bir konu/bölümün cevabı başkasının
   yerine geçiyor).
3. §8'deki 4 cevaplanamayan "X nedir" sorusu (`corpus.jsonl`'de tam adıyla
   kayıt yok).

---

# 8. BİLİNEN ENGLELLER

- **4 "X nedir" sorusu cevaplanamıyor**: `kadin`, `siber guvenlik`, `kuantum
  bilgisayarlar`, `fotografik` → `corpus.jsonl`'de tam adıyla kayıt yok.
- ~~Çalışan VS Code debug sunucusu (PID 7064, port 5000)~~ → **ÇÖZÜLDÜ**
  (29.09 sonrası ölçüldü: PID 7064 yok, port 5000'de dinleyen yok). Artık
  `corpus.jsonl` bozma riski yok.
- 587 test ~270 saniye sürüyor; ölçüm aracı çalıştırırken `train_llm.py`'ye dokunma.

---

# 9. DOSYA HARİTASI

| dosya | satır | içerik |
|---|---|---|
| `build_book_pairs.py` | docstring | **ÖLÇÜLEN KAYNAK TAVANI (609)**, kanıtlı |
| `kaggle_start.sh` | 1-50, 60-140, 149 | Kullanım, env değişkenleri, asıl eğitim komutu |
| `train_llm.py` | 185, 207, 210 | `MAX_SEQ_LEN=256`, **`KB_TEXT_CHARS=300`**, `SEED=7` |
| `train_llm.py` | 213, 232, 837 | `TOKEN_PER_PAIR`, `ENC_CIFT_SN`, `MS_PER_PAIR` (hepsi ölçülmüş) |
| `train_llm.py` | 485 | `make_batches` (PAD budama + `_pack_encoded`) |
| `train_llm.py` | 737, ~760 | `build_kb_lut`, `rag_context_stats` (29.09 düzeltmesi) |
| `train_llm.py` | 874 | `sure_ve_hesapla` — **MAX_PAIRS tavanının tek doğruluk kaynağı** |
| `train_llm.py` | ~1090 | RAG yazdırma satırı (yalnızca log) |
| `train_llm.py` | 386 | `_cache_fp` |
| `train_llm.py` | 920, 1021-1023 | `load_chatgrow_pairs`, RAG eşiği (`use_corpus`) |
| `brain.py` | 1047 | 40 sınıf kuralı (`>6` desen → sohbet) |
| `brain.py` | 764 | `knowledge_bias=1.2` |
| `brain.py` | 2035 | decoding — 4 eksende ölçüldü, hiçbiri kazandırmadı (§7) |
| `brain.py` | 2098, 2145 | kapı distinct-letter düzeltmesi |
| `eval_llm.py` | 400+ | `compare_reports` decoding denetimi |
| `corpus.py` | 188, 303, 368, 486, 921 | `Corpus`, `_load_index_cache`, `load`, `_ensure_embeddings`, `search(query, k=2)` |
| `app.py` | 340-410, 442-455, 517-554 | `learn_from_internet` (düzeltildi), `fallback_answer`, model logları |
| `autogrow.py` | 57 | "num_intents 40'da sabit kalır" notu |
| `model_kur.py` | — | güvenli kurulum (`--check`, rollback) |
| `model/yeni_2909/` | — | üretimdeki model + `kaggle_train.log` (5777 bayt) |
| `model/kaggle_2909/` | — | önceki koşu logu |
| `model/yedek/20260929_031340/` | — | tek rollback noktası (29.09 öncesi) |
| `tools/` | — | 6 ölçüm aracı + README |
| `scripts/sync_chatgrow.ps1` | — | yerel ChatGrow senkronu |
| `colab/nextgen_llm_colab.ipynb` | — | Colab alternatifi |

---

# 10. POWERSHELL TUZAKLARI (bu ortamda defalarca ısırıldı)

- `"x" % y` **modülo** yapar → ondalık için **`-f`** kullan
- `-f` ile Türkçe locale'de **virgül** basar → `InvariantCulture` gerek
- `Select-String` eşleşmezse **exit code 1** verir (hata değil!)
- `Set-Content -Encoding utf8` **BOM** yazar
- **Heredoc `<<` yok** → commit mesajını dosyaya yaz, `git commit -F dosya`
- `python -c "...çok satır..."` tırnak kaçışını bozuyor → `%TEMP%\opencode\`
  altına `.py` dosyası yaz, `python "$env:TEMP\opencode\xxx.py"` ile çalıştır
- `Join-Path $env:TEMP 'a' 'b'` 3 argüman **kabul etmez** → değişkenle birleştir
- Geçici dosya: `C:\Users\cxc\AppData\Local\Temp\opencode\`

---

# 11. TEST / CI

- `pytest` **yok** → `python -m unittest discover -s tests -p "test_*.py"`
- ~270 saniye, **587 test OK (1 skip)**
- CI (`ci.yml`, Python 3.12) yalnız **`pip install numpy requests flask`** yapar →
  `torch` bağımlı testler `@requires_torch` ile **skip** edilir (aksi halde
  `unittest.loader._FailedTest` modülü düşürüp tüm suite'i kırmızı eder).
  `requires_torch` 5 test dosyasında kullanılıyor.
- Test taramasında docstring ve diziler kod sayılmamalı
- **Kod tarama nöbetçileri** (sayı sabitlerine bağlı kalibrasyon; sabitleri
  değiştirirsen testleri de güncelle): `n / 6000.0`, `n * 0.16`,
  `n * TOKEN_PER_PAIR` (boşluklu **ve** boşluksuz), `MS_PER_PAIR=`
  (kaggle_start.sh'da olmamalı — bash mantığı Python'da test edilemez)

---

# 12. ÇALIŞTIRMA TUZAKLARI

- `eval_llm.py` `train_llm.py`'yi import ettiği için **ölçüm sürerken
  `train_llm.py`'ye dokunma**
- `app.load_bot()` `global bot` kullanıyor, **döndürmüyor** → `app.bot` kullan
  (yanlış testte `NoneType` hatası verdi)
- `Corpus.search(query, k=2)` — **`top_k` parametresi yok**
- PowerShell'de `'%s' % (x, y)` içinde `%%.1f` yazarsan **"not all arguments
  converted"** hatası verir: `%%` kaçış olduğu için o alan dönüşüm saymaz.
  Bu tuzak 29.09'da 3 kez ölçüm betiğini düşürdü (biri 10 dakikalık 3 kolü
  kaybettirdi). Gerçek yüzde için `%.1f%%` + değeri ayrı argüman olarak ver.
- Ölçüm betiğinde **kol biter bitmez diske yaz**, sonra raporla: raporlama
  satırındaki hata tüm üretimleri yok ediyor.

---

# 13. KALICI ÖLÇÜM ÇIKTILARI

`C:\Users\cxc\AppData\Local\Temp\opencode\`:
- `sweep_*.json` — decoding sweep sonuçları (6 ayar × n=400)
- `uretim_karsilastir.py`, `kod_sweepi_cikti.txt`, `commit_mesaj7.txt`
- `t_rag.py` — RAG sayaç regresyon testi (geçici)
- `t_shadow.py` — 29.09 shadow ölçümü (ESKİ %9,3 / YENI %52,9)
- `kitap_tam.jsonl` + `kitap_tam.txt` — 609 byte-aynı kanıtı
- `devam_promtu.md` — bu dosyanın ilk sürümü (artık güncel değil, **bu dosya kanonik**)

### 29.09 sonrası ölçüm betikleri ve çıktıları (hepsi canlı ölçüm)

| betik | ne ölçtü | çıktı |
|---|---|---|
| `olc_baglam.py` | RAG kapsamı, bağlam/yanıt kırpma oranı, token/çift | `olc_baglam_cikti.txt` |
| `olc_kopya_sinyali.py` | altın yanıtın bağlamdan kopyalanması (ASCII katlamalı) | `olc_kopya_cikti.txt`, `olc_ornek_ciftler.jsonl` |
| `olc_token_dogrula.py` | `TOKEN_PER_PAIR`: elle formül / canlı `encode_llm` | `olc_token_cikti.txt` |
| `olc_model_tarafi.py` | paired bağlam açık/kapalı + gruplar arası | `olc_model_tarafi.json`, `olc_model_cikti.txt` |
| `olc_sadakat_neden.py` | kaybedilen altın kelimeleri: bağlamda var mı? | (başka betikten) |
| `olc_sweep3.py` | `knowledge_bias` 0/1,2/3,0 + **GEÇERSİZ** oracle | `olc_sweep3.json` |
| `olc_sweep4.py` | başlıklı oracle (geçersiz) + birleşik | `olc_sweep4.json` |
| `olc_nll.py` | öğretmen koşullu NLL + argmax doğruluğu | `olc_nll.json` |
| `olc_kaynak_karsilastir.py`, `olc_kaynak_esit_butce.py` | eğitim / üretim bağlamı (eşit bütçe) | `olc_kaynak_karsilastir.json` |
| `olc_sweep7.py` | sıcaklık 0,7/0,3/0,05 | `olc_sweep7.json` |
| `olc_sweep10.py` | `rep_penalty` 0,4/0,2/0,0 | `olc_sweep10.json` |
| `olc_altin_kalite.py` | altın yanıt kalitesi (sohbet / bilgi) | (stdout) |
| `olc_dogal_kapsam.py` | doğallaştırma içerik koruması | `olc_dogal_kapsam.json` |
| `olc_diskursor.py` | diskursör işareti: hedeflerde ve çıktılarda | (stdout) |
| `olc_naturalize_hata.py` | `naturalize.py` hata sınıfları (%1,06) | (stdout) |

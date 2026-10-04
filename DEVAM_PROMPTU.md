# Devam Promptu — Nextgen AI (Nextgen_API)

> **Bu dosya bir yapay zekâya devam promptu olarak verilir.**
> Aşağıdaki "KISA PROMPT" bölümünü yeni sohbete yapıştır, sonra bu dosyayı
> okumasını söyle. Geri kalanı senin için.

---

# KISA PROMPT (yeni sohbete yapıştır)

```
Bu repo için devam ediyoruz. Önce DEVAM_PROMPTU.md dosyasının TAMAMINI oku
(bash: cat DEVAM_PROMPTU.md).Dosyada yazılı olan her şeyi kanıt kabul et, kendi tahminini
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

## 6.6 "40 sohbet sınıfı" **kasıtlı bir karar DEĞİLDİR** (02.10.2026 ölçümüyle düzeltildi)

**ESKİ KAYIT YANLIŞTI:** "40 sohbet sınıfı kasıtlı, `num_intents: 40` sabit" deniyordu.
Ölçüm bunun **bir seçim değil, bir yan etki** olduğunu gösteriyor.

`brain.py` kuralı (artık `brain._sohbet_mi`, :1036): `len(patterns) > 6` → sohbet,
`≤6` → bilgi. AutoGrow bilgi intent'lerini tam 6 desenli Wikipedia şablonundan
ürettiği için **sohbet hiç büyüyemiyor** — sohbet intent'i ekleyen bir boru hattı
yok. Yani 40 sayısı, "sohbeti kısmak istedik" seçimi değil, "başka türlü
üretemiyoruz" sonucudur.

### Ölçülen taban (02.10.2026, canlı `intents.json`)

| ölçüm | değer |
|---|---|
| toplam intent | 16.225 |
| sohbet sınıfı (>6 desen) | **40** |
| bilgi intent'i (≤6 desen) | 16.185 |
| bilgi intent'lerinin 6 desenli şablona uyanı | 16.182 / 16.185 = **%99,98** |
| sohbet intent'lerinin **en az** desen sayısı | **8** |
| tam 7 desenli intent | **0** (eşik ile sınıf arası boşluk yok) |
| sohbetin eğitim verisine katkısı | 1.083 / 302.506 çift = **%0,358** |

Bilgi intent'lerinin 3'ü şablona **uymuyor** ve **sohbet dili** taşıyor —
bunlar sızıntı, aşağıda.

### "Eşiği düşür" yanlış yön — kanıtlı

Eşik 6'ya düşürülse (`>5`, yani ≥7 desen) sohbet sınıfı sayısı **değişmez**
(çünkü 7 desenli intent yok) ama **16.185 bilgi intent sınıflandırıcıya girer**
(ölçüldü: 40 → **16.225 sınıf**). Yani "esnet" sayıyla yapılırsa felakettir.

### Ölçülmüş sızıntı: 6 desenli olup sohbet olan 3 sınıf

`kavram_tanimi` ("mizah nedir"), `tavsiye_isteme` ("bana ne önerirsin"),
`gelecek_planlari` ("geleceğim beni endişelen") → 6 desenli oldukları için
**sınıflandırıcıya hiç girmiyor**, hiç yakalanamıyor.

### Uygulanan düzeltme: sayı değil, **açık işaret**

`brain.py` `_sohbet_mi(intent)`: `tur == 'sohbet'` → sohbet, `tur == 'bilgi'` →
bilgi, **yoksa eski sayma kuralı** (geriye uyumlu, mevcut veride **40 aynı 40**).
Eşik **düşürülmedi**. Ölçülen: 3 sızıntı intent'ine işaret konunca **40 → 43**,
istenenden başka **hiçbir bilgi intent taşınmadı** (16.182/16.185 korundu).
`conversational_data` yalnızca **eğitimde** çağrılır ve `intent_tags`
`bot_data.json`'a yazılır → bu değişiklik **mevcut modeli etkilemez**, sonraki
eğitimi etkiler.

Testler: `tests/test_core.py::TestTwoLayerArchitecture` → `test_sohbet_mi_*` (6 test),
`test_sohbet_mi_gercek_veride_sonuc_degismedi` regresyon kilidi.

**KALAN:** `intents.json`'a bu 3 intent'e `"tur": "sohbet"` yazmak gerekiyor —
veri dosyası olduğu için **bilinçli ve ayrı** bir adım; ölçüm seti hazır
(`tools/soru_listesi_sohbet.json`, `sinif_tasi` grubu) etkisini ölçmek için.

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

## 6.10 Üretim yolu ölçümü 3 hata düzeltildi (01.10.2026) — **BU BÖLÜM ESKİ SAYILARI GEÇERSIZ KILAR**

`tools/uretim_olc.py` "iki koşu birebir aynı sonucu verir" diyordu. **Bu iddia
yanlıştı** ve üç ayrı hata vardı. Üçü de ölçüldü, üçü de düzeltildi.

### Hata 1 — Sadakat metriği Türkçe harfleri SİLİYORDU
`icerik_kelimeler` `re.sub(r'[^a-z0-9 ]+', ' ', lower())` kullanıyordu:

```
"sarkinin"  -> ['sarkinin']   (ASCII metin bozulmadan kalir)
"sarkının"  -> ['ark']        (ş ve noktali I SILINIR)
```

Model çıktısı **ASCII**, bilgi metni (`kb`) **Türkçe** olduğu için iki taraf
**hiç eşleşemiyordu**; sadakat yapay olarak düşük ölçülüyordu. Artık projenin
kanonik katlaması `normalize.ascii_normalize` kullanılıyor.

| | aynı 50 soru, aynı model, aynı üretimler |
|---|---|
| raporun yazdığı sadakat | %33,3 |
| **gerçek sadakat** | **%57,1** |

> **README ve 29.09 notlarındaki "%22–31 sadakat" bu yüzden düşüktür.** O
> sayılar gerçek kalite değil, normalizasyon hatasının sonucudur.

### Hata 2 — Soru listesi veri sürümüne bağlıydı
Sorular `knowledge_map.jsonl`'den `i % 600` **satır adımıyla** seçiliyordu. Ama
`knowledge_map.jsonl` CI'da (`kbmap.yml`) her gün yeniden üretiliyor: satır
sayısı değişmese bile satır **içeriği** değişiyor → aynı adım **farklı
soruları** seçiyor.

Satır numarası kanıtı: 01.10 adımı tam 600 (0, 600, 1200, 2400…), 29.09 adımı
~604 (0, 600, 1200, **2404, 3008, 3612**…).

**Ölçülen sonuç: 29.09 raporu ile 01.10 raporunda yalnızca 3/50 ortak soru.**
Yani "ekrana çıkan metin %52 → %76" gibi **zaman serisi iddiaları geçersizdi**
— 47/50 soru değişmişti.

Artık sorular `tools/soru_listesi.json`'dan okunur. Dosya repodadır, veri
dosyası **değildir** (ölçüm tanımıdır) ve **elle değiştirilmemelidir**.

### Hata 3 — `np.random.seed(7)` bilgi metnini KONTROL ETMİYORDU (asıl hata)
`brain.py:1467` ve `brain.py:1469` → `random.choice(responses)`. Bilgi
intent'inde sorgu kelimeleri yanıtla eşleşmediği için `best_score <= 0` olur →
**yanıtlar listesinden rastgele biri seçilir** ve bu metin LLM'e bilgi olarak
girer. Python `random` modülü NumPy tohumundan **bağımsızdır**.

Doğrulama zinciri:
1. `corpus.search` deterministik mi? **EVET** — 12/12 aynı, hem aynı süreç
   içinde hem iki ayrı süreç arasında. Retrieval suçlu değil.
2. Sınıflandırma etiketi değişiyor mu? **HAYIR** — her iki çağrıda da aynı
   (`cesma`, `waldner`, `umeklidinyum`, …).
3. O halde fark nerede? **Aynı süreç içinde aynı soruyu ikinci kez sorunca
   `kb` metni 6/11 soruda değişiyor.** Değişen metinler aynı intent'in
   farklı yanıt varyantları (doğallaştırılmış "kısaca/aynı zamanda" dolgulu
   varyant vs. temiz orijinal).

Düzeltme: `uretim_olc.py` her sorudan önce `np.random.seed(SEED)` **ve**
`random.seed(SEED)` çağırır.

**Doğrulama:** `det_a` ve `det_b` iki ayrı koşu → 6 alanın 6'sı da 50/50
birebir aynı (kb, üretim metinleri, kapı kararları, kullanıcıya yanıt,
`uretilildi`, rapor sadakatı). Araç artık gerçekten deterministik.

### Ölçüm değişti, model değişmedi — güncel taban (01.10 modeli, 50 sabit soru)

| ölçüm | 29.09 notları (geçersiz) | 01.10 ölçülen |
|---|---|---|
| boş kova ("bilgim yok") | ~%40 | **%0** (0/50) |
| deneme kabul oranı | %34 | **%47** (71/150) |
| ekrana üretim geçen soru | %52 | **%72** (36/50) |
| sadakat | %22–31 | **%60** (n=36) |
| sadakat < %34 | — | 7/36 |

### Bulgu (01.10): "kapı en sadık metinleri çöpe atıyor" — **VE BU YORUM YANLIŞTI**

Aşağıdaki tablo 01.10'da ölçüldü ve "en sadık metinler (%97) atılıyor"
yorumu yapıldı. **Bu yorum yanlıştır**; §6.12'de n-gram ölçümüyle çürütüldü.
Tabloyu burada bırakıyorum çünkü aynı hatanın tekrarlanmaması gerekiyor:
**sadakat tek başına kalite ölçütü değildir.**

| kapı kararı | n | sadakat (düzeltilmiş ölçüm) |
|---|---|---|
| KABUL EDİLEN | 71 | %59,2 |
| RET: "özgünlük yok" | 37 | **%97,4** |
| RET: "konu kelimesi yok" | 27 | %9,2 |
| RET: "konu bileşimi düşük" | 15 | %18,3 |

> **Sadakat bu noktada bir kopya dedektörüdür, kalite ödülü değildir.**
> sadakat = üretilen metnin içerik kelimelerinden kb'de BULUNANLARIN oranı.
> Metin kb'yi kopyaladıkça sadakat **yükselir**. Yani %97 sadakat = "neredeyse
> tamamı kb'den kelime" = **kopya**. Kapının reddettiği metinlerde yüksek
> sadakat iyi işaret değildir. §6.12'de 1-4 gram BLEU ile ölçüldü.

## 6.11 Depo testleri: 3 kırmızı test **ÇÖZÜLDÜ** → 587 test OK (01.10.2026 21:0x)

**Başlangıç:** `python -m unittest discover -s tests` → 587 test, **3 hata**. Üçü de benim
değişikliklerimden değil, **CI'ın veriyi büyütmesinden**. İki ayrı neden ölçüldü:

**Bitiş (01.10 21:05): `Ran 587 tests in 511.4s` → `OK (skipped=1)`.**
Üç test de düzeltildi: eski "kırpma aktif olmalı" varsayımı yerine gerçek
invaryant yazıldı (`test_butce_veriyi_zararli_asmaz`), hardcode `kadin nedir`
listesi yerine canlı veriden türetildi. **Üretim koduna dokunulmadı.**

> Not: süre ~270 sn değil, **511 sn** (veri büyüdükçe artıyor).
> Sonraki ölçüm (zaman tavanı düzeltmesinden sonra): **589 test, 452,7 sn** —
> §6.16'da iki test daha eklendi ve `MS_PER_HAM_CIFT_EPOCH` ölçüme bağlandı.

### Neden 1 (2 test) — `MAX_PAIRS` kırpması **tamamen ölmüş**

`train_llm.coz_max_pairs` yalnızca `18,85 × intent_sayısı` kullanıyor
(`train_llm.py:939`); ctx-temeli `3,08 × benzersiz_ctx` karşılaştırması
**hiç yapılmıyor** (`:245` `assertLess(butce, uretilen)` bunu zaten istiyor).

| tarih | intent | ham çift | bütçe | kırpma |
|---|---|---|---|---|
| 28.09 ölçüm | 6.364 | 162.975 | 120.000 | **%73,6** (gerçek darlık) |
| 01.10 koşusu (08:00) | 14.300 | 270.178 | 269.555 | %99,8 (623 çift) |
| 01.10 yerel ölçüm | 14.647 | 276.016 | 276.096 | **yok — bütçe veriyi aşıyor** |
| **01.10 koşusu (23:26)** | **15.408** | **288.802** | **290.441** | **yok — bütçe %+0,57 FAZLA** |

Kırpma dört haftada **%73,6 → %99,8 → yok → yok** eridi. Sabit `18,85`
veriyla büyümüyor: gerçek çift/intent oranı 28.09'da 25,6 iken son koşuda
**288.802/15.408 = 18,74**. Yani `18,85 × 15.408 = 290.441` ile ham çift
`288.802` **yine neredeyse tam denk** (+1.639 çift fazla). Bu üçüncü
bağımsız ölçüm: sabit karpan veriyle birlikte kayan bir sayı, tesadüf değil.
Bir sonraki koşuda bütçe hiçbir şeyi kırpamaz (+%0,57 çift = daha uzun
epoch — §6.15'te bu artık zaman riski olarak ölçüldü).

> **Düzeltilmedi — karar senin.** Seçenekler: (a) `coz_max_pairs` iki formülün
> **min()**'ini alsın (bugün 273.042 → kırpma geri gelir, 2.974 çift düşer);
> (b) karpanı ölçüp güncelle; (c) kırmızı kalsın, "darboğaz taşındı" işareti.
> (a) ve (b) **eğitim veri bütçesini değiştirir** → ayrı ölçüm ister.

### Neden 2 (1 test) — `kadin_tanim` artık veride

`test_pattern_dizini` "kadin nedir" veride yok diye varsayıyor; CI
`kadin_tanim` intent'ini eklemiş (6 desen, 1 yanıt). Testin varsayımı eskidi,
kod doğru çalışıyor.

**Ders (proje çapında):** testler ölçülmüş sabitleri hardcode ediyor ve veri
otomatik büyüyor → depo kendi kırmızı çubuğunu kendi üretiyor. Ölçüm
disiplininde "veri sürümünü raporla" kuralı teste de yazılmalı.

---

## 6.12 Özgünlük kapısı eşiği taraması — paired, 8 eşik (01.10.2026)

Soru: `ozgunluk >= %15` kuralı (`brain.py:2126` sohbet, `:2163` bilgi)
`%5-8`'e çekilirse ne getirir?

### Yöntem: neden 1 koşu yeterli
Üretim yolu **3 adayı her zaman** üretir ve geçenler arasından **en uzunu**
seçer (`brain.py:2033-2042`). Yani aday metinleri de seçim kuralı da eşikten
bağımsızdır → tek koşuda adayları kaydedip eşiği değiştirerek **yeniden
hesaplamak** birebir aynıdır. 8 eşik × 8 koşu yerine 1 koşu.

Kapının kopyası önce **doğrulandı**: eldeki rapordaki 150 adayda asıl
`brain._accept_kb_rephrase` kararıyla **150/150 aynı**. Sapma olsaydı tarama
geçersiz sayılacaktı.

### Tarama sonucu (50 sabit soru, tek koşu)

| eşik | aday kabul | ekrana üretim | sadakat |
|---|---|---|---|
| **0,15 (bugün)** | 71/150 | 36/50 = %72 | %60 |
| 0,12 – 0,10 | 75/150 | 38/50 = %76 | %62 |
| 0,08 | 80/150 | 38/50 = %76 | %62 |
| 0,06 – 0,05 | 85–86/150 | 39/50 = %78 | %63 |
| 0,03 | 87/150 | 40/50 = %80 | %64 |
| 0,00 (kural tamamen kalktı) | 108/150 | 43/50 = %86 | %70 |

İlk bakışta "eşiği çek, sadakat %60 → %70, ekrana çıkan %72 → %86" gibi
görünüyor. **Bu yanıltıcı.**

### İkinci ölçüm: sadakat bir kopya dedektörü

Sadakat = üretilen metnin içerik kelimelerinden kb'de bulunanların oranı.
Metin kb'yi **kopyaladıkça sadakat yükselir**. Yani yüksek sadakat, bu
metrikte "iyi" değil **"kopya"** demektir. Bu yüzden ayrıca
`eval_llm.bleu(kb, gen)` (1-4 gram) ölçüldü.

**Ölçüm tuzağı (ikinci kez aynı hata):** ilk hesapta ham Türkçe/ASCII
karşılaştırması yapıldı → bleu **0,10–0,20** çıktı, ama bu yapay olarak
düşüktü: model çıktısı ASCII, kb Türkçe; `menevsiye` ile `menevşiye` farklı
token sayılıyor. (01.10'da sadakat metriğinde bulunan hatanın aynısı.)
**İki taraf da `ascii_normalize` katlandıktan sonra:**

| grup | n | novel | **bleu_ASCII** | sadakat |
|---|---|---|---|---|
| 0,15'te KABUL (bugün ekrana çıkan) | 71 | 0,452 | **0,362** | %59,8 |
| 0,05'te YENİ kabul | 15 | 0,091 | **0,700** | %93,6 |
| 0,00'da YENİ kabul | 22 | 0,002 | **0,866** | %100,0 |

Yani bugün ekrana çıkan metinler kb'ye göre **0,36** kopya; eşiği 0,05'e
çekince ekrana girecek metinler **0,70**, kural tamamen kalkınca **0,87**.

### Karar: eşik ÇEKİLMEDİ

Gerekçe iki ölçülmüş olguya dayanıyor:

1. Eşiği çekmek ekrana **daha çok kopya** metin koyuyor (0,36 → 0,70).
2. Yeni kabul edilen metinler gözle de bozuk:
   `"Konusmadigimiz seyler var,, yani turk sarkici..."` (çift virgül),
   `"Istanbul universitesi orman, şöyle ki fakultesi bahcekoy..."`
   (yan cümle düzeni bozulmuş), `"e ile uyesi"` (`EXILE` bozulmuş).

### Asıl bulgu: kapının yanlış yeri

Kapı reddettiğinde ekrana giden metin **ham kb**'dir — yani kendi ölçütüne
göre **bleu 1,000**, yani %100 kopya. Kapı 0,70 kopyalı bir metni reddedip
kullanıcıya 1,00 kopya metin gösteriyor.

> **Ölçülebilir çelişki:** eşiği çekmek değil, **reddedildiğinde gösterilen
> yedek metni** düzeltmek kazandırır. Bu, §7.3'teki sıradaki ölçülebilir aday.

## 6.13 `normalize.ascii_normalize` BÜYÜK HARF BOZUYOR (01.10.2026)

Bulgu: `brain.py:1467/1469` incelenirken görüldü, kök neden `normalize.py:16-19`
ve docstring'in **tersini** yapıyor.

```
EXILE -> EXiLE | AI -> Ai | IT -> iT | GENERATIONS -> GENERATiONS
Çesma -> cesma | Üniversite -> universite | Irak -> irak | Isparta -> isparta
```

Kök neden: `'I': 'i'` ve `'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'Ö': 'o', 'Ş': 's',
'Ü': 'u'` — hepsi **küçük** ASCII'ye çeviriyor. Docstring "buyuk/kucuk korunur"
diyor.

### Ölçülen etki alanı (canlı veri)

| | sayı |
|---|---|
| bozulan token (sadece büyük/küçük) | **34.697** |
| corpus tokenlarına oranı | **%0,484** |
| en çok bozulan | `II→ii` (3.542), `I→i` (3.342), `III→iii` (1.057), `FIFA→FiFA` (974), `Irak→irak` (923), `COVID→COViD` (492) |

Romen rakamları, kısaltmalar ve özel adlar (Isparta, Irak, Illinois, Island,
Ipomoea) en çok etkilenenler — yani **varlık retrieval'ının** taşıdığı içerik.

### Ama kullanıcıya görünen hasar küçük

| | etkilenen |
|---|---|
| kullanıcıya giden 50 yanıt | **2 (%4)** |
| üretilen 150 aday | 8 (%5) |

**DÜZELTİLMEDİ — karar senin.** Nedenleri:
- `normalize.py` **tek kaynak**: brain, corpus, generator, seqgen hepsi
  kullanıyor. Değiştirmek retrieval + eğitim verisini aynı anda etkiler;
  etki alanı ölçülmeden dokunulmaz (proje kuralı).
- Tutarlılık şu an **doğru**: sorgu da metin de aynı fonksiyondan geçtiği
  için `"isparta"` ↔ `"Isparta"` eşleşmesi bozulmuyor. Hasar yalnızca
  **görüntülemede** (`deasciify` sözlüğü kirkin token'ı bulamıyor → `EXiLE`).

### Reddedilen hipotez (ölçüldü, doğru çıkmadı)
`"EXILE" → "e ile"` çıktısının bu hatadan geldiği varsayıldı → **yanlış**.
Ölçüldü: corpus'ta `EXILE` yalnızca **7 kez** geçiyor (3 `EXILE`, 4 `Exile`);
`"e ile"` örneklerinin 2'si de meşru Türkçe (`"ya da e ile ifade edilen"`).
Yani bu, **nadir varlık başarısızlığı**; normalize hatasının zincir etkisi değil.
Hipotez ölçülmeden yazılsaydı yanlış kayda geçecekti.

---

## 6.14 YENİ EĞİTİM ÖLÇÜLDÜ: val loss İYİLEŞTİ, EKRANA ÇIKAN METİN AZALDI (01.10.2026 23:26 koşusu)

**Bu, projedeki ilk "proxy iyileşirken gerçek kötüleşir" olayıdır.**

### Kurulum (ölçümden önce)
`model/nde-irma.zip` (62,7 MB) içinde `llm_model.json` + `llm_model_weights.npz`
+ `kaggle_train.log`. `model_kur.py` ile kuruldu (yedekle → doğrula → atomik
değiştir → doğrula). Aynı mimari: **101 tensor, 16.905.856 parametre, config
birebir aynı**; ağırlıklar tamamen farklı (max mutlak fark 6.696) → gerçekten
yeni bir eğitim.

### Koşu karşılaştırması
| | önceki koşu (01.10 08:00) | yeni koşu (01.10 23:26) |
|---|---|---|
| intent | 14.300 | **15.408** |
| ham çift | 270.178 | **288.802** |
| MAX_PAIRS | 269.555 | 290.441 |
| kırpma | %99,8 | **%100,0 → kırpma YOK** |
| eğitim çifti | 1.291.579 | **1.377.895** |
| RAG | %50,1 | **%47,1** |
| **val en iyi** | 0,3274 @ ep 6 | **0,3129 @ ep 8** |
| acc | 0,918 | **0,925** |
| süre | 338,8 dk | **427,1 dk** |

val loss **%4,4 iyileşti**, acc **+0,7 puan**. Eğitim daha uzun sürdü ve daha
çok veri gördü. **Metriklere bakılırsa kazanmış.**

### Kullanıcının gördüğü ölçüm: KAYIP
50 sabit soru, aynı veri, aynı seed (7), aynı araç — sadece ağırlıklar farklı:

| | eski model | yeni model | fark |
|---|---|---|---|
| **ekrana üretim** | **36/50 = %72** | **26/50 = %52** | **−20 puan** |
| sadakat | %59,6 (n=36) | %55,7 (n=26) | −3,8 puan |
| deneme kabulü | 71/150 = %47 | 55/150 = %36 | −16 deneme |
| boş kova | 0 | 0 | — |

**Karar: yeni model geri alındı.** `model/llm_model.json` =
`model/yedek/0110_2326/llm_model.json` (01.10 08:00 modeli). Yeni model
`model/nde_irma_0110_2326/` ve `model/yedek/geri_alindi_0110c/` altında duruyor.

### Ölçüm güvenilirliği (bunlar kanıtlandı, varsayılmadı)
- **kb metni 50/50 soruda birebir aynı** → fark yalnızca üretimden.
- `uretim_det_a` ≡ `uretim_det_b` ≡ `uretim_eski_0110c` (**50/50 yanıt aynı**)
  → ölçüm tamamen deterministik; **−20 puan gürültü değil, model kaynaklı.**
- İki raporun generator metin sayısı birebir aynı (134.863) → 20:09'da
  yazılan yeni `chatgrow_hf_20261001_1954.jsonl` ikisine de girmemiş, karışma yok.
- Yanıtlar birbirinden **22–40% benzer** (yeni model kopyalamıyor, farklı
  metinler üretiyor) → düşüş "ezberleme" değil.

### Ret nedenleri kaydı (kapı değil, üretim değişmiş)
| ret grubu | eski | yeni | fark |
|---|---|---|---|
| özgünlük yok | 37 | 38 | +1 |
| **konu kelimesi yok** | 27 | **35** | **+8** |
| **konu bileşimi çok düşük** | 15 | **22** | **+7** |
| toplam red | 79 | 95 | +16 |

Kopyalama değil: yeni model **konu kelimesini/metnini taşımayan** cümleler
üretiyor. Örnek (`waldner`): yeni metinler kb'deki `"hermann waldner"`,
`"14 eylül 1908"`, `"sektörünün lider"` bilgisini taşıyor; eski metinler
`"Aldner holding gmbh, merkezi almanya'nın ..."` gibi konudan kopuk.

### Yeni modelin ekran metinleri daha temiz (bu yönde iyi)
`cesma`, `franz bohme`, `tcg ufuk`, `mapillary`, `altın rengi yaprakbülbülü`,
`niCARAGUA arması` gibi 18 soruda yeni model **kopyalama yapmadığı** için
ham kb'yi (`"çesma (rusça: ), 1880'lerde ..."`) ekrana koydu. Kapı reddetti,
sistem canned'a düştü, **kullanıcı bozuk metin gördü**. Bu §6.12'deki
"kapı reddedince ekrana ham kb çıkıyor" bulgusunun ikinci kanıtı.

### ANA BULGU: `val loss` bu sistemde üretim kalitesini ölçmüyor
Val loss **token tahmin hatası** ölçer (bir sonraki token'ı ne kadar iyi
bilmesi). Ekrana çıkma ise **kapının 3 denemede birini kabul etmesi** +
**metnin konu taşıması**. Bunlar farklı sorular. Yeni koşu ikisinde de
iyileşti, birinde kötüleşti.

**Önerilen sonraki ölçüm (yapılmadı):** val loss + üretim metriği birlikte
raporlanmalı; `kaggle_train.txt` val eğrisi tek başına başarı ölçütü
sayılmamalı. `train_llm.py` val eğrisi sonuç üretmiyor — erken durdurma
var ama "bu koşu daha kötü" bilgisi kayboluyor.

---

---

## 6.15 Zaman tavanı **fazla** tahmin ediyor (01.10.2026 koşusundan ölçüldü)

> **01.10 sonrası düzeltme:** başlıkta "%3,75 KAT" yazıyordu. Doğru türetilmiş ama **karşılaştırma tabanı** eksikti: 540 dk'nın *tamamı* kullanılmıştı. Kodun kendi politikası (×0,75 − encode) uygulanınca hata **%5,34 KAT**. Düzeltildi ve **§6.16'da asıl hatanın birim hatası olduğu ölçülerek gösterildi** — sabitlerin güncellenmesi tek başına yetersizmiş.

**`sure_ve_hesapla()` oturumu taşırma riski için var. Bugün ölçüldü ki
bugün hâlâ güvenli, ama veri büyüdükçe sessizce taşıracak.**

### Logdan ölçülen gerçek (kaggle_train.txt, 01.10 23:26)
| | değer |
|---|---|
| MAX_PAIRS bütçesi | 290.441 |
| zaman tavanı | **1.251.733 → bütçeye TEĞMEDİ** |
| ham çift | 288.802 (bütçenin **%23,1**'i) |
| **DİKKAT: log modeli işaretlemiyor** | `kaggle_train.txt` bu koşunun Kaggle çıktısının logudur ancak **hangi model mimarisiyle** eğitildiğini (d_model/blocks/heads) içinde yazmıyor. `model/llm_model.json` ile **zorunlu olarak eşleştir** (mevcut aktif: `d_model=384, num_blocks=6`). Log aynı isimle birden çok koşuya ait olabilir; logu **yere kopyalamadan önce** her zaman model mimarisiyle çapraz doğrula. |
| kırpma payı | **+1.639 çift (%+0,57)** |
| işlevsel açılış sonrası | 303.892 (×1,0523) |
| doğal varyant (×5) sonrası | 1.377.895 (×4,5342) |
| train / val | 1.240.102 / 137.793 |
| token | 125,1M (100,88 tok/çift; `TOKEN_PER_PAIR=100` → sapma %+0,88) |
| epoch toplamı (12) | 25.620,4 sn = **427,0 dk** |
| BPE encode | 2.420,0 sn = **40,3 dk** |
| **gerçek duvar saati** | **≥ 467,3 dk / 540 dk = ≥ %86,5** |

`encode` logdaki "Toplam eğitim süresi"ne **dahil değil** (o sadece epoch
toplamı). Veri hazırlığı (klon, BPE öncesi işlemler) **hiç ölçülmüyor** →
gerçek oturum tüketimi **≥ %86,5**.

### `train_llm.py` sabitleri gerçekte ne kadar yanlış
| sabit | varsayım | ölçülen | hata |
|---|---|---|---|
| `MS_PER_PAIR` (`:858`) | 1,5139 ms/çift | **1,7217 ms/çift** | **%12,1 fazla hızlı** |
| `ENCODE_DK` (`:879`) | 26,0 dk | **40,3 dk** | **%35,5 fazla hızlı** |
| `VARSAYILAN_BOSLUK` (`:892`) | %75 (405 dk) | epoch başına 31,58 dk ayrılmış, **gerçek 35,58 dk** | **%+12,7** |

### Formülün dediği vs gerçek
```
sure_ve_hesapla(540 dk, 12 epoch) = 1.251.733 çift      ← formül
540 dk'da ölçülmüş hızla sığacak   =   333.703 çift      ← gerçek (tam 540 dk)
                                    → formül %3,75 KAT fazla
```
> ⚠️ **Bu blok yalnız başına okunmasın.** %3,75 KAT **540 dk'nın tamamını**
> kullanan karşılaştırmadır. Kodun kendi politikası (×0,75 − encode) uygulanınca
> doğru sınır **234.207** ve hata **%5,34 KAT**. Düzeltildi → **§6.16**.

**Neden büyük fark var (ölçüldü):** formül `ms_per_pair`'ı doğrudan
uyguluyor, ama **çift sayısı bütçe birimi (pre-expansion), süre ise
post-expansion (×4,77) üzerinden ölçülmüş.** Yani formül birimsiz.
`MS_PER_PAIR` yorumunu `1,7217 × 4,7711 = 8,215 ms/bütçe-çifti` olarak
düzeltse bile, `bosluk` ve `encode` sapmaları kalıyor.

> **Bu teşhis doğru çıktı ama eksikti:** §6.16'da ölçüldü ki `MS_PER_PAIR`'ı
> bile doğru değere çekmek (`1,7217`) tavanı yalnızca **%12,1** düşürüyor —
> **yetersiz.** Asıl düzeltme sabitin birimini değiştirmekti.

### Bu ne zaman patlar
Veri büyümesi ölçüldü: 270.178 → 288.802 = **%+6,9 / eğitim koşusu**.
Aynı hızla **3 koşu sonra** (≈ 3 gün) gerçek sığan çift 333.703'e ulaşır ve
oturum taşar. `MS_PER_PAIR` ve `ENCODE_DK` düzeltilirse sınır ~%15 geri
çekilir.

**Karar verildi → §6.16.** (Bu blok 01.10 23:56'da "karar verilmedi"
diye duruyordu. Ölçüm sonucu: iki seçeneğin de **tek başına yetersiz**
olduğunu gösterdi; asıl düzeltme birim hatasıydı. Ayrıca "%3,75 KAT"
rakamının karşılaştırma tabanı eksikti — §6.16'da düzeltildi.)

---

## 6.16 Zaman tavanı: **BİRİM HATASI** bulundu ve düzeltildi (01.10 2026)

> ⚠️ **Bu bölümdeki iki sayı §6.18 ile DÜZELTİLDİ:** `8,0911` →
> **7,3927**, tavan `234.207` → **256.333** (encode iki kez sayılıyordu).
> Aşağıdaki "Kırpma artık gerçekten çalışıyor — bedeli ölçüldü" tablosu
> da **hatalı**: kırpma %0,9 değil **%2,08 kapsamayı** siliyordu.
> Ayrıntı ve doğru ölçümler için **§6.18**.

§6.15 "formül birimsiz" diye teşhis koydu ama **iki seçeneği de ölçmemişti**.
İkisi de ölçüldü: **ikisi de tek başına yetersiz.**

#### Ölçüm (`tools/sure_olc.py`, kaynak `kaggle_train.txt`)

| seçenek | tavan (540 dk, 12 epoch) | önceki tavanın kaçı | yeterli mi |
|---|---|---|---|
| **VAZIF** (sahadaki kod) | **1.251.733** | — | ✗ |
| A) sadece `MS_PER_PAIR` → 1,7217 | 1.100.682 | %88,0 | **HAYIR** (sadece %12,1 düşürür) |
| A′) emniyet payı `× 0,80` | 880.546 | %70,4 | **HAYIR** |
| B) ham çift sabiti (8,0911 ms/ham/epoch) | **234.207** | **%18,7** | **EVET** |

**A seçeneğinin yetersizliği ölçümle kanıtlandı** — bu yüzden
`MS_PER_PAIR`'ı tek başına güncellemek yanlış çözümdü.

#### "3,75 KAT" rakamı düzeltmesi

§6.15'teki **%3,75 KAT** rakamı **doğru türetilmiş ama yanlış tabanla
karşılaştırılmıştı**: 540 dk'nın **tamamını** kullandı (333.703 çift). Kodun
kendi politikası oturumun **%75'ini** veriye ayırıyor ve `ENCODE_DK`'yı düşüyor:

```
540 dk × 0,75 − 26 dk encode = 379 dk  →  234.207 ham çift
```

Yani **kodun kendi politikasına göre hata %5,34 KAT** (1.251.733 / 234.207).
İki rakam da doğru, farklı tabanlar. Kayıtta **%5,34** esas alındı.

#### Uygulanan düzeltme

- Yeni sabit **`MS_PER_HAM_CIFT_EPOCH = 8.0911`** (ms / ham çift / epoch).
  Expansion'ı (4,7711) ve token uzunluğunu **içine sindiren tek ölçüm**:
  `28.040,4 sn (12 epoch + encode) / 288.802 ham çift / 12 = 8,0911 ms`.
- `sure_ve_hesapla` artık `ms_per_ham=MS_PER_HAM_CIFT_EPOCH` kullanıyor ve
  docstring'e **birim sözleşmesi** (ham çift = pre-expansion) yazıldı.
- `MS_PER_PAIR` **değiştirilmedi** (1,5139) — artık bütçe formülünü beslemiyor
  ama `test_olculen_sabitler_birlikte_tutarli` onu `kaggle_start.sh`
  yorumundaki ölçüme bağlıyor. `MS_PER_PAIR`'ın yorumundaki "9 saatlik
  oturuma ~1,46 MILYON çift sığıyor" cümlesi **yanlıştı** ve düzeltildi:
  hata sabitte değil **birimdeydi**.

#### Kırpma artık gerçekten çalışıyor — bedeli ölçüldü

| | sınırsız | zaman tavanı 234.207 | kayıp |
|---|---|---|---|
| ham çift | 276.016 | 234.207 | **%15,1** |
| benzersiz ctx | 88.650 | 87.829 | **%0,9** |
| benzersiz yanıt | 45.705 | 45.645 | **%0,1** |

**%15,1 çift gidiyor ama %0,9 kapsama.** Çift sayısı kombinasyoneldir
(N pattern × M yanıt); önce tekrarlar kesiliyor. Zamana karşı alınan
bedel **ucuz** — bu ölçüm kararın temeli.

#### Testler artık döngüsel değil

| test | önce | şimdi |
|---|---|---|
| `test_tavan_oturum_butcesini_asmaz` | `MS_PER_PAIR` ile **döngüsel** | `MS_PER_HAM_CIFT_EPOCH` ile birim-doğru |
| `test_sure_ve_hesapla_temel_deger` | `MS_PER_PAIR` türetiyordu, `>1.000.000` zorunluydu | 234.207 bekleniyor, bant 150.000–400.000 |
| `test_zaman_tavani_olculen_sinirda` | **YOK** | **YENİ** — sabiti **ölçülen 97,09 ms/ham**'a bağlar; üst sınır 333.699 (540 dk), alt sınır `288.802 × %75` (01.10'da bitişi **kanıtlanmış** hacim) |
| `test_simdiki_butce_tavana_siuyor` | `coz_max_pairs(yaz=False)` → **tavan hiç uygulanmıyordu** | `ust_tavan=tavan` geçirir, gerçek boru hattını yansıtır |
| `test_kirpma_kapsamayi_agirmeden_atmiyor` | "kırpma aktif olmalı" (yanlış varsayım) | **YENİ** — kırpma benzersiz ctx kapsamasını ≥%99 tutuyor mu |

**Bulunmayan boşluk:** hiçbir test sabitlerin **doğru** olduğunu
ölmüyordu — hepsi formülü aynı sabitlerle yeniden hesaplıyordu.
`test_olculen_sabitler_birlikte_tutarli` 29.09'daki 4,5 KAT'lık hatayı
kapatmıştı ama **doğru birim yanlış sayıyı meşru kılıyordu.**

---

## 6.17 Yedek metin ölçümü: kapı reddedince çıkan ham kb **DÜZELTİLMEDİ** (01.10 2026)

`tools/yedek_olc.py`, 50 sabit soru × 4 varyant (tries × sıcaklık),
`olcum_raporlari/yedek_olc_0110.json`.

#### Ekran oranı ve sadakat

| varyant | tries | temp | üretim | sadakat | aday | kabul |
|---|---|---|---|---|---|---|
| bugün (taban) | 3 | 0,7 | **%72,0** (36/50) | %59,6 | 150 | 71 (%47,3) |
| t6_t07 | **6** | 0,7 | **%86,0** (43/50) | %57,1 | 300 | 140 (%46,7) |
| t3_t09 | 3 | **0,9** | %76,0 (38/50) | %57,3 | 150 | 68 (%45,3) |
| t6_t09 | 6 | 0,9 | **%86,0** (43/50) | %54,0 | 300 | 133 (%44,3) |

**Kilit bulgu: kaldıraç `tries`, sıcaklık değil.**
`tries 3→6` ekran oranını **%72→%86 (+7 soru)** çıkarıyor, **kapı kuralına
dokunmadan**. Sıcaklık 0,7→0,9 tek başına sadece +2 soru veriyor.
Bedeli: aday sayısı **×2,0** (üretim süresi ~2 kat) ve sadakat −2,4 puan.

#### "Kapı reddedince bozuk metin çıkıyor" SEZGİSİ ÖLÇÜMDE YANLIŞ ÇIKTI

Ekrana çıkan metin, kaynağına göre kalite işaretleriyle karşılaştırıldı
(4 varyant birleşik):

| işaret | **modelin metni** (n=160) | **ham kb yedegi** (n=40) |
|---|---|---|
| dolgu ("daha fazla detay…") | **%21,9** | **%7,5** |
| küçük harfle başlıyor | %0,0 | %7,5 |
| yarım cümle (kapanış yok) | **%8,1** | **%0,0** |
| ortalama karakter | 119,8 | 93,1 |

**Ham kb, modelin kendi metninden 2,9 KAT daha az dolgu içeriyor ve hiç
yarım cümle yok.** Beklenen tam tersiydi. Yani §6.12'deki "ham kb
ekrana koyuluyor" şikâyetinin **asıl kaynağı ham kb değil**; sorun
modelin ürettiği metnin içinde (dolgu %21,9, yarım cümle %8,1).

**Ham kb'nin tek kusuru:** %7,5 küçük harfle başlıyor (40 metinde 3).
Bu **veri** kusuru (`knowledge_map.jsonl`), araçla düzeltilmez —
veri dosyaları ölçüm araçları tarafından yazılmaz.

#### Karar verilmedi — `brain.py` değişikliği pahalı

`tries 3→6` **+14 puan** ekran oranı veriyor, ama:
1. Aday sayısı 150→300 → **üretim süresi ~2 kat**. Bu süre **ölçülmedi**
   (`yedek_olc.py` zaman damgası yazmıyor) → "ne kadar yavaşlar" bilmiyoruz.
2. Sadakat −2,4 puan düşüyor. Ekran oranı ile sadakat arasındaki
   denge noktası **tanımlı değil**.
3. `brain.py` üretim kodu; `eval_llm.py` ve ölçüm araçları import ediyor.

**Önerilen (ölçüme dayalı) sıra:** önce `yedek_olc.py`'ye süre damgası
ekle (deterministik kalır), sonra `tries` kararını **gecikme ölçümüyle**
birlikte vermek. Sıcaklık 0,9 **elenmiştir** (+2 soru için sadakat
−2,2 puan; `tries` aynı bedeli 3,5 KAT daha fazla kazançla veriyor).

---

## 6.18 Zaman tavanı: **İKİNCİ** birim hatası + kırpma **%2,08 kapsamayı** siliyordu (01.10–02.10 2026)

> **§6.16'daki iki sayı bu bölümle düzeltildi:** `MS_PER_HAM_CIFT_EPOCH`
> **8,0911 değil 7,3927**, tavan **234,207 değil 256,333**. §6.16'daki
> "kırpma bedeli %0,9 kapsama" ölçümü de **hatalı** (aşağıda).

### (a) Encode iki kez sayılıyordu — `tools/sure_olc.py`'nin kendi hatası

`MS_HAM` hesaplanırken encode de içine katılıyordu:

```
sure_olc.py (ESKI):  MS_HAM = EPOCH_SN*1000/HAM + ENCODE_SN*1000/HAM  = 97,0921 ms
sure_olc.py (YENI):  MS_HAM = EPOCH_SN*1000/HAM                        = 88,7127 ms
sure_ve_hesapla:      kalan_dk = oturum_dk*bosluk - encode_dk          <-- encode TEKRAR
```

Encode `sure_ve_hesapla` tarafından **zaten** düşülüyor; `MS_HAM`'e de
katılınca **iki kez** sayıldı. Sabit **%9,45 yüksek** → tavan gereksiz
dar. **İki hata da aynı yöndeydi** (tavanı daraltıyor).

| | ölçülen | sabit | tavan |
|---|---|---|---|
| encode'la | 97,0921 ms/ham | 8,0911 | 234.207 |
| **encode'sız (doğru)** | **88,7127 ms/ham** | **7,3927** | **256.333** |

**Ders (araca değil üretime de yazıldı):** sabiti türeten araç ile onu
kullanan kod arasındaki **her muhasebe kalemi** karşılıklı sayılmamalı.
`test_olcum_araclari.py` aracın bu satırı üretmesini kapatıyor.

### (b) `origin/main` verisi **yeni**ydi — yerel bayattı (merge)

Yerel `main` **ahead 2 / behind 8** idi ve iki commit'i de veri
dosyalarına dokunmuştu. "Reset 42.000 satır veri kaybı yapar" sezgisi
**ÖLÇÜMDE YANLIŞ ÇIKTI**:

| | bilgi intent |
|---|---|
| yerel çalışma ağacı / HEAD | 14.647 |
| `origin/main` | **16.225** (+1.578) |

Merge **çakışmasız** (`git merge-tree --write-tree` → exit 0) ve
`intents.json`/`corpus.jsonl`/`corpus_ids.jsonl`/`knowledge_map.jsonl`
merge sonrası `origin/main`'in **birebir aynısı**. Yani merge veriyi
**ileri** götürdü. Push'un tek veri etkisi: 3 `chatgrow_hf_*.jsonl`
(01.10 08:04/11:00/19:54, 0,19 MB) — deponun mevcut düzeni zaten bu
(origin'de 24.09–29.09'dan 8 tane var).

### (c) ANA BULGU: kırpma kapsamayı **sessizce** bozuyordu

`seqgen.load_pairs` son satırları: `random.Random(3).shuffle(pairs)` →
`pairs[:max_pairs]`. Rastgele kesme. **Medyan ctx'in yalnızca 3 çifti**
(min 1, maks 36) olduğu için küçük bağlamlar tamamen siliniyor:

| kap | önce: ctx % | **sessiz ctx** | sonra: ctx % | sessiz ctx |
|---|---|---|---|---|
| **256.333 (yeni tavan)** | 97,92 | **2.039** | **100,00** | **0** |
| ×0,90 | 95,90 | 4.028 | 100,00 | 0 |
| ×0,80 | 92,75 | 7.113 | 100,00 | 0 |
| ×0,70 | 88,38 | 11.398 | 100,00 | 0 |
| ×0,50 | 74,88 | 24.652 | 100,00 | 0 |
| ×0,35 (89.716 < 98.116 ctx) | — | — | 91,44 | 8.400 = 98.116−89.716 ✓ |

**Çift sayısı birebir aynı** → süre ve eğitim maliyeti değişmiyor; sadece
**hangi** çiftlerin seçildiği değişiyor. Yöntem: önce her ctx'den 1 çift
garanti, sonra kalan bütçe rastgele. Tohum 3/7/11/42'de de %100.

**Bu bir regresyondu, kazanç değil:** son gerçek koşu **%100,0 kapsama**
ile bitmişti (`[butce] ham 288.802 cift -> ... 100.0% kapsama`).
§6.16'daki düzeltme uygulansaydı bir sonraki koşuda 2.039 bağlam
**sıfır eğitim verisi** alacaktı.

**Asıl kazanç: kapsama artık zaman tavanından BAĞIMSIZ.** Tablodaki
%100, veri tabanının fiyatı değil, `butce / ctx` oranından gelir; bütçe
ctx sayısının altına düşünce kırpma lehine açıkça devreye giriyor.

### (d) Testler

| test | önce | şimdi |
|---|---|---|
| `test_kirpma_kapsamayi_agirmeden_atmiyor` | ≥**%99** eşiği (kirpma %97,85'te kırmızı) | **kesin sözleşme:** `benzersiz ctx == min(kap, benzersiz ctx)` |
| `test_kirpma_yariya_inse_de_kapsama_durur` | — | **YENİ** — ×1,0 / ×0,75 / ×0,5'te bütce dolar mı, kapsama durur mu |

**%99 eşiği neden bırakıldı:** kirpma sessizce 1.000 ctx kaybetseydi
geçerdi. Sözleşme ya sağlanır ya sağlanmaz.

Tam paket: **589 → 590 test**, `test_autogrow_kapi` 26 → **27**.

---

## 6.19 SOHBET HATTI ÖLÇÜLDÜ: boru hattında **hiç sohbet kaynağı yok** (02.10.2026)

Kullanıcı "sohbet sınıfı kalitesini artırmak için internetten kaynak ekle" dedi.
Kaynak eklemeden önce **ölçüldü**: mevcut hatta sohbet verisi ne kadar var?

### (a) "sohbet" adlı dosyaların çoğu sohbet DEĞİL

25.009 chatgrow çiftinin kaynak dağılımı (dosya adları + içerik örnekleri):

| dosya | çift | gerçekte ne |
|---|---|---|
| `chatgrow_hf_sohbet.jsonl` | 5.616 | matematik + film tanımı |
| `chatgrow_hf_2026*.jsonl` (14) | 16.800 | matematik / muhakeme / analiz |
| `chatgrow_discourse_pardus*.jsonl` (2) | 2.191 | **gerçek** Türkçe sohbet, ama teknik (Pardus Linux) |
| `chatgrow_kitap.jsonl` | 609 | kitap/şiir, parçalanmış |
| **`chatgrow_sohbet.jsonl`** | **198** | **tek gerçek gündelik sohbet dosyası** |

Eğitim karmasında sohbet payı **%1,69**, bilgi **%98,31**.

### (b) HF'daki "sohbet" kaynağı **sohbet değil** — kaynak kaynak çekilerek ölçüldü

`fetch_hf_turkish.py` 3 kaynak çekiyor; birinin adı sohbet diyordu. Her biri
**ayrı ayrı** çekildi (aynı tohum 7, `--max-pairs 300`, TEMP'e yazıldı):

| kaynak | 664 hamdan | dedup sonrası | süre |
|---|---|---|---|
| `tascib/turkish-instruction` | 664 | 613 | 26 sn |
| `erythropygia/ThinkingData-200K-Turkish` | 664 | **324** (kopya-ctx **314**) | 513 sn |
| `kilicai/turkish-sft-multi-turn-dialogue-10k` | 664 | 524 | 75 sn |

`kilicai/...multi-turn-dialogue-10k` örnekleri:

```
"hukumle ilgili asagidaki anlatimlardan hangisi"  -> hukuk coklu secim
"ky-024 hall effect sensor karti..."             -> teknik
"basliklari detaylandirarak anlat ve"            -> google ads stratejisi
```

**Adı "multi-turn dialogue" olan kaynak sohbet üretmiyor.** Üstelik en az
veren kaynak o. Yani hatta bağlı "sohbet kaynağı" yok.

### (c) Köken kaydı yok — seyreltme fark edilemiyordu

`fetch_hf_turkish.py` çıktıya yalnızca `query`/`answer` yazıyor (`:588`),
`source` alanını **düşürüyor**. Hangi çiftin hangi kaynaktan geldiği üretimde
bilinmiyor → sohbet kaynağının payının eridiği fark edilememiş. Kaynak
ağırlığı da kontrol edilmiyor: her kaynak `max_pairs*2+64` ham çekiyor,
sonra **verimine göre orantılı** karıştırılıp `max_pairs`'a kırpılıyor.

### (d) `chatgrow.py` Reddit çekicisi **hiç çalışmamış**

`DEFAULT_SUBREDDITS = ["r/Turkey"]` kodda duruyor; depoda **hiç
`chatgrow_reddit_*.jsonl` yok** → 0 kez çalışmış. Hazır ama hiç
kullanılmamış kaynak.

### SONUÇ — kaynak eklemeden önceki gerçek taban

Sohbet kalitesi için ölçüm tabanı **yoktu**: sabit 50 sorunun **50'si de**
bilgi sorusuydu, tek sohbet sorusu yok. "Sohbet %72" denilen her sayı
**bilgi-only**. Yeni ölçüm aracı: `tools/sohbet_olc.py` +
`tools/soru_listesi_sohbet.json` (40/40 sınıfın tamamı + 5 kenar soru).
Taban sonucu: `olcum_raporlari/sohbet_taban_0110.json`.

### Araca düzeltilen İKİ hata (rapor güvenilirliği)

**Hata 1 — "sınıf taşı" filtresi yanlış alana bakıyordu.** Beklenen etiket yerine
*tahmin edilen* etikete bakıyordu; 6 adet yanlış tahmin yanlışlıkla sınıf taşı
sayılıyordu. Doğru kriter **beklenen** etiketin sınıfta olması.
İlk koşunun "SINIF TASI: 7 soru" sayısı **geçersizdir**.

**Hata 2 — ölçüm kendi kendini kirletiyordu (daha ciddi).** Tek geçişte her soru
için önce `predict()` sonra `get_response()` çağrılıyordu; bir önceki sorunun
`get_response`'i **sonraki** sorunun `predict`'ini değiştiriyordu.

Bunu ölçerek doğrulandı (`predict()` iki kez arka arkaya → **45/45 aynı**, yani
`predict` kendi başına deterministik; kirlilik yalnızca `get_response →
predict` yönünde):

| soru | kirli koşu | temiz koşu |
|---|---|---|
| "izleyecek bir şey arıyorum" | `ari hjelm` | `tavsiye_isteme` |
| "havadaki uçak şu an neredeyse" | `an giang` | `1985 balikesir ucak kazasi` |

**Çözüm:** ölçüm **iki geçişe** ayrıldı — (1) sınıflandırma, hiçbir
`get_response` çağrısı yapılmadan; (2) üretim. Ayrıca "etiket olmayan tahmin"
iddiası **tamamen düştü**: temiz koşuda 45 sorunun **44'ü** geçerli etiket,
tek istisna `Anlayamadim` (meşru sentinel — artık istisna olarak tanınıyor).
Yani `patron bebek: yine is basinda`, `garipler` gibi görünenler **gerçek bilgi
intent etiketleriymiş**, bozuk çıktı değil.

Ana ölçü artık **kullanılabilir isabet**: tahmin doğru **ve** sınıf gerçekten
var. `brain.py:1746` `chosen_tag in self.intent_tags` kapısı açılmazsa doğru
tahmin de işe yaramıyor.

---

## 6.20 NORMALIZE BÜYÜK HARF + SOURCE ALANI + 3 SIZINTI INTENT DÜZELTMESİ (02.10.2026)

### 6.20.1 `normalize.ascii_normalize` büyük harf koruma düzeltmesi

**Sorun:** Docstring "buyuk/kucuk korunur" diyordu ama kod `'Ç': 'c', 'İ': 'i'` yapıyordu.
Tüm Türkçe büyük harfler (Ç, Ğ, İ, I, Ö, Ş, Ü) ASCII küçük harfe çevriliyordu.

**Ölçülen etki (öncesi):**
- Korpus token'larında büyük harf + non-ASCII içeren: 231.614
- Büyük harf tamamen kaybolan (küçülmüş): 58.479 (**%25,25**)
- En çok bozulan: İstanbul (2.542), Şubat (2.027), Üniversitesi (1.918)...
- deasciify'de acronym yanılgısı: İSTANBUL → "ISTANBUL" normalize → `w.isupper()`=True → kısaltma sanılıp sözlükten siliniyordu → model "istanbul" ürettiğinde "İstanbul" olmuyordu.

**Düzeltme:**
- `normalize.py:16-19`: `'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'I': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'`
- `brain.py:827`: acronym tespiti `w.isupper() and w.isascii()` → sadece ASCII büyük harf (NATO, AI, FIFA) kısaltma sayılır.

**Ölçülen etki (sonrası):**
- Büyük harf kaybı: **%0,0** (231.614 token, hepsi korundu)
- deasciify: "istanbul" → "İstanbul", "ankara" → "Ankara", "nato" → "NATO" (doğru)
- Retrieval tutarlılığı: sorgu + corpus aynı normalize'den geçtiği için **değişmedi** (her ikisi de "İstanbul" → "Istanbul")

### 6.20.2 `fetch_hf_turkish.py` `source` alanı eklendi

**Sorun:** §6.19(c) - HF kaynaklarından gelen verinin kökü kaybediliyordu (sadece `query`/`answer` yazılıyordu).

**Düzeltme:**
- Çıktı formatı: `{"query":..., "answer":[...], "source":"tascib/turkish-instruction"}`
- `dedupe_pairs_with_source` fonksiyonu eklendi — dedup/karıştırma sonrası kaynak korunuyor.
- Çoklu kaynak desteği: `--source` tekrarlanabilir, her çift kendi kaynağını taşır.

### 6.20.3 3 sızıntı intent'e `"tur": "sohbet"` eklendi

**Sorun:** §6.6 - `kavram_tanimi`, `tavsiye_isteme`, `gelecek_planarı` 6 desenli olduğu için bilgi sayılıyor, sınıflandırıcıya girmiyordu.

**Düzeltme:** `intents.json`'a her birine `"tur": "sohbet"` eklendi.
- Eski kural (desen >6): 40 sohbet sınıfı
- Yeni kural (açık işaret): 43 sohbet sınıfı (+3, sıfır sızıntı)
- Testler güncellendi: `test_sohbet_mi_gercek_veride_sonuc_degismedi` (eski kuralın üst kümesi), `test_sohbet_mi_sizinti_siniflar_gercekten_sohbet_di` (artık sohbete giriyor doğrulaması).

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

**Bu bölüm 01.10 23:56'da güncellendi: aşağıdaki ilk iki madde artık
BİLİNMEYEN değil, ÖLÇÜLMÜŞTÜR (§6.14, §6.15).**

~~`MS_PER_PAIR = 1,5139` ölçülmedi, GPU gerekiyordu.~~
→ **01.10 koşusundan ölçüldü: gerçek 1,7217 ms/eğitim-çifti (%12,1 daha yavaş).**
`ENCODE_DK = 26,0` → **ölçüldü: 40,3 dk (%35,5 daha yavaş)**.
~~`MS_PER_PAIR`'ın hangi çift sayımına ait olduğu belirsiz.~~
→ **ÇÖZÜLDÜ (§6.16): `eğitim` çifti. Formül `ham` çift sayıyordu →
birim hatası, 5,34 kat. `MS_PER_HAM_CIFT_EPOCH` eklendi ve `sure_ve_hesapla`
düzeltildi.** Tavan 1.251.733 → 234.207 → **256.333**.
~~`MS_HAM` encode'u içine katıyordu ama `sure_ve_hesapla` zaten düşüyor.~~
→ **ÇÖZÜLDÜ (§6.18a): encode iki kez sayılıyordu → sabit %9,45 yüksek.**
Doğru sabit **7,3927** (encode hariç). **İki hata da aynı yöndeydi.**
~~Kırpma bazı benzersiz ctx'leri tamamen susturuyor.~~
→ **ÖLÇÜLDÜ ve DÜZELTİLDİ (§6.18c):** `shuffle → kes` yeni tavanda
**2.039 benzersiz ctx'i (%2,08) sıfır eğitim verisiyle** bırakıyordu
(son gerçek koşuda kapsama %100,0'dı → **regresyon**). Yeni yöntem:
önce her ctx'den 1 çift. Çift sayısı ve süre aynı, kapsama **her bütçede
%100**. Kapsama artık zaman tavanından **bağımsız**.
Ayrıca: **sabitlerin doğruluğunu ölçen test YOKTU** (hepsi formülü aynı
sabitlerle yeniden hesaplıyordu) → `test_zaman_tavani_olculen_sinirda` yazıldı.

### Hâlâ ölçülmemiş (gerçekten bilinmeyenler)
1. **`val loss` neden üretimi ölçmüyor?** (§6.14) — yeni koşuda val iyileşti
   (+%4,4) ama ekrana çıkan metin %72→%52 düştü. **Mekanizma ölçülmedi.**
   Aday hipotezler (hiçbiri kanıtlanmadı): (a) model kb metnine çok yakın
   metin üretiyor, kapı "özgünlük yok" deyip reddediyor → eski modelin
   ürettiği "kopya" metinleri de reddediyordu ama biraz daha az;
   (b) ret nedeni kayması gösteriyor ki asıl değişen "konu kelimesi yok"
   (+8) ve "konu bileşimi çok düşük" (+7) → yeni model **konu taşımayan**
   cümleler üretiyor. Bu ikisi ölçüldü; **neden** ölçülmedi.
2. **`embed` std büyümesi (0,183→0,196) sadakati etkiliyor mu?** İlişki
   kurulmadı — 2 nokta, başka koşu yok.
3. ~~**Yedek metin kalitesi** — `tools/yedek_olc.py` yazıldı, çalıştırılmadı.~~
   → **ÖLÇÜLDÜ (§6.17).** Ham kb **daha temiz** çıktı: dolgu %7,5 vs
   modelin %21,9'i; yarım cümle %0,0 vs %8,1. Sezgi yanlıştı.
   Kalan bilinmeyen: `tries 3→6` **gecikme maliyeti ölçülmedi**.
4. **`normalize.py:16-19` ascii_normalize** etki alanı (retrieval tutarlılığı).
5. Doğallaştırma varlık adı bozuyor (`alyson hannigan` → *aleis denisof*);
   %1,6 varyantta içerik kapsamı <%50.
6. Aynı ctx için farklı yanıtlar karışıyor (bir konu/bölümün cevabı başkasının
   yerine geçiyor).
7. §8'deki 4 cevaplanamayan "X nedir" sorusu (`corpus.jsonl`'de tam adıyla
   kayıt yok).

### Ölçüm tarihçesi (hangi rapor ne zaman ölçüldü)
| rapor | etiket | veri sürümü | deterministik |
|---|---|---|---|
| `uretim_baza_0110.json` | 01.10 09:22 | eski normalizasyon | ❌ geçersiz |
| `uretim_duzeltilmis_0110.json` | 01.10 09:46 | düzeltilmiş | kısmi |
| `uretim_det_a.json` | 01.10 10:02 | düzeltilmiş + seed | ✅ |
| `uretim_det_b.json` | 01.10 10:07 | **aynı** | ✅ (≡ det_a) |
| `kapi_ozgunluk_0110.json` | 01.10 10:41 | eşik taraması | ✅ |
| `uretim_yeni_0110b.json` | 01.10 20:0x | **yeni model (23:26)** | ✅ |
| `uretim_eski_0110c.json` | 01.10 20:2x | **eski model (08:00), aynı veri** | ✅ (≡ det_a) |
| `yedek_olc_0110.json` | 01.10 23:5x | 01.10 modeli (08:00) | ✅ (seed 7) |
| (doğrudan log) `kaggle_train.txt` | 01.10 23:26 | — | ✅ (zaman tavanı ölçümü, §6.16) |


## 7.3 Sıradaki adım (02.10.2026 güncellendi)

Artık ölçüm güvenilir (§6.10 üç hata düzeltildi, araç deterministik
doğrulandı). Sıra şöyle:

1. ✅ **Ölçümü karşılaştırılabilir yap** — soru listesi sabitlendi, sadakat
   normalizasyonu düzeltildi, `random.seed` eklendi. Doğrulandı (6/6 alan
   50/50 aynı).
2. ✅ **Özgünlük kapısını çek** — **ÖLÇÜLDÜ, ÇEKİLMEDİ** (§6.12).
   8 eşik paired tarandı: eşik 0,15→0,05 ekrana çıkan metnin kb'ye göre
   kopyalık oranını **0,36 → 0,70**'e çıkarıyor. Yeni kabul edilenler gözle
   bozuk. Gerçek bulgu: kapı reddedince ekrana **ham kb** çıkıyor (bleu
   **1,000**) → kazanç eşikten değil **yedek metinden**.
3. ✅ **Yeni eğitimi ölç** — **ÖLÇÜLDÜ, GERİ ALINDI** (§6.14). val loss
   %4,4 iyileşti (0,3274→0,3129) ama ekrana çıkan metin **%72→%52** düştü.
   Deterministik olduğu doğrulandı (`det_a`≡`det_b`≡`eski_0110c`, 50/50).
   **Ana bulgu: `val loss` bu sistemde üretim kalitesini ölçmüyor.**
4. ✅ **MAX_PAIRS kırpma erimesi** — yeni koşuda da **%100,0, kırpma YOK**
   (§6.14, bütçe +%0,57 fazla). Bağımsız doğrulama.
5. ✅ **Yedek metni ölç** — **ÖLÇÜLDÜ (§6.17).** 50 sabit soru × 4 varyant.
   Sezgi yanlış çıktı: **ham kb modelin metninden daha temiz** (dolgu %7,5 vs
   %21,9; yarım cümle %0,0 vs %8,1). Gerçek kaldıraç **kapı değil `tries`**:
   `tries 3→6` ekran oranını **%72→%86** çıkarıyor, kapı kuralına dokunmadan.
6. ✅ **Zaman tavanı** — **ÖLÇÜLDÜ ve DÜZELTİLDİ (§6.16 + §6.18).** Asıl hata
   **birim hatasıydı** (eğitim çifti ↔ ham çift), sabit yanlış değildi.
   Tavan 1.251.733 → 234.207 → **256.333**. `MS_PER_HAM_CIFT_EPOCH` =
   8,0911 → **7,3927** (encode iki kez sayılıyordu, §6.18a).
7. ✅ **Kırpma kapsamayı sessizce bozuyordu — DÜZELTİLDİ (§6.18c).**
   `shuffle → kes` medyan-3-çiftli ctx'leri siliyordu: yeni tavanda
   **2.039 benzersiz ctx (%2,08) sıfır eğitim verisi** alıyordu, son gerçek
   koşuda ise kapsama **%100,0**'dı. Yeni yöntem: önce her ctx'den 1 çift.
   Çift sayısı aynı → süre aynı; kapsama **her bütçede %100** (×0,5'e kadar).
   Kapsama artık **zaman tavanından bağımsız**. Test: %99 eşiği yerine
   **kesin sözleşme** (`ctx == min(kap, ctx)`) + 1 yeni test.
8. ✅ **`origin/main` verisi yeniydi** — yerel `main` ahead 2 / behind 8 idi,
   "reset veri kaybı yapar" sezgisi **ölçümde yanlış çıktı** (14.647 vs
   16.225 intent). Merge çakışmasız, veri `origin/main`'in aynısı (§6.18b).
9. ✅ **`tries 3→6` süre maliyeti ÖLÇÜLDÜ** (`yedek_olc 0110b`). Beklenen
   bedel **ölçülerek doğrulandı**: `t6_t07` ekran oranı **%72→%86 (+16 puan)**,
   soru başına süre **×2,18**, sadakat **−0,4 puan**. Sıcaklık 0,9 elendi
   (`tries`'i 6'ya çıkarmadan ekran oranı +0 puan). **Karar kullanıcıda:**
   +16 puan için ×2,18 süre kabul edilebilir mi?
10. ✅ **Sohbet hattı ölçüldü ve taban kuruldu** (§6.19). Sabit sohbet seti
    (`tools/soru_listesi_sohbet.json`, **40/40 sınıf**) + ölçüm aracı
    (`tools/sohbet_olc.py`) yazıldı. **Bulgu: hatta sohbet kaynağı yok**
    (adı "sohbet" olan HF kaynağı sohbet üretmiyor; gerçek gündelik sohbet
    198 çift; Reddit çekicisi kodda duruyor ama 0 kez çalışmış).
11. ✅ **"40 sınıf kasıtlı" kaydı DÜZELTİLDİ** (§6.6) — kasıtlı karar değil,
    AutoGrow'un 6-desen şablonunun yan etkisi. Kural **sayıdan açık işarete**
    (`tur` alanı) taşındı, geriye uyumlu (40 → 40), +3 sınıf kazandı, sıfır sızıntı.
12. ✅ **`intents.json`'a 3 sızıntı intent'ine `"tur": "sohbet"` EKLENDİ** (02.10.2026)
    — `kavram_tanimi`, `tavsiye_isteme`, `gelecek_planlari` artık sohbet sınıfında.
    Testler güncellendi (`test_sohbet_mi_gercek_veride_sonuc_degismedi`,
    `test_sohbet_mi_sizinti_siniflar_gercekten_sohbet_di`). Sohbet sınıfı 40→43.
13. ✅ **`fetch_hf_turkish.py` `source` ALANI EKLENDİ** (02.10.2026) — Çıktı JSONL
    artık `{"query":..., "answer":[...], "source":"<dataset>"}` içeriyor.
    `dedupe_pairs_with_source` eklendi, kaynak bilgisi dedup/karıştırma sonrası da korunuyor.
14. ✅ **`normalize.ascii_normalize` BÜYÜK HARF BOZMA HATASI DÜZELTİLDİ** (02.10.2026)
    — Eski: `'Ç': 'c', 'İ': 'i'` (büyük→küçük). Yeni: `'Ç': 'C', 'İ': 'I'` (büyük/büyük korunur).
    Etki: korpus token'larında büyük harf kaybı **%25,25 → %0,0**. Acronym tespiti
    `brain.py`'de `w.isascii()` eklendi → Türkçe TAM BÜYÜK kelimeler (İSTANBUL)
    artık yanlışlıkla kısaltma sayılmıyor, deasciify sözlüğünde duruyor.
    Test `test_ascii_normalize` güncellendi. Tüm testler yeşil (122/122).
15. ✅ **YENİ SOHBET KAYNAKLARI EKLENDİ** (02.10.2026) — HF'den 3 gerçek sohbet veri seti:
    - `3nesdeniz/turkish-daily-dialogues-5k`: 5K çok-dönüşlü günlük diyalog (market, alışveriş vb.)
    - `SoAp9035/everyday-conversations-tur`: Çok-dönüşlü günlük sohbetler (bilgi+sorular)
    - `Renicames/turkish-law-chatbot`: Hukuk alanında QA (yüksek kalite, tek-dönüş)
    Yeni parser'lar: `pairs_from_daily_dialogues`, `pairs_from_everyday_conversations`,
    `pairs_from_law_chatbot`. Çıktı `chatgrow_hf_chat_new.jsonl` (500 çift test).
    `source` alanı korundu → hangi çift hangi kaynaktan geldiği izlenebilir.
16. ✅ **`qa_score` v3 — KOPYALAMA ODULLANDIRMESİ AZALTILDI** (02.10.2026)
    - v2: `%50 tmean + %25 fluency + %15 copy_bleu + %10 length_fit`
    - v3: `%65 tmean + %25 fluency + %5 copy_bleu + %5 length_fit`
    - `copy_bleu` ağırlığı **%15 → %5** (kopyalama teşviki minimize edildi)
    - `tmean` (gold_recall + topic_k) ağırlığı **%50 → %65** (fidelity/icerik kapsama ANA eksen)
    - `METRIC_VERSION = 3` (eval_llm.py, test_eval_llm.py güncellendi)
    - Neden: §6.1'de kanıtlandı — copy_bleu ve gold_recall zıt eksenler; kopyalama düşürürken
      gold_recall da düşüyordu. Metrik kopyalamayı ödüllüyordu, model "ezber" yapıyordu.
17. ✅ **MODEL BÜYÜME ALTYAPISI HAZIR: Gradient Accumulation + 2 Oturumlu Kaggle** (02.10.2026)
    - `train_llm.py`: `--grad-accum N` eklendi (loss/grad_accum, step % grad_accum == 0'da step)
    - `kaggle_start.sh`: `LLM_GRAD_ACCUM`, `LLM_CAP`, `LLM_BLOCKS`, `LLM_DP_OFF` env değişkenleri
    - Resume mekanizması zaten var (`llm_ckpt.pt` → optimizer state + epoch + best_val)
    - Kaggle 2 oturumlu eğitim: Oturum 1 (epoch 1-8) → checkpoint indir → Oturum 2 (epoch 9-16) resume
    - Test: `--dry-run --grad-accum 2 --batch-size 64` çalışıyor
18. ⬜ **MODEL BÜYÜME: d=512, 8 blok (~33.4M tied)** — Kaggle'de çalıştırılacak
    - Mevcut: d=384, 6 blok, 16.9M param, 12 epoch ≈ 467 dk
    - Hedef: d=512, 8 blok, 33.4M param (tied), ~2x parametre
    - Strateji: `LLM_CAP=512 LLM_BLOCKS=8 LLM_GRAD_ACCUM=2 LLM_DP_OFF=1` + epoch 16 (2 oturum)
    - `train_llm.py` ve `kaggle_start.sh` hazır — sadece Kaggle'de çalıştırmak kalıyor
19. ⬜ `build_crawl_corpus.py` → 8 blok / d=512 (büyüttükten sonra)

### 1.10 eğitim koşusu (üretimdeki model 01.10 08:00 modeli DEĞİL)

| | değer |
|---|---|
| mimari | decoder-only, pre-LN, 6 blok, d=384, 8 baş, ff=1536, **tied** |
| parametre | **16.905.856** (embed 6.144.000 = **%36,4**; 6 blok 10.646.784) |
| çift | `MAX_PAIRS` 290.441 (ham 288.802, **kırpma yok**) → ×5 → **1.377.895** |
| RAG bağlamlı çift | **%47,1** (106.273 benzersiz ctx, 13,0 çift/ctx) |
| token | 125,1M (100,88 tok/çift; `TOKEN_PER_PAIR=100` → **+%0,88**) |
| en iyi val | **0,3129 @ epoch 8**, acc 0,925 — **ama ekrana üretim %52 (§6.14)** |
| süre | 427,1 dk epoch + 40,3 dk encode = **≥ 467,3 dk** (oturum %86,5) |
| parity torch↔numpy | 5,0e-05 |
| **durum** | **GERİ ALINDI** — ekrana üretim %72→%52 düştü |

**Uyarı — iki dropout var, karıştırma:** `--dropout 0.10` → **residual** dalı
(`train_llm.py:370`); **attention** olasılık dropout'u **sabit 0,05**
(`train_llm.py:377`), bayraktan bağımsız.

**Ölçüldü (artık iddia değil):** `embed` std yeni koşuda **0,1959** (eskisi
0,1831, **+%,7**). `head_b` **yok** — çıkış `embed^T` (tied), doğrulandı.
std oranı (yeni/eski): medyan **1,014**, min 0,910, max 1,463 → ölçekler
sağlıklı, patoloji yok. Ancak **embed ölçeğinin büyümesi sadakat düşüşüyle
ilişkili olabilir — bu ilişki ölçülmedi, varsayım olarak yazılmadı.**


# 8. BİLİNEN ENGLELLER

- **4 "X nedir" sorusu cevaplanamıyor**: `kadin`, `siber guvenlik`, `kuantum
  bilgisayarlar`, `fotografik` → `corpus.jsonl`'de tam adıyla kayıt yok.
- ~~Çalışan VS Code debug sunucusu (PID 7064, port 5000)~~ → **ÇÖZÜLDÜ**
  (29.09 sonrası ölçüldü: PID 7064 yok, port 5000'de dinleyen yok). Artık
  `corpus.jsonl` bozma riski yok.
- **590 test, `OK (skipped=1)`, 364,7 sn** (02.10 2026 ölçümü, tam paket).
  Süre **makineye bağlı**: 01.10'da 452,7 sn, 02.10'da 364,7 sn. Ölçüm aracı
  çalıştırırken `train_llm.py`'ye dokunma (`eval_llm.py` import ediyor).
- **`origin/main` verisi yeniydi** — yerel `main` 02.10'da **ahead 2 /
  behind 8** idi; "reset 42.000 satır veri kaybı yapar" sezgisi
  **ölçümde yanlış çıktı**: yerel 14.647, `origin/main` **16.225** bilgi
  intent. Merge **çakışmasız**, veri dosyaları `origin/main`'in aynısı
  oldu (§6.18b).
- **Kök dizinde 13 untracked tek-seferlik script** (30.09/01.10'dan kalma).
  Ölçüldü (01.10): **13/13 untracked**, CI yalnızca `-s tests` taradığı için
  **13/13 CI-dışı**. 4 tanesi **veri dosyasına YAZIYOR**:
  `add_4_intents.py`, `add_4_km.py`, `add_corpus_entries.py`,
  `fix_intent_responses.py`. Kalan 9 yalnızca okur.
  **Karar: SİLİNMEDİ.** Ölçülen zararı sıfır; silmek geri alınamaz ve
  kullanıcının dosyaları. Tek risk: biri **elle** çalıştırılırsa veri bozar.
  (Bunları silmek istersen ayrıca söyle.)

---

# 9. DOSYA HARİTASI

| dosya | satır | içerik |
|---|---|---|
| `build_book_pairs.py` | docstring | **ÖLÇÜLEN KAYNAK TAVANI (609)**, kanıtlı |
| `kaggle_start.sh` | 1-50, 60-140, 149 | Kullanım, env değişkenleri, asıl eğitim komutu |
| `train_llm.py` | 185, 207, 210 | `MAX_SEQ_LEN=256`, **`KB_TEXT_CHARS=300`**, `SEED=7` |
| `train_llm.py` | 213, 232, 858 | `TOKEN_PER_PAIR`, `ENC_CIFT_SN`, `MS_PER_PAIR` (**eğitim çifti** — bütçe formülünü artık beslemiyor) |
| `train_llm.py` | 944 | **`MS_PER_HAM_CIFT_EPOCH = 7,3927`** — bütçe formülünün tek kullandığı sabit (encode **hariç**; §6.16 + §6.18a) |
| `seqgen.py` | 412-455 | **KIRPMA: önce her ctx'den 1 çift, sonra kalan bütçe.** Eski `shuffle → kes` 2.039 benzersiz ctx'i (%2,08) susturuyordu; çift sayısı aynı, süre aynı, kapsama **%100** (§6.18c) |
| `train_llm.py` | 485 | `make_batches` (PAD budama + `_pack_encoded`) |
| `train_llm.py` | 737, ~760 | `build_kb_lut`, `rag_context_stats` (29.09 düzeltmesi) |
| `train_llm.py` | 952 | `sure_ve_hesapla` — **MAX_PAIRS tavanının tek doğruluk kaynağı** (birim: ham çift; `encode_dk`'yı **bir kez** düşer) |
| `train_llm.py` | ~1090 | RAG yazdırma satırı (yalnızca log) |
| `train_llm.py` | 386 | `_cache_fp` |
| `train_llm.py` | 920, 1021-1023 | `load_chatgrow_pairs`, RAG eşiği (`use_corpus`) |
| `brain.py` | 1047 | 40 sınıf kuralı (`>6` desen → sohbet) |
| `brain.py` | 764 | `knowledge_bias=1.2` |
| `brain.py` | 2035 | decoding — 4 eksende ölçüldü, hiçbiri kazandırmadı (§7) |
| `brain.py` | 1467, 1469 | **`random.choice(responses)`** — bilgi metni seçimi; NumPy tohumundan bağımsız, ölçümü bozuyordu (§6.10 hata 3) |
| `tools/uretim_olc.py` | 93 | `icerik_kelimeler` — Türkçe harf **silme** hatası düzeltildi (§6.10 hata 1) |
| `tools/sure_olc.py` | — | **zaman tavanını `kaggle_train.txt`'ten ölçer.** İKİ birim hatasını buldu: eğitim çifti ↔ ham çift (§6.16) ve **encode'un iki kez sayılması** (§6.18a). `MS_HAM` artık encode içermez; kapsama **canlı veriden** ölçülür. Sabiti yeniden ölçmek için çalıştırılır |
| `tools/yedek_olc.py` | — | kapı reddedince ekrana ne çıktığını ölçer; `tries`×`temp` 4 varyant (§6.17) |
| `tools/model_ab.py` | — | "hangi modeli kuralım" kararını ölçümle verdirir (modeli kurmaz) |
| `tools/soru_listesi.json` | — | **SABİT 50 soru.** Elle değiştirme; değişirse raporlar kıyaslanamaz |
| `brain.py` | 2098, 2145 | kapı distinct-letter düzeltmesi |
| `eval_llm.py` | 400+ | `compare_reports` decoding denetimi |
| `corpus.py` | 188, 303, 368, 486, 921 | `Corpus`, `_load_index_cache`, `load`, `_ensure_embeddings`, `search(query, k=2)` |
| `app.py` | 340-410, 442-455, 517-554 | `learn_from_internet` (düzeltildi), `fallback_answer`, model logları |
| `autogrow.py` | 57 | "num_intents 40'da sabit kalır" notu |
| `model_kur.py` | — | güvenli kurulum (`--check`, rollback) |
| `model/yeni_2909/` | — | üretimdeki model + `kaggle_train.log` (5777 bayt) |
| `model/kaggle_2909/` | — | önceki koşu logu |
| `model/yedek/20260929_031340/` | — | tek rollback noktası (29.09 öncesi) |
| `tools/` | — | 9 ölçüm aracı + README |
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
- **364,7 saniye**, **590 test OK (skipped=1)** — tam yeşil (02.10 2026).
  Önceki ölçüm 01.10'da 452,7 sn / 589 test idi; süre makineye bağlı,
  iki sayı da kayıtlı. **590 = 589 + 1**
  (`test_kirpma_yariya_inse_de_kapsama_durur` — §6.18d).
- **`test_kirpma_kapsamayi_agirmeden_atmiyor`** artık %99 eşiği değil
  **kesin sözleşme** ölçüyor: `benzersiz ctx == min(kap, benzersiz ctx)`.
  %99 eşiği, kirpma sessizce 1.000 ctx kaybetseydi de geçerdi.
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

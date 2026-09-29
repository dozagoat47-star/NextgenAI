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
| Disk | 2,1 GB → **1,47 GB** (temizlik sonrası); `model/` 329 MB |

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

Akıcılık **düşmez** + altın kapsama en fazla **0,015** düşer + fark paired t-testinde
anlamlı (**|t| ≥ 2**). Aksi halde değişiklik atılır.

---

# 3. KESİN KISITLAR

- **Veri dosyalarına elle dokunma**: `intents.json`, `knowledge_map.jsonl`,
  `corpus.jsonl`, `corpus_ids.jsonl`, `chatgrow_*.jsonl`.
  Sadece **üreten scriptler** (`build_book_pairs.py`, `autogrow.py`, `chatgrow.py`),
  sync betiği (`scripts/sync_chatgrow.ps1`) ve CI değiştirilebilir. Veri dosyası
  üreticiyi **çalıştırarak** yeniden üretilir, elle düzenlenmez.
- `d_model` / blok sayısı küçültülemez.
- Testler veri dosyalarına dokunmamalı.
- `model/` git'e girmez → üzerine yazmadan/taşımadan önce **tarihli yedek** zorunlu.
  Yeni model ölçülmeden `model/llm_model.*` üzerine yazılmaz. `model_kur.py` güvenli
  yol (`--check`, küçültme yasağı, rollback, `model/yedek/<ts>/`).
- **Force push yasak.** Reddedilirse: `git fetch` → `git rebase origin/main` → normal push.
  (`data/chatgrow` sync'i + CI günde 3-4 kez commit+push yapıyor, çakışma normal.)
- Kullanıcı **kısa cevap istiyor.** Belirsizlikte tahmin etme, sor.
  "Her adımı bitirdiğinde dur ve sonraki adımı ne yapacağını söyle."
- İki iş isteniyorsa **bir kafayla** ikisini birden yap (kullanıcı açıkça böyle istedi).

---

# 4. EĞİTİM: KAGGLE'DE NASIL YAPILIYOR

Eğitim **Kaggle'da** yapılır, GPU ile. Yerel NumPy eğitimi saatler sürer.

## Donanım / kota
- **Accelerator: GPU P100** (veya **T4x2**).
- Kaggle haftada **30 saat** ücretsiz GPU.
- Oturum süresi pratikte **9 saat**; `kaggle_start.sh` bunu `OTURUM_DK=540` olarak varsayar.
- 29.09 ölçümü (d=384/6 blok, T4x2, max_seq 256): 12 epoch ≈ 279 dk ≈ 4,7 saat.

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
| `!bash kaggle_start.sh verify` | dry-run doğrulama, GPU gerekmez, **TAM encode yapılmaz** | ~1-2 dk |
| `!bash kaggle_start.sh bench` | 1 epoch zamanlama (cache/encode + 1 epoch birlikte ölçülür) | ~10 dk + 1 epoch |
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
| `LLM_CAP` / `LLM_BLOCKS` | 384 / 6 | ~22,9M parametre |
| `LLM_SEQ` | 256 | `MAX_SEQ_LEN` ile **AYNI olmalı** (tek kaynak) |
| `LLM_TIE` | 1 | 0 → `--untie-embeddings` (23,0M → 16,8M, dosya 87,6 → 64,1 MB) |
| `LLM_WD` | 0.01 | AdamW, yalnız ağırlık matrisleri |
| `LLM_DROPOUT` | 0.10 | |
| `LLM_NATURAL` | 5 | doğal çoğaltma çarpanı |
| `LLM_CAP=256 LLM_BLOCKS=4` | — | küçük deney örneği |

## Ölçülmüş zaman sabitleri (`train_llm.py`)

| sabit | değer | nasıl ölçüldü |
|---|---|---|
| `ENC_CIFT_SN` | 664,5 çift/sn | 1.024.172 çift, encode 1541 sn (921.748 train + 102.424 val) |
| `MS_PER_PAIR` | 1,5139 ms/çift | 921.748 çift, ortalama 1395 sn/epoch (val olan epoch'lar ~1435, val atlananlar ~1338 → **tek epoch seçmek %7 yanıltırdı**, ortalama alındı) |
| `TOKEN_PER_PAIR` | 89,3 token | kirpmalı, RAG + kb-map açık |

**Dikkat (28.09 hatası):** MS_PER_PAIR küçük saymak bütçeyi **BÜYÜTÜYOR** (12 epoch
396 dk yerine gerçekte 431 dk ister → oturum kesilirdi). Zaman bütçesi pratikte bağlayıcı
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
RAG bağlamlı çift   : %52,9   (log "%2,7" demişti — HATALI, bkz. §6.2)
BPE vocab 16.000 · 100,3M token · 8 epoch (patience ile erken kesildi)
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

## 6.4 Disk temizliği — 676 MB, kanıtlayarak silindi
2 Kaggle zip'i (içerik diskte doğrulandı), byte-aynı yedek kopyalar, 0 referanslı model
dosyaları, 4 `llm_data_*.npz` (gitignore'lu, Kaggle klonunda var olamaz).
`model/` 941 → 329 MB. Tek rollback noktası: `model/yedek/20260929_031340`.
İki `kaggle_train.log` korundu.

## 6.5 Korpus önbellekleri (1.038 MB) SİLİNMEDİ — ölçüldü
Önbellekli yükleme **18,4 sn** vs önbelleksiz **273,8 sn**. Çöp değil, her açılışta
4,3 dk kazandıran hız önbelleği. Karar `corpus.py` `load()` docstring'ine yazıldı.

## 6.6 40 sohbet sınıfı kasıtlı
`brain.py:1047` kuralı: `len(patterns) > 6` → sohbet, `≤6` → bilgi. AutoGrow bilgi
intent'lerini tam 6 desenli Wikipedia şablonundan üretiyor → `num_intents: 40` sabit
(`autogrow.py:57` yorumu bunu doğruluyor). `intents.json` = 11.557 intent
(40 sohbet + 11.517 bilgi). Büyüme retrieval katmanına gidiyor.

Ölçüm: 11.557 intent, 40'ı desen>6, 11.517'si desen≤6 (tam 6 desenli şablon).
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

# 7. ANA ÇIKARIM (sıradaki adımı belirleyen)

**"Bilgi verisi yok" tezi ölçümle çürüdüldü.** Eğitim çiftlerinin **yarısı (%52,9)
zaten bilgi bağlamı taşıyor.** Kopyalama sorunu ve düşük sadakat (%22–31), bu çiftlerin
yarısında modele **"kopyala" sinyali** verilmesinden kaynaklanıyor olabilir.
Kitap verisini büyütmek bu yüzreğe dokunmuyor.

**Sıradaki adım (onay bekliyor — henüz ölçülmedi):**
`KB_TEXT_CHARS=300` ile bağlamın kırpılma oranını, ve bilgi-bağlamlı çiftlerdeki
`copy_bleu`'yu bilgi'siz çiftlerle karşılaştırmak. Kopyalamanın gerçekten bu grupta
yoğunlaştığını doğrularsa hedef decoding'den veri tarafına (paragraf yeniden ifade)
kayacak.

Yeni sohbette **ilk adım bu ölçümü kurmak ve taban değerleri almak.**

---

# 8. BİLİNEN ENGLELLER

- **4 "X nedir" sorusu cevaplanamıyor**: `kadin`, `siber guvenlik`, `kuantum
  bilgisayarlar`, `fotografik` → `corpus.jsonl`'de tam adıyla kayıt yok.
- Kullanıcının **çalışan VS Code debug sunucusu** (PID 7064, port 5000) hâlâ eski kodu
  çalıştırıyor. Yeniden başlatılana kadar bir bilgi sorusu `corpus.jsonl`'i bozabilir.
  (2 kez bildirildi, yapılmadı.)
- 587 test ~270 sn sürüyor; ölçüm aracı çalıştırırken `train_llm.py`'ye dokunma.

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
| `brain.py` | 2035 | decoding — **DOKUNMA** |
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
- ~270 sn, **587 test OK (1 skip)**
- CI (`ci.yml`, Python 3.12) yalnız **`pip install numpy requests flask`** yapar →
  `torch` bağımlı testler `@requires_torch` ile **skip** edilir (aksi halde
  `unittest.loader._FailedTest` modülü düşürüp tüm suite'i kırmızı eder).
  `requires_torch` 5 test dosyasında kullanılıyor.
- Test taramasında docstring ve diziler kod sayılmamalı
- **Kod tarama nöbetçileri** (`kaggle_start.sh`/`train_llm.py` sayılarına bağlı
  kalibrasyondur — dokunma/bozma): `n / 6000.0`, `n * 0.16`,
  `n * TOKEN_PER_PAIR` (boşluklu **ve** boşluksuz), `MS_PER_PAIR=`
  (kaggle_start.sh'da yasak)

---

# 12. ÇALIŞTIRMA TUZAKLARI

- `eval_llm.py` `train_llm.py`'yi import ettiği için **ölçüm sürerken
  `train_llm.py`'ye dokunma**
- `app.load_bot()` `global bot` kullanıyor, **döndürmüyor** → `app.bot` kullan
  (yanlış testte `NoneType` hatası verdi)
- `Corpus.search(query, k=2)` — **`top_k` parametresi yok**

---

# 13. KALICI ÖLÇÜM ÇIKTILARI

`C:\Users\cxc\AppData\Local\Temp\opencode\`:
- `sweep_*.json` — decoding sweep sonuçları (6 ayar × n=400)
- `uretim_karsilastir.py`, `kod_sweepi_cikti.txt`, `commit_mesaj7.txt`
- `t_rag.py` — RAG sayaç regresyon testi (geçici)
- `t_shadow.py` — 29.09 shadow ölçümü (ESKİ %9,3 / YENI %52,9)
- `kitap_tam.jsonl` + `kitap_tam.txt` — 609 byte-aynı kanıtı
- `devam_promtu.md` — bu dosyanın ilk sürümü (artık güncel değil, **bu dosya kanonik**)

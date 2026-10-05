# Devam Promptu — Nextgen AI (Nextgen_API)

> **Bu dosya bir yapay zekâya devam promptu olarak verilir.**
> Aşağıdaki "KISA PROMPT" bölümünü yeni sohbete yapıştır, sonra bu dosyayı
> okumasını söyle. Geri kalanı senin için.
>
> **Not (05.10):** Bu dosya 02.10 sonrası temizlenmiştir — eski ölçümlerin
> adım adım düzeltilme tarihçesi silinmiştir. Kararların *nedeni* ve *güncel
> sayılar* kalmıştır. Geçmiş tarihçe git'tedir.

---

# KISA PROMPT (yeni sohbete yapıştır)

```
Bu repo için devam ediyoruz. Önce DEVAM_PROMPTU.md dosyasının TAMAMINI oku
(bash: cat DEVAM_PROMPTU.md). Dosyada yazılı olan her şeyi kanıt kabul et, kendi
tahminini ondan önce koyma. Başladığında ilk işi dosyanın "Sıradaki adım"
bölümünden al.
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
| HEAD | `bb2f11a` (05.10.2026) |
| Bağımlılıklar | `requirements.txt`: numpy>=1.24, flask>=2.3, requests>=2.31, openpyxl>=3.1 |
| Eğitim verisi | `intents.json` ~**20.000** intent (2.10'da 16.225, 29.09'da 11.149 idi) |
| **Çalışma tarzı** | Kodu elle düzelt, tahminle atlama; iddiasının sayısı olsun. Karar vermeden önce ölç, ölçtüğünü payla. Belirsizlikte sor. |

## Mimari (README'den)

| Katman | Dosya | Görev |
|---|---|---|
| Sınıflandırıcı | `transformer.py`, `brain.py` | Transformer encoder (NumPy) — sohbet niyetlerini sınıflandırır |
| Bilgi arama | `corpus.py`, `knowledge.py` | RAG-lite: LSA/SVD + PPMI + BM25; bilinmeyen soru Wikipedia'dan |
| Üretim | `generator.py`, `seqgen.py`, `seq2seq.py`, `llm.py` | Anchor-sabit parafraz + koşullu LSTM üreteç + Seq2Seq transformer |
| Öğrenme | `finetune.py` | LoRA adaptörüyle anlık intent ekleme/çıkarma (`/learn`, `/forget`) |
| Sunucu | `app.py` | Flask web arayüzü, yönlendirme kuralları |

İki katmanlı çalışma prensibi:
- **Sohbet niyetleri** `tur == 'sohbet'` (açık işaret; yoksa desen>6 eski kuralı) ile sınıflandırılır.
- **Bilgi niyetleri** (Wikipedia şablonlu, otomatik büyüyen küme) IDF anahtar-kelime retrieval ile.
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
- Ölçüm sana ters düşerse **ölçümü kabul et**, tahmini değil.
- Rastgele tahmin yaptığını fark edersen **açıkça söyle** ve düzelt. Suçlayıcı ton
  kullanma; ölçüme dön.

## Karar kuralı (sonuçlara bakmadan ÖNCE sabit — 29.09'da belirlendi)

Varsayılan eşik: akıcılık **düşmez** + altın kapsama en fazla **0,015** düşer +
fark paired t-testinde anlamlı (**|t| ≥ 2**). Bu, karşılaştırmaları tutarlı
tutmak için — tek başına yasak değil, ölçülen şeyi nasıl değerlendireceğimiz.

**Ama ölçüm sana başka bir şey gösteriyorsa kuralı değiştirebilirsin** — yeter
ki değişikliği gerekçelendir, güncelle ve dosyaya yaz.

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
  (bkz. §2 karar kuralı).
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
- 01.10 ölçümü (d=384/6 blok, T4x2, max_seq 256): 12 epoch ≈ 427 dk + 40,3 dk encode.

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
| `!bash kaggle_start.sh train` | asıl eğitim | ~4,7 saat (12 epoch) |

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
| `LLM_PATIENCE` | 4 | **epoch cinsinden** |
| `LLM_OTURUM_DK` | 540 | oturum süresi → `MAX_PAIRS` tavanını hesaplar |
| `LLM_CAP` / `LLM_BLOCKS` | 384 / 6 | ~16,9M parametre |
| `LLM_SEQ` | 256 | `MAX_SEQ_LEN` ile **AYNI olmalı** (tek kaynak) |
| `LLM_TIE` | 1 | 0 → `--untie-embeddings` |
| `LLM_WD` | 0.01 | AdamW, yalnız ağırlık matrisleri |
| `LLM_DROPOUT` | 0.10 | residual dropout; attention dropout sabit 0,05 |
| `LLM_NATURAL` | 5 | doğal çoğaltma çarpanı |
| `LLM_GRAD_ACCUM` | 1 | gradient accumulation (2 oturumlu eğitim için) |

## Ölçülmüş zaman sabitleri (`train_llm.py`)

| sabit | değer | not |
|---|---|---|
| `TOKEN_PER_PAIR` | 100,0 | canlı `encode_llm`, eğitim popülasyonu |
| `MS_PER_PAIR` | 1,5139 ms/eğitim-çifti | artık bütçe formülünü beslemiyor |
| `MS_PER_HAM_CIFT_EPOCH` | **7,3927** ms / ham çift / epoch | **bütçe formülünün tek sabit kaynağı** (encode hariç) |
| `ENCODE_DK` | ~40,3 dk (ölçüldü) | 29.09'deki 26 yanıltıcıydı |

## `max_seq 256` neden ölçülmüş özel bir sayı

`kb_budget = max_seq - max_ctx - 8`; küçük değerde yanıta yer kalmaz ve RAG yanıtları
çarpılır. `knowledge_map.jsonl` gerçek karışımıyla:

| max_seq | RAG yanıtı kırpıldı |
|---|---|
| 128 | **%87** ← eski (hatayı yarattı) |
| 192 | %37 |
| 256 | **%3** ← seçilen |

## CI (GitHub Actions) — veri hattı otomatik

`.github/workflows/`: `autogrow.yml`, `autogrow-deep.yml`, `chatgrow.yml`,
`crawl_corpus.yml`, `kbmap.yml`, `ci.yml`, `opencode.yml`, `opencode-scheduled.yml`

Bunlar veri dosyalarını **kendi üreticileriyle** üretip **doğrudan `main`'e push eder** →
`git pull`/`rebase` sırasında çakışma normaldir, force push yasak.

---

# 5. ANA HEDEF VE GÜNCEL ÖLÇÜMLER

Hedef: modelin kalitesini artırmak. Güncel taban (50 sabit soru, `tools/uretim_olc.py`):

| ölçüm | değer | kaynak |
|---|---|---|
| ekrana üretim | **%72** (36/50) | 01.10 ölçümü (düzeltilmiş) |
| sadakat | **%60** | 01.10 ölçümü (düzeltilmiş) |
| deneme kabulü | **%47** (71/150) | 01.10 |
| boş kova ("bilgim yok") | **%0** | 01.10 |
| ekrana çıkan metin (yedek dahil) | model %72, yedek metin ~%7,5 dolgu | §6.17 |

> 29.09'da "%22–31 sadakat / ~%40 bilgim yok" deniyordu; ikisi de ölçüm
> hatasıydı (normalizasyon + deterministik yok). Gerçek taban yukarıdaki.

---

# 6. YAPILMIŞ VE KANITLANMIŞ İŞLER (tekrar etme)

## 6.1 Kopyalama decoding ile çözülmez (kanıtlandı)
6 decoding ayarı tarandı (n=400), paired t-testi: **hiçbiri geçmedi**.
Aşırı ayar kontrol olarak t=+2,13 anlamlı çıktı → ölçüm gücü doğrulandı.
`brain.py:2035` decoding ayarlarına **dokunulmadı** (kanıt yok).

## 6.2 Ölçüm hataları düzeltildi
- Eval `kb=0.0`, üretim `1.2` → `compare_reports` artık 8 decoding anahtarını
  karşılaştırıp farklıysa uyarıyor (`7f2d110`).
- RAG kapsam ölçümü `tek_ctx/toplam` diyordu → çift bazlı **%52,9** (`11bac35`).
- Sadakat metriği Türkçe harfleri siliyordu → `ascii_normalize` ile **%60** gerçek.
- Soru listesi veri sürümüne bağlıydı → sabit liste `tools/soru_listesi.json`.
- `random.seed` eklendi, ölçüm artık deterministik (6/6 alan 50/50 aynı).

## 6.3 Kitap hattı ÖLÇÜLDÜ ve KAPALI — kaynak tükenmiş
`--max-pairs 8000` hedefi ulaşılamaz: tam kapasite koşusu depodaki 609
çiftle **byte-aynı** (sha256 `71251cb0ac3d6965`). 8 kategoride 128 sayfa,
Türk Wikisource'ta daha zengin nesir kategorisi yok. 609 çift / 223.815 taban
= **%0,27**. Bu hat akıcılık kaynağı olamaz. `build_book_pairs.py`
continuation üretiyor, bilgi değil.

## 6.4–6.5 Disk / önbellek kararları
- Disk: 2,1 → 1,47 GB temizlik (676 MB kanıtlayarak silindi). `model/` 329 MB,
  tek rollback: `model/yedek/20260929_031340`.
- Korpus önbellekleri **silinmedi**: açılış 273,8 sn → 18,4 sn kazandırıyor.

## 6.6 + 6.20.3 Sohbet sınıflandırması: **sayıdan açık işarete**
Eskiden "40 sohbet sınıfı kasıtlı" deniyordu. Ölçüm: sohbet sınıfı desen>6
kuralının yan etkisi; bilgi intent'leri 6 desenli şablonla üretildiği için
sohbet hiç büyüyemiyor. Eşik düşürmek felaket: ≥7 desenli intent yok, ama
16.185 bilgi intent'i sınıflandırıcıya girerdi.

Düzeltme: `brain.py` `_sohbet_mi(intent)` — `tur == 'sohbet'` → sohbet,
`tur == 'bilgi'` → bilgi, yoksa eski sayma kuralı (geriye uyumlu).
`intents.json`'a 3 sızıntı intent'ine (`kavram_tanimi`, `tavsiye_isteme`,
`gelecek_planlari`) `"tur": "sohbet"` eklendi → sohbet **40 → 43**, sıfır sızıntı.

## 6.7 Ölçüm araçları
`tools/`: `uretim_olc.py`, `uretim_karsilastir.py`, `kapi_ab.py`, `kendi_cumlesi.py`,
`token_olc.py`, `kalite_olc.py`, `uretim_olc` sabit yolu `__file__`'dan türetiyor.
Çıktı `olcum_raporlari/`'na gidiyor (gitignore'lu).

## 6.8–6.9 Kapı ve app.py düzeltmeleri
- Kapı distinct-letter hatası: kabul %14→%39, ekrana %24→%58 (`7f2d110` öncesi).
- `app.py:374` veri bozma hatası düzeltildi (`bd63543`), 11 test eklendi.

## 6.10 Üretim yolu ölçüm hataları (3'ü de düzeltildi)
1. Sadakat metriği Türkçe harfleri **siliyordu** → `ascii_normalize`; gerçek sadakat
   %33 değil **%57–60**. (README'deki eski %22–31 geçersiz.)
2. Soru listesi satır-adımıyla seçiliyordu, CI veriyi değiştirince sorular değişiyordu
   → sabit `tools/soru_listesi.json`; 29.09 vs 01.10 raporunda 3/50 ortak soru.
3. `np.random.seed` Python `random`'ı kontrol etmiyordu → `random.seed` da eklendi,
   bilişim artık 50/50 deterministik.

**Kilit bulgu:** sadakat bir kopya dedektörüdür — yüksek sadakat "kopya" demek,
kalite ödülü değil. Kapı reddettiği metinlerde %97 sadakat kopyaydı.

## 6.11 Depo testleri: 3 kırmızı test düzeltildi
`MAX_PAIRS` kırpması ölmüştü (bütçe veriyi aşıyor), `kadin_tanim` veride
var ama test hardcode etmiyordu. Testler artık canlı veriden türetiliyor,
üretim koduna dokunulmadı. 589 → 590 test OK.

## 6.12 Özgünlük kapısı eşiği taraması (01.10)
Eşik 0,15→0,05: ekrana çıkan metin kb'ye göre kopyalık 0,36→0,70, yeni
kabul edilenler gözle bozuk. **KARAR: eşik çekilmedi.** Gerçek çelişkili
bulgu: kapı reddedince ekrana **ham kb** çıkıyor (bleu 1,000) — kazanç
eşikten değil **yedek metinden**. Ama §6.17'de ölçüldü: ham kb aslında
modelin metninden daha temiz (dolgu %7,5 vs %21,9). Asıl problem modelin
kendi metninde.

## 6.13 + 6.20.1 Normalize büyük harf bozması — DÜZELTİLDİ (02.10)
`normalize.py` büyük harfleri küçük ASCII'ye çeviriyordu (`'I': 'i'`).
Düzeltme: `'Ç':'C', 'İ':'I', 'I':'I'` — büyük korunur. Etki: büyük harf
kaybı korpusta **%25,25 → %0,0**. `brain.py:827` acronym tespiti
`w.isascii()` ile sınırlandı (İSTANBUL artık kısaltma sanılmıyor).
Retrieval tutarlılığı değişmedi çünkü sorgu da metin de aynı fonksiyondan geçiyor.

## 6.14 Yeni eğitim: val iyileşti, ekran metni azaldı
Yeni koşu (01.10 23:26): val 0,3274→0,3129, acc 0,918→0,925. **Ama** ekrana
üretim %72→%52, sadakat −3,8. Model **geri alındı**. Ana bulgu:
**`val loss` bu sistemde üretim kalitesini ölçmüyor** — token tahmini vs.
kapının kabul ettiği üretken metin farklı sinyaller.

## 6.15–6.18 Zaman tavanı: iki birim hatası + sessiz kırpma — HEPSI DÜZELTİLDİ
- **Birim hatası 1:** bütçe formülü eğitim çifti sayıyordu, süre ham çifte göreydi
  → %5,34 KAT fazla. Sabit `MS_PER_HAM_CIFT_EPOCH` eklendi.
- **Birim hatası 2:** encode süresi iki kez sayılıyordu → sabit 8,0911→**7,3927**,
  tavan 234.207→**256.333**.
- **Kırpma sessizliği:** `shuffle→kes` medyan-3-çiftli ctx'leri siliyordu →
  2.039 benzersiz ctx (%2,08) **sıfır eğitim verisi** alıyordu. Yeni: önce her
  ctx'den 1 çift, sonra kalan bütçe. Çift sayısı ve süre aynı, kapsama **her bütçede %100**.

Tavanın tek doğruluk kaynağı `train_llm.py:952 sure_ve_hesapla()`; `MS_PER_PAIR`
artık bütçe formülünü beslemiyor.

## 6.17 Yedek metin ölçümü (01.10)
50 sabit soru × 4 varyant (`tries`×`temp`). Kilit bulgu: **kaldıraç `tries`,
sıcaklık değil.** `tries 3→6` ekran oranını %72→%86 (+14 puan), sadakat −2,4,
süre ~×2 (ölçüldü: ×2,18). Sıcaklık 0,9 elendi (+2 soru, sadakat −2,2).
**Karar kullanıcıda:** +16 puan için ×2,18 süre kabul mü?

## 6.19 Sohbet hattı ölçümü (02.10)
25.009 chatgrow çifti incelendi: `chatgrow_hf_*` 16.800 çift **matematik/muhakeme**,
"sohbet" adlı HF kaynağı sohbet **üretmiyor**, gerçek gündelik sohbet yalnızca
`chatgrow_sohbet.jsonl` **198 çift**, Reddit çekicisi kodda duruyor ama **0 kez
çalışmış**. Eğitim karması %1,69 sohbet / %98,31 bilgi. Yeni ölçüm tabanı:
`tools/sohbet_olc.py` + `tools/soru_listesi_sohbet.json`.

## 6.20 02.10 düzeltmeleri (hepsi ölçüldü)
1. `normalize` büyük harf koruma (§6.13). ✅
2. `fetch_hf_turkish.py` çıktıya `source` alanı eklendi; `dedupe_pairs_with_source`. ✅
3. 3 sızıntı intent'e `"tur": "sohbet"` (§6.6). ✅
4. `qa_score` v3: copy_bleu ağırlığı %15→%5, tmean %50→%65 (`METRIC_VERSION=3`). ✅
5. `train_llm.py` `--grad-accum N`, `kaggle_start.sh` `LLM_GRAD_ACCUM` →
   2 oturumlu Kaggle eğitim altyapısı hazır. ✅
6. Yeni sohbet kaynakları (HF): daily-dialogues, everyday-conversations,
   law-chatbot → `chatgrow_hf_chat_new.jsonl`, `source` korunuyor. ✅

---

# 7. ANA ÇIKARIM VE ÖĞRENİLENLER

- Kopyalama (`copy_bleu`) **decoding ile çözülmez**, decoding tükendi (§7, §6.1).
- `val loss` **üretim kalitesini ölçmüyor** (§6.14) — bir sonraki token tahmin
  hatası vs. kapının kabul ettiği üretken metin.
- Sadakat yüksekse metin **kopya** olabilir (§6.10) — tek başına kalite ölçütü değil.
- Yedek metin (kapı reddedince ham kb) **modelin metninden daha temiz** (§6.17).
- Zaman tavanı iki birim hatası taşıyordu; artık ham çift birimiyle doğru (§6.15–6.18).
- Sohbet verisi hatta **neredeyse yok** (%1,69) ve kaynak etiketsiz (§6.19, §6.20.2).

## 7.2 Açık kusurlar (ölçülmedi / düzeltilmedi)
- `naturalize.py` metin bozulması: varyantların **%1,06**'sı hatalı
  (bitişik `x.y` %0,64, yapıştırılmış işaret %0,42). Doğrulaması 4,7 saatlik
  Kaggle ister, kalite etkisi kanıtlanmadı.
- Eğitim/üretim bağlam uyuşmazlığı: aynı 250 sorunun **0/250**'sinde
  eğitimdeki `knowledge_map` metni ile üretimdeki `Corpus.search` metni aynı.
- Doğallaştırma varlık adı bozuyor (`alyson hannigan` → *aleis denisof*);
  %1,6 varyantta içerik kapsamı <%50.
- Aynı ctx için farklı yanıtlar karışıyor.
- `val loss` neden üretimi ölçmüyor — mekanizma bilinmiyor.
- `embed` std büyümesi (0,183→0,196) sadakati etkiliyor mu — 2 nokta, başka koşu yok.
- 4 "X nedir" sorusu cevaplanamıyor: `kadin`, `siber guvenlik`,
  `kuantum bilgisayarlar`, `fotografik` → `corpus.jsonl`'de tam adıyla kayıt yok.

## 7.3 Sıradaki adım (05.10.2026)

Sabit duran, ölçülmüş ve doğrulanmış işler 6. bölüme taşındı. Kalan:

1. ⬜ **Model büyüme: d=512, 8 blok (~33.4M tied)** — Kaggle'de çalıştırılacak.
   Altyapı hazır (`--grad-accum`, `LLM_GRAD_ACCUM`, 2 oturumlu resume).
   Mevcut: d=384, 6 blok, 16,9M, 12 epoch ≈ 467 dk. Strateji:
   `LLM_CAP=512 LLM_BLOCKS=8 LLM_GRAD_ACCUM=2 LLM_DP_OFF=1` + epoch 16.
2. ⬜ `tries 3→6` kararını kullanıcıya sor: +16 puan ekran için ×2,18 süre kabul mü?
3. ⬜ Sohbet kaynağı gerçekten sohbet üreten kanala çekilmeli (Reddit `r/Turkey`
   kodda hazır ama 0 kez çalıştı; yeni HF kaynakları eklendi, etkisi ölçülmedi).
4. ⬜ `build_crawl_corpus.py` → 8 blok / d=512 (büyüttükten sonra).
5. ⬜ `intents.json` büyüyor (11k→16k→20k). Veri tabanı **küçültülebilir mi**
   (ölçümlü): (a) dedup/kalite filtresi, (b) `MAX_PAIRS` tavanının veri
   bütçesine yetip yetmediği, (c) sohbet:bilgi dengesi `tur` alanıyla kota.
   Silmek yerine önce ölç — CI hâlâ besliyor.

---

# 8. BİLİNEN ENGLELLER

- **4 "X nedir" sorusu cevaplanamıyor** (§7.2).
- **590 test, `OK (skipped=1)`, ~365 sn** (02.10 tam paket). CI `ci.yml`
  yalnız `pip install numpy requests flask` yapar → `torch` bağımlı testler
  `@requires_torch` ile skip edilir.
- **`origin/main` verisi yeniydi** (02.10'da yerel ahead 2 / behind 8) —
  merge çakışmasız, veri `origin/main`'in aynısı oldu.
- **Kök dizinde 13 untracked tek-seferlik script** (30.09/01.10). 4'ü veri
  dosyasına **yazıyor**. Karar: **silinmedi**, zararı ölçülen sıfır; elle
  çalıştırılırsa veri bozabilir.
- Kök dizinde `_test_*.jsonl`, `*.log`, `prof*` gibi geçici dosyalar —
  git'e girmiyor, temizlenebilir (kullanıcı isterse).

---

# 9. DOSYA HARİTASI

| dosya | içerik |
|---|---|
| `build_book_pairs.py` | **ÖLÇÜLEN KAYNAK TAVANI (609)**, kanıtlı |
| `kaggle_start.sh` | kullanım, env değişkenleri, asıl eğitim komutu |
| `train_llm.py` | `MAX_SEQ_LEN=256`, `KB_TEXT_CHARS=300`, `SEED=7`, `MS_PER_HAM_CIFT_EPOCH=7,3927`, `sure_ve_hesapla`, `rag_context_stats`, `--grad-accum` |
| `seqgen.py` | **KIRPMA: önce her ctx'den 1 çift, sonra kalan bütçe** (kapsama %100) |
| `brain.py` | `knowledge_bias=1.2`, decoding (§7'de tükendi), `_sohbet_mi` (`tur` alanı), kapı distinct-letter, acronym tespiti |
| `eval_llm.py` | `compare_reports` decoding denetimi, `METRIC_VERSION=3` |
| `corpus.py` | `Corpus`, `_load_index_cache`, `load`, `_ensure_embeddings`, `search(query, k=2)` |
| `app.py` | `learn_from_internet`, `fallback_answer`, model logları |
| `autogrow.py` | Wikipedia'dan otomatik bilgi |
| `model_kur.py` | güvenli kurulum (`--check`, rollback) |
| `tools/uretim_olc.py` | 50 sabit soru üretim ölçümü (`icerik_kelimeler` düzeltildi) |
| `tools/sure_olc.py` | zaman tavanı ölçümü (2 birim hatasını buldu) |
| `tools/yedek_olc.py` | kapı reddedince çıkan metni ölçer |
| `tools/sohbet_olc.py` | sohbet sınıflandırma/üretim tabanı |
| `tools/soru_listesi.json` | **SABİT 50 soru** — elle değiştirme |
| `tools/soru_listesi_sohbet.json` | 40/40 sınıf + 5 kenar soru |
| `model/yeni_2909/` | üretim modeli + log |
| `model/kaggle_2909/` | önceki koşu logu |
| `model/yedek/20260929_031340/` | tek rollback noktası (29.09 öncesi) |

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
- **~365 sn**, **590 test OK (skipped=1)** — 02.10 2026 tam paket.
- `test_kirpma_kapsamayi_agirmeden_atmiyor` artık %99 eşiği değil **kesin
  sözleşme**: `benzersiz ctx == min(kap, benzersiz ctx)`.
- CI (`ci.yml`, Python 3.12) yalnız **`pip install numpy requests flask`** yapar →
  `torch` bağımlı testler `@requires_torch` ile skip edilir.
- **Kod tarama nöbetçileri** (sabit değişince testleri de güncelle):
  `n / 6000.0`, `n * 0.16`, `n * TOKEN_PER_PAIR`, `MS_PER_PAIR=`
  (kaggle_start.sh'da olmamalı).

---

# 12. ÇALIŞTIRMA TUZAKLARI

- `eval_llm.py` `train_llm.py`'yi import ettiği için **ölçüm sürerken
  `train_llm.py`'ye dokunma**
- `app.load_bot()` `global bot` kullanıyor, **döndürmüyor** → `app.bot` kullan
- `Corpus.search(query, k=2)` — **`top_k` parametresi yok**
- PowerShell'de `'%s' % (x, y)` içinde `%%.1f` yazarsan "not all arguments
  converted": `%%` kaçış olduğu için o alan dönüşüm saymaz. Gerçek yüzde için
  `%.1f%%` + değeri ayrı argüman olarak ver.
- Ölçüm betiğinde **kol biter bitmez diske yaz**, sonra raporla.

---

# 13. KALICI ÖLÇÜM ÇIKTILARI

`C:\Users\cxc\AppData\Local\Temp\opencode\` ve `olcum_raporlari/`:
- `sweep_*.json` — decoding sweep (6 ayar × n=400)
- `uretim_det_a/b.json`, `uretim_eski_0110c.json` — deterministik doğrulama
- `yedek_olc_0110.json` — kapı yedek metni
- `kaggle_train.txt` — 01.10 23:26 koşu logu (zaman tavanı ölçümü)
- `kitap_tam.jsonl` + `kitap_tam.txt` — 609 byte-aynı kanıtı
- `devam_promtu.md` — bu dosyanın ilk sürümü (artık güncel değil, **bu dosya kanonik**)

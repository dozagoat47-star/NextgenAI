# Nextgen AI

Sıfırdan yazılmış (NumPy-only, hazır ML kütüphanesi yok) Türkçe sohbet asistanı.

> **Bir yapay zekâya devam ettirmek istiyorsan:** `DEVAM_PROMPTU.md` dosyasını
> okut. Kısa yönerge dosyanın başındaki "KISA PROMPT" bölümünde.

## Mimari

| Katman | Dosya | Görev |
|---|---|---|
| Sınıflandırıcı | `transformer.py`, `brain.py` | Transformer encoder (NumPy) — sohbet niyetlerini sınıflandırır (40 sınıf) |
| Bilgi arama | `corpus.py`, `knowledge.py` | RAG-lite: LSA/SVD + PPMI + BM25; bilinmeyen soru Wikipedia'dan |
| Üretim | `llm.py`, `seq2seq.py`, `seqgen.py`, `generator.py` | Üretim hattı sırayla dener: **LLM → Seq2Seq → LSTM**; hepsi aynı kalite kapısından geçer |
| Öğrenme | `finetune.py` | LoRA adaptörüyle anlık intent ekleme/çıkarma (`/learn`, `/forget`) |
| Sunucu | `app.py` | Flask web arayüzü, yönlendirme kuralları |

Çalışma prensibi:
- **Sohbet niyetleri** (deseni >6 olan intent'ler) sınıflandırıcıyla çözülür.
- **Bilgi niyetleri** (Wikipedia şablonlu, otomatik büyüyen küme) IDF anahtar-kelime
  retrieval ile.
- Üretim katmanı ana üreticidir; yanıt kalite kapısından geçemezse bir sonrakine düşer.
- Hiçbiri yetmezse `corpus` → internet (Wikipedia) fallback'i; internetten gelen bilgi
  `corpus.jsonl`'a kaydedilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Eğitim

> **Eğitim Kaggle'da GPU ile yapılır** (`kaggle_start.sh`). Yerel NumPy eğitimi
> saatler sürer, pratikte değildir.

Temel eğitim hattı yalnızca `intents.json` kullanır; `chatgrow_*.jsonl` dosyaları
otomatik olarak eğitime eklenmez. `knowledge_map.jsonl` ile RAG eğitimi,
`train_llm.py` içinde henüz uygulanmadığı için bu hat tarafından kullanılmaz.
Eğitim hedefi otoregresif sonraki-token tahminidir: doğruluk ölçümü de
yanıt token'lerini tahmin eden önceki konumlarda yapılır. Bu hedef düzeltmesini
içeren kodla eğitim sıfırdan başlatılmalı; önceki sürümün checkpoint'i
devam ettirilmemelidir.

Kaggle.com → New Notebook. **Ayarlar:** Internet **ON**, Accelerator **GPU P100** (veya T4x2).

```python
# 1. hücre
!git clone https://github.com/dozagoat47-star/NextgenAI.git
%cd NextgenAI
!python -m pip install --quiet numpy
```
```python
# 2. hücre
!bash kaggle_start.sh train
```

Sonra **Save (Version)** → **Output** sekmesi → **Download All**. İndirilen zip'teki
`llm_model.json` + `llm_model_weights.npz` dosyalarını birlikte `model/` klasörüne kopyala.

Betik üç modda çalışır:

| komut | ne yapar | süre |
|---|---|---|
| `!bash kaggle_start.sh verify` | dry-run doğrulama, GPU gerekmez, tam encode yapılmaz | ~1-2 dk |
| `!bash kaggle_start.sh bench` | 1 epoch zamanlama | ~10 dk + 1 epoch |
| `!bash kaggle_start.sh train` | asıl eğitim | ~4,7 saat (12 epoch) |

Ayar için ortam değişkenleri: `LLM_EPOCHS` (12), `LLM_PATIENCE` (4), `LLM_CAP` (384),
`LLM_BLOCKS` (6), `LLM_SEQ` (256), `LLM_DROPOUT` (0.10), `LLM_NATURAL` (5),
`LLM_OTURUM_DK` (540). Ayrıntı ve ölçülmüş süre sabitleri `DEVAM_PROMPTU.md` §4'te.

Ayrıntılar ve alternatif yollar için `DEVAM_PROMPTU.md` §4'e bak.

## Model dosyaları

| dosya | boyut | ne |
|---|---|---|
| `model/llm_model.json` + `llm_model_weights.npz` | 0,19 MB + 67,6 MB | **ana üretici** (Kaggle'da eğitilir) |
| `model/seq2seq_model.json` | 34,6 MB | Seq2Seq encoder-decoder üreteç |
| `model/seq_model.json` | 8,4 MB | LSTM karakter üreteci |
| `model/model.json` + `model_weights.npz` | 0,3 MB + 4,8 MB | niyet sınıflandırıcı (40 sınıf) |
| `model/bot_data.json` | 1,4 MB | bot verisi |

> `model/` git'e girmez. Üzerine yazmadan önce `model_kur.py --check` ile kontrol et;
> betik küçültmeyi engeller ve `model/yedek/<tarih>/` altına yedek alır.

`colab/` altındaki notebook'lar **ikincil yoldur**; güncel akış Kaggle'dır
(`kaggle_start.sh`). Colab klasörü yalnızca yan modellerin (`transformer.py`,
`seqgen.py`, `seq2seq.py`) notebook'larını ve LLM için yedek yolu tutar.
Ayrıntı: `colab/README.md`.

## Çalıştırma

```bash
python app.py        # http://localhost:5000
```

| uç | gövde | sonuç |
|---|---|---|
| `POST /chat` | `{"message": "..."}` | `{"response": "..."}` |
| `POST /predict` | `{"text": "..."}` | `{"suggestions": [...]}` (en fazla 6 öneri) |
| `POST /learn` | `{"entries": [{"tag", "patterns", "responses"}]}` | LoRA ile öğretir |
| `POST /forget` | `{"tag": "..."}` | LoRA intentini geri alır |
| `GET /status` | — | model/corpus durumu |

## Test

```bash
python -m unittest discover -s tests -v
```

## Veri hattı

- `autogrow.py` — Wikipedia'dan otomatik bilgi toplar, `intents.json` + `corpus.jsonl`'ı büyütür
- `clean_intents.py` — toplanan ham veriyi temizler
- `build_book_pairs.py` — Wikisource'tan continuation çiftleri (`chatgrow_kitap_*.jsonl`)
- `chatgrow.py` — sohbet çiftleri
- `tools/` — ölçüm araçları (`tools/README.md`'e bak)

Veri hattı CI'da da otomatik çalışır (`.github/workflows/`) ve `main`'e kendisi push eder.

## Bilinen sınırlar

- ~%40 oranında "bilgim yok" cevabı veriyor — veri eksiği, kodla çözülemez
- Üretimde sadakat (fidelity) %22–31; kopyalama eğilimi ölçülmüş sorun
- Kitap/Wikisource hattı kaynak tükenmiş: 609 çift (bkz. `build_book_pairs.py` docstring)
- **Decoding tükendi (ölçüldü, n=250 paired t-testi):** `knowledge_bias` (0/1,2/3,0),
  sıcaklık (0,7/0,3/0,05) ve `rep_penalty` (0,4/0,2/0,0) hiçbiri `gold_recall`'ı
  anlamlı değiştirmiyor. Model bağlamı kullanıyor (öğretmen koşullu argmax
  %68,2 → %86,7) ve bağlam cevabın %60,0'ını taşıyor, ama örneklemeli üretim
  altının %30,8'ini veriyor → darboğaz üretim biçimi. Ayrıntı: `DEVAM_PROMPTU.md` §7

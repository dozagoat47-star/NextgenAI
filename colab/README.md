# Nextgen AI - Colab ile GPU Eğitimi (PyTorch -> NumPy export)

Yerel NumPy eğitimi yavaş olduğundan aynı transformer mimarisi Google
Colab'da **PyTorch + GPU** ile eğitilir; ağırlıklar yerelin anladığı
`model.json` (NumPy, float32) formatına aktarılır. `app.py`, `brain.py`
hiç değişmeden çalışmaya devam eder.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `nextgen_transformer_colab.ipynb` | Eğitim notebook'u (self-contained) |
| `../tests/test_core.py::test_transformer_export_map` | Export şeması round-trip regresyon testi |

## Kullanım

1. `intents.json`, `brain.py`, `transformer.py` dosyalarını Colab'ın
   sol Files panelinden `/content` klasörüne yükle.
2. Notebook'u aç (`File > Upload notebook`) ve `Runtime > Run all`.
3. Eğitim bitince son hücredeki talimatla indir:
   - `model.json`
   - `model_weights.npz`
   - `bot_data.json`
4. Yerelde bu üçünü `model/` klasöründeki dosyaların üstüne kopyala.
   (Eski bozuk `model.json` ezilir; `model/_backup_feedforward/` yedeği
   dokunulmaz.)
5. Test: `python excel_predict.py` veya `python app.py`.

## Checkpoint / Resume (oturum kesilmesine dayanıklı)

- Her **500 optimizasyon adımında** (ve her epoch sonunda):
  `MyDrive/NextgenAI/checkpoint.pt`
- En iyi doğruluklu model: `MyDrive/NextgenAI/best.pt`
- Eğitim bitince: `MyDrive/NextgenAI/final.pt`
- Oturum kesildiyse notebook'u yeniden aç, setup hücrelerini çalıştır —
  `RESUME=True` olduğundan eğitim hücresi en son checkpoint'ten devam eder.
- Çekpoint içeriği: model ağırlıkları, optimizer (Adam) momentleri,
  global adım sayısı, epoch, en iyi doğruluk ve torch/numpy RNG durumu
  (rastgelelik aynı yerden sürer; karıştırma sırası korunur).

## Doğrulama

Notebook'un son hücresi, eğitilen torch modelini NumPy referans modeline
yükleyip `|torch - numpy|` farkını ölçer (< 2e-5 beklenir) ve birkaç örnek
soruya intent tahmini basar. Export sonrası yerelde de
`python -m unittest discover -s tests -v` çalıştırılarak
`test_transformer_export_map` doğrulanabilir.

## Parametreler (notebook'un "setup" hücresinde)

```
EPOCHS=500, BATCH_SIZE=32, LR_BASE=1e-3, WARMUP=200, LR_MIN=0.1
WEIGHT_DECAY=1e-4, GRAD_CLIP=5.0, PATIENCE=30, CKPT_EVERY=500
```

Bu değerler, yerelde NumPy ile doğrulanmış stabil kombinasyondur
(önceki ıraksama bug'larının ardından; bkz. `transformer.py`).

## İki katmanlı mimari: sohbet + bilgi ayrımı

Eski 791 intent'lik model %1.75'e takılıyordu (753 bilgi intent'i aynı
"X nedir" şablonuyla oluşturulmuş; 6.7 örnek/sınıf). Notebook artık:

1. **Sohbet sınıflandırması:** Deseni >6 olan intent'ler (gerçek veride
   38 adet) siniflandırıcıya öğretilir. `num_intents` 38'e iner, doğruluk
   hızla yükselir.
2. **Bilgi retrieval'i:** Kalan 753 bilgi intent'i `intents`/`intent_kws`
   içinde kalır; `bot_data.json` bunları içerir. Yerelde
   `ChatBot._select_knowledge` bilgi sorularını ters dizin (inverted index)
   ile yanıtlar — siniflandırıcıya hiç dokunmaz.
3. `get_response` akışı: bilgi sorusunda güven düşükse bilgi retrieval'e
   sorulur; bulunursa canned bilgi yanıtı, yoksa netleştirme.

Bu yüzden notebook'taki `bot.conversational_data(data)` filtresi
`intent_tags`'i 38'e indirir ama `intents`/`intent_kws`'i 791'de bırakır;
export sonrası yereldeki `brain.py` aynı bot_data ile iki katmanlı çalışır.

---

## SeqGen LSTM Yeniden Eğitimi (sorgu-koullu, PyTorch+GPU)

### Nedir?

Eski seq_model.json tag-koulluydu (örn. "hava_durumu" → anlamsız jenerik
metin). Yeni notebook **gerçek kullanıcı sorgusuyla** (örn. "hava nasil olacak")
koşullu LSTM eğitir → Türkçe tutarlı ve konuya uygun yanıt üretir.

### Dosyalar

| Dosya | Açıklama |
|---|---|
| `nextgen_seqgen_colab.ipynb` | SeqGen LSTM eğitim notebook'u (PyTorch + GPU) |

### Kullanım

1. `intents.json` ve `seqgen.py` dosyalarını `/content` altına yükle.
2. `nextgen_seqgen_colab.ipynb`'i aç, `Runtime > Run all`.
3. Eğitim bitince son hücreden `model/seq_model.json`'ı indir.
4. Yerelde `model/seq_model.json`'ı kopyala.
5. `brain.py` seqen'i artık kullanıcının sorgusuyla örnekliyor; model
   otomatik aktif olur, kalite kapısından geçemeyen çıktı canned cevaba
   düşer (güvenli fallback).

### Hiperparametreler

```
HIDDEN=256, EPOCHS=250, BATCH_SIZE=128, LR_BASE=2e-3, LR_MIN=0.01
GRAD_CLIP=5.0, PATIENCE=15, MAX_PAIRS=20000, DROPOUT=0.15
```

### EOS durdurma sinyali

- `seqgen.encode_pair` artık her yanıt sonuna `<EOS>` ekler; model yanıtın
  bittiğini öğrenir (kayıp son pozisyonda da hesaplanır).
- `seqgen.sample` EOS üretebilecek (eskiden EOS yasaktı) ve seçilince
  `break` eder; `max_len=90` ile sınırlı güvenlik calibi vardır.
- Dropout yalnızca PyTorch **eğitiminde** uygulanır; `eval()`/export'ta kapalı
  olduğundan NumPy parity bozulmaz.

---

## Seq2Seq Encoder-Decoder Eğitimi (sorgu-koullu, PyTorch+GPU)

### Nedir?

SeqGen LSTM'nin daha güçlü evrimi: **(kullanıcı sorgusu) -> (doğal Türkçe
yanıt)** ilişkisini öğrenen, numaralandırılabilir karakter-seviyesi transformer
**encoder-decoder**. LSTM'e kıyasla uzun bağımlılıkları, konu bağlamını ve
yanıt yapısını çok daha iyi modeller (copy-paste sorununa asıl çözüm budur).

### Dosyalar

| Dosya | Açıklama |
|---|---|
| `nextgen_seq2seq_colab.ipynb` | Seq2Seq eğitim notebook'u (PyTorch + GPU) |
| `../seq2seq.py` | NumPy inference modülü (ileri geçiş + örnekleme) |
| `../tests/test_core.py::TestSeq2Seq` | Kayıt/yükleme + örnekleme regresyonu |

### Nasıl çalışır?

1. Veri `seqgen.load_pairs(..., use_query=True)` ile (pattern, response)
   çiftleridir; `build_vocab` karakter sözlüğünü kurar (PAD/BOS/EOS = 0/1/2).
2. Torch modeli `seq2seq.Seq2Seq` ile **birebir aynı cebir**: paylaşılan
   embedding, sinusoidal konum (×1/sqrt(d)), N encoder + N decoder bloğu
   (Pre-LN, GELU, causal + cross-attn), final LN + lineer kafa. Parametre
   adları (örn. `b0_Wq`, `db0_Wqc`, `head`) düz attribute olarak tutulur →
   `named_parameters` doğrudan NumPy `params` anahtarlarına eşlenir.
3. Teacher-forcing CE kaybı yalnızca yanıt pozisyonlarında; Adam + warmup +
   cosine, grad-clip, erken durdurma. Checkpoint `seq2seq_ckpt.pt`.
4. Son hücrede **parity testi**: torch forward vs NumPy `_encode` +
   `_decoder_logits` farkı < 1e-3 doğrulanır (dropout eval'da kapalı olduğundan
   birebir eşit olmalıdır).

### Kullanım

1. `intents.json`, `seqgen.py`, `seq2seq.py` dosyalarını `/content` altına
   yükle.
2. `nextgen_seq2seq_colab.ipynb`'i aç, `Runtime > Run all`.
3. Eğitim bitince `model/seq2seq_model.json`'ı indir.
4. Yerelde `model/seq2seq_model.json`'ı kopyala.
5. `brain._try_seq_rephrase` artık **önce seq2seq'i** dener (char-level
   encoder-decoder), çıktı kalite kapısını geçemezse veya dosya yoksa
   **SeqGen LSTM'e** düşer; ikisi de yoksa canned cevaba döner.
6. Manuel test:
   ```python
   from seq2seq import load_seq2seq
   m = load_seq2seq()
   print(m.sample('hava nasil olacak', temperature=0.9, top_k=14))
   ```

### Hiperparametreler

```
D_MODEL=128, NUM_BLOCKS=4, NUM_HEADS=4, FF_MULT=3, DROPOUT=0.10
BATCH_SIZE=64, EPOCHS=400, LR_BASE=1e-3, LR_MIN=0.1, WARMUP=200
GRAD_CLIP=5.0, PATIENCE=20, MAX_PAIRS=20000
MAX_ENC_LEN=40, MAX_DEC_LEN=48, TARGET_MAX_LEN=42
```

### Önemli uyum notları

- Cross-attn çıktı adları kodda `db{i}_Wqc/Wkc/Wvc/Woc` + `db{i}_bqc/bkc/bvc/boc`
  (docstring ile aynı; export haritası bunlarla birebir eşleşir).
- `embed` satır 0 (PAD) hem NumPy hem Torch'ta sabit 0 tutulur.
- Dropout yalnızca eğitimde; `eval()`/export parity'si bozulmaz.
- `seq2seq.py` yalnızca ileri geçiş yapar; backprop yok (eğitim için Torch).

## Alternatif: Lightning AI ile eğitim (`train_seq2seq.py`)

Notebook'a gerek yok; bağımsız script `train_seq2seq.py` (kök dizinde) aynı
eğitimi terminalde koşar. Avantajları:
- **Kalıcı disk**: checkpoint restarta/copmaya rağmen durur; script otomatik
  kaldığı yerden devam eder (Colab `/content`'e benzeri yok).
- Arka plan çalıştırma; tarayıcı kapanınca eğitim sürer.
- Ücretsiz kredi (ayda 15, ~1 dolar = 1 kredi); T4 ~0.2-1.2 kredi/saat.

Çalıştırma: Studio'ya 5 dosyayı yükle (`intents.json`, `seqgen.py`,
`seq2seq.py`, `normalize.py`, `train_seq2seq.py`), GPU (T4) seç, sonra:

```bash
python train_seq2seq.py --epochs 250   # veya 400
SMOKE=1 python train_seq2seq.py        # 2 adım hız testi (CPU/GPU)
```

Çıktı: `seq2seq_model.json` -> yerel `model/` klasörüne kopyala.
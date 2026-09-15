# Nextgen AI

Sıfırdan yazılmış (NumPy-only, hazır ML kütüphanesi yok) Türkçe sohbet asistanı.
Üç katmanlı mimari: intent sınıflandırma (transformer) + yerel bilgi retrieval (RAG-lite) + üretim katmanı.

## Mimari

| Katman | Dosya | Görev |
|---|---|---|
| Sınıflandırıcı | `transformer.py`, `brain.py` | Transformer encoder (NumPy) — sohbet niyetlerini sınıflandırır |
| Bilgi arama | `corpus.py`, `knowledge.py` | RAG-lite: LSA/SVD + PPMI + BM25; bilinmeyen soru Wikipedia'dan |
| Üretim | `generator.py`, `seqgen.py`, `seq2seq.py` | Anchor-sabit parafraz (bigram) + koşullu LSTM üreteç + Seq2Seq transformer encoder-decoder |
| Öğrenme | `finetune.py` | LoRA adaptörüyle anlık intent ekleme/çıkarma (`/learn`, `/forget`) |
| Sunucu | `app.py` | Flask web arayüzü, yönlendirme kuralları |

İki katmanlı çalışma prensibi:
- **Sohbet niyetleri** (deseni >6 olan intentler) transformer ile sınıflandırılır.
- **Bilgi niyetleri** (Wikipedia şablonlu 750+) IDF anahtar-kelime retrieval ile cevaplanır.
- Model güven veremezse `corpus` -> internet (Wikipedia) fallback'ine düşer; internetten gelen bilgi `corpus.jsonl`'a kaydedilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Eğitim

> Eğitim Google Colab'da GPU ile yapılır; yerel NumPy eğitimi saatler sürer.

1. Eğitilmiş model `model/` dizininde hazır gelir: `model/model.json` + `model/model_weights.npz` + `model/bot_data.json` (+ opsiyonel `model/lora.json`, `model/seq_model.json`, `model/seq2seq_model.json`).
2. Yeniden eğitmek için `colab/nextgen_transformer_colab.ipynb` (GPU) kullanın, çıktıyı `model/`'e kopyalayın.
3. Yerel NumPy eğitimi (yavaş, önerilmez): `python train.py --epochs 500 --lr 0.001`
4. Karakter üreteci (Seq2Seq encoder-decoder): `colab/nextgen_seq2seq_colab.ipynb` (önerilen)  
5. Karakter üreteci (LSTM fallback): `colab/nextgen_seqgen_colab.ipynb` veya `python seqgen.py`

## Çalıştırma

```bash
python app.py        # http://localhost:5000
```

API uçları:
- `POST /chat` — `{"message": "..."}` -> `{"response": "..."}`
- `POST /learn` — `{"entries": [{"tag", "patterns", "responses"}]}` (LoRA ile öğretir)
- `POST /forget` — `{"tag": "..."}` (LoRA intentini geri alır)
- `GET /status` — model/corpus durumu

## Test

```bash
python -m unittest discover -s tests -v
```

## Veri hattı

- `autogrow.py` — Wikipedia'dan otomatik bilgi toplar, `intents.json` + `corpus.jsonl`'ı büyütür.
- `clean_intents.py` — toplanan ham veriyi temizler (Latin dışı yazım, kesik biyografi, zararlı tag).

## Bilinen sınırlar / yön

- Çekirdek bir dil modeli değil, sınıflandırıcı + retrieval'dır; akıl yürütme, kurgu, çok adımlı mantık sınırlıdır.
- Yanıt üretimi parafraz esaslıdır; Seq2Seq üreteç kopyalanan yanıtları koşullu yeniden üretir (tam semantik üretim değil).
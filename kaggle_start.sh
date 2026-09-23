#!/usr/bin/env bash
# Kaggle Notebook (P100/T4x2 GPU, haftada 30 sa ucretsiz) - Nextgen LLM egitimi.
#
# Kullanimi (Kaggle'da):
#   1) kaggle.com -> New Notebook ac, ad ver.
#   2) Ayarlar (yukari sag) -> Internet: ON  |  Accelerator: GPU P100 (veya T4x2)
#   3) Ilk kod hucresine YAPISTIR (repo klonu + bagimlilik):
#         !git clone https://github.com/dozagoat47-star/NextgenAI.git
#         %cd NextgenAI
#         !python -m pip install --quiet numpy
#   4) Ikinci hucresine:
#         !bash kaggle_start.sh train 250
#      (once deneme istersen:  !bash kaggle_start.sh verify   )
#      (1 epoch suresi olcmek icin:  !bash kaggle_start.sh bench )
#   5) Egitim sonrasi indirme hucresi (asagidaki INDIRME notuna bak).
#
#   Veri NOTU: train_llm MAX_PAIRS=70000 cifti natural 5 ile ~330k cifte
#   cikarir -> epoch basina sure eskiye gore ~5.5x. erken durdurma (patience)
#   otomatik keser; sure endiseleniyorsan 250 yerine 80 ver.
#
#   Veriyi/intents'i degistirdiysen repo'ya push ettikten sonra yine 1. adim
#   (clone) yeterli - tum dosyalar taze gelir.

set -euo pipefail
cd /kaggle/working/NextgenAI

MODE="${1:-verify}"
EPOCHS="${2:-250}"

CGARG=''
if compgen -G 'chatgrow_*.jsonl' > /dev/null; then
  echo "[0/3] ChatGrow verisi bulundu, egitim hattina eklenecek."
  CGARG="--chatgrow $(ls chatgrow_*.jsonl | tr '\n' ' ')"
fi

# Kapasite (varsayilan d=384 / 6 blok ~22.9M; env ile asilabilir)
DPARGS="${LLM_CAP:+--d-model $LLM_CAP} ${LLM_BLOCKS:+--num-blocks $LLM_BLOCKS}"

DONE=''
case "$MODE" in
  train)
    echo "[1/3] RAG egitim (natural 5, epochs=$EPOCHS, d=384/6 blok) -> llm_model.json"
    python train_llm.py --rag --kb-map knowledge_map.jsonl --natural 5 $CGARG \
      --epochs "$EPOCHS" --batch-size 64 --val-every 2 $DPARGS 2>&1 | tee kaggle_train.log
    DONE='yes'
    ;;
  bench)
    echo "[1/3] 1-epoch zamanlama (cache/encode + 1 epoch, birlikte olculur)"
    python train_llm.py --rag --kb-map knowledge_map.jsonl --natural 5 $CGARG \
      --epochs 1 --batch-size 64 --val-every 1 $DPARGS 2>&1 | tee kaggle_bench.log
    echo ""
    echo "[2/3] Son egitim satiri (epoch suresi '| NN.Ns' bolumundedir):"
    grep 'epoch ' kaggle_bench.log | tail -1
    echo "[3/3] Encode ilk seferde ~10 dk ayri, sonra onbellegi kullanilir."
    echo "      Duvar suresi icin Kaggle 'Cell executed in NHmNs' degerine bak."
    ;;
  verify)
    echo "[1/3] dry-run dogrulama (GPU gerekmez, ~1-2 dk; TAM encode YAPILMAZ)"
    echo "      Ayni veri bayraklari -> onbellek parmak izi bench/train ile ayni."
    python train_llm.py --dry-run --rag --kb-map knowledge_map.jsonl --natural 5 \
      --batch-size 64 --limit-pairs 4000 $CGARG $DPARGS
    echo "[2/3] OK - ilk-kelime hizalama ve RAG hatti hazir."
    echo "[3/3] Tam egitim icin:  !bash kaggle_start.sh train 250"
    ;;
  *)
    echo "Bilinmeyen mod: $MODE  (verify | train)"
    exit 2
    ;;
esac

if [ -n "$DONE" ]; then
  echo ""
  echo "[2/3] Cikti zip'e aliniyor (model: llm_model.json + llm_model_weights.npz)..."
  mkdir -p /kaggle/working/cikti
  cp -f llm_model.json llm_model_weights.npz /kaggle/working/cikti/ 2>/dev/null || true
  cp -f kaggle_train.log /kaggle/working/cikti/ 2>/dev/null || true
  ls -la /kaggle/working/cikti/
  (cd /kaggle/working/cikti && zip -q -9 /kaggle/working/nde-irma.zip \
     llm_model.json llm_model_weights.npz kaggle_train.log 2>/dev/null || \
     (tar -czf /kaggle/working/nde-irma.tar.gz \
        llm_model.json llm_model_weights.npz kaggle_train.log 2>/dev/null || true))
  echo "[3/3] INDIRME: Asagidaki sekmelerden birini kullan:"
  echo "  - /kaggle/working/nde-irma.zip  (ya da .tar.gz)"
  echo "  Kaggle'da dosya indirme: Notebook'u SAVE (Version) yapinca Output"
  echo "  sekmesinden 'Download All' ile iner; veya dosya adlariyla aratip tek tek."
  echo "  Iki dosya birlikte model/ klasorune kopyalanir (llm_model.json + llm_model_weights.npz)."
fi

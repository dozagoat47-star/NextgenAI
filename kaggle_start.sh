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
#   5) Egitim sonrasi indirme hucresi (asagidaki INDIRME notuna bak).
#
#   Veriyi/intents'i degistirdiysen repo'ya push ettikten sonra yine 1. adim
#   (clone) yeterli - tum dosyalar taze gelir.

set -euo pipefail
cd /kaggle/working/NextgenAI

MODE="${1:-verify}"
EPOCHS="${2:-250}"

DONE=''
case "$MODE" in
  train)
    echo "[1/3] RAG egitim (epochs=$EPOCHS) -> model/llm_model.json"
    python train_llm.py --rag --kb-map knowledge_map.jsonl --natural 3 \
      --epochs "$EPOCHS" 2>&1 | tee kaggle_train.log
    DONE='yes'
    ;;
  verify)
    echo "[1/3] dry-run dogrulama (GPU gerekmez, ~1 dk)"
    python train_llm.py --dry-run --rag --kb-map knowledge_map.jsonl
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
  echo "[2/3] Cikti zip'e aliniyor (webfetch'te indirme sorunu olmasin diye)..."
  mkdir -p /kaggle/working/cikti
  cp -f model/llm_model.json /kaggle/working/cikti/ 2>/dev/null || true
  cp -f kaggle_train.log /kaggle/working/cikti/ 2>/dev/null || true
  (cd /kaggle/working/cikti && zip -q -9 /kaggle/working/nde-irma.zip llm_model.json kaggle_train.log 2>/dev/null || \
     (tar -czf /kaggle/working/nde-irma.tar.gz llm_model.json kaggle_train.log 2>/dev/null || true))
  echo "[3/3] INDIRME: Asagidaki sekmelerden birini kullan:"
  echo "  - /kaggle/working/nde-irma.zip  (ya da .tar.gz)"
  echo "  Kaggle'da dosya indirme: Notebook'u SAVE (Version) yapinca Output"
  echo "  sekmesinden 'Download All' ile iner; veya dosya adlariyla aratip tek tek."
fi

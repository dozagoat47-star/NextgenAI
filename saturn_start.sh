#!/usr/bin/env bash
# Saturn Cloud (Jupyter Server / Terminal) nextgen_llm egitimi baslaticisi.
#
# Kullanim:
#   bash saturn_start.sh                      # clone + pip + dry-run dogrulama
#   bash saturn_start.sh train                # clone + pip + RAG egitimi (250 epoch)
#
# Notlar:
#  - Git/terminal kimligini Saturn Cloud resource'unda GitHub ile acin (repo public).
#  - GPU yoksa/limitliyse kucuk epoch ile baslayin: bash saturn_start.sh train 60

set -euo pipefail
cd ~/git 2>/dev/null || mkdir -p ~/git && cd ~/git

REPO="https://github.com/dozagoat47-star/NextgenAI"
DIR="NextgenAI"

if [ ! -d "$DIR/.git" ]; then
    echo "[1/4] git clone $REPO"
    git clone "$REPO"
fi
cd "$DIR"

echo "[2/4] pip bagimliliklari"
pip install --quiet --upgrade numpy
python -c "import numpy; print('numpy', numpy.__version__)"

MODE="${1:-verify}"
EPOCHS="${2:-250}"

CGARG=''
if [ -f chatgrow_sohbet.jsonl ]; then
  echo "[3/4] ChatGrow verisi bulundu, egitim hattina eklenecek."
  CGARG='--chatgrow chatgrow_sohbet.jsonl'
fi

case "$MODE" in
  train)
    echo "[3/4] RAG egitimi basliyor (epochs=$EPOCHS) -- llm_model.json uretecek"
    python train_llm.py --rag --kb-map knowledge_map.jsonl --natural 3 --epochs "$EPOCHS" --batch-size 64 --val-every 2 $CGARG 2>&1 | tee saturn_train.log
    ;;
  *)
    echo "[3/4] dry-run dogrulama"
    python train_llm.py --dry-run --rag --kb-map knowledge_map.jsonl $CGARG
    ;;
esac

echo "[4/4] Bitti."
echo "Egitim tamamlandiysa: model/llm_model.json -> bilgisayarina indir"
echo "  (Saturn Cloud dosya paneli / notebook'tan indirme yapilabilir)."

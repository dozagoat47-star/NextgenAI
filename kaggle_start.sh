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
#         !bash kaggle_start.sh train
#      (EPOCH vermezsen EPOCHS=12 kullanilir; veri butcesi intents ile
#       buyur, sure butcesi 1,4 milyon cift -> pratikte baglamaz)
#      (once deneme istersen:  !bash kaggle_start.sh verify   )
#      (1 epoch suresi olcmek icin:  !bash kaggle_start.sh bench )
#   5) Egitim sonrasi indirme hucresi (asagidaki INDIRME notuna bak).
#
#   SURE NOTU (d=384/6 blok, T4x2, max_seq 256, 29.09 kosusu). IKI AYRI
#   olcum var, karistirma: biri ENCODE hizi, digeri EPOCH hizi.
#
#   (1) ENCODE OLCUMU  -- OLCULDU: 1024172 cift, encode 1541 sn
#       (921.748 train + 102.424 val)
#       -> 664,5 cift/sn = train_llm.ENC_CIFT_SN
#   (2) EPOCH OLCUMU   -- EPOCH OLCUMU: 921748 cift, ort 1395 sn/epoch
#       (val olan epoch'lar ~1435 sn, val atlananlar ~1338 sn;
#        tek epoch secmek %7 yaniltirdi, ORT alindi)
#       -> 1,5139 ms/cift = train_llm.MS_PER_PAIR
#       12 epoch = 16.744 sn = 279,1 dk. Gercekten 12. epoch'a kadar
#       tamamlandi (en iyi val 0,3959 @ 8. epoch, acc 0,913).
#
#   DIKKAT: 28.09 olcumu 315.883 CIFTTE alinmisti (585 cift/sn, 1,3884
#   ms/cift) ve veri 3,2 KAT buyuyunca iki sabit de bayatlaydi. Yanlisi
#   yondu: MS_PER_PAIR kucuk saymak butceyi BUYUTUYORDU (12 epoch 396 dk
#   yerine gercekte 431 dk ister -> oturum kesilirdi).
#
#   DIKKAT 3 (02.10.2026): DIKKAT 2'nin buldugu birim hatasinin IKINCISI
#   bulundu ve o da duzeltildi -- bu sefer hata sure_olc.py'nin KENDISINDE
#   idi: MS_HAM'a encode katiliyordu, ama sure_ve_hesapla encode'u
#   kalan_dk'dan zaten dusuyor -> encode IKI KEZ sayiliyordu, sabit
#   %9,45 yuksekti. Dogru sabit 7,3927 ms/ham/epoch; tavan 234.207 ->
#   256.333. OLCUM: tools/sure_olc.py, DEVAM_PROMPTU.md 6.18.
#
#   DIKKAT 2 (01.10.2026): yukaridaki "zaman butcesi pratikte baglayici
#   DEGIL" yorumu YANLISTI ve kaldirildi. Asil hata sabit yanlisi degil
#   BIRIM hatasiydi: MS_PER_PAIR egitim cifti, tavan ham cift sayiyordu
#   (expansion 01.10'da OLCULDU: 4,7711). Tavan gercek sinirdan 5,34 KAT
#   uzaktaydi ve 29.09'dan beri hic baglamiyordu. Artik HAM cift sabiti
#   kullaniliyor ve tavan GERCEKTEN bagliyor.
#   OLCUM: tools/sure_olc.py, DEVAM_PROMPTU.md 6.16.
#   Erken durdurma (patience) val yukselmeye baslayinca keser; EPOCH
#   vermezsen 12 kullanilir, yine olusturulabilir. Daha uzun egitim istersen
#   LLM_EPOCHS=150 gibi ver ve Kaggle oturum suren yeterli olsun.
#
#   ONCEKILERE DOKUNMA: eski kosularda patience "kotu val OLCUMU" sayiyordu,
#   --val-every 2 ile birlikte tolerans 12 epoch'a cikiyordu; 10 epoch'lik
#   olculmus kosuda val 4. epoch'tan yukselmeye baslamis olmasina ragmen
#   erken durdurma HIC tetiklenememisti. Artik patience EPOCH cinsindendir
#   (bkz. train_llm.py --patience) ve --val-every yalnizca maliyeti etkiler.
#
#   Veriyi/intents'i degistirdiysen repo'ya push ettikten sonra yine 1. adim
#   (clone) yeterli - tum dosyalar taze gelir.

set -euo pipefail
cd /kaggle/working/NextgenAI

MODE="${1:-verify}"
# Butce: patience=2 ile kosu ilk val yukselmesinde (~6. epoch) bitecegi
# icin 70 yerine 12 yeter. DIKKAT: EPOCHS artik yalnizca ust sinir DEGIL --
# lr_horizon = min(EPOCHS, patience+20) oldugu icin EPOCHS lr programini da
# belirler. 12 -> lr_horizon=12, yani LR butun butce boyunca LR_MIN'e iner
# (eski 10 epoch'lik kosunun lr_horizon=10'u ile kiyaslanabilir).
EPOCHS="${2:-${LLM_EPOCHS:-12}}"
# Sabir EPOCH cinsinden (bkz. train_llm.py) ve --val-every 2 ile 1 kotu
# val OLCUMU = 2 epoch demektir.
#
# 29.09 OLCUMU: patience=2 ile kosu 6. epoch'ta BITTI, val kaybi 0,5827
# (4. epoch) -> 0,5901 (6. epoch), yani %1,3 artti. Ama ayni noktada acc
# HALA YUKSELIYORDU (0,876 -> 0,885) ve train kaybi hizla iniyordu
# (0,4784 -> 0,3425). lr_horizon=12 idi, yani kosu LR'nin tam inmedigi
# yerde kesildi; 27.09 kosusu 12 epoch'a tamamlamisti. Tek kotu olcum
# gurultuyle erken durduruyor.
#
# 4 = iki kotu olcum (4 epoch). lr_horizon etkilenmez:
# min(EPOCHS=12, patience+20=24) = 12 -> LR programi AYNI kalir, sadece
# erken durma gevser. 12 epoch x ~18 dk = ~3,6 saat (9 saat oturuma sigar).
PATIENCE="${LLM_PATIENCE:-4}"

echo "[0/3] Veri kapsami: yalnizca intents.json; ChatGrow dosyalari egitime alinmayacak."
echo "      RAG/knowledge_map egitim yolu dogrulanana kadar kapali."

# ---------------- VERI BUTCESI: SURE TAVANI (28.09) -----------------------
# MAX_PAIRS artik veriden OTOMATIK hesaplaniyor (train_llm.py
# coz_max_pairs): amaci 'butun ciftleri kullanmak' degil, 'benzersiz
# ctx'nin cogunu kapsamak'. Intentler buyudugu icin butce de buyur
# (6.364 intent -> ~120.000; 20.000 intent -> ~377.000).
#
# BURADA sure tamani hesaplanir cunku oturum suresi ve EPOCHS sadece
# burada biliniyor.Uc olculmus sabit kullanilir:
#   MS_PER_HAM_CIFT_EPOCH  8,0911 ms / HAM cift / epoch  (01.10 23:26)
#   ENCODE_DK              26,0 dk  (bir kez)
#   VARSAYILAN_BOSLUK      0,75    (oturumun %75'i veriye)
#
# DIKKAT (01.10.2026 duzeltmesi): once MS_PER_PAIR (1,5139 ms) kullaniliyordu
# ama o bir EGTIM cifti (post-expansion) maliyetidir; tavan ise HAM cift
# (pre-expansion) sayar. Expansion 01.10'da OLCULDU: 4,7711. Birim hatasi
# tam o kadar -> tavan gercek sinirdan 5,34 KAT uzaktaydi ve 29.09'dan beri
# HICBIR ZAMAN baglamiyordu. Eski yorumlardaki "7,31 dk/epoch" ve "9 dk
# encode" sayilari da bu hatanin yanlis oldugunu SANYAN kaleydi; ikisi de
# kullanilmiyor. OLCUM: tools/sure_olc.py, ayrinti DEVAM_PROMPTU.md 6.16.
#
# TAVAN FORMULU train_llm.sure_ve_hesapla()'da TEK DOGRULUK KAYNAGI olarak
# yazilidir (bash'ta yazilsaydi test edilemezdi; ilk yazimda saat->dk cevirisi
# iki kez yapildigi icin tavan 316.000 yerine 9.575 cift cikti ve egitimi
# mahvedecekti). Burada sadece oturum bilgisi (9 saat) ve EPOCHS aktarilir.
OTURUM_DK="${LLM_OTURUM_DK:-540}"
# Tavan ve kullanilan iki sabit KODDAN okunur, bash'a YAZILMAZ; yoksa log
# satiri koddaki degisikligi yansitmaz ve kaggle_train.txt olcum kaynagi
# olarak yaniltir (29.09'da "7,31 dk/epoch" yazan yorum bu yuzden elle
# guncellenmisti ve kodla arasi ayrilmisti).
MPCAP=$(python -c "import train_llm as t; print(t.sure_ve_hesapla(oturum_dk=$OTURUM_DK, epochs=$EPOCHS))" 2>/dev/null || echo 60000)
MSHAM=$(python -c "import train_llm as t; print(t.MS_PER_HAM_CIFT_EPOCH)" 2>/dev/null || echo 0)
ENCDK=$(python -c "import train_llm as t; print(t.ENCODE_DK)" 2>/dev/null || echo 0)
echo "[0/3] Veri butcesi tavani: $MPCAP cift ($OTURUM_DK dk oturum, $EPOCHS epoch,"
echo "      %75 kullanim, ${ENCDK} dk encode; ${MSHAM} ms/HAM-cift/epoch olcusunden)"
MPCARGS="--max-pairs-cap $MPCAP"

# Kapasite: varsayilan d=512 / 8 blok (~68M). Env ile asilabilir:
#   LLM_CAP=512 LLM_BLOCKS=8 bash kaggle_start.sh train ...
# max-seq-len 256 OZEL BIR SAYI DEGIL, olcumle secildi (knowledge_map.jsonl,
# 1000 RAG ornegi + 68794 ciftin gercek karmasi):
#   kb_budget = max_seq - max_ctx - 8 oldugu icin kucuk degerde YANIT yer
#   kalmaz ve RAG yanitlari kirpilir. RAG isabeti %57.
#     max_seq  ort uzunluk  islem   RAG yaniti kirpildi
#        128       110     1.00x        %87   <- eski (bu hatayi yaratti)
#        192       137     1.24x        %37
#        256       143     1.30x        %3    <- secilen
#   Artan islem yalnizca %30: uzunluk-kirpimli batch'ler (train_llm.py
#   _pack_encoded) RAGsiz 68794 ciftin ort uzunlugu degismiyor.
#   Uretimde de on-ek 128'de 26 -> 256'da ~160 token bosluk birakir.
#   Bu train_llm.py MAX_SEQ_LEN varsayilaniyla AYNI olmali (tek kaynak);
#   Colab notebooku da bu degere bagli.
DPARGS="--d-model ${LLM_CAP:-512} --num-blocks ${LLM_BLOCKS:-8} --max-seq-len ${LLM_SEQ:-256}"

# Duzenlestirme: gomme<->cikis bagliligi + AdamW. Varsayilanlar train_llm.py
# ile ayni; burada ACIK yazilir ki Kaggle logu kendi kendini belgelensin.
#   baglilik : 23.0M -> 16.8M parametre (-%26.8), dosya 87.6 -> 64.1 MB.
#               Kapatmak icin: LLM_TIE=0
#   AdamW    : wd=0.01, yalnizca agirilik matrislerine; bias/LayerNorm/gomme
#              cezasiz. Kapatmak icin: LLM_WD=0
REGARGS="--weight-decay ${LLM_WD:-0.01}"
if [ "${LLM_TIE:-1}" = "0" ]; then
  REGARGS="$REGARGS --untie-embeddings"
fi

# Dropout ve dogal cogaltma. Ikisi de ezberlemeyi GECIKTIRMANIN kaldiracidir:
#   9.855 farkli sorudan 350.301 satir uretiliyor (soru basina ~36 tekrar) ve
#   model 16.9M parametre. 12 epoch'lik kosuda val 2. epoch'tan sonra
#   monoton yukseliyor (0.4728 -> 0.5177), yani model kapasitesini asiyor.
#   Daha yuksek dropout ve daha az varyant bu noktayi GECIKTIRIR ama tavanı
#   KALDIRMAZ: bilgi tasiyan farkli soru sayisini artirmak gerekir.
#   Varsayilanlar mevcut uretim ayarlaridir; deney icin:
#     LLM_DROPOUT=0.2 LLM_NATURAL=2 bash kaggle_start.sh train
DROPOUT="${LLM_DROPOUT:-0.10}"
NATURAL="${LLM_NATURAL:-5}"
GRAD_ACCUM="${LLM_GRAD_ACCUM:-1}"
REGARGS="$REGARGS --dropout $DROPOUT"
# Gradient accumulation: --grad-accum N. Effective batch = batch_size * N.
# Tek GPU (LLM_DP_OFF=1) icin: LLM_GRAD_ACCUM=2 --batch-size 64 -> eff_batch 128, VRAM yarilanir.

DONE=''
case "$MODE" in
  train)
    echo "[1/3] intents.json egitimi (natural $NATURAL, dropout=$DROPOUT, grad_accum=$GRAD_ACCUM, epochs=$EPOCHS, patience=${PATIENCE} epoch, d=${LLM_CAP:-512}/${LLM_BLOCKS:-8}) -> llm_model.json"
    # --val-every 2 yalnizca VAL MALIYETI icin (olculmus: epoch 7.5 -> 7.1 dk).
    # Erken durdurma esigini ETKILEMEZ: patience artik epoch cinsinden.
    python train_llm.py --natural "$NATURAL" \
      --epochs "$EPOCHS" --patience "$PATIENCE" \
      --batch-size 64 --val-every 2 --grad-accum "$GRAD_ACCUM" $DPARGS $REGARGS $MPCARGS 2>&1 | tee kaggle_train.log
    DONE='yes'
    ;;
  bench)
    echo "[1/3] 1-epoch zamanlama (cache/encode + 1 epoch, birlikte olculur)"
    python train_llm.py --natural "$NATURAL" \
      --epochs 1 --batch-size 64 --val-every 1 --fresh --grad-accum "$GRAD_ACCUM" $DPARGS $REGARGS $MPCARGS 2>&1 | tee kaggle_bench.log
    echo ""
    echo "[2/3] Son egitim satiri (epoch suresi '| NN.Ns' bolumundedir):"
    grep 'epoch ' kaggle_bench.log | tail -1
    echo "[3/3] Encode ilk seferde ~10 dk ayri, sonra onbellegi kullanilir."
    echo "      Duvar suresi icin Kaggle 'Cell executed in NHmNs' degerine bak."
    echo "      Ort 7.31 dk/epoch ise 9h icin en fazla ~70 epoch:"
    echo "        !bash kaggle_start.sh train"
    ;;
  verify)
    echo "[1/3] dry-run dogrulama (GPU gerekmez, ~1-2 dk; TAM encode YAPILMAZ)"
    echo "      Veri kapsami yalnizca intents.json; ChatGrow ve RAG kullanilmaz."
    python train_llm.py --dry-run --natural "$NATURAL" \
      --batch-size 64 --limit-pairs 4000 $DPARGS $REGARGS $MPCARGS
    echo "[2/3] OK - intents.json egitim hatti hazir."
    echo "[3/3] Tam egitim icin:  !bash kaggle_start.sh train"
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

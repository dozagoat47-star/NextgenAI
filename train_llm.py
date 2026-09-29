"""Nextgen AI - Decoder-only LLM (llm.py) egitimi (bagimsiz script).

Kaggle/Colab/Lightning AI uyumlu PyTorch egitim scripti. Nucleo inference
(llm.py) ile Torch modeli BIREBIR ayni cebiri calistirir; dropout yalnizca
training'de, eval'da kapali -> parity bozulmaz.

Veri formati (tek zincir, otoregresif):
    <PAD> <BOS> <sorgu> <SEP> <yanit> <EOS>
    <PAD> <BOS> <sorgu> <SEP> <bilgi-parcasi> <SEP> <yanit> <EOS>   (--rag)
  Kayip YALNIZCA yanit pozisyonlarinda; --rag ile bilgi parcasi bağlam
  olur -> model verilen bilgiden OZGUN cumle kurmayi (akil yurutme) ogrenir.

Kullanim (yerel dogrulama icin torch gerektirmez):
  python train_llm.py --dry-run [--rag] [--natural K]   # veri hattini dogrula
  SMOKE=1 python train_llm.py                            # 2 adim CPU hiz testi

Hiz (Kaggle T4 2x):
  - DINAMIK batch: her sekans PAD kuyrugu budanarak gercek uzunluga kirpilir
    ve benzer uzunluklar kume halinde paketlenir -> transformator yalnizca
    gerekli uzunlukta cosar. Verinin cogu max_seq'ten kisa oldugu icin islem
    ~1.5-2x hizlanir (600sn -> ~300sn/epoch deneyimi).
  - --batch-size 64 (artik varsayilan), --val-every 2 ile val gecisleri
    yarilanir. Kucuk modelde DataParallel senkron ucreti batch'i asabilir;
    tek GPU denemek icin: LLM_DP_OFF=1 python train_llm.py ...

Kaggle'da egitim (GPU notebook):
  python train_llm.py --rag --natural 5 --kb-map knowledge_map.jsonl   # oneri (d=256)
  python train_llm.py --rag --natural 5 --kb-map knowledge_map.jsonl --d-model 384 --num-blocks 6
  python train_llm.py --rag --natural 3 --kb-map knowledge_map.jsonl --max-ctx-len 48 --max-seq-len 256
  python train_llm.py --epochs 400 --patience 40 --batch-size 64 --val-every 2
Veri boyu: MAX_PAIRS=70000 ham cift N5 ile ~330k cift -> epoch basina sure eski
(60k cift) 60/gore ~5.5x artar; erken durdurma (patience) devrede -> genelde
cok daha azda durur, asiriya kacmaz. Sure endiseleniyorsan --epochs 80 --patience 12.

ERKEN DURDURMA: val LOSS uzerinden calisir (VAL_IMP=5e-4; en iyi val
kaybini bu kadar altina cekemeyen her val epoch'u 'iyilesme yok' sayar).
Val kaybi yukselmeye basladiginda sayac dolar ve patience sonrasi eğitim
DURUR; en iyi val kaybindaki agirliklarla kapanir. --patience duyarliligi
ayarlar (orn. --patience 6 -> sert, --patience 20 -> rahat).

COSINE OGRENME HIZI UFKU: --epochs DEGIL. Pratikte egitim --epochs'ten cok
once (val plato -> ~patience*val_every) biter; ufuk --epochs kalirsa LR o
noktaya kadar inmis gibi gorunmez (cosine faktoru ~1) ve egitim tepede
biter. Varsayilan ufuk: min(epochs, patience*val_every + 20) -- yani
cosine, erken durdurma noktasindan biraz SONRA tamamlanir. Boylece LR
gercekten iniyor, val daha gec yukseliyor, daha iyi genelleme cikiyor.
Elle ayarlamak icin: --lr-horizon N.

DOGRULAMA BOLMESI: train/val ayrimi CIFT (pair) seviyesinde DEGIL, SORGU
(ctx) seviyesindedir -- group_split. load_pairs her pattern'i birden cok
yanitla eslestirir, naturalize_pairs her cifti k varyanta bolerken ctx'yi
sabit tutar; bu yuzden pair seviyesinde bolmede val'in TAMAMI train'de de
bulunur (olculdu: %100 sizinti) ve val kaybi ezberlemeyi goremez.

Kapasite flag'leri: --d-model --num-blocks --num-heads --ff-mult
  --max-ctx-len --max-seq-len --batch-size --lr-base. Varsayilani d=256'dir;
  eski (d=128) model dosyalari veri ogesi tasidigi icin geriye donuk yuklenir.

DUZENLESTIRME (varsayilan acik, kapatmak icin flag):
  - GOMME<->CIKIS BAGLILIGI: head = embed^T. Cikis matrisi ayri saklanmaz;
    23.0M -> 16.8M parametre (-%26.8), agirlik dosyasi 87.6 -> 64.1 MB.
    Yeni egitimde varsayilan ACIK. Kapatmak: --untie-embeddings. Eski (baglanmamis)
    modeller bayrak yok sayilarak aynen calismaya devam eder.
  - AdamW: --weight-decay (varsayilan 0.01). Ceza YALNIZCA agirilik
    matrislerine; bias, LayerNorm ve gomme/cikis CEZASIZ kalir (gomme seyrek
    tablodur, ceza kullanilmayan tokenlari kalici sifira cekerdi).
    --weight-decay 0 -> cezasiz AdamW.

Ciktilari:
  <SAVE_DIR>/llm_ckpt.pt    -> kaldigi yerden devam (restart/copma guvenli)
  <export-dir>/llm_model.json + llm_model_weights.npz
Notebook'ta SAVE_DIR='../working' ayarla; indirilen iki dosyayi yerel
model/ klasorune kopyala (llm.load_llm otomatik agar).
SAVE_DIR varsayilani script klasoru; ortam degiskeni ile asilabilir.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    # Dikkat: burada torch.cuda.is_available() CAĞRILMAZ. Cok-cekirdekli
    # BPE-encode 'fork' kullanir; canli CUDA baglamiyla fork edilirse
    # tilekilenebilir. CUDA ancak prepare_data (fork) BITTIKTEN sonra acilir.
except Exception:
    torch = nn = None
    HAVE_TORCH = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from seqgen import RESP_CHARS_MAX, clean_chars, load_pairs
from llm import PAD, LLM, build_llm_vocab, encode_llm, load_tokenizer
from naturalize import naturalize_pairs

# ---------------- hiperparametreler (numpy inference ile AYNI mimari)
# Varsayilanlar "buyuk" kapasite icindir (d=256); eski model dosyalari veri
# ogesi oldugu icin geriye donuk yuklenmeye devam eder. Kaggle'da daha
# buyuk model icin:  python train_llm.py --d-model 384 --num-blocks 6
D_MODEL = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
FF_MULT = 4
# Varsayilan duzenleme 0.10. DIZENLEME DENEYI icin `--dropout` ile degistirilebilir
# (bkz. --dropout yardimi): val egrisi 2. epoch'tan sonra monoton yukseldigi
# icin (12 epoch'lik kosu: 0.4728 -> 0.5177) modelin kapasitesi veriyi asiyor.
DROPOUT = 0.10
BATCH_SIZE = 64
LR_BASE = 1e-3
LR_MIN = 0.1
WARMUP = 200
PATIENCE = 6      # erken durdurma (val loss tabanli): val kaybi VAL_IMP kadar
#                  # altina inemeyen art arda ~bu kadar EPOCH'ta DURUR -> val
#                  # yukselmeye basladiginda plato yakalanir. EPOCH cinsindendir
#                  # (kotu val OLCUMLERI val_every kadar epoch biriktirir), boylece
#                  # --val-every maliyet dumesi olarak kalir, esegi DEGISTIRMEZ.
VAL_IMP = 5e-4   # ckpt-secimi + ERKEN-DURDURMA toleransi: val kaybi eski
# rekorun bu kadar altina inemiyorsa 'iyilestme yok' sayilir (best_state /
# en iyi kayipta dondurulur; patience uzerinde kalirsa egitim durur).
LR_HORIZON_PAD = 20   # cosine UFKU, erken durdurma noktasindan bu kadar SONRA
#                      # olmalidir. Hesap: ufuk = patience + PAD.
#                      # Aksi halde ufuk --epochs'a takilir (or. 70) ve val 10.
#                      # epoch'ta dururken LR 1e-3'ten 9.4e-4'e inmis gibi
#                      # olur (cosine hic calismaz) -> egitim tepede biter.
GRAD_CLIP = 5.0
WEIGHT_DECAY = 0.01   # AdamW ayrik cezasi. 0.01: 16.9M parametre / 68.8k
                      # gercek ornek. Gomme ve norm/bias cezasiz (bkz.
                      # optimizer kurulumu). 0 yapmak asiri uyuma hic mudahal
                      # vermiyordu.
TIE_EMBED = True      # gomme <-> cikis bagliligi (varsayilan acik)
CKPT_FREQ = 1   # her epoch kaydedilir -> Colab kesilse bile max ~1 epoch kayip, resume aninda
MAX_PAIRS = 0   # 0 = OTOMATIK (veriden olcerak, coz_max_pairs). >0 elle tavan.
                  # Bu bir veri butcesidir, epoch suresi ayari DEGILDIR; sure
                  # tavana kaggle_start.sh'den gelir (--max-pairs-cap). Kod
                  # butceyi hesaplar, insan unutmaz.
                  #
                  # ESKI DEGER 70.000 ve yorumu "ham ciftlerin TAMAMI
                  # kullanilir (intents.json: ~68.654)" idi. Yani deger
                  # veriye gore konmus, epoch suresi icin degil. Ama veri
                  # buyuyunce 6.364 intent'te uretilen cift 162.975'e cikti
                  # ve 70.000 yine %57'sini kirpiyor -> yorum gecersiz.
                  #
                  # OLCUM: bir intent N pattern x M cevap -> N*M cift uretiyor.
                  # Bilgi N+M'de, N*M'de degil. 6.364 intent'te:
                  #   benzersiz ctx 38.954 | benzersiz cevap 23.366
                  # Ince tarama (benzersiz ctx / cevap artisi):
                  #   70.000   31.848  (baz)      22.570  (baz)
                  #   90.000   35.287  +%10,8     23.163  +%2,6
                  #  120.000   38.048  +%19,5     23.359  +%3,5
                  #  140.000   38.760  +%21,7     23.366  +%3,5
                  #  162.975   38.954  +%22,3     23.366  +%3,5
                  # => 120.000'te benzersiz ctx'nin %97,7'si, benzersiz
                  # cevabin %99,97'si kapsanir. 120k -> 163k arasi +%37
                  # sure verip yalnizca +%2,4 ctx ve +%7 CEVAP getiriyor;
                  # 200.000/300.000/400.000 tamamen bos yazma.
                  #
                  # NEDEN SABIT DEGIL: intents 6.364 -> 20.000 olunca
                  # benzersiz ctx ~122.000'e cikar ve 120.000 yeniden darbo
                  # gaz olur. Karpanlar olcumle sabitlendi; butce intents
                  # ile birlikte buyur.
                  #
                  # SURE: 7,31 dk/epoch (MAX_PAIRS=70.000'de, kaggle_start.sh
                  # yorumunda OLCULEN) dogrusal olceklendiyor. Tavan
                  # kaggle_start.sh'de hesaplanir (9 saatlik oturum ve EPOCHS
                  # orada bilindigi icin) ve --max-pairs-cap ile gelir.
                  # Dogrulama:  !bash kaggle_start.sh bench
MAX_PAIRS_CTX_CARPAN = 3.08      # olcum: 120.000 / 38.954 benzersiz ctx
MAX_PAIRS_INTENT_CARPAN = 18.85  # olcum: 120.000 / 6.364 intent
                                  # (ikisi ayni olcumu ifade eder; biri
                                  #  dogrudan olculmus, digeri turetilmis)
MAX_CTX_LEN = 48    # sorgu icin token butcesi (olcum: gercek sorgu max 25 token
                    # -> 48 asilir, hic kesme yok; soru butcesini kucultmek
                    # bilgi/yanit yerine degil, bos yere yer acar)
MAX_SEQ_LEN = 256   # toplam sekans uzunlugu. TEK KAYNAK: kaggle_start.sh ve
                    # colab notebooku da buna bagli (128'de RAG yanitlarinin
                    # %87'si kirpilirdi). Olcum (knowledge_map 1000 ornek,
                    # 68794 ciftin gercek karmasi, RAG isabeti %57):
                    #   max_seq  ort  islem   kirpilan RAG yaniti
                    #      128   110  1.00x      %87   <- eski, kotu
                    #      192   137  1.24x      %37
                    #      256   143  1.30x      %3    <- burasi
                    # Artan islem %30'dur: _pack_encoded PAD kuyrugunu budayip
                    # uzunluga gore kumeler, RAGsiz ciftlerin ort uzunlugu
                    # degismez. Uretimde on-ek boslugu 26 -> ~160 token.
                    #
                    # NOT: yanit tarafi artik butceye uygun -- seqgen.RESP_CHARS_MAX
                    # = 204 karakter, ve BPE'de en uzun yanit 88 TOKEN cikiyor
                    # (olculdu, 4.000 yanit). Yani yanit asla tasmaz; kalan
                    # ~112 token RAG bilgisine kalir (kb_budget=200). Tablodaki
                    # "ort 143" 70 karakterlik yanitlarla olculmustu; yeni
                    # veriyle biraz yukselir, tasma olusmaz.
CTX_CHARS = 64      # sorgu icin KARAKTER butcesi: load_pairs ctx_len + kb LUT anahtar
#                    # uzunlugu. 40-char kesim 689 pattern'i kirpiyordu (data kaybi);
#                    # 64'te yalnizca 31 uzun pattern kesilir. kb anahtari da AYNI
#                    # kesim uzunluguyla uretilir -> RAG isabeti 65% -> 76%.
KB_TEXT_CHARS = 300 # kb parcasindan kullanilacak karakter sayisi (200 -> 300;
#                    # seq 192, kb butce ~136 token -> zengin bilgi koullandirmasi,
#                    # inference'ta brain'in ilettigi ~500 karaktere yaklasir).
SEED = 7

# ---- VERI HAZIRLAMA OLÇÜMLERI (29.09, 3.000 ciftin GERÇEK encode'u) ----
TOKEN_PER_PAIR = 89.3    # cift basina GERCEK token (kirpmali, RAG+kb-map acik,
                         # max_seq 256). Olcum: encode_llm -> PAD kuyrugu
                         # budanmis gercek uzunluk. Maske (loss) 38,6 token.
                         #
                         # 29.09 OLCUMU: 3 tohum (11/23/37) x 1000 cift =
                         # 89,27 +- 0,96 (SE). Yani %95 aralik 87,4-91,1.
                         # Onceki sabit 108,8di ve %18 YANLISTI.
                         #
                         # NEDEN 28.09 KABUL EDILMEDI: kaggle_train.log'daki
                         # "100,3M token" bir olcum DEGIL, sabitin kendisiyle
                         # carpimi (bkz. asagida yazan print). 921.748 x 108,8
                         # = 100,3M -> dongusel, hicbir sey dogrulamaz. Log'a
                         # bakip sabiti "dogrulandim" sanmak iki hatayi ust
                         # uste bindirirdi. Tek dogruluk kaynagi CANLI encode.
                         #
                         # ONCEDE '%.1fM' bicimiyle n * 0.16 yaziliyordu:
                         # 0,16 "binler" cinsinden ama M "milyon" demek ->
                         # 315.883 ciftte 50.541M yaziyordu, GERCEK 34,4M.
                         # 1.471 KAT HATA. Duzeltildi, olcum sabitlendi.
ENC_CIFT_SN = 664.5     # encode hizi, cift/sn (4 cekirdek, Kaggle T4x2).
                        # KAYNAK: 29.09 kosusu, UC BAGIMSIZ OLcum:
                        #   train 921.748 cift -> 1385 sn = 665,6 cift/sn
                        #   val   102.424 cift ->  156 sn = 656,6 cift/sn
                        #   toplam 1.024.172   -> 1541 sn = 664,5 cift/sn
                        # Onceki deger 585,0 DU (315.883 ciftten) ve %12
                        # YAVASESTI: sure tahmini 1.575 sn yerine gercek
                        # 1.541 sn (fazla kotu = guvenli yon, ama yine de
                        # yanlis). 29.09 verisi 2,9 KAT buyuk oldugu icin
                        # guncellenmediyse sure uyari metni yaniltirdi.
                        #
                        # ONCEDE n / 6000.0 vardi ve YORUMU "'dakika' birimi"
                        # diyordu: bolum sonucu SANIYE idi, dakika degil ->
                        # 315.883 ciftte "~52,6 dk" yaziyordu, GERCEK 9 dk.
                        # 5,8 KAT HATA. (1/0,16 = 6,25 ~ 6000: iki sabit de
                        # ayni yanlis olcekten geliyordu.)

SAVE_DIR = os.environ.get('SAVE_DIR', BASE)
os.makedirs(SAVE_DIR, exist_ok=True)
CKPT = os.path.join(SAVE_DIR, 'llm_ckpt.pt')
INTENTS = os.path.join(BASE, 'intents.json')


if HAVE_TORCH:
    class TorchLLM(nn.Module):
        """llm.LLM ile birebir AYNI cebir (PyTorch). Parametre adlari
        numpy 'params' anahtarlariyla eslesecek sekilde duz attribute tutulur
        (embed, b{i}_Wq/bk/bv/bo/ln/W1/W2, out_ln_g/b, head, head_b).
        Dropout yalnizca training'de; eval'da kapali -> NumPy parity bozulmaz.
        """
        def __init__(self, V, d_model=D_MODEL, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                     ff_mult=FF_MULT, max_seq_len=MAX_SEQ_LEN, drop=DROPOUT,
                     tie_embeddings=True):
            super().__init__()
            self.V, self.D = V, d_model
            self.N, self.H = num_blocks, num_heads
            self.hd = d_model // num_heads
            self.ff = ff_mult * d_model
            self.rsqrt = self.hd ** -0.5
            self.drop = drop
            # GOMME <-> CIKIS BAGLILIGI
            # embed (V,d) ve head (d,V) ayri ayri 6.14M parametre; modelin
            # %53.4'u yalnizca bu ikisi. Baglandiginda head = embed^T olur ve
            # 6.14M parametre (%26.7) tasarruf edilir. 23M -> 16.9M.
            # Baslangic olcegi zaten ayniydi (embed: randn*0.02, head: he*0.02)
            # ve Adam her parametreyi olcekleyerek guncelledigi icin cift
            # yonlu gradyan ayrica bir olcek carpimi gerektirmez.
            # head_b (cikis biasi) BAGLANMAZ: bias gommenin transpozunda
            # tasinamiyor, ayrica bir vektor olarak kalir.
            self.tie_embeddings = bool(tie_embeddings)

            def he(shape, scale=None):
                if scale is None:
                    scale = math.sqrt(2.0 / shape[0])
                return torch.randn(*shape) * scale

            with torch.no_grad():
                self.embed = nn.Parameter(torch.randn(V, d_model) * 0.02)
                pe = torch.zeros(max_seq_len, d_model)
                pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
                dim = torch.arange(d_model // 2, dtype=torch.float32)
                div = 10000.0 ** (2.0 * dim / d_model)
                pe[:, 0::2] = torch.sin(pos / div)
                pe[:, 1::2] = torch.cos(pos / div)
                pe = pe / math.sqrt(max(d_model, 1))
            self.register_buffer('pos_enc', pe)

            for i in range(num_blocks):
                for n, dims in [('Wq', (d_model, d_model)), ('Wk', (d_model, d_model)),
                                ('Wv', (d_model, d_model)), ('Wo', (d_model, d_model))]:
                    setattr(self, f'b{i}_{n}', nn.Parameter(he(dims)))
                    setattr(self, f'b{i}_b{n[1:]}', nn.Parameter(torch.zeros(1, d_model)))
                setattr(self, f'b{i}_ln1_g', nn.Parameter(torch.ones(1, d_model)))
                setattr(self, f'b{i}_ln1_b', nn.Parameter(torch.zeros(1, d_model)))
                setattr(self, f'b{i}_W1', nn.Parameter(he((d_model, self.ff))))
                setattr(self, f'b{i}_b1', nn.Parameter(torch.zeros(1, self.ff)))
                setattr(self, f'b{i}_W2', nn.Parameter(he((self.ff, d_model), scale=0.02)))
                setattr(self, f'b{i}_b2', nn.Parameter(torch.zeros(1, d_model)))
                setattr(self, f'b{i}_ln2_g', nn.Parameter(torch.ones(1, d_model)))
                setattr(self, f'b{i}_ln2_b', nn.Parameter(torch.zeros(1, d_model)))

            self.out_ln_g = nn.Parameter(torch.ones(1, d_model))
            self.out_ln_b = nn.Parameter(torch.zeros(1, d_model))
            if not self.tie_embeddings:
                self.head = nn.Parameter(he((d_model, V), scale=0.02))
            self.head_b = nn.Parameter(torch.zeros(1, V))

        def _ln(self, x, g, b):
            return torch.nn.functional.layer_norm(
                x, (x.size(-1),), g.reshape(-1), b.reshape(-1))

        @staticmethod
        def _causal_attn(q, k, v, rsqrt, heads, hd, drop_p=0.0, training=False):
            """Causal self-attention: torch SDPA (is_causal=True). Uygun GPU'da
            Flash/MemoryEfficient backend; degilse hizlandirilmis math backend.
            Eval'da dropout 0 -> numpy (llm.py) parity bozulmaz."""
            B, T, d = q.shape
            Q = q.reshape(B, T, heads, hd).transpose(1, 2)
            K = k.reshape(B, T, heads, hd).transpose(1, 2)
            V = v.reshape(B, T, heads, hd).transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None,
                dropout_p=drop_p if training else 0.0,
                is_causal=True)
            return o.transpose(1, 2).reshape(B, T, d)

        def _ffn(self, x, i):
            W1, b1 = getattr(self, f'b{i}_W1'), getattr(self, f'b{i}_b1')
            W2, b2 = getattr(self, f'b{i}_W2'), getattr(self, f'b{i}_b2')
            return torch.nn.functional.gelu(x @ W1 + b1, approximate='tanh') @ W2 + b2

        def forward(self, X):
            """X:(B,T) token ids -> logits:(B,T,V) (tam sekans, otoregresif)."""
            T = X.shape[1]
            x = self.embed[X]
            x = x + self.pos_enc[:T][None]
            if self.training and self.drop > 0:
                x = torch.nn.functional.dropout(x, self.drop)
            for i in range(self.N):
                pre = self._ln(x, getattr(self, f'b{i}_ln1_g'), getattr(self, f'b{i}_ln1_b'))
                q = pre @ getattr(self, f'b{i}_Wq') + getattr(self, f'b{i}_bq')
                k = pre @ getattr(self, f'b{i}_Wk') + getattr(self, f'b{i}_bk')
                v = pre @ getattr(self, f'b{i}_Wv') + getattr(self, f'b{i}_bv')
                a = self._causal_attn(q, k, v, self.rsqrt, self.H, self.hd,
                                      drop_p=0.05, training=self.training)
                x = x + a @ getattr(self, f'b{i}_Wo') + getattr(self, f'b{i}_bo')
                pre = self._ln(x, getattr(self, f'b{i}_ln2_g'), getattr(self, f'b{i}_ln2_b'))
                x = x + self._ffn(pre, i)
            h = self._ln(x, self.out_ln_g, self.out_ln_b)
            if self.tie_embeddings:
                return h @ self.embed.t() + self.head_b
            return h @ self.head + self.head_b


def llm_loss(logits, tgt, mask):
    """Otoregresif next-token: logits[t] -> tgt[t+1] (KENDI token'i degil)."""
    lg = torch.log_softmax(logits.float(), dim=-1)
    nxt = torch.full_like(tgt, PAD)
    nxt[:, :-1] = tgt[:, 1:]
    nll = lg.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
    return -(nll * mask).sum() / mask.sum().clamp(min=1.0)


def _cache_fp(tokenizer, kb_map_path, n_pairs, NATURAL, RAG,
              max_ctx_len, max_seq_len, batch_size, vocab):
    """Veri ondeklenti parmak izi: veri/tokenizer/kb-map degisince yeniden
    encode edilir; ayniysa ondeklent onbellegi (npz) kullanilir.

    RESP_CHARS_MAX DAHILDIR: yanit tavani degisince veri icerigi degisir ama
    cift SAYISI degismez (clean_chars kirpar, elmez). Sayi/sekil/seed ayni
    kaldigi icin bu bilesen olmadan parmak izi CAKISIR ve onbellek eski
    kirpma ile encode edilmis veriyi geri yukler -> tavan degisimi sessizce
    hicbir seye donusurdu.
    """
    h = hashlib.md5()
    # 'fmt:4' -> v4 cache: duz (flat) duz arrays, pickle YOK, PAD kuyrugu
    # budanmis (dinamik uzunluklu batch). v1-v3 object-array/160-PAD npz'leri
    # Kaggle'da 10 dk'lik yukleme takilmalari yapiyordu; v3 tek seferde
    # okunurdu, v4 ayni duz okuyu + daha kucuk tensorde cosar. format
    # degisince eski cache gecersiz -> ilk koşuda 1 kez encode.
    h.update(('fmt:4|%d|%d|%d|%d|%d|%d|%d|%d|r%d' % (n_pairs, NATURAL, int(RAG),
                                                    max_ctx_len, max_seq_len,
                                                    batch_size, SEED,
                                                    CTX_CHARS,
                                                    RESP_CHARS_MAX)).encode('utf-8'))
    if kb_map_path and os.path.exists(kb_map_path):
        with io.open(kb_map_path, 'rb') as f:
            h.update(f.read(2_000_000))
    if tokenizer is not None:
        h.update(('tok:%d:%d' % (len(tokenizer), len(tokenizer.merges))).encode('utf-8'))
    else:
        h.update(('char:%d' % (len(vocab) if vocab else 0)).encode('utf-8'))
    return h.hexdigest()[:16]


# ---- cok cekirdekli BPE-encode (ilk veri hazirligini hizlandirir).
# encode_llm saf/deterministiktir; sira korundugu icin cikti birebir ayni
# kalir, ondeklent parmak izi degismez. Yalnizca Linux (fork) -> Kaggle/Colab.
_mp_dummy = None
_mp_ctx = {}


def _mp_init(dummy_, ctx_map_):
    global _mp_dummy, _mp_ctx
    _mp_dummy = dummy_
    _mp_ctx = ctx_map_


def _mp_enc(item):
    ctx, resp = item
    return encode_llm(_mp_dummy, ctx, resp, context=_mp_ctx.get(ctx))


def _toplam_token(enc_all, pad=None):
    """enc_all -> GERCEK toplam token sayisi (PAD oncesi, kirpma sonrasi).

    NEDEN: encode sureci log'da 'n * TOKEN_PER_PAIR' yaziyordu, yani sabitin
    kendisiyle carpimi. Bu dongusel bir "olcum" idi: sabit yanlissa log da
    yanlisi kanitlardi ve 29.09'da 108,8 olan sabit %18 fazla oldugu halde
    "dogrulanmis" sanildi. Burada gercekten sayiyoruz; boylece TOKEN_PER_PAIR
    her kosuda kendi kendini denetler.

    encode_llm her sekansi max_seq'e PAD'ler; PAD kuyrugu budanmis gercek
    uzunluk sayilir (testle ayni tanim).
    """
    pad = PAD if pad is None else pad
    toplam = 0
    for item in enc_all or ():
        seq = item[0] if isinstance(item, (tuple, list)) else item
        if hasattr(seq, 'tolist'):
            seq = seq.tolist()
        try:
            toplam += seq.index(pad)
        except ValueError:
            toplam += len(seq)
    return toplam


def _pack_encoded(enc_all, B):
    """[(seq, mask)] -> uzunluk-kirpimli, sirali-kumeli batch listesi.

    encode_llm her sekansi max_seq'e PAD'ler; burada PAD kuyrugu budanarak
    GERCEK uzunluk bulunur ve benzer uzunluktaki sekanslar ayni batch'e
    paketlenir. Transformator boylece yalnizca gerekli uzunlukta cosar
    (PAD pozisyonlari zaten maskesiz; kesme matematigi DEGISTIRMEZ).

    Dondurulen her batch (B, T) numpy; T = icindeki en uzun icerik. Toplam
    maske toplami (= egitim kaybinin gorecekleri) birebir korunur.
    """
    items = []
    for seq, mask in enc_all:
        seq = np.asarray(seq)
        mask = np.asarray(mask, dtype=np.float32)
        L = int(np.argmax(seq == PAD)) if (seq == PAD).any() \
            else int(seq.shape[0])
        L = max(1, L)
        items.append((L, seq[:L].copy(), mask[:L].copy()))
    items.sort(key=lambda it: it[0])
    if not items:
        return []
    batches = []
    for s in range(0, len(items), B):
        block = items[s:s + B]
        T = block[-1][0]
        X = np.zeros((len(block), T), np.int64)
        M = np.zeros((len(block), T), np.float32)
        for r, (_, x, m) in enumerate(block):
            X[r, :x.shape[0]] = x
            M[r, :m.shape[0]] = m
        batches.append((X, M))
    return batches


def make_batches(pairs_, B, dummy, ctx_map):
    """(sorgu, yanit) listesini DINAMIK (uzunluk-budanmis) batch'lere cevirir.

    Ilk-encode cok cekirdekli (fork -> Kaggle/Colab) ya da sirali; cikti
    sira korundugu icin ondeklent deterministiktir. Paketleme _pack_encoded
    ile PAD kuyruklari budanarak + benzer uzunluguna gore kumelenerek yapilir.
    Her 25k cift'te ilerleme satiri basilir -> Kaggle'da encode'un hic
    "takilmis" gorunmemesi ve kullanicinin sureyi gormesi saglanir.
    """
    items = sorted(pairs_, key=lambda pr: len(pr[0]))
    n = len(items)
    est_sn = n / ENC_CIFT_SN   # olculen hiz; SN cinsinden
    print(f'BPE-encode basliyor: {n} cift (~{est_sn:.0f} sn = '
          f'{est_sn / 60:.1f} dk, tek sefer; sonra onbellek kullanilir)',
          flush=True)
    t0e = time.time()
    enc_all = None
    if (sys.platform.startswith('linux') and n >= 20000
            and os.environ.get('LLM_MP_OFF') is None):
        try:
            import multiprocessing as mp
            nw = max(2, min(4, os.cpu_count() or 2))
            with mp.get_context('fork').Pool(
                    nw, initializer=_mp_init, initargs=(dummy, ctx_map)) as p:
                enc_all = [None] * n
                done = 0
                for i, r in enumerate(p.imap(_mp_enc, items, chunksize=4096)):
                    enc_all[i] = r
                    done += 1
                    if done % 25000 == 0 or done == n:
                        print(f'  encode: {done}/{n} (%.0fs)' % (time.time() - t0e),
                              flush=True)
            canli = _toplam_token(enc_all)
            print('BPE-encode %d cift (%.1fM token, CANLI OLcum) %d cekirdekle %.0fs'
                  % (n, canli / 1e6, nw, time.time() - t0e), flush=True)
            if canli:
                # Sabit burada kendini dogrular: log artik n*TOKEN_PER_PAIR
                # yazmiyor, GERCEK sayiyi yaziyor. Ikisi ayrilirsa encode
                # yolu degismistir ve TOKEN_PER_PAIR bayatlamistir.
                tpp = canli / float(n)
                print('  -> canli token/cift = %.1f | TOKEN_PER_PAIR = %.1f '
                      '| fark %+.1f%%'
                      % (tpp, TOKEN_PER_PAIR,
                         100.0 * (tpp - TOKEN_PER_PAIR) / TOKEN_PER_PAIR),
                      flush=True)
        except Exception as e:
            enc_all = None
            print('paralel encode atlandi (sirali):', str(e)[:120], flush=True)
    if enc_all is None:
        enc_all = []
        for j, (ctx, resp) in enumerate(items):
            enc_all.append(encode_llm(dummy, ctx, resp,
                                      context=ctx_map.get(ctx)))
            if (j + 1) % 25000 == 0 or (j + 1) == n:
                print(f'  encode: {j + 1}/{n} (%.0fs)' % (time.time() - t0e),
                      flush=True)
    return _pack_encoded(enc_all, B)


def masked_acc(logits, tgt, mask):
    nxt = torch.full_like(tgt, PAD)
    nxt[:, :-1] = tgt[:, 1:]
    ok = ((logits.argmax(-1) == nxt) & mask.bool())
    return ok.sum().item() / mask.sum().clamp(min=1.0).item()


def early_stop_step(vl, best_val, bad, val_every, patience, val_imp=VAL_IMP):
    """Erken durdurma sayacini ilerletir. SAF fonksiyon (test edilebilir).

    Doner: (bad, improved, stop)

    Semantik: --patience ve --val-every da EPOCH cinsindendir.
    Kotu bir val OLCUMU sayaci val_every kadar ARTIRIR. Boylece
    --val-every yalnizca val MALIYETINI etkiler; durdurma esigini degistirmez.

    Neden boyle (olculmus hata): once sayac kotu val OLCUMU sayiyordu.
    --val-every 2 ile birlikte patience=6 -> 12 epoch'lik tolerans
    demekti; 10 epoch'lik olculmus kosuda val 4. epoch'tan itibaren
    yukselmeye baslamis olmasina ragmen erken durdurma HIC tetiklenemedi
    (3 kotu olcum birikti, 6 gerekiyordu). --val-every saf bir maliyet
    dumesi oldugu icin bu birim degisimi dogru olan.

    val_imp: iyilesme esigi. Eski rekorun bu kadar ALTINA inemiyorsa
    'iyilestme yok' sayilir (gurultuye karsi koruma).
    """
    if vl < best_val - val_imp:
        return 0, True, False
    bad += val_every
    return bad, False, bad >= patience


def effective_stop_epoch(patience, val_every, epochs):
    """Verilen ayarlarla en kocaman kaçıncı epoch'ta durulabilir (ust sinir).

    patience EPOCH, val_every EPOCH. Kotu olcum basi val_every kadar
    biriktiginden: gereken kotu OLCUM = ceil(patience / val_every), yani
    durdurma en geç `ceil(patience/val_every) * val_every` epoch sonra
    (iyilesme hic olmazsa) tetiklenir. epochs'tan kucukse TETIKLENEMEZ.
    """
    if val_every < 1:
        val_every = 1
    if patience < 0:
        patience = 0
    n_meas = -(-patience // val_every)          # ceil
    return min(epochs, n_meas * val_every), n_meas * val_every


def min_stoppable_epoch(patience, val_every, fresh=True):
    """Durdurmanin TETIKLENEBILDIGI en kencar epoch (alt sinir) -- taze kosu.

    Tepe noktasinin olabilecegi en kencar yeri varsayar: ilk val olcumu her
    zaman yeni kayittir, dolayisiyla kotu sayac oradan baslar ve TETIKLENEMEZ
    bir butce bulmanin tek yolu budur. Resume'da `bad` sayaci devredildigi
    icin bu hesap gecerli degildir -- orada bütce kontrolu ayri sorudur.

    Gercek olcum izgarasi (egitim dongusuyle birebir):
        do_val = (ep % val_every == 0 or start_ep == 0 and ep == 1)
    Taze kosuda (start_ep == 0) 1. epoch'ta bir olcum olur ve o olcum her
    zaman KAYIT acar (best_val = 1e9). Yani ilk KOTU olcum:
      * val_every == 1 -> 2. epoch'ta (1. epoch hem baslangic hem katsiyi
        birden karsiladigi icin kayit acmak zorundadir)
      * val_every >= 2 -> val_every. epoch'ta
    Kotu olcum basi val_every epoch biriktirdiginden, tetiklenmek icin
    k = max(1, ceil(patience/val_every)) kotu olcum gerekir ve durus
    epoch'u = ilk_kotu_olcum + (k - 1) * val_every olur.

    Iki yan etki:
      * patience <= 0 -> k = 1, yani ilk kotu olcumde durulur. effective_stop_epoch
        ise 0 dondurur (mutlak epoch olarak yanlis 0 verirdi).
      * val_every == 1 -> durus = 1 + max(1, patience). effective_stop_epoch
        toleransi (patience) dondurur ve MUTLAK epoch'u 1 eksik bildirir.
    """
    if val_every < 1:
        val_every = 1
    if patience < 0:
        patience = 0
    if not fresh:
        raise ValueError('sadece taze kosu (fresh=True) icin gecerlidir')
    k = max(1, -(-patience // val_every))      # ceil, en az 1 olcum
    first_bad = 2 if val_every == 1 else val_every
    return first_bad + (k - 1) * val_every


def refine_resp(r, maxc=MAX_SEQ_LEN - MAX_CTX_LEN - 4):
    r = (r or '').strip()
    if len(r) < 10:
        return None
    if len(r) > maxc:
        cut = 0
        for i in range(15, min(len(r), maxc + 8)):
            if r[i] in '.?:!':
                cut = i
        r = r[:cut + 1].strip() if cut > 0 else r[:maxc].strip()
    if not r:
        return None
    return r


def group_split(pairs, val_frac=0.1, seed=SEED):
    """Ayni SORGUYA (ctx) ait TUM ciftleri tek tarafa koyan split.

    Neden gerekli: veri uretimi grup halinde cogalttigi icin ayni ctx yuzlerce
    kez geciyor -- seqgen.load_pairs her pattern'i tum yanitlarla eslestirir
    (pats x resps), naturalize_pairs her cifti k dogal varyanta bolerken
    ctx'yi SABIT tutar. Cift (pair) seviyesinde bolunurse val'in TAMAMI
    train'de de bulunur (olculdu: 33.088/33.088 = %100 sizinti) -> val kaybi
    'ezberlemeyi gorme' yetenegini kaybeder, erken durdurma yan sinyal okur.

    Bu bolme ctx uzerinden grup yapar: bir grubun butunu ya val'de ya
    train'de. Deterministik (seed). Greedy: karistirilmis gruplar sirayla
    val'e eklenir; hedefi asacak grup train'de kalir (kucuk sapma olur ama
    oran ~= val_frac).

    Doner: (tr_pairs, va_pairs, va_ctx). va_ctx, degerlendirme betiginin
    train'e sizan val ciftlerini elemek icin kullanabilecegi grup anahtarlari.
    """
    groups = {}
    for idx, (ctx, _resp) in enumerate(pairs):
        groups.setdefault(ctx, []).append(idx)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target = max(1, int(val_frac * len(pairs)))
    val_idx, val_keys, n_val = [], [], 0
    for g in order:
        if n_val >= target:
            break
        val_idx.extend(groups[g])
        val_keys.append(g)
        n_val += len(groups[g])
    val_set = set(val_idx)
    tr_pairs = [p for i, p in enumerate(pairs) if i not in val_set]
    va_pairs = [p for i, p in enumerate(pairs) if i in val_set]
    return tr_pairs, va_pairs, set(val_keys)


FUNCTIONAL_OPENERS = frozenset("""
    merhaba selam selamlar selamet hey merhabalar gunaydin gunaydinlar
    iyi iyiyim iyiym hosgeldin hosgeldiniz hosbulduk
    evet tabii tabi tabii ki elbette peki aynen kesinlikle dogru
    harika super guzel cok cok guzel muhtesem bayagi baya
    bir bu su o ben benim biz sana size lutfen rica ederim rica
    paylasayim vereyim anlatayim soyleyeyim yazayim bakayim dusunelim
    vaktim seve seve memnuniyetle elbette ki
    no tamam olur olur tabi evet tabii
""".split())


def stabilize_first_words(pairs, seed=SEED):
    """IŞLEVSEL açılışların İLK KELİMESİNİ moda sabitler (içerik korunur).

    Ocak (pattern) başına yanıtlar farklı IŞLEVSEL açılış kelimeleriyle
    başladığında (merhaba/selam/selamlar, evet/tabii/elbette, harika/süper...)
    model ilk yanıt-tokenini (o=0) ayırt edemez -> ilk-token doğruluğu
    düşer, bu token sampling'i zehirler. Burada YALNIZCA işlevsel açılış
    kelimeleri o pattern için mod işlevsel kelimeye hizalanır; İÇERİK
    taşıyan ilk kelimeler (tarif adı, isim, sayı...) olduğu gibi bırakılır
    (anlam bozulmaz). naturalize ilk kelimeyi zaten korur -> uyumludur.

    intents.json DEĞİŞTİRİLMEZ; dönüşüm yalnızca eğitim çiftlerinde olur.
    """
    by_ctx = {}
    for i, (ctx, _resp) in enumerate(pairs):
        by_ctx.setdefault(ctx, []).append(i)
    changed = 0
    for _ctx, idxs in by_ctx.items():
        func_counts = {}
        for i in idxs:
            w = pairs[i][1].strip().split()
            if not w:
                continue
            w0 = w[0].strip('.,;:!?…"\'()').lower()
            if w0 in FUNCTIONAL_OPENERS:
                func_counts[w0] = func_counts.get(w0, 0) + 1
        if len(func_counts) < 2:
            continue
        mode = max(func_counts, key=func_counts.get)
        for i in idxs:
            parts = pairs[i][1].strip().split()
            if not parts:
                continue
            w0 = parts[0].strip('.,;:!?…"\'()').lower()
            if w0 not in FUNCTIONAL_OPENERS:
                continue
            cap = parts[0][:1].isupper()
            newfirst = mode if not cap else mode[:1].upper() + mode[1:]
            parts[0] = newfirst
            pairs[i] = (pairs[i][0], ' '.join(parts))
            changed += 1
    print(f'islevsel acilis sabitendi ({changed} yanit)', flush=True)
    return pairs


def build_kb_lut(path, ctx_chars=CTX_CHARS):
    """knowledge_map.jsonl -> {egitim-ctx: bilgi} sozlugu.

    Dosyadaki 'ctx' anahtari bpe.clean_text (Turkce imlali) uzayindadir;
    EGITIM ctx'si ise seqgen.clean_chars (ASCII + kesik). Dogrudan esleme
    Turkce harf iceren/kirpilan desenlerde kaciyordu (RAG isabeti %65). Lut
    anahtarlari da ayni clean_chars(ctx, CTX_CHARS) ile uretilir -> train/
    eval/kb anahtar uzayi BIREBIR eslesir. Dosyanin kendisi degismez (brain
    ham sorguyla dosya anahtarina vurmaya devam eder; ascii fallback orada).
    """
    lut = {}
    if not os.path.exists(path):
        return lut
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('ctx') and row.get('text'):
                key = clean_chars(row['ctx'], ctx_chars)
                if key:
                    lut[key] = row['text']
    return lut


def rag_context_stats(pairs_, ctx_map):
    """RAG kapsamini CIFT uzerinden olcer -> (baglamli_cift, tek_ctx, toplam).

    29.09 DUZELTMESI. Sayici once benzersiz ctx sayiyordu, payda ise TOPLAM
    cift idi; bu ikisi ayni sey olmadigi icin oran gercekten ~14 kat kucuk
    cikiyordu. Dogal varyantlar (--natural K) ctx'i degistirmez, ayrica
    load_pairs ayni desen icin birden cok cift uretiyor -> bir ctx ortalama
    5-14 kez gecir. 29.09 logu "27.732/1.024.172 = %2,7" dedi ve bundan
    "RAG neredeyse hic kullanilmiyor, kitap/bilgi katkisi yok" sonucu
    cikarildi. Gercek olcum: %52,8 (dogrudan sayimla teyit edildi).

    Doner: (n_cift_baglamli, n_benzersiz_ctx, n_cift_toplam)
    """
    toplam = len(pairs_)
    baglamli = sum(1 for ctx, _ in pairs_ if ctx_map.get(ctx))
    return baglamli, len(ctx_map), toplam


def load_chatgrow_pairs(path, ctx_len=CTX_CHARS, resp_len=None, max_pairs=20000):
    """chatgrow.py/seed ciktisi -> (sorgu, yanit) ciftleri.

    path: tek dosya yolu ya da dosya yollari listesi. Birden cok dosya
    (sohbet + discourse + birlesik) sirayla okunur; ayni (ctx, resp) cifti
    TEK kez eklenir -> birlesik dosya discourse ile ayni satirlari iceriyor
    olsa bile veri tekrarlanmaz (tohum 11'e bagli karistirma sonrasinda bile
    deterministik).

    Kayit bicimi: {"query": ..., "answer": [..]} (answer tek dize de olabilir).
    ctx/yanit, intents hattiyla AYNI normalizasyondan gecer (clean_chars:
    ascii + kucuk harf + kisaltma) -> train/eval/llm_inference uzayi birebir.
    Her sorgu, her yanitla bir cift olur; shisha sabit tohumla karistirilir.

    resp_len=None -> RESP_CHARS_MAX (204) kullanilir, yani intents hattinin
    TAVANI. Daha once 140 idi ve intents hatti 70 idi: model ayni kosuda iki
    farkli kirpma aliskanligi goruyordu (70 ve 140 karakter). Artik ikisi de
    tek butcden gelir; bkz. seqgen.RESP_CHARS_MAX.
    """
    if resp_len is None:
        resp_len = RESP_CHARS_MAX
    paths = [path] if isinstance(path, str) else list(path)
    seen = set()
    pairs = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with io.open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                q = rec.get('query') or rec.get('soru')
                ans = rec.get('answer') or rec.get('answers')
                if isinstance(ans, str):
                    ans = [ans]
                if not q or not ans:
                    continue
                ctx = clean_chars(q, ctx_len)
                if len(ctx) < 6:
                    continue
                for a in ans:
                    r = clean_chars(a, resp_len)
                    if len(r) < 6:
                        continue
                    key = (ctx, r)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append(key)
    rng = random.Random(11)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


MS_PER_PAIR = 1.5139  # ms/cift. KAYNAK: 29.09 kaggle kosusu (OLCULDU).
                      # 12 epoch suresi sn: 1437,4 1434,8 1337,3 1435,0
                      # 1337,2 1433,8 1337,8 1437,5 1339,1 1437,5 1339,0
                      # 1437,8 -> ORT 1395,35 sn/epoch (val olan epoch'lar
                      # ~1435 sn, val atlananlar ~1338 sn; tek epoch secmek
                      # %7 yaniltirdi). Egitim cifti 921.748 ->
                      # 1.395.350 ms / 921.748 = 1,5139 ms/cift.
                      #
                      # ONCEKI DEGER 1,3884 DU 315.883 ciftlik eski kosudan
                      # kaliyordu ve %11 EKSIK sayiyordu. Yanlisi yondu:
                      # butce 12 epoch'u 396 dk'ya sigdirirken gercekte
                      # 396 x 1,5139/1,3884 = 431 dk isterdi -> oturum
                      # KESILIRDI. Veri butcesinin altinda kaldigi icin bu
                      # kosuda ismadi; veri buyunce isteyecekti.
                      #
                      # DANGER: burada ilk yazimda 70.000 cift varsayildi ve
                      # 6,266 ms/cift cikti — 4,5 KAT YANLIStI ve zaman
                      # tavanini gereksiz yere 316.000'a cekti (halbuki
                      # 9 saatlik oturuma ~1,46 MILYON cift sigiyor).
                      # Olcum satiri okunmadan varsayim yapilmayacak:
                      # test_olculen_sabitler_birlikte_tutarli bunu kapatir.
ENCODE_DK = 26.0      # ilk kez BPE encode, bir kez. 29.09 OLCUMU:
                      # 1.024.172 cift -> 1385 + 156 = 1541 sn = 25,7 dk.
                      # Onceki deger 9,0 DU ve 315.883 cift icindi; veri
                      # 3,2 KAT buyuyunce 26 dk gercekken 9 dk varsayildi.
                      # Yanlis YONDU: encode suresi egitim butcesinden
                      # DUSULUYOR, demek ki kucuk saymak fazla cift
                      # vaat etmek demek (17 dk = butcenin %4,3'u).
                      #
                      # BILINEN KISIT: bu SABIT ama encode suresi n ile
                      # ORANTILI. sure_ve_hesapla cift sayisini URETIR,
                      # encode suresi ise o sayiya BAKAR -> tam dongusel
                      # degil, kaba bir ust sinir. Veri buyunce bu deger
                      # tekrar olculmelidir.
VARSAYILAN_BOSLUK = 0.75   # oturumun %75'i veriye, %25'i bosluk/erteleme


def sure_ve_hesapla(oturum_dk=540, epochs=12, encode_dk=ENCODE_DK,
                    bosluk=VARSAYILAN_BOSLUK, ms_per_pair=MS_PER_PAIR):
    """SURE TAVANI: kac cift bu oturumda sigar? (cift sayisi, dk cinsinden)

    NEDEN BURADA: oturum suresi ve EPOCHS sadece kaggle_start.sh'da
    biliniyor ama formul bash'ta yazilirsa test EDILEMEZ. Tek dogruluk
    kaynagi burada; kaggle_start.sh bunu cagirir.

    DANGER: saat->dk cevirisi iki kez yapilirsa sonuc 1000 kere kucuk
    olur (ilk yazimda boyle bir hata olustu: 316.000 yerine 9.575 cift,
    yani egitim 120.000 -> 9.575 cifte dusertilirdi). Bu yuzden asagida
    dakika cinsinden islem yapilir ve test taban degerleri dogrular.
    """
    kalan_dk = oturum_dk * bosluk - encode_dk
    if kalan_dk <= 0 or epochs <= 0:
        return 0
    dk_per_epoch = kalan_dk / epochs
    # dk -> sn -> ms, sonra ms_per_pair'a bol. Tek ceviri; iki kez
    # yapildiginda sonuc 1000 kere kuculur.
    return int(dk_per_epoch * 60 * 1000 / ms_per_pair)


def coz_max_pairs(intents_path=None, ust_tavan=0, yaz=True):
    """Veri butcesini intents SAYISINDAN hesapla (0 = otomatik).

    Neden sabit degil: MAX_PAIRS'in isi 'butun ciftleri kullanmak' degil,
    'benzersiz ctx'nin cogunu kapsamak'. Olcum (28.09) su ivme:
        intents 6.364 -> butce 120.000 -> benzersiz ctx kapsamasi %97,7
    Ayni yapi intents 20.000'e cikarsa benzersiz ctx ~122.000 olur ve sabit
    120.000 yeniden darbo gaz olur. Bu yuzden butce intents ile buyur.

    ust_tavan: kaggle_start.sh'den gelen SURE tavani (--max-pairs-cap).
    Veri ihtiyaci buyuk, zaman yoksa TAVAN KAZANIR ve bu yazdirilir —
    sessizce kismamak yerine gorunur kilinir.
    """
    import json as _json
    yol = intents_path or INTENTS
    n = 0
    if os.path.exists(yol):
        with io.open(yol, 'r', encoding='utf-8') as f:
            n = len(_json.load(f).get('intents', []))
    if MAX_PAIRS > 0:            # elle tavan varsa ona saygi
        n_butce, kaynak = MAX_PAIRS, 'elle (MAX_PAIRS)'
    else:
        n_butce = int(round(MAX_PAIRS_INTENT_CARPAN * n))
        kaynak = 'otomatik (%s x %d intent)' % (MAX_PAIRS_INTENT_CARPAN, n)
    tavan = ust_tavan or 0
    if tavan and n_butce > tavan:
        if yaz:
            print('[butce] veri butcesi %s > zaman tavani %s: TAVAN '
                  'KAZANDI, %s cift kullanilacak (ihtiyac: %s)'
                  % (format(n_butce, ','), format(tavan, ','),
                     format(tavan, ','), format(n_butce, ',')), flush=True)
        n_butce = tavan
    if yaz:
        print('[butce] MAX_PAIRS=%s (%s, zaman tavani %s)'
              % (format(n_butce, ','), kaynak,
                 format(tavan, ',') if tavan else 'yok'), flush=True)
    return n_butce


def prepare_data(RAG, NATURAL=0, tokenizer=None, kb_map_path=None,
                 FIRST_WORD_STABILIZE=True, max_ctx_len=MAX_CTX_LEN,
                 max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE,
                 chatgrow_path=None, limit_pairs=0, max_pairs_cap=0):
    """Veri + RAG hattini HAZIRLAR (yalnizca numpy; torch gerektirmez).
    --dry-run bu fonksiyonu calistirip dogrular; egitim de ayni yolu kullanir.

    NATURAL > 0 ise her (sorgu, yanit) cifti, yanitin dogal varyantlariyla
    cogaltilir (naturalize_pairs): model ayni icerigi pek cok dogal sekilde
    ifade etmeyi ogrenip kopya-yerine-canli-sohbet icin veri kazanir.

    limit_pairs > 0 (dry-run/verify) ise naturalize SONRASI ciftler bu
    sayiya budanir -> 10 dk'lik TAM BPE-encode YAPILMAZ ve onbellek
    yazilmaz (dogrulama ~1-2 dk surer; gercek egitim encode'u bench/train'de
    bir kez yapilir, oradaki cache sonraki calistirmalarda kullanilir).

    kb_map_path verilirse (enrich_intents.py uretimi knowledge_map.jsonl)
    desen->bilgi parcasini CANLI corpus.search yerine ezberlenmis haritadan
    alir: deterministik (RAG hit oranini %100'e surdugumuz bilgi intent'leri
    icin guvenli) ve corpus shisha yayilirken hizlidir. Harita anahtarlari
    build_kb_lut ile EGITIM ctx'siyle AYNI uzayda (clean_chars, CTX_CHARS)
    normalize edildigi icin isabet %65 -> %76 olur; bilgi metni ise
    KB_TEXT_CHARS (300) karakterle sinirli.

    tokenizer (BPETokenizer) verilirse BPE modu kullanilir (subword vocab);
    yoksa eski karakter sozlugu (build_llm_vocab) kullanilir."""
    assert os.path.exists(INTENTS), f'intents.json bulunamadi: {INTENTS}'
    butce = coz_max_pairs(INTENTS, ust_tavan=max_pairs_cap)
    pairs = load_pairs(INTENTS, max_pairs=butce, use_query=True,
                       ctx_len=CTX_CHARS)
    # Butcenin isini GERCEKTEN yaptigini olc: benzersiz ctx kapsamasi.
    # Karpanlar (MAX_PAIRS_INTENT_CARPAN) olcumle sabitlendi; intents
    # yapisi degisirse bu sayi saptar ve saptamasi GORUNUR olur.
    _ham = load_pairs(INTENTS, max_pairs=10 ** 9, use_query=True,
                      ctx_len=CTX_CHARS)
    _tum_ctx = set(c for c, _ in _ham)
    _alinan_ctx = set(c for c, _ in pairs)
    # NOT: .format() kullanildi, %-bicimleme DEGIL: yuzde isareti icin
    # '%%' kacisi gerekirdi ve bu dosyada defalarca '%' ile biten
    # bozuk mesajlar uretti.
    print('[butce] ham {0:,} cift -> butce {1:,} ({2:.1f}%); benzersiz ctx '
          '{3:,}/{4:,} = {5:.1f}% kapsama, benzersiz cevap {6:,}'.format(
              len(_ham), len(pairs),
              100.0 * len(pairs) / max(1, len(_ham)),
              len(_alinan_ctx), len(_tum_ctx),
              100.0 * len(_alinan_ctx) / max(1, len(_tum_ctx)),
              len(set(r for _c, r in pairs))), flush=True)
    del _ham, _tum_ctx, _alinan_ctx
    if chatgrow_path:
        cg = load_chatgrow_pairs(chatgrow_path)
        src = chatgrow_path if isinstance(chatgrow_path, str) \
            else ', '.join(chatgrow_path)
        if cg:
            print('chatgrow sohbet cifti: %d (kaynak: %s)' %
                  (len(cg), src), flush=True)
            pairs = pairs + cg
        else:
            print('chatgrow kaynak bos ya da yok: %s' % src, flush=True)
    pairs = [(ctx, rr) for ctx, r in pairs if (rr := refine_resp(r)) is not None]
    if FIRST_WORD_STABILIZE:
        n_before = len(pairs)
        pairs = stabilize_first_words(pairs)
        print(f'islevsel acilis sabitlendi: {n_before} cift'
              f' (yalnizca islevsel acilislar -> mod; icerik korunur),'
              f' o=0 entropisi dusuruldu', flush=True)
    if NATURAL > 0:
        n_before = len(pairs)
        pairs = naturalize_pairs(pairs, k=NATURAL)
        print(f'dogal varyant: {n_before} -> {len(pairs)} cift'
              f' (varyant: {NATURAL})', flush=True)
    if limit_pairs > 0 and len(pairs) > limit_pairs:
        # dry-run/verify: TAM encode YOK. Toplamdan DETERMINISTIK (SEED)
        # ornekle -> pipeline dogrulamasinin suresi ~1-2 dk kalir.
        rng = np.random.RandomState(SEED)
        pick = rng.choice(len(pairs), limit_pairs, replace=False)
        pick.sort()
        pairs = [pairs[i] for i in pick]
        print('verify ornekleme: %d cift (limit-pairs %d, tam encode degil)'
              % (len(pairs), limit_pairs), flush=True)
    print('egitim cifti (sorgu, yanit):', len(pairs), flush=True)

    vocab = None
    if tokenizer is None:
        all_text = []
        for ctx, resp in pairs:
            all_text.append(ctx)
            all_text.append(resp)
        vocab = build_llm_vocab(all_text)
        print('karakter sozlugu:', len(vocab), flush=True)

    # Dummy modeli yalnizca encode icin (c2i/tokenizer + uzunluk bilgisi)
    dummy = LLM(vocab, d_model=4, num_blocks=1, num_heads=1,
                max_ctx_len=max_ctx_len, max_seq_len=max_seq_len,
                seed=SEED, tokenizer=tokenizer)
    if tokenizer is not None:
        print('BPE tokenizer: vocab =', len(tokenizer), flush=True)

    tr_pairs, va_pairs, va_ctx = group_split(pairs, val_frac=0.1, seed=SEED)
    print('split (ctx-grup bazli): train %d | val %d | val grubu %d/%d'
          % (len(tr_pairs), len(va_pairs), len(va_ctx),
             len({c for c, _ in pairs})), flush=True)

    # ---- RAG: her (sorgu, yanit) ciftine ilgili bilgi parcasi
    ctx_map = {}
    corpus = None
    kb_pre = {}
    use_corpus = False
    if kb_map_path and os.path.exists(kb_map_path):
        kb_pre = build_kb_lut(kb_map_path)
        print('kb-map yuklendi (normalize-ctx: %d desen)' %
              len(kb_pre), flush=True)
    if RAG:
        if kb_pre:
            # Harita deterministiktir: corpus'a hic dokunmadan desen->bilgi
            # dogrudan alinir -> Colab/Kaggle ilk calistirmada dev corpus
            # embeddingi kurmaz (yoksa ~10 dk CPU beklentisi). Elesma
            # olmayan desen bilgi-parcasiz kalir (cani zaklamaz).
            print('RAG kaynagi: kb-map (corpus yuklenmez - hizli)', flush=True)
        else:
            use_corpus = True
            try:
                from corpus import Corpus
                corpus = Corpus()
                corpus.load()
                print('RAG corpus yuklendi (parca:', len(corpus.chunks), ')', flush=True)
            except Exception as e:
                corpus = None
                print('RAG corpus yuklenemedi, bilgi-parcasiz egitim:', e, flush=True)

    def kb_for(ctx):
        if ctx in kb_pre:
            return kb_pre[ctx]
        if corpus is None and not use_corpus:
            return None
        try:
            chunk = corpus.search(ctx)
            if not chunk:
                return None
            text = ((chunk.get('title') or '') + '. ' +
                    (chunk.get('text') or ''))[:KB_TEXT_CHARS]
            return text if len(text.strip()) >= 20 else None
        except Exception:
            return None

    if RAG:
        # ONCE: her benzersiz ctx icin baglam cozulur.
        for ctx, _ in tr_pairs + va_pairs:
            if ctx in ctx_map:
                continue
            ctx_map[ctx] = kb_for(ctx)
        # SONRA: oran CIFT uzerinden hesaplanir.
        #
        # OLCULEN HATA (29.09): sayici benzersiz ctx sayiyordu, payda ise
        # TOPLAM cift idi. Dogal varyantlar (--natural 5) ctx'i degistirmez,
        # ayrica load_pairs ayni desen icin birden cok cift uretiyor -> bir
        # ctx ortalama 13,8 kez gecir. "hits" bu yuzden cift sayisinin
        # ~1/14'u idi ve oran gercek yerine 13,8 KAT KUCUK cikiyordu.
        # 29.09 logu: 27.732/1.024.172 = %2,7 -> "RAG neredeyse hic
        # kullanilmiyor" yanlis cikarimi. Gercek olcum: %52,8.
        # Yanlisin bedeli: kitap/bilgi katkisi %2,7 sanildi ve "veri yok"
        # denildi; oysa ciftlerin YARISI bilgi baglami tasiyor.
        # Duzeltme yalnizca YAZDIRIR; ctx_map icerigi ve egitim girdisi
        # bit bit aynidir (onbellek parmak izi de degismez).
        _baglamli, _tek_ctx, _toplam = rag_context_stats(tr_pairs + va_pairs,
                                                       ctx_map)
        print('RAG baglamli cift : {}/{} = {:.1f}%  ({} benzersiz ctx, '
              'ortalama {:.1f} cift/ctx)'.format(
                  _baglamli, _toplam,
                  100.0 * _baglamli / max(1, _toplam),
                  _tek_ctx, _toplam / max(1, _tek_ctx)), flush=True)

    # ---- veri ondeklenti: ayni veri+tokenizerla tekrar cagrildiginda
    # (Colab resume / dry-run sonrasi egitim) 10dk'lik BPE-encode ATLANIR.
    fp = _cache_fp(tokenizer, kb_map_path, len(pairs), NATURAL, RAG,
                   max_ctx_len, max_seq_len, batch_size, vocab)
    CACHE = os.path.join(SAVE_DIR, 'llm_data_%s.npz' % fp)
    if limit_pairs > 0:
        # verify/dry-run: TAM encode YOK. Cache YUKLENMEZ ve YAZILMAZ
        # (gercek egitim onbellegi farkli fp ile bench/train'de olusur).
        tr = make_batches(tr_pairs, batch_size, dummy, ctx_map)
        va = make_batches(va_pairs, batch_size, dummy, ctx_map)
        print('train batch:', len(tr), '| val batch:', len(va), flush=True)
        return {'vocab': vocab, 'tokenizer': tokenizer,
                'tr': tr, 'va': va, 'ctx_map': ctx_map, 'val_ctx': va_ctx}
    if os.path.exists(CACHE):
        try:
            print('ondeklent yukleniyor: %s (%.0f MB) ...' % (
                os.path.basename(CACHE), os.path.getsize(CACHE) / 1e6), flush=True)
            # v4: DUZ diziler (pickle/object-array YOK) -> tek seferde okunur.
            with np.load(CACHE) as z:
                Xg, Mg = z['Xtr'], z['Mtr']
                Xvg, Mv = z['Xva'], z['Mva']
                sht, shv = z['sh_tr'], z['sh_va']
            ntr, nva = len(sht), len(shv)
            tr0 = [(Xg[i, :sht[i, 0], :sht[i, 1]],
                    Mg[i, :sht[i, 0], :sht[i, 1]]) for i in range(ntr)]
            va0 = [(Xvg[i, :shv[i, 0], :shv[i, 1]],
                    Mv[i, :shv[i, 0], :shv[i, 1]]) for i in range(nva)]
            print('veri ondeklenti kullanildi:', os.path.basename(CACHE),
                  '(%d+%d batch)' % (ntr, nva), flush=True)
            return {'vocab': vocab, 'tokenizer': tokenizer,
                    'tr': tr0, 'va': va0, 'ctx_map': ctx_map, 'val_ctx': va_ctx}
        except Exception as e:
            print('ondeklent yuklenemedi, yeniden encode:', e, flush=True)

    # Dinamik batch'ler (PAD kuyrugu budanir, benzer uzunluklar kumelenir)
    # modul-duzey make_batches ile kurulur (birim test edilebilir).
    tr = make_batches(tr_pairs, batch_size, dummy, ctx_map)
    va = make_batches(va_pairs, batch_size, dummy, ctx_map)
    print('train batch:', len(tr), '| val batch:', len(va), flush=True)
    # 29.09: once 30 krk'a SERT kesip gosteriyordu; ekranda yarim cumle
    # gorununce veri bozuk sanildi. Kirpma isareti (… ) + GERCEK uzunluk
    # yazilir, boylece ekrandaki kisaltma veriden ayirt edilir.
    def _goster(s, n=30):
        s = clean_chars(s, None)
        return (s[:n] + u'…') if len(s) > n else s
    _c0, _r0 = tr_pairs[0]
    print('ornek cift (%d/%d krk): (%r, %r)' % (len(_c0), len(_r0),
                                                _goster(_c0), _goster(_r0)),
          flush=True)
    try:
        # v4: pickle YOK. Tum batch'ler (Bmax, T0)_cadde TEK duz diziye pad'lenir;
        # gercek (B,T) olculeri sh_tr/sh_va ile saklanir. Dinamik budama sonrasi
        # T0 artisma gore cosan max gercek uzunluk duzeyi; pad-kuyruk maskesi 0
        # -> loss'a KATILMAZ, egitim matematigi birebir ayni.
        Bmax = batch_size
        T0 = max(x.shape[1] for x, _ in tr)
        T0v = max(x.shape[1] for x, _ in va)
        sh_tr = np.array([[x.shape[0], x.shape[1]] for x, _ in tr], dtype=np.int32)
        sh_va = np.array([[x.shape[0], x.shape[1]] for x, _ in va], dtype=np.int32)
        Xtr = np.zeros((len(tr), Bmax, T0), dtype=np.int32)
        Mtr = np.zeros_like(Xtr, dtype=np.float32)
        for i, (x_, m_) in enumerate(tr):
            B, T = x_.shape
            Xtr[i, :B, :T] = x_.astype(np.int32)
            Mtr[i, :B, :T] = m_
        Xva = np.zeros((len(va), Bmax, T0v), dtype=np.int32)
        Mva = np.zeros_like(Xva, dtype=np.float32)
        for i, (x_, m_) in enumerate(va):
            B, T = x_.shape
            Xva[i, :B, :T] = x_.astype(np.int32)
            Mva[i, :B, :T] = m_
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.savez_compressed(CACHE, Xtr=Xtr, Mtr=Mtr, Xva=Xva, Mva=Mva,
                            sh_tr=sh_tr, sh_va=sh_va)
        print('ondeklent yazildi (v4):', os.path.basename(CACHE),
              '| T0=%d T0v=%d | toplam %.0f MB' % (
                  T0, T0v, os.path.getsize(CACHE) / 1e6), flush=True)
    except Exception as e:
        print('ondeklent yazilamadi (devam):', e, flush=True)
    return {'vocab': vocab, 'tokenizer': tokenizer,
            'tr': tr, 'va': va, 'ctx_map': ctx_map, 'val_ctx': va_ctx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--rag', action='store_true',
                    help='corpus.jsonl riddaren bilgi-parcalariyla koullu egitim')
    ap.add_argument('--kb-map', default=None, metavar='PATH',
                    help='enrich_intents.py uretimi knowledge_map.jsonl; '
                         'RAG desen->parca eslemesini corpus.search yerine '
                         'bu haritadan alir (deterministik, --rag ile birlikte)')
    ap.add_argument('--max-pairs-cap', type=int, default=0, metavar='N',
                    help='veri butcesinin SURE tavani. 0 = tavan yok. '
                         'kaggle_start.sh oturum suresi ve EPOCHS bilindigi '
                         'icin bunu hesaplayip gonderir; veri ihtiyaci '
                         'tavani asarsa TAVAN KAZANIR ve yazdirilir '
                         '(MAX_PAIRS veri butcesidir, sure ayari degil)')
    ap.add_argument('--natural', type=int, default=0, metavar='K',
                    help='her cevabin K dogal varyantiyla veriyi buyut (orijinal dahil)')
    ap.add_argument('--chatgrow', default=None, nargs='+', metavar='PATH',
                    help='chatgrow.py ciktisi chatgrow_sohbet.jsonl; gercek '
                         'sohbet ciftlerini (sorgu->yanit) egitim verisine '
                         'ekler. BIRDEN COK dosya: --chatgrow a.jsonl b.jsonl '
                         '(tekrar eden ciftler otomatik elenir)')
    ap.add_argument('--limit-pairs', type=int, default=0, metavar='N',
                    help='>0 ise naturalize SONRASI ciftler N de budanir; '
                         'dry-run/verify icin: TAM encode atlanir (~1-2 dk), '
                         'onbellek yazilmaz. Eg<itimde kullanilmaz.')
    ap.add_argument('--dry-run', action='store_true',
                    help='torch olmadan veri/RAG hattini dogrula ve cik')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='erken durdurmada calinmasi gereken iyilesmesiz EPOCH sayisi. '
                         'EPOCH cinsindendir: --val-every 2 ile 2 deyince TEK kotu val '
                         'OLCUMU (= 2 epoch) tolere edilir, yani val TEK SEFER yukselince '
                         'durus olur. --val-every 2 ile 6 deyince 3 kotu olcum (= 6 epoch) '
                         'tolere edilir. --val-every yalnizca maliyeti etkiler, bu esigi '
                         'DEGISTIRMEZ.')
    ap.add_argument('--d-model', type=int, default=D_MODEL, help='gizli boyut')
    ap.add_argument('--num-blocks', type=int, default=NUM_BLOCKS, help='transformer blok sayisi')
    ap.add_argument('--num-heads', type=int, default=NUM_HEADS, help='dikkat kafa sayisi')
    ap.add_argument('--ff-mult', type=int, default=FF_MULT, help='FFN genisletme carpani')
    ap.add_argument('--max-ctx-len', type=int, default=MAX_CTX_LEN,
                    help='sorgu icin token butcesi (secenek: 48)')
    ap.add_argument('--max-seq-len', type=int, default=MAX_SEQ_LEN,
                    help='toplam sekans uzunlugu (secenek: 192)')
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--lr-base', type=float, default=LR_BASE)
    ap.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                    help='AdamW ayrik cezasi (0 = kapat). Gomme, bias ve '
                         'LayerNorm her zaman cezasiz kalir.')
    ap.add_argument('--dropout', type=float, default=DROPOUT,
                    help='Gizli katman dropout orani (0 = kapat). Eval\'da '
                         'her zaman kapali, bu yuzden numpy parity degismez.\n'
                         'DIZENLEME DENEYI icin: veri 9.855 farkli sorudan '
                         'uretiliyor (350.301 satir = soru basina ~36 tekrar) '
                         've model 16.9M parametre; 12 epoch\'lik kosuda val '
                         '2. epoch\'tan sonra monoton yukseliyor (0.4728 -> '
                         '0.5177) yani ezberliyor. Daha yuksek dropout '
                         'ezberlemeyi GECIKTIRIR.')
    ap.add_argument('--untie-embeddings', dest='tie_embed', action='store_false',
                    default=TIE_EMBED,
                    help='gomme/cikis bagligini KAPATIR (head ayri saklanir, '
                         '+6.1M parametre). Bagli varsayilandir.')
    ap.add_argument('--val-every', type=int, default=1,
                    help='val gecisini her N epochda bir yap (2 -> val maliyeti '
                         'yarilanir, epoch suresi kisalir). SADECE maliyet etkisi: '
                         'erken durdurma esigini DEGISTIRMEZ.')
    ap.add_argument('--lr-horizon', type=int, default=0, metavar='N',
                    help='cosine ogrenme hizi UFKU (epoch). 0 (varsayilan) = otomatik: '
                         'min(epochs, patience + 20). Otomatik secenek, LRnin erken '
                         'durdurma noktasina kadar gercekten inmesini garanti eder. '
                         'Cok dusuk -> LR erken flattening yapar; cok yuksek '
                         '(orn. --epochs) -> decay yine calismaz.')
    ap.add_argument('--export-dir', default=None,
                    help='llm_model.json + _weights.npz ciktisi (varsayilan: SAVE_DIR)')
    ap.add_argument('--fresh', action='store_true',
                    help='mevcut checkpoint yok sayilir, sifirdan basla')
    args = ap.parse_args()
    EPOCHS = args.epochs
    RAG = args.rag
    NATURAL = args.natural
    patience = args.patience
    val_every = max(1, args.val_every)
    dm, nb, nh, ff = args.d_model, args.num_blocks, args.num_heads, args.ff_mult
    mxc, mxs = args.max_ctx_len, args.max_seq_len
    bs, lr_base = args.batch_size, args.lr_base
    wd, tie_embed = args.weight_decay, args.tie_embed
    drop = args.dropout
    export_dir = args.export_dir or SAVE_DIR
    # Cosine ufku. Gercek bitis noktasi --epochs degil, ~patience civaridir
    # (patience artik EPOCH cinsinden; once val olcumu sayiyordu ve val_every
    # ile katlaniyordu). Ufuk bunun oncesinde kalirsa LR o noktaya kadar hic
    # inmez (cosine faktoru ~1) ve egitim tepede sonlanir. PAD, erken
    # durdurmanin biraz GEC tetiklenmesine izin verir -> LR gercekten inmis olur.
    if args.lr_horizon > 0:
        lr_horizon = args.lr_horizon
    else:
        lr_horizon = min(EPOCHS, patience + LR_HORIZON_PAD)
    if dm % nh != 0:
        raise SystemExit(f'--d-model {dm} --num-heads {nh} ile bolunebilir olmali')
    os.makedirs(export_dir, exist_ok=True)

    print('CONFIG: d_model=%d blocks=%d heads=%d ff_mult=%d '
          'max_ctx=%d max_seq=%d batch=%d val_every=%d lr=%.1e export=%s' % (
              dm, nb, nh, ff, mxc, mxs, bs, val_every, lr_base, export_dir),
          flush=True)
    # patience ve val_every'yi AYRIK goster: ikisi carpilirsa kullanici
    # gercek epoch butcesini yanlis hesaplar. Kotu val OLCUMU sayisi de
    # yazilir (gercekte kac olcumun "iyilesme yok" sayilacagi) ve toplam
    # epoch cinsinden TOLERANS gosterilir -- bu ikisi ayni sey DEGIL:
    # val_every izaraya uymadigi icin toleransi patience'i asabilir
    # (val_every=4, patience=6 -> 2 olcum x 4 = 8 epoch).
    _n_meas = max(1, -(-patience // val_every))
    _toler = _n_meas * val_every
    print('CONFIG: epochs=%d patience=%d EPOCH val_every=%d '
          '-> kotu val olcumu toleransi: %d olcum = %d epoch'
          % (EPOCHS, patience, val_every, _n_meas, _toler), flush=True)
    print('CONFIG: lr_horizon=%d (LR %.1e -> %.1e)' % (
        lr_horizon, lr_base, lr_base * LR_MIN), flush=True)
    print('CONFIG: optimizer=AdamW wd=%.4f | gomme<->cikis bagi=%s' % (
        wd, tie_embed), flush=True)
    # Erken durdurmanin GERCEKTEN tetiklenebilmesi icin epoch butcesi
    # yeterli olmali. Aksi halde kosu zorlamayla biter ve val'in yukselmeye
    # baslamasi fark edilmeden epoch butcesi tukenir -- 10 epoch'lik olculmus
    # kosuda tam olarak bu oldu (val 4. epoch'tan itibaren yukselmisti).
    #
    # Esik, toleranstan DEGIL dogrudan izgaradan turetilir; ikisi
    # val_every=1'de ve patience=0'da ayrilir (bkz. min_stoppable_epoch).
    _min_ep = min_stoppable_epoch(patience, val_every)
    if _min_ep > EPOCHS:
        # Butce toleransi karsilayacak kadar yok -> HIC TETIKLENEMEZ,
        # hangi veri cikarsa ciksin.
        _lvl = 'KRITIK'
        _msg = 'HIC TETIKLENEMEZ'
        _need = _min_ep + 2 * val_every
    elif _min_ep == EPOCHS:
        # yalnizca son epoch'ta, tam toleransla; hicbir epoch tasarruf etmez.
        _lvl = 'UYARI'
        _msg = 'en fazla SON epochta tetiklenir'
        _need = EPOCHS + 2 * val_every
    else:
        _lvl = None
    if _lvl:
        print('%s: epochs=%d, patience=%d (EPOCH), val_every=%d. Kotu val '
              'OLCUMU basi %d epoch biriktirir (toplam %d olcum = %d epoch) '
              '-> tepe noktasindan sonra durulur. Ilk val olcumu her zaman '
              'kayit oldugu icin durdurma en kencar %d. epochta tetiklenebilir: '
              'bu butceyle %s. En az %d epoch ver.'
              % (_lvl, EPOCHS, patience, val_every, val_every,
                 _n_meas, _toler, _min_ep, _msg, _need),
              file=sys.stderr, flush=True)

    if args.dry_run:
        d = prepare_data(RAG, NATURAL=NATURAL, tokenizer=load_tokenizer(),
                         kb_map_path=args.kb_map, max_ctx_len=mxc, max_seq_len=mxs,
                         batch_size=bs, chatgrow_path=args.chatgrow,
                         limit_pairs=args.limit_pairs,
                         max_pairs_cap=args.max_pairs_cap)
        ex = next((c for c in d['ctx_map'].values() if c), None)
        print('DRY-RUN OK: train batch', len(d['tr']), '| val batch',
              len(d['va']), '| tokenizer', d['tokenizer'].vocab_size
              if d['tokenizer'] else len(d['vocab']), flush=True)
        if RAG:
            print('RAG ornek baslam:', (ex or '')[:80].replace('\n', ' '), flush=True)
        return 0

    if not HAVE_TORCH:
        print('Eksik: PyTorch kurulu degil. Egitim Colab/Lightning AI ortaminda '
              'calistirilir; burada yalnizca --dry-run ile veri hattini '
              'dogrulayabilirsin.', flush=True)
        return 2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ---------------- veri  (Bu noktaya kadar CUDA HIC AYAGA KALKMAZ;
    # fork tabanli cok-cekirdekli BPE-encode bu sayede guvenli)
    d = prepare_data(RAG, NATURAL=NATURAL, tokenizer=load_tokenizer(),
                     kb_map_path=args.kb_map, max_ctx_len=mxc, max_seq_len=mxs,
                     batch_size=bs, chatgrow_path=args.chatgrow,
                     max_pairs_cap=args.max_pairs_cap)
    vocab = d['vocab']
    tok = d['tokenizer']
    V = tok.vocab_size if tok is not None else len(vocab)
    tr, va = d['tr'], d['va']

    # CUDA bu NOKTAIYLA baslar: encode fork'lari bitmis, birden fazla GPU'yla
    # DataParallel oncesi her cihaz sicakligi tek tek dogrulanir (kilitlenme
    # yerine aninda hata dondurur).
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count() if DEVICE == 'cuda' else 0
    if DEVICE.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        for g in range(n_gpu):
            t = torch.tensor(1.0, device='cuda:%d' % g)
            _ = (t + 1).item()
        print('CUDA sicak: %d GPU dogrulandi' % n_gpu, flush=True)
    print('PyTorch', torch.__version__, '| device:', DEVICE,
          '| GPU:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '-',
          '(Count: %d)' % n_gpu,
          '| AMP:', 'fp16' if DEVICE == 'cuda' else 'off',
          '| SAVE_DIR:', SAVE_DIR, '| RAG:', RAG, '| patience:', patience, flush=True)

    trX = [torch.from_numpy(b[0]).long().to(DEVICE) for b in tr]
    trM = [torch.from_numpy(b[1]).float().to(DEVICE) for b in tr]
    vaX = [torch.from_numpy(b[0]).long().to(DEVICE) for b in va]
    vaM = [torch.from_numpy(b[1]).float().to(DEVICE) for b in va]

    # ---------------- model + resume
    model = TorchLLM(V, d_model=dm, num_blocks=nb, num_heads=nh, ff_mult=ff,
                     max_seq_len=mxs, drop=drop, tie_embeddings=tie_embed
                     ).to(DEVICE)

    # ---------------- optimizer: AdamW + duzenlestirme (weight decay)
    # Model 23M->16.9M parametre, gercek (benzersiz) ornek ~68.8k; dogal
    # varyantlar ayni 68.8k'yi 5 katina cikariyor -> belirgin asiri uyum
    # riski. Onceki kurulum `torch.optim.Adam(..., lr=lr_base)` idi:
    # weight_decay=0 ve AYRILMIS (decoupled) olmayan ceza. AdamW'ye gecmek
    # iki seyi birden duzeltir: (a) gercek bir ceza, (b) gradyana eklenen
    # L2 yerine parametreye dogrudan uygulanan ayrik (decoupled) ceza --
    # uyarlanabilir optimizerlarda bu iki sey ayni degildir.
    #
    # Ceza YALNIZCA agirilik matrislerine uygulanir; standart pratik:
    #   - bias'lar ve LayerNorm katsayilari (olcek kararlari) cezasiz,
    #     bunlari kurmak ozellikle kotu,
    #   - gomme/cikis (embed) cezasiz: 16000x384 seyrek tablo, her satiri
    #     yalnizca o token goruldugunde guncellenir; ceza -> kullanilmayan
    #     tokenlari surekli sifira cekerek kalici bozar.
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        # Bu modelde vektorler (bias, LayerNorm gamma) (1, d) seklinde
        # tutulur -- yani ndim=2'dir, bu yuzden ndim<2 tek basina YETMEZ
        # (b0_bq gibi dikdortgen bias'lar cezaya girerdi). Vektor olma
        # olcumu shape[0]==1'dir; gomme ayrica istisna (V,d) seyrek tablo.
        is_vector = prm.ndim < 2 or prm.shape[0] == 1
        if is_vector or name == 'embed':
            no_decay.append(prm)
        else:
            decay.append(prm)
    # betas/eps PyTorch VARAYILANINDA birakildi (0.9/0.999, 1e-8): eski
    # `Adam` ayarlariyla ayni. Boylece bu kosudaki TEK fark duzenlestirmedir;
    # beta/eps'i de degistirmek isteyen ayri bir deney olurdu.
    opt = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': wd},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr_base)
    n_dec = sum(p.numel() for p in decay)
    n_nod = sum(p.numel() for p in no_decay)
    print('optimizer: AdamW wd=%.4f | cezali %d (%.1fM) | cezasiz %d (%.1fM) '
          '| gomme bagli: %s' % (wd, n_dec, n_dec / 1e6,
                                 n_nod, n_nod / 1e6, model.tie_embeddings),
          flush=True)

    # AMP (fp16): T4 Tensor Core'lari devreye girer (~2x). Master agirliklar
    # FP32 kalir (GradScaler) -> export/parity etkilenmez. Veri zaten egitimin
    # basinda CUDA'ya tek seferde tasindigi icin DataLoader num_workers/pin_memory
    # darboğaz DEGILDIR (per-step transfer yok); o yuzden eklenmedi.
    use_amp = DEVICE.startswith('cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = None
    best_val = 1e9
    best_acc = 0.0
    bad = 0
    start_ep = 0
    step = 0
    tot_steps = lr_horizon * len(trX)
    # veri parmak izi: cift sayisi + natural + uzunluklar + BOLME STRATEJISI ->
    # veri degisince eski checkpoint otomatik atlanir (eski veriyle egitilmis
    # devam etmez). 'gs1' = ctx-grup bazli split; eski (pair seviyesi, sizintili)
    # checkpoint'lar 'gs0' ile isaretlidir ve otomatik reddedilir.
    # 'r%d' = RESP_CHARS_MAX: yanit tavani 70 -> 204 degistiginde cift sayisi,
    # natural, mxc/mxs ve CTX_CHARS AYNI kaldigi icin (clean_chars kirpar,
    # elmez) parmak izi bu bilesen olmadan cakisir ve 70 karakterle egitilmis
    # modelden sessizce devam edilirdi -> 204 deneyi tamamen bos olurdu.
    data_fp = '%d-%d-%d-%d-b4-c%d-gs1-r%d' % (len(trX), NATURAL, mxc, mxs,
                                              CTX_CHARS, RESP_CHARS_MAX)

    if args.fresh:
        print('Uyari: --fresh verildi, mevcut checkpoint yok sayilir '
              '(sifirdan basliyorum, eski dosya korunur).', flush=True)
    elif os.path.exists(CKPT):
        cp = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        arch = cp.get('arch', {})
        same_arch = (arch.get('d_model') == dm and arch.get('num_blocks') == nb
                     and arch.get('num_heads') == nh and arch.get('ff_mult') == ff
                     and arch.get('max_seq_len') == mxs and arch.get('V') == V
                     # Baglilik degistiyse checkpoint'in 'head' anahtari ya
                     # fazladir ya da yoktur -> state_dict yuklemesi PATLAR.
                     # Bu yuzden mimari uyumuna dahil edilir.
                     and bool(arch.get('tied_embeddings', False))
                     == bool(model.tie_embeddings)
                     # Dropout AGIRLIKLARI DEGISTIRMEZ (sadece egitim
                     # davranisini duzenler), bu yuzden state_dict yuklemesi
                     # patlamaz. Yine de mimari uyumuna dahil: duzenleme
                     # deneyi karsilastirmasini bulaniklastirmamak icin
                     # FARKLI dropout ile eski yarim egitimli modelden devam
                     # etmek yerine sifirdan baslanmali. Aksi halde "dropout
                     # 0.20 daha iyi" cikarimi, aslinda 0.10 ile yarim
                     # egitilmis bir modelin devami olurdu.
                     and arch.get('drop') == drop)
        same_data = cp.get('data') == data_fp
        if not same_arch or not same_data:
            print('Uyari: mevcut checkpoint eski (mimari-uyum: %s, '
                  'veri-uyum: %s, beklenen data=%s). '
                  'Sifirdan basliyorum (eski dosya korunur).' % (
                      same_arch, same_data, data_fp), flush=True)
        else:
            model.load_state_dict(cp['model'])
            opt.load_state_dict(cp['opt'])
            best_val, best_state, start_ep, step = cp['best_val'], cp['best_state'], cp['epoch'], cp['step']
            best_acc = cp.get('best_acc', 0.0)
            # Sabir sayaci da tasinmali. Kaganlde oturum 9 saatlik kosunun
            # ortasinda kopabiliyor; sayac sifirlanirsa her resume'da model
            # ek patience*val_every epoch daha tolerere ediliyor ve
            # "gercek bitis noktasi" sessizce kayar. Eski checkpoint'lerde
            # 'bad' yok -> 0 (daha genis tolerans, guvenli yon).
            cp_ve = cp.get('val_every', val_every)
            if cp_ve == val_every:
                bad = int(cp.get('bad', 0))
            else:
                # val_every degismis: kotu OLCUM sayisini epoch'a cevir ki
                # esik degissin.
                bad = int(round(int(cp.get('bad', 0)) * cp_ve / max(1, val_every)))
            best_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}
            # step, ESKI ufka gore sayildigi icin yeni tot_steps'i asabilir
            # (ufuk kisisince prog>1 olur ve cosine anlamsiz bir LR verir).
            # Tamponla: ufka sabitle, LR_min uzerinden devam etsin.
            if step > tot_steps:
                print('Uyari: checkpoint step %d > yeni lr ufku %d step; '
                      'step ufka sabitleniyor (LR -> LR_MIN).' % (step, tot_steps),
                      flush=True)
                step = tot_steps
            print('Devam: epoch', start_ep, '| step', step,
                  '| best val:', round(best_val, 4),
                  '| best acc:', round(best_acc, 3),
                  '| kotu epoch:', bad, '/ esik', patience, flush=True)

    # ---- coklu GPU sarmaci (DataParallel) -------------------------------------
    # torch.compile VARSAYILAN KAPALI: bu parametre/buffer tabanli modulde cloud
    # GPU'sunda 'embed' AttributeError ve DP.train() RecursionError verdigi icin
    # hicbir kosulda otomatik denenmez. Yalnizca acik istekle (LLM_COMPILE=1)
    # denenir; o durumda bile warmup-backward zirhi hatayi epoch oncesi yakalar.
    raw = model                       # duz TorchLLM yedegi (fallback)

    def _set_mode(training):
        m = model
        while hasattr(m, 'module'):
            m = m.module
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        m.training = bool(training)
    compiled = False
    dp = False
    if (DEVICE.startswith('cuda')
            and tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 0)
            and os.environ.get('LLM_COMPILE') == '1'):
        try:
            import torch._dynamo as _dyn
            _dyn.config.suppress_errors = True
            _dyn.config.cache_size_limit = 128
        except Exception:
            pass
        try:
            model = torch.compile(model, dynamic=True)
            compiled = True
            print('torch.compile aktif (kernel derleme)', flush=True)
        except Exception as e:
            print('torch.compile atlandi:', str(e)[:140], flush=True)
    if DEVICE.startswith('cuda') and torch.cuda.device_count() > 1 \
            and os.environ.get('LLM_DP_OFF') is None:
        try:
            model = torch.nn.DataParallel(model)
            dp = True
            print('DataParallel: %d GPU kullaniliyor (batch parcalaniyor)' %
                  torch.cuda.device_count(), flush=True)
        except Exception as e:
            model = model.module if hasattr(model, 'module') else model
            print('DataParallel atlandi (tek GPU ile devam):', str(e)[:140], flush=True)
    if compiled:
        try:
            with torch.no_grad():
                probe = torch.randint(0, max(2, V), (1, 16), device=DEVICE)
                _set_mode(False)
                _ = model(probe)
            _set_mode(True)
            p2 = torch.randint(0, max(2, V), (1, 16), device=DEVICE)
            out = model(p2)
            out.sum().backward()
            if hasattr(model, 'module'):
                mm = model.module
            else:
                mm = model
            if hasattr(mm, '_orig_mod'):
                for _p in mm._orig_mod.parameters():
                    _p.grad = None
            else:
                for _p in mm.parameters():
                    _p.grad = None
            print('kernel on-isinmasi OK (ders/derleme calisiyor)', flush=True)
        except Exception as e:
            print('kernel on-isinmasi hatali, derlenmemis modele donuyorum:',
                  str(e)[:140], flush=True)
            model = raw
            if dp and torch.cuda.device_count() > 1:
                try:
                    model = torch.nn.DataParallel(model)
                except Exception:
                    pass
    # DataParallel/kernel sarmasindan sonra DUZ (module prefix'siz) anahtarlar:
    # checkpoint/export her zaman buradan beslenir -> tek GPU'da sorunsuzlasir.
    base = model.module if hasattr(model, 'module') else model

    # ---------------- egitim
    t0_all = time.time()
    done = False
    for ep in range(start_ep + 1, EPOCHS + 1):
        _set_mode(True)
        t0 = time.time()
        order = list(range(len(trX)))
        random.Random(ep).shuffle(order)
        tl = 0.0
        for bi in order:
            step += 1
            cur = lr_base
            if step <= WARMUP:
                cur = lr_base * (step / WARMUP)
            else:
                prog = (step - WARMUP) / max(1, tot_steps - WARMUP)
                cur = lr_base * (LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g['lr'] = cur
            opt.zero_grad()
            with torch.autocast('cuda', torch.float16) if use_amp \
                    else contextlib.nullcontext():
                loss = llm_loss(model(trX[bi]), trX[bi], trM[bi])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            if os.environ.get('SMOKE') and step >= 2:
                print('SMOKE OK:', float(loss.item()), flush=True)
                return 0
        tl /= len(trX)

        _set_mode(False)
        # --val-every N: val gecisi yalnizca her N epochda bir yapilir
        # (val, train-isleminin yarisi kadar tutar; N>1 epoch suresini kisaltir).
        do_val = (ep % val_every == 0 or start_ep == 0 and ep == 1)
        if do_val:
            vl = va_acc = 0.0
            with torch.no_grad():
                for x, m in zip(vaX, vaM):
                    lg = model(x)
                    vl += llm_loss(lg, x, m).item()
                    va_acc += masked_acc(lg, x, m)
            vl /= len(vaX)
            va_acc /= len(vaX)
            print(f'epoch {ep:3d}/{EPOCHS} | train {tl:.4f} | val {vl:.4f} | acc {va_acc:.3f} | '
                  f'{time.time()-t0:.1f}s | lr {cur:.5f} | step {step}/{tot_steps}', flush=True)
        else:
            vl = best_val
            print(f'epoch {ep:3d}/{EPOCHS} | train {tl:.4f} | (val atlandi) | '
                  f'{time.time()-t0:.1f}s | lr {cur:.5f} | step {step}/{tot_steps}', flush=True)

        # EARLY-STOP: val LOSS tabanli, tolerans EPOCH cinsinden -> early_stop_step
        if do_val:
            bad, improved, stop = early_stop_step(vl, best_val, bad, val_every,
                                                  patience, VAL_IMP)
            if improved:
                # gercek kayip iyilesmesi -> rekor guncellenir
                best_val = vl
                best_acc = va_acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in base.state_dict().items()}
            elif stop:
                print(f'[llm] Erken durdurma: val loss yukselmeye basladi '
                      f'(son {bad} epoch iyilesme yok, esik {patience}). '
                      f'En iyi val: {best_val:.4f} | best acc: {best_acc:.3f}',
                      flush=True)
                done = True
        if ep % CKPT_FREQ == 0 or done:
            torch.save({'epoch': ep, 'step': step, 'model': best_state,
                        'opt': opt.state_dict(), 'best_val': best_val,
                        'best_state': best_state, 'best_acc': best_acc,
                        'bad': bad, 'val_every': val_every,
                        'arch': {'d_model': dm, 'num_blocks': nb, 'num_heads': nh,
                                 'ff_mult': ff, 'max_seq_len': mxs, 'V': V,
                                 'drop': drop,
                                 'tied_embeddings': base.tie_embeddings},
                        'data': data_fp}, CKPT)
            print(f'  checkpoint -> {CKPT}', flush=True)
        if done:
            break

    print('\nToplam egitim suresi: %.1f dk' % ((time.time() - t0_all) / 60), flush=True)

    # ---------------- export (numpy inference ile uyumlu compact NPZ format)
    if best_state is None:
        # uc durum: acc hic (0.0) ile rekor kirmadan erken durma -> son hal kullan
        best_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    base.load_state_dict(best_state)
    _set_mode(False)
    data = {
        'arch': 'llm',
        'V': V,
        'd_model': dm, 'num_blocks': nb, 'num_heads': nh,
        'ff_mult': ff,
        'max_ctx_len': mxc, 'max_seq_len': mxs,
        'tied_embeddings': bool(base.tie_embeddings),
        'weights_file': 'llm_model_weights.npz',
    }
    if tok is not None:
        data['tok_mode'] = 'bpe'
        data['vocab'] = None
        data['tokenizer'] = {
            'specials': tok.specials,
            'chars': tok.chars,
            'merges': [list(m) for m in tok.merges],
        }
    else:
        data['tok_mode'] = 'char'
        data['vocab'] = vocab
    dest = os.path.join(export_dir, 'llm_model.json')
    with io.open(dest, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    weights_dest = os.path.join(export_dir, 'llm_model_weights.npz')
    np.savez(weights_dest,
             **{k: np.asarray(v, dtype=np.float32) for k, v in best_state.items()})
    print('model yazildi:', dest, flush=True)
    print('agirliklar yazildi:', weights_dest, flush=True)

    # --------- parity 1: dogrudan tensor -> numpy (best_state uzerinden)
    x = trX[0][:4]
    with torch.no_grad():
        torch_logits = model(x).float().numpy() if DEVICE == 'cpu' else \
            model(x).cpu().float().numpy()
    np_model = LLM(vocab, d_model=dm, num_blocks=nb, num_heads=nh,
                   ff_mult=ff, max_ctx_len=mxc, max_seq_len=mxs,
                   seed=SEED, tokenizer=tok)
    np_model.tied_embeddings = bool(base.tie_embeddings)
    np_model.params = {k: np.asarray(v, np.float32) for k, v in best_state.items()}
    numpy_logits = np_model.forward(x.detach().cpu().numpy())
    diff = float(np.max(np.abs(torch_logits - numpy_logits)))
    print(f'parity dogrudan: {diff:.6f} (beklenen < 1e-3)', flush=True)
    assert diff < 1e-3, f'Parity bozuk: {diff}'
    if base.tie_embeddings:
        assert 'head' not in best_state, 'bagli modelde head ayri saklanmamali'
        print('baglilik dogrulandi: head anahtari yok, cikis = embed^T', flush=True)

    # --------- parity 2: export round-trip (data + npz -> from_dict -> forward)
    json_model = LLM(['<PAD>', '<BOS>', '<SEP>', '<EOS>']).from_dict(
        data, weights_path=weights_dest)
    json_logits = json_model.forward(x.detach().cpu().numpy())
    diff2 = float(np.max(np.abs(torch_logits - json_logits)))
    print(f'parity JSON round-trip: {diff2:.6f} (beklenen < 1e-3)', flush=True)

    # --------- agirlik karsilastirmasi: her param key icin max-abs fark
    worst_key, worst_val = '', 0.0
    for k in sorted(set(json_model.params) & set(best_state)):
        d = float(np.max(np.abs(json_model.params[k] -
                                np.asarray(best_state[k].cpu().numpy(), np.float32))))
        if d > worst_val:
            worst_val = d
            worst_key = k
    print(f'agirlik en buyuk fark: {worst_key} = {worst_val:.3e}', flush=True)

    # --------- val-batch karsilastirmasi: ilk val batch'te torch vs json numpy
    if va:
        xv, mv = va[0]
        with torch.no_grad():
            tv = model(torch.from_numpy(xv).long().to(DEVICE))
            tv = tv.cpu().float().numpy()
        nv = json_model.forward(xv)
        diff3 = float(np.max(np.abs(tv - nv)))
        xv_t = torch.from_numpy(xv).long()
        mv_t = torch.from_numpy(mv).float()
        tv_acc = masked_acc(torch.from_numpy(tv), xv_t, mv_t)
        nv_acc = masked_acc(torch.from_numpy(nv), xv_t, mv_t)
        print(f'val-batch: torch acc={tv_acc:.3f} | json-numpy acc={nv_acc:.3f} | logit-fark={diff3:.3e}', flush=True)

    if diff2 >= 1e-3:
        print('[HATA] JSON round-trip parity tutarsiz!', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
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
# --- NCCL deadlock prevention: MUST BE SET BEFORE torch import ---
import os
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
os.environ.setdefault('NCCL_TIMEOUT', '3600')
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

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
    import torch.nn.functional as F
    import torch.distributed as dist
    HAVE_TORCH = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    torch = nn = F = dist = None
    HAVE_TORCH = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from seqgen import RESP_CHARS_MAX, clean_chars, load_pairs
from llm import PAD, LLM, build_llm_vocab, encode_llm, load_tokenizer
from naturalize import naturalize_pairs

# ---------------- TorchLLM: PyTorch mirror of NumPy LLM ----------------
if HAVE_TORCH:
    class TransformerBlock(nn.Module):
        """Single transformer block with pre-LN architecture."""
        def __init__(self, d_model, num_heads, ff_dim, drop):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.rsqrt = 1.0 / math.sqrt(self.head_dim)
            self.drop = drop
            
            # Pre-LN 1
            self.ln1 = nn.LayerNorm(d_model)
            # Pre-LN 2
            self.ln2 = nn.LayerNorm(d_model)
            
            # Attention projections
            self.Wq = nn.Linear(d_model, d_model, bias=False)
            self.Wk = nn.Linear(d_model, d_model, bias=False)
            self.Wv = nn.Linear(d_model, d_model, bias=False)
            self.Wo = nn.Linear(d_model, d_model, bias=False)
            self.bq = nn.Parameter(torch.zeros(1, d_model))
            self.bk = nn.Parameter(torch.zeros(1, d_model))
            self.bv = nn.Parameter(torch.zeros(1, d_model))
            self.bo = nn.Parameter(torch.zeros(1, d_model))
            
            # FFN
            self.W1 = nn.Linear(d_model, ff_dim)
            self.W2 = nn.Linear(ff_dim, d_model)
            self.b1 = nn.Parameter(torch.zeros(1, ff_dim))
            self.b2 = nn.Parameter(torch.zeros(1, d_model))
            
            # Pre-LN scale/shift (NumPy style: g*x + b)
            self.ln1_g = nn.Parameter(torch.ones(1, d_model))
            self.ln1_b = nn.Parameter(torch.zeros(1, d_model))
            self.ln2_g = nn.Parameter(torch.ones(1, d_model))
            self.ln2_b = nn.Parameter(torch.zeros(1, d_model))
            self.b1 = nn.Parameter(torch.zeros(1, ff_dim))
            self.b2 = nn.Parameter(torch.zeros(1, d_model))
            
            # Init
            for name in ('Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2'):
                nn.init.normal_(getattr(self, name).weight, std=0.02)
        
        def forward(self, x, rsqrt):
            B, T, d = x.shape
            H, hd = self.num_heads, self.head_dim
            
            # Pre-LN 1 + Attention
            x_ln = F.layer_norm(x, (self.d_model,), self.ln1.weight, self.ln1.bias, eps=1e-5)
            x_ln = self.ln1_g * x_ln + self.ln1_b
            
            Q = (x_ln @ self.Wq.weight.t() + self.bq).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            K = (x_ln @ self.Wk.weight.t() + self.bk).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V = (x_ln @ self.Wv.weight.t() + self.bv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) * rsqrt
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            # -1e4 instead of -1e9 to avoid FP16 overflow (half max ~ -65504)
            scores.masked_fill_(mask, -1e4)
            attn = torch.softmax(scores, dim=-1)
            out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
            x = x + (out @ self.Wo.weight.t() + self.bo)
            
            # Pre-LN 2 + FFN
            x_ln = F.layer_norm(x, (self.d_model,), self.ln2.weight, self.ln2.bias, eps=1e-5)
            x_ln = self.ln2_g * x_ln + self.ln2_b
            
            h = F.gelu(x_ln @ self.W1.weight.t() + self.b1)
            h = F.dropout(h + self.b1, p=self.drop, training=self.training)
            x = x + (h @ self.W2.weight.t() + self.b2)
            
            return x


if HAVE_TORCH:
    class TorchLLM(nn.Module):
        """PyTorch mirror of llm.LLM (NumPy). Same architecture, same init."""
        def __init__(self, V, d_model=256, num_blocks=4, num_heads=8, ff_mult=4,
                     max_seq_len=256, drop=0.10, tie_embeddings=True):
            super().__init__()
            self.V = int(V)
            self.d_model = int(d_model)
            self.num_blocks = int(num_blocks)
            self.num_heads = int(num_heads)
            self.ff_mult = int(ff_mult)
            self.max_seq_len = int(max_seq_len)
            self.drop = float(drop)
            self.tie_embeddings = bool(tie_embeddings)
            assert self.d_model % self.num_heads == 0
            self.head_dim = self.d_model // self.num_heads
            self.ff_dim = self.ff_mult * self.d_model
            self.rsqrt = float(1.0 / math.sqrt(self.head_dim))

            # Embedding
            self.embed = nn.Embedding(self.V, self.d_model, padding_idx=0)
            nn.init.normal_(self.embed.weight, std=0.02)
            with torch.no_grad():
                self.embed.weight[0].zero_()

            # Positional encoding
            pe = torch.zeros(self.max_seq_len, self.d_model)
            pos = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
            dim = torch.arange(self.d_model // 2, dtype=torch.float32)
            div = 10000.0 ** (2.0 * dim / self.d_model)
            pe[:, 0::2] = torch.sin(pos / div)
            pe[:, 1::2] = torch.cos(pos / div)
            pe = pe / math.sqrt(max(self.d_model, 1))
            self.register_buffer('pos_enc', pe)

            # Transformer blocks
            self.blocks = nn.ModuleList()
            for i in range(self.num_blocks):
                blk = TransformerBlock(self.d_model, self.num_heads, self.ff_dim, self.drop)
                # Custom init to match NumPy
                for name in ('Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2'):
                    nn.init.normal_(getattr(blk, name).weight, std=0.02)
                self.blocks.append(blk)

            # Output norm
            self.out_ln = nn.LayerNorm(self.d_model)
            self.out_ln_g = nn.Parameter(torch.ones(1, self.d_model))
            self.out_ln_b = nn.Parameter(torch.zeros(1, self.d_model))

            # Output head
            if not self.tie_embeddings:
                self.head = nn.Linear(self.d_model, self.V, bias=False)
                nn.init.normal_(self.head.weight, std=0.02)
                self.head_b = nn.Parameter(torch.zeros(1, self.V))
            else:
                self.head = None
                self.head_b = nn.Parameter(torch.zeros(1, self.V))

        def forward(self, x):
            # Handle 1D input (single sequence)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            B, T = x.shape
            x = self.embed(x) + self.pos_enc[:T].unsqueeze(0)

            for blk in self.blocks:
                x = blk(x, self.rsqrt)

            h = F.layer_norm(x, (self.d_model,), self.out_ln.weight, self.out_ln.bias, eps=1e-5)
            h = self.out_ln_g * h + self.out_ln_b
            if self.tie_embeddings:
                logits = h @ self.embed.weight.t() + self.head_b
            else:
                logits = h @ self.head.weight.t() + self.head_b
            return logits
else:
    TorchLLM = None

# ---------------- hiperparametreler (numpy inference ile AYNI mimari) ----------------
D_MODEL = 512
NUM_BLOCKS = 8
NUM_HEADS = 8
FF_MULT = 4
DROPOUT = 0.10
BATCH_SIZE = 64
LR_BASE = 1e-3
LR_MIN = 0.1
WARMUP = 200
PATIENCE = 6
VAL_IMP = 5e-4
LR_HORIZON_PAD = 20
GRAD_CLIP = 5.0
WEIGHT_DECAY = 0.01
TIE_EMBED = True
CKPT_FREQ = 1
MAX_PAIRS = 0
MAX_PAIRS_CTX_CARPAN = 3.08
MAX_PAIRS_INTENT_CARPAN = 18.85
MAX_CTX_LEN = 48
MAX_SEQ_LEN = 256
SEED = 7
# Kaggle Output paneli yalnizca /kaggle/working/ kokunu gosterir.
# Script git klonunun icinden calisir (/kaggle/working/NextgenAI/) ->
# ciktilari ust dizine yonlendirmezse Output'ta gorunmez.
_KAGGLE_OUT = '/kaggle/working'
SAVE_DIR = _KAGGLE_OUT if os.path.isdir(_KAGGLE_OUT) else os.environ.get('SAVE_DIR', BASE)
CTX_CHARS = 48


def _log(msg, *args, **kwargs):
    """Print with immediate flush for Kaggle real-time logs."""
    kwargs.setdefault('flush', True)
    print(msg, *args, **kwargs)


def prepare_data(RAG, NATURAL=0, tokenizer=None, kb_map_path=None,
                 max_ctx_len=MAX_CTX_LEN, max_seq_len=MAX_SEQ_LEN,
                 batch_size=64, chatgrow_path=None, limit_pairs=0,
                 max_pairs_cap=0, intents_path=None):
    """chatgrow/intent verisinden (sorgu, yanit) ciftleri uretir.
    CPU tensors dondurur; GPU'ya batch isleme tasinir."""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    
    if intents_path is None:
        intents_path = os.path.join(BASE, 'intents.json')
    
    # Load pairs from intents.json
    # max_pairs=0 means no limit (use default 20000 in load_pairs)
    effective_max_pairs = max_pairs_cap if max_pairs_cap > 0 else 20000
    pairs = load_pairs(intents_path, max_pairs=effective_max_pairs)
    _log(f'Loaded {len(pairs)} pairs from intents.json')
    
    # Add chatgrow pairs if provided
    if chatgrow_path:
        from seqgen import load_chatgrow_pairs
        cg_pairs = load_chatgrow_pairs(chatgrow_path, ctx_len=max_ctx_len, 
                                        resp_len=RESP_CHARS_MAX, max_pairs=limit_pairs)
        pairs.extend(cg_pairs)
        _log(f'chatgrow pairs added: {len(cg_pairs)}')
    
    # Naturalize pairs
    if NATURAL > 0:
        pairs = naturalize_pairs(pairs, k=NATURAL, seed=SEED)
        _log(f'naturalize: {len(pairs)} pairs after x{NATURAL}')
    
    # Limit pairs
    if limit_pairs > 0:
        pairs = pairs[:limit_pairs]
    
    # Split train/val by context groups (no leakage)
    ctx_groups = {}
    for ctx, resp in pairs:
        if ctx not in ctx_groups:
            ctx_groups[ctx] = []
        ctx_groups[ctx].append(resp)
    
    ctx_list = list(ctx_groups.keys())
    random.Random(SEED).shuffle(ctx_list)
    split_idx = int(len(ctx_list) * 0.9)
    train_ctxs = set(ctx_list[:split_idx])
    val_ctxs = set(ctx_list[split_idx:])
    
    train_pairs = [(ctx, resp) for ctx, resp in pairs if ctx in train_ctxs]
    val_pairs = [(ctx, resp) for ctx, resp in pairs if ctx in val_ctxs]
    
    _log(f'train: {len(train_pairs)} pairs, val: {len(val_pairs)} pairs')
    
    # Tokenize pairs
    def encode_pair(ctx, resp):
        # Format: <BOS> ctx <SEP> resp <EOS>
        ids = [tokenizer.bos_id] + tokenizer.encode(ctx) + [tokenizer.sep_id] + tokenizer.encode(resp) + [tokenizer.eos_id]
        return ids
    
    tr = []
    for ctx, resp in train_pairs:
        ids = encode_pair(ctx, resp)
        # Mask: 1 for response tokens, 0 for context
        ctx_ids = [tokenizer.bos_id] + tokenizer.encode(ctx) + [tokenizer.sep_id]
        mask = [0] * len(ctx_ids) + [1] * (len(ids) - len(ctx_ids))
        tr.append((ids, mask))
    
    va = []
    for ctx, resp in val_pairs:
        ids = encode_pair(ctx, resp)
        ctx_ids = [tokenizer.bos_id] + tokenizer.encode(ctx) + [tokenizer.sep_id]
        mask = [0] * len(ctx_ids) + [1] * (len(ids) - len(ctx_ids))
        va.append((ids, mask))
    
    _log(f'Encoded: train {len(tr)}, val {len(va)}')
    
    # Get vocab from tokenizer
    vocab = list(tokenizer.vocab().values()) if hasattr(tokenizer, 'vocab') else None
    
    return {'vocab': vocab, 'tokenizer': tokenizer, 'tr': tr, 'va': va}


def llm_loss(logits, targets, mask):
    """Cross entropy loss only on response positions (mask=1)."""
    # logits: (B, T, V), targets: (B, T), mask: (B, T)
    # EOS token (id=3) gets higher weight to encourage proper termination
    eos_weight = 5.0
    weight = torch.ones(logits.size(-1), device=logits.device)
    weight[3] = eos_weight  # EOS token id=3
    loss_fct = nn.CrossEntropyLoss(ignore_index=PAD, reduction='none', weight=weight)
    loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss = loss.view_as(targets) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def masked_acc(logits, targets, mask):
    """Accuracy on response positions only (fully vectorized, GPU-resident).

    Accepts any mix of batch / unbatched tensors:
      logits  – (B, T, V) or (T, V)
      targets – (B, T) or (T,)
      mask    – (B, T) or (T,)
    Returns a Python float (single .item() call at the end).
    """
    # --- 1. preds from logits ------------------------------------------
    preds = logits.argmax(dim=-1)          # (B, T) or (T,)

    # --- 2. Normalise to 2-D (B, T) ------------------------------------
    if preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    # --- 3. Align sequence lengths on dim-1 -----------------------------
    min_len = min(preds.size(1), targets.size(1), mask.size(1))
    preds   = preds[:, :min_len]
    targets = targets[:, :min_len]
    mask    = mask[:, :min_len]

    # --- 4. Compute accuracy (single kernel, single .item()) -----------
    mask_b  = mask.bool()
    total   = mask_b.sum()
    correct = ((preds == targets) & mask_b).sum()
    return (correct.float() / total.float().clamp(min=1)).item()


def main():
    # Prevent multiprocessing re-import issues (DataParallel/workers)
    if hasattr(torch, 'multiprocessing'):
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--rag', action='store_true')
    ap.add_argument('--kb-map', default=None, metavar='PATH')
    ap.add_argument('--max-pairs-cap', type=int, default=0, metavar='N')
    ap.add_argument('--natural', type=int, default=0, metavar='K')
    ap.add_argument('--chatgrow', default=None, nargs='+', metavar='PATH')
    ap.add_argument('--limit-pairs', type=int, default=0, metavar='N')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--d-model', type=int, default=D_MODEL)
    ap.add_argument('--num-blocks', type=int, default=NUM_BLOCKS)
    ap.add_argument('--num-heads', type=int, default=NUM_HEADS)
    ap.add_argument('--ff-mult', type=int, default=FF_MULT)
    ap.add_argument('--max-ctx-len', type=int, default=MAX_CTX_LEN)
    ap.add_argument('--max-seq-len', type=int, default=MAX_SEQ_LEN)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--lr-base', type=float, default=LR_BASE)
    ap.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    ap.add_argument('--dropout', type=float, default=DROPOUT)
    ap.add_argument('--untie-embeddings', dest='tie_embed', action='store_false', default=TIE_EMBED)
    ap.add_argument('--val-every', type=int, default=1)
    ap.add_argument('--lr-horizon', type=int, default=0, metavar='N')
    ap.add_argument('--grad-accum', type=int, default=1, metavar='N')
    ap.add_argument('--export-dir', default=None, metavar='PATH')
    ap.add_argument('--fresh', action='store_true')
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
    grad_accum = args.grad_accum
    export_dir = args.export_dir or SAVE_DIR

    dp_off = os.environ.get('LLM_DP_OFF') == '1'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count() if DEVICE == 'cuda' else 0
    
    if DEVICE.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        for g in range(n_gpu):
            t = torch.tensor(1.0, device=f'cuda:{g}')
            _ = (t + 1).item()
        _log(f'CUDA sicak: {n_gpu} GPU dogrulandi')
    
    _log(f'PyTorch {torch.__version__} | device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE=="cuda" else "-"} (Count: {n_gpu}) | AMP: {"fp16" if DEVICE=="cuda" else "off"} | SAVE_DIR: {SAVE_DIR} | RAG: {RAG} | patience: {patience} | dp_off: {dp_off}')

    _log('Veri hazirlaniyor...')
    intents_path = os.path.join(BASE, 'intents.json')
    d = prepare_data(RAG, NATURAL=NATURAL, tokenizer=load_tokenizer(),
                     kb_map_path=args.kb_map, max_ctx_len=mxc, max_seq_len=mxs,
                     batch_size=bs, chatgrow_path=args.chatgrow,
                     max_pairs_cap=args.max_pairs_cap, intents_path=intents_path)
    vocab = d['vocab']
    tok = d['tokenizer']
    V = tok.vocab_size if tok is not None else len(vocab)
    tr, va = d['tr'], d['va']

    if args.dry_run:
        _log('DRY-RUN OK')
        return 0

    if not HAVE_TORCH:
        _log('Eksik: PyTorch kurulu degil.')
        return 2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = TorchLLM(V, d_model=dm, num_blocks=nb, num_heads=nh, ff_mult=ff,
                     max_seq_len=mxs, drop=drop, tie_embeddings=tie_embed)
    
    if DEVICE.startswith('cuda') and n_gpu > 1 and not dp_off:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
        _log(f'DataParallel: {n_gpu} GPU kullaniliyor')
    model = model.to(DEVICE)

    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        is_vector = prm.ndim < 2 or prm.shape[0] == 1
        base_name = name.replace('module.', '') if hasattr(model, 'module') else name
        if is_vector or base_name == 'embed':
            no_decay.append(prm)
        else:
            decay.append(prm)
    
    opt = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': wd},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr_base)

    use_amp = DEVICE.startswith('cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    CKPT = os.path.join(SAVE_DIR, 'llm_ckpt.pt')
    best_state = None
    best_val = 1e9
    best_acc = 0.0
    bad = 0
    start_ep = 0
    step = 0
    
    data_fp = f'{len(tr)}-{NATURAL}-{mxc}-{mxs}-b4-c{CTX_CHARS}-r{RESP_CHARS_MAX}'
    
    if args.fresh:
        _log('Uyari: --fresh verildi, sifirdan basliyorum')
    elif os.path.exists(CKPT):
        cp = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        arch = cp.get('arch', {})
        same_arch = (arch.get('d_model') == dm and arch.get('num_blocks') == nb
                     and arch.get('num_heads') == nh and arch.get('ff_mult') == ff
                     and arch.get('max_seq_len') == mxs and arch.get('V') == V
                     and bool(arch.get('tied_embeddings', False)) == bool(tie_embed)
                     and arch.get('drop') == drop)
        same_data = cp.get('data') == data_fp
        if not same_arch or not same_data:
            _log(f'Uyari: checkpoint eski (arch: {same_arch}, data: {same_data}). Sifirdan basliyorum.')
        else:
            model.load_state_dict(cp['model'])
            opt.load_state_dict(cp['opt'])
            best_val, best_state, start_ep, step = cp['best_val'], cp['best_state'], cp['epoch'], cp['step']
            best_acc = cp.get('best_acc', 0.0)
            cp_ve = cp.get('val_every', 1)
            if cp_ve == 1:
                bad = int(cp.get('bad', 0))
            else:
                bad = int(round(int(cp.get('bad', 0)) * cp_ve / max(1, 1)))
            best_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}
            if step > tot_steps:
                _log(f'Uyari: checkpoint step {step} > ufuk {tot_steps}; step sabitleniyor.')
                step = tot_steps
            _log(f'Devam: epoch {start_ep} | step {step} | best val: {best_val:.4f} | bad: {bad}')

    trX = [torch.from_numpy(np.array(x, dtype=np.int64)) for x, _ in tr]
    trM = [torch.from_numpy(np.array(m, dtype=np.float32)) for _, m in tr]
    vaX = [torch.from_numpy(np.array(x, dtype=np.int64)) for x, _ in va]
    vaM = [torch.from_numpy(np.array(m, dtype=np.float32)) for _, m in va]

    # Training - Use proper DataLoader for efficient batching
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert to tensors and pad to max_seq_len
    def pad_sequence(seq, max_len, pad_value):
        if seq.shape[0] < max_len:
            return torch.cat([seq, torch.full((max_len - seq.shape[0],), pad_value, dtype=seq.dtype)])
        return seq[:max_len]
    
    _log("Preparing padded tensors for DataLoader...")
    trX_padded = torch.stack([pad_sequence(x, mxs, PAD) for x in trX])
    trM_padded = torch.stack([pad_sequence(m, mxs, 0.0) for m in trM])
    vaX_padded = torch.stack([pad_sequence(x, mxs, PAD) for x in vaX])
    vaM_padded = torch.stack([pad_sequence(m, mxs, 0.0) for m in vaM])
    
    # Create datasets and dataloaders (num_workers=0, pin_memory=False as requested)
    train_dataset = TensorDataset(trX_padded, trM_padded)
    val_dataset = TensorDataset(vaX_padded, vaM_padded)
    
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False
    )
    
    # Training
    t0_all = time.time()
    done = False
    tot_steps = EPOCHS * len(train_loader)  # simplified
    
    # Memory cleanup before training
    import gc
    gc.collect()
    if DEVICE.startswith('cuda'):
        torch.cuda.empty_cache()
    _log("Eğitim döngüsü başlıyor...")
    
    for ep in range(start_ep + 1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tl = 0.0
        step_in_epoch = 0
        
        for x_batch, m_batch in train_loader:
            step += 1
            step_in_epoch += 1
            if step <= WARMUP:
                cur = lr_base * (step / WARMUP)
            else:
                prog = (step - WARMUP) / max(1, EPOCHS * len(train_loader) - WARMUP)
                cur = lr_base * (LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g['lr'] = cur
            
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            m_batch = m_batch.to(DEVICE, non_blocking=True)
            
            opt.zero_grad()
            with torch.autocast('cuda', torch.float16) if use_amp else contextlib.nullcontext():
                loss = llm_loss(model(x_batch), x_batch, m_batch) / grad_accum
            
            scaler.scale(loss).backward()
            
            if step % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            tl += loss.item() * grad_accum
            
            if os.environ.get('SMOKE') and step >= 2:
                _log(f'SMOKE OK: {float(loss.item() * grad_accum)}', flush=True)
                return 0
        
        # Flush remaining gradients
        if step % grad_accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        
        tl /= step_in_epoch
        do_val = (ep % val_every == 0 or ep == 1)
        if do_val:
            model.eval()
            vl = va_acc = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for x_v, m_v in val_loader:
                    x_v = x_v.to(DEVICE, non_blocking=True)
                    m_v = m_v.to(DEVICE, non_blocking=True)
                    with torch.autocast('cuda', torch.float16) if use_amp else contextlib.nullcontext():
                        lg = model(x_v)
                    vl += llm_loss(lg, x_v, m_v).item()
                    va_acc += masked_acc(lg, x_v, m_v)
                    n_val_batches += 1
            vl /= max(n_val_batches, 1)
            va_acc /= max(n_val_batches, 1)
            _log(f'epoch {ep:3d}/{EPOCHS} | train {tl:.4f} | val {vl:.4f} | acc {va_acc:.3f} | {time.time()-t0:.1f}s | lr {cur:.5f} | step {step}')
        else:
            _log(f'epoch {ep:3d}/{EPOCHS} | train {tl:.4f} | (val atlandi) | {time.time()-t0:.1f}s | lr {cur:.5f} | step {step}')
        
        if do_val:
            if vl + VAL_IMP < best_val:
                best_val = vl
                best_acc = va_acc
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    _log(f'Erken durdurma: epoch {ep}')
                    break
        
        if ep % CKPT_FREQ == 0 or done:
            torch.save({
                'epoch': ep, 'step': step, 'model': best_state,
                'opt': opt.state_dict(), 'best_val': best_val,
                'best_state': best_state, 'best_acc': best_acc,
                'bad': bad, 'val_every': 1,
                'arch': {'d_model': dm, 'num_blocks': nb, 'num_heads': nh,
                         'ff_mult': ff, 'max_seq_len': mxs, 'V': V,
                         'drop': drop, 'tied_embeddings': True},
                'data': data_fp
            }, os.path.join(SAVE_DIR, 'llm_ckpt.pt'))
            _log(f'checkpoint saved')

    _log(f'Toplam egitim suresi: {(time.time() - t0_all) / 60:.1f} dk')

    # --- DISA AKTARMA: llm_model.json + llm_model_weights.npz (llm.load_llm uyumlu) ---
    # YALNIZCA egitim SONRASI calisir: en iyi agirliklar (best_state) PyTorch
    # state_dict'ten NumPy inference formatina cevrilir. Egitim dongusune,
    # mimariye ve hesaplara hicbir sekilde dokunulmaz.
    json_path = os.path.join(export_dir, 'llm_model.json')
    npz_path = os.path.join(export_dir, 'llm_model_weights.npz')
    export_ok = False
    try:
        sd = {}
        for _k, _v in best_state.items():
            if _k.startswith('module.'):
                _k = _k[len('module.'):]
            sd[_k] = _v.detach().cpu().float()

        def _n(t):
            return t.numpy().astype(np.float32, copy=False)

        params = {'embed': _n(sd['embed.weight'])}
        for i in range(nb):
            p = f'blocks.{i}.'
            for n in ('Wq', 'Wk', 'Wv', 'Wo'):
                # torch: x @ W.T  ==  numpy: x @ p  ->  p = W.T
                params[f'b{i}_{n}'] = _n(sd[p + n + '.weight']).T.copy()
            for n in ('q', 'k', 'v', 'o'):
                params[f'b{i}_b{n}'] = _n(sd[p + 'b' + n])
            # torch: (g * (norm*W + B)) + b  ==  numpy ln(x; g', b') :
            #   g' = g*W ,  b' = g*B + b   (sonuc birebir ayni)
            for n in ('ln1', 'ln2'):
                params[f'b{i}_{n}_g'] = _n(sd[p + n + '_g'] * sd[p + n + '.weight'])
                params[f'b{i}_{n}_b'] = _n(sd[p + n + '_g'] * sd[p + n + '.bias'] + sd[p + n + '_b'])
            params[f'b{i}_W1'] = _n(sd[p + 'W1.weight']).T.copy()
            params[f'b{i}_b1'] = _n(sd[p + 'b1'])
            params[f'b{i}_W2'] = _n(sd[p + 'W2.weight']).T.copy()
            params[f'b{i}_b2'] = _n(sd[p + 'b2'])
        params['out_ln_g'] = _n(sd['out_ln_g'] * sd['out_ln.weight'])
        params['out_ln_b'] = _n(sd['out_ln_g'] * sd['out_ln.bias'] + sd['out_ln_b'])
        if not tie_embed:
            params['head'] = _n(sd['head.weight']).T.copy()
        params['head_b'] = _n(sd['head_b'])

        if tok is not None:
            llm_np = LLM(d_model=dm, num_blocks=nb, num_heads=nh, ff_mult=ff,
                         max_ctx_len=mxc, max_seq_len=mxs, tokenizer=tok)
        else:
            llm_np = LLM(vocab=vocab, d_model=dm, num_blocks=nb, num_heads=nh,
                         ff_mult=ff, max_ctx_len=mxc, max_seq_len=mxs)
        llm_np.params = params
        llm_np.tied_embeddings = bool(tie_embed)
        os.makedirs(export_dir, exist_ok=True)
        header = llm_np.to_dict(weights_file='llm_model_weights.npz',
                                include_params=False)
        # from_dict/to_dict bunu yazar/okur; bayrak yuklemede sadece
        # _validate_params icin gerekli (bagli modelde head beklenmez).
        header['tied_embeddings'] = bool(tie_embed)
        with io.open(json_path, 'w', encoding='utf-8') as f:
            json.dump(header, f, ensure_ascii=False)
        np.savez(npz_path, **{k: np.asarray(v, dtype=np.float32)
                              for k, v in params.items()})
        export_ok = True
        _log(f'Model export edildi: {json_path} '
             f'({os.path.getsize(json_path) / 1048576:.2f} MB) + {npz_path} '
             f'({os.path.getsize(npz_path) / 1048576:.1f} MB)')
    except Exception as e:
        _log(f'Uyari: model export edilemedi: {e}')

    # --- Egitim ciktilerini SAVE_DIR icinde zip'le (Kaggle Output icin) ---
    # 3 temel dosya: llm_ckpt.pt + llm_model.json + llm_model_weights.npz
    # + varsa kaggle_train.log. Zip icin bos kalmamasini dogrulariz.
    try:
        import zipfile
        zip_path = os.path.join(SAVE_DIR, 'model_outputs.zip')
        os.makedirs(SAVE_DIR, exist_ok=True)
        files = []
        ckpt_path = os.path.join(SAVE_DIR, 'llm_ckpt.pt')
        if os.path.exists(ckpt_path):
            files.append(ckpt_path)
        if export_ok:
            files.extend([json_path, npz_path])
        for _d in (SAVE_DIR, BASE):
            _lp = os.path.join(_d, 'kaggle_train.log')
            if os.path.exists(_lp):
                files.append(_lp)
                break
        if files:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fp in files:
                    zf.write(fp, os.path.basename(fp))
            _names = {os.path.basename(f) for f in files}
            _missing = [n for n in ('llm_ckpt.pt', 'llm_model.json',
                                    'llm_model_weights.npz') if n not in _names]
            _log(f'Ciktilar ziplendi: {zip_path} ({len(_names)} dosya, '
                 f'{os.path.getsize(zip_path) / 1048576:.1f} MB): '
                 + ', '.join(sorted(_names)))
            if _missing:
                _log(f'Uyari: zip 3 temel dosyanin tamamina sahip DEGIL, '
                     f'eksik: {", ".join(_missing)}')
        else:
            _log('Uyari: zip icin dosya bulunamadi')
    except Exception as e:
        _log(f'Uyari: zip olusturulamadi: {e}')

    # --- Kaggle Outputs: /kaggle/working/ icine kaydet + zip'le ---
    # (Kaggle notebook'ta Output sekmesinden indirmek icin)
    try:
        import zipfile
        work_dir = os.path.join(os.path.expanduser("~"), "kaggle", "working")
        os.makedirs(work_dir, exist_ok=True)

        # 1) llm_model.json - model yapilandirmasi
        config_path = os.path.join(work_dir, "llm_model.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        # 2) llm_model_weights.npz - agirliklar
        weights_path = os.path.join(work_dir, "llm_model_weights.npz")
        np.savez(weights_path, **weights)   # weights => train_llm.py'de olusturulan dict of ndarray

        # 3) ZIP paketle
        zip_path = os.path.join(work_dir, "model_outputs.zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(config_path, arcname="llm_model.json")
            zf.write(weights_path, arcname="llm_model_weights.npz")
            log_path = os.path.join(work_dir, "kaggle_train.log")
            if os.path.exists(log_path):
                zf.write(log_path, arcname="kaggle_train.log")
        _log(f'Kaggle ciktilari hazirlandi: {zip_path}')
    except Exception as e:
        _log(f'Uyari: Kaggle cikti hazirlanamadi: {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
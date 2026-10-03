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
    import torch.distributed as dist
    HAVE_TORCH = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    torch = nn = dist = None
    HAVE_TORCH = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from seqgen import RESP_CHARS_MAX, clean_chars, load_pairs
from llm import PAD, LLM, build_llm_vocab, encode_llm, load_tokenizer
from naturalize import naturalize_pairs

# ---------------- hiperparametreler (numpy inference ile AYNI mimari) ----------------
D_MODEL = 256
NUM_BLOCKS = 4
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
SAVE_DIR = BASE
CTX_CHARS = 48


def _log(msg):
    """Print with immediate flush for Kaggle real-time logs."""
    print(msg, flush=True)


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
    pairs = load_pairs(intents_path, max_pairs=max_pairs_cap)
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
        pairs = naturalize_pairs(pairs, NATURAL, tokenizer)
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
    loss_fct = nn.CrossEntropyLoss(ignore_index=PAD, reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss = loss.view_as(targets) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def masked_acc(logits, targets, mask):
    """Accuracy on response positions only."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask.bool()
    return correct.sum().float() / mask.sum().clamp(min=1)


def main():
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

    # Training
    t0_all = time.time()
    done = False
    tot_steps = EPOCHS * len(trX)  # simplified
    
    for ep in range(start_ep + 1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        order = list(range(len(trX)))
        random.Random(ep).shuffle(order)
        tl = 0.0
        
        for bi in order:
            step += 1
            if step <= WARMUP:
                cur = lr_base * (step / WARMUP)
            else:
                prog = (step - WARMUP) / max(1, EPOCHS * len(trX) - WARMUP)
                cur = lr_base * (LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g['lr'] = cur
            
            x = trX[bi].to(DEVICE, non_blocking=True)
            m = trM[bi].to(DEVICE, non_blocking=True)
            
            opt.zero_grad()
            with torch.autocast('cuda', torch.float16) if use_amp else contextlib.nullcontext():
                loss = llm_loss(model(x), x, m) / grad_accum
            
            scaler.scale(loss).backward()
            
            if step % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            tl += loss.item() * grad_accum
            
            if os.environ.get('SMOKE') and step >= 2:
                _log(f'SMOKE OK: {float(loss.item() * grad_accum)}')
                return 0
        
        if step % grad_accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        
        tl /= len(trX)
        
        do_val = (ep % val_every == 0 or ep == 1)
        if do_val:
            model.eval()
            vl = va_acc = 0.0
            with torch.no_grad():
                for x_v, m_v in zip(vaX, vaM):
                    x_v = x_v.to(DEVICE, non_blocking=True)
                    m_v = m_v.to(DEVICE, non_blocking=True)
                    lg = model(x_v)
                    vl += llm_loss(lg, x_v, m_v).item()
                    va_acc += masked_acc(lg, x_v, m_v)
            vl /= len(vaX)
            va_acc /= len(vaX)
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
"""train_llm dinamik batch (PAD budama + uzunluk kumeleme) testleri.

Optimizasyonun dogrulugu: PAD kuyrugu budanip benzer uzunluklar paketlense de
1) maske toplami (= egitim kaybinin gordugu isaretler) birebir korunur,
2) transformator onundeki logitler budanan aralikta DEGISMEZ (kesme dogru).
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import PAD, LLM, encode_llm
from train_llm import (TorchLLM, _pack_encoded, early_stop_step,
                       effective_stop_epoch, make_batches,
                       min_stoppable_epoch)

_VOCAB = ['<PAD>', '<BOS>', '<SEP>', '<EOS>',
          'm', 'e', 'r', 'h', 'a', 'b', ' ', 'n', 's', 'i', 'l', 'y',
          'q', 'u', 'g', 'z', 'd', 'o']

# Torch tarafi testleri icin kucuk yapi (V, d_model, num_blocks).
_V, _D, _N = 24, 16, 2


def _dummy(max_seq=24):
    return LLM(_VOCAB, d_model=8, num_blocks=2, num_heads=2,
               max_ctx_len=10, max_seq_len=max_seq, seed=7)


class TestPackEncoded(unittest.TestCase):
    def test_pad_tail_is_trimmed(self):
        m = _dummy(24)
        seq, mask = encode_llm(m, 'merhaba', 'iyiyim')
        full_L = int(np.argmax(seq == PAD)) if (seq == PAD).any() else len(seq)
        self.assertLess(full_L, 24)      # kisa cift PAD'e bakmadan budanir
        batches = _pack_encoded([(seq, mask)], 4)
        self.assertEqual(batches[0][0].shape[1], full_L)
        self.assertEqual(np.asarray(mask).sum(), batches[0][1].sum())

    def test_lengths_bucketed_non_decreasing(self):
        m = _dummy(32)
        pair = [('merhaba nasilsin', 'iyiyim tesekkur ederim sen nasilsin'),
                ('selam', 'merhaba bugun nasilsin'),
                ('nasilsin', 'memnuniyetle yardimci olurum'),
                ('hava', 'bugun hava cok guzel ve gunesli')]
        enc = [encode_llm(m, c, r, max_seq=32) for c, r in pair]
        batches = _pack_encoded(enc, 2)
        widths = [b[0].shape[1] for b in batches]
        # Benzer uzunluklar kumelenir: ilk batch EN KISA sekanslari icerir
        self.assertEqual(widths, sorted(widths))
        for X, M in batches:
            self.assertEqual(X.shape, M.shape)
            self.assertLessEqual(X.shape[1], 32)

    def test_total_mask_preserved(self):
        m = _dummy(20)
        pair = [('merhaba', 'iyiyim tesekkur ederim'),
                ('hava nasil', 'bugun hava cok guzel'),
                ('selam', 'merhaba')]
        enc = [encode_llm(m, c, r) for c, r in pair]
        total_before = sum(np.asarray(mask).sum() for _, mask in enc)
        batches = _pack_encoded(enc, 2)
        total_after = sum(float(M.sum()) for _, M in batches)
        self.assertAlmostEqual(float(total_before), total_after)

    def test_forward_math_equivalent_after_trim(self):
        m = _dummy(24)
        seq, _mask = encode_llm(m, 'merhaba nasilsin', 'iyiyim tesekkur')
        L = int(np.argmax(seq == PAD)) if (seq == PAD).any() else len(seq)
        full = seq[None, :]
        trim = seq[:L][None, :]
        f_full = m.forward(full)
        f_trim = m.forward(trim)
        # Budanan aralik icindeki logitler birebir ayni olmali
        np.testing.assert_allclose(f_full[:, :L], f_trim, atol=1e-6)


class TestMakeBatches(unittest.TestCase):
    def test_end_to_end_shapes(self):
        m = _dummy(32)
        pairs = [('merhaba', 'iyiyim tesekkur ederim'),
                 ('hava nasil', 'bugun hava cok guzel olacak'),
                 ('selam', 'merhaba iyiyim'),
                 ('yunus', 'yapay zeka nedir bilgi'),
                 ('ai', 'yapay zeka gelisiyor')]
        batches = make_batches(pairs, 2, m, {})
        n_rows = sum(b[0].shape[0] for b in batches)
        self.assertEqual(n_rows, len(pairs))
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(b[0].shape[1] <= 32 for b in batches))


class TestTiedEmbeddings(unittest.TestCase):
    """Gomme <-> cikis bagliligi: tasarruf, esdegerlik, uyumluluk."""

    def _tied(self):
        return TorchLLM(_V, d_model=_D, num_blocks=_N, num_heads=2,
                       ff_mult=2, max_seq_len=24, tie_embeddings=True).eval()

    def _untied(self):
        return TorchLLM(_V, d_model=_D, num_blocks=_N, num_heads=2,
                       ff_mult=2, max_seq_len=24, tie_embeddings=False).eval()

    def test_head_not_stored_and_params_saved(self):
        tied, untied = self._tied(), self._untied()
        self.assertNotIn('head', tied.state_dict())
        self.assertIn('head', untied.state_dict())
        n_tied = sum(p.numel() for p in tied.parameters())
        n_untied = sum(p.numel() for p in untied.parameters())
        # Tasarruf tam olarak V*d (cikis matrisi).
        self.assertEqual(n_untied - n_tied, _V * _D)

    def test_matches_untied_with_head_transposed(self):
        """Bagli forward, head=embed^T olan ayri modelle BIT Ayni olmali."""
        tied = self._tied()
        untied = self._untied()
        sd = tied.state_dict()
        sd['head'] = sd['embed'].t().contiguous()
        untied.load_state_dict(sd)
        x = torch.randint(0, _V, (3, 12))
        with torch.no_grad():
            self.assertTrue(torch.equal(tied(x), untied(x)))

    def test_head_actually_used(self):
        """head yok sayilmiyor: rastgele head degisince cikti degismeli."""
        tied = self._tied()
        untied = self._untied()
        sd = tied.state_dict()
        sd['head'] = sd['embed'].t().contiguous()   # esdeger kurulum
        untied.load_state_dict(sd)
        x = torch.randint(0, _V, (2, 10))
        with torch.no_grad():
            before = untied(x)
            sd = untied.state_dict()
            sd['head'] = torch.randn_like(sd['head'])
            untied.load_state_dict(sd)
            self.assertGreater((before - untied(x)).abs().max().item(), 1e-3)

    def test_embed_gets_gradient_from_both_paths(self):
        """Baglilik tek yonlu degil: embed, cikis yolundan da gradyan alir."""
        m = self._tied()
        m.train()
        m.embed.grad = None
        x = torch.randint(0, _V, (2, 8))
        torch.nn.functional.cross_entropy(
            m(x).reshape(-1, _V),
            torch.randint(0, _V, (16,))).backward()
        self.assertIsNotNone(m.embed.grad)
        self.assertGreater(m.embed.grad.norm().item(), 0.0)

    def test_torch_numpy_parity_tied(self):
        """Bagli modelde torch -> numpy parity (export yolu)."""
        m = self._tied()
        np_m = LLM(list(range(_V)), d_model=_D, num_blocks=_N, num_heads=2,
                   ff_mult=2, max_seq_len=24, seed=7)
        np_m.tied_embeddings = True
        np_m.params = {k: np.asarray(v.detach().cpu(), np.float32)
                       for k, v in m.state_dict().items()}
        x = torch.randint(0, _V, (4, 12))
        with torch.no_grad():
            torch_logits = m(x).float().numpy()
        diff = float(np.max(np.abs(torch_logits - np_m.forward(x.numpy()))))
        self.assertLess(diff, 1e-3)

    def test_untied_flag_defaults_false_for_fresh_numpy_model(self):
        """Taze LLM() cagrisi eski (ayri head) davranisini korumali."""
        self.assertFalse(LLM(_VOCAB, d_model=8, num_blocks=2, num_heads=2,
                             max_ctx_len=10, max_seq_len=24, seed=7
                             ).tied_embeddings)


class TestNumpyTiedLoad(unittest.TestCase):
    """llm.py tarafi: bayrak okuma + yanlis eslesme reddi."""

    def _params(self, drop_head):
        m = LLM(list(range(_V)), d_model=_D, num_blocks=_N, num_heads=2,
                ff_mult=2, max_seq_len=16, seed=5)
        p = dict(m.params)
        if drop_head:
            p.pop('head')
        return p

    def _header(self, **extra):
        h = {'V': _V, 'd_model': _D, 'num_blocks': _N, 'num_heads': 2,
             'ff_mult': 2, 'max_ctx_len': 8, 'max_seq_len': 16,
             'tok_mode': 'char', 'vocab': list(range(_V))}
        h.update(extra)
        return h

    def _load(self, params, header):
        tmp = tempfile.mkdtemp()
        wp = os.path.join(tmp, 'w.npz')
        np.savez(wp, **{k: np.asarray(v, np.float32) for k, v in params.items()})
        return LLM(['x']).from_dict(header, weights_path=wp)

    def test_tied_loads_without_head(self):
        m = self._load(self._params(drop_head=True),
                       self._header(tied_embeddings=True))
        self.assertTrue(m.tied_embeddings)

    def test_tied_header_with_untied_weights_rejected(self):
        """Bagli header + ayri NPZ sessizce yanlis uretirdi; reddedilmeli."""
        with self.assertRaises(ValueError) as cm:
            self._load(self._params(drop_head=False),
                       self._header(tied_embeddings=True))
        self.assertIn('head', str(cm.exception))

    def test_untied_still_loads(self):
        m = self._load(self._params(drop_head=False),
                       self._header(tied_embeddings=False))
        self.assertFalse(m.tied_embeddings)

    def test_legacy_header_without_flag_loads(self):
        """Eski deploy edilen model (bayrak yok) bozulmamali."""
        m = self._load(self._params(drop_head=False), self._header())
        self.assertFalse(m.tied_embeddings)

    def test_untied_missing_head_rejected(self):
        with self.assertRaises(ValueError) as cm:
            self._load(self._params(drop_head=True),
                       self._header(tied_embeddings=False))
        self.assertIn('head', str(cm.exception))


class TestTiedExport(unittest.TestCase):
    """Kaggle export yolu: basit dis durum -> NPZ + header -> from_dict."""

    def _trained(self, tie=True, steps=2):
        m = TorchLLM(_V, d_model=_D, num_blocks=_N, num_heads=2, ff_mult=2,
                     max_seq_len=24, tie_embeddings=tie)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        for _ in range(steps):
            opt.zero_grad()
            x = torch.randint(0, _V, (3, 10))
            torch.nn.functional.cross_entropy(
                m(x).reshape(-1, _V), torch.randint(0, _V, (30,))).backward()
            opt.step()
        # Export oncesi dropout KAPALI olmali (train_llm.py: _set_mode(False)).
        # Train modunda kalsak parity rastgele bozulur.
        m.eval()
        return m, {k: v.detach().cpu().clone()
                   for k, v in m.state_dict().items()}

    def test_export_roundtrip_end_to_end(self):
        m, best_state = self._trained(tie=True)
        self.assertNotIn('head', best_state)      # export onayi (Kaggle yolu)
        tmp = tempfile.mkdtemp()
        wp = os.path.join(tmp, 'llm_model_weights.npz')
        np.savez(wp, **{k: np.asarray(v, np.float32)
                        for k, v in best_state.items()})
        header = {
            'arch': 'llm', 'V': _V, 'd_model': _D, 'num_blocks': _N,
            'num_heads': 2, 'ff_mult': 2, 'max_ctx_len': 10, 'max_seq_len': 24,
            'tied_embeddings': bool(m.tie_embeddings),
            'weights_file': 'llm_model_weights.npz',
            'tok_mode': 'char', 'vocab': list(range(_V)),
        }
        loaded = LLM(['x']).from_dict(header, weights_path=wp)
        self.assertTrue(loaded.tied_embeddings)
        x = torch.randint(0, _V, (4, 12))
        with torch.no_grad():
            torch_logits = m(x).float().numpy()
        diff = float(np.max(np.abs(torch_logits - loaded.forward(x.numpy()))))
        self.assertLess(diff, 1e-3)

    def test_untied_export_roundtrip_end_to_end(self):
        m, best_state = self._trained(tie=False)
        self.assertIn('head', best_state)
        tmp = tempfile.mkdtemp()
        wp = os.path.join(tmp, 'llm_model_weights.npz')
        np.savez(wp, **{k: np.asarray(v, np.float32)
                        for k, v in best_state.items()})
        header = {'V': _V, 'd_model': _D, 'num_blocks': _N, 'num_heads': 2,
                  'ff_mult': 2, 'max_ctx_len': 10, 'max_seq_len': 24,
                  'tied_embeddings': False, 'tok_mode': 'char',
                  'vocab': list(range(_V))}
        loaded = LLM(['x']).from_dict(header, weights_path=wp)
        x = torch.randint(0, _V, (4, 12))
        with torch.no_grad():
            torch_logits = m(x).float().numpy()
        diff = float(np.max(np.abs(torch_logits - loaded.forward(x.numpy()))))
        self.assertLess(diff, 1e-3)


class TestEarlyStopping(unittest.TestCase):
    """--patience EPOCH cinsindendir; --val-every esigi DEGISTIRMEZ.

    Regresyon gerekcesi: 10 epoch'lik olculmus Kaggle kosusunda val kaybi
    4. epoch'tan itibaren yukselmeye baslamisti (0.5492 -> 0.5667 -> 0.5736 ->
    0.5796) ama erken durdurma HIC tetiklenmedi; kosu zorlamayla bitti.
    Nedeni: sayac kotu val OLCUMU sayiyordu. --val-every 2 ile patience=6
    -> 12 epoch'lik tolerans; sadece 3 kotu olcum birikti (gereken 6).
    """

    # kaggle_train.txt'teki 10 epoch'lik olculmus kosu, epoch bazinda.
    # None = val atlandi (val_every=2).
    _VAL10 = [0.6955, 0.5663, None, 0.5492, None,
              0.5667, None, 0.5736, None, 0.5796]

    # Her epoch'a val OLÇUMU olan yogun dizi (val_every=1 testleri icin).
    # 4. epoch'ta en iyi, sonra monotonik yükselis.
    _DENSE = [1.0, 0.70, 0.60, 0.5492] + [0.5492 + 0.01 * i for i in range(1, 27)]

    @staticmethod
    def _grid(per_epoch, val_every, start_ep=1):
        """--val-every izarasina oturan (epoch, val) olcum listesi uretir."""
        return [(e, v) for e, v in enumerate(per_epoch, start=1)
                if v is not None and (e % val_every == 0 or e == start_ep)]

    def _run(self, meas, patience=6, val_every=2, val_imp=5e-4, legacy=False):
        """Olcum listesini oynatir.

        legacy=True: v1 semantigi (kotu OLCUM sayaci 1 artirir) -- hatanin
        kendisi, karsilastirma icin.

        Doner: (best_val, best_epoch, stop_epoch, bad_at_stop)
        """
        best_val, best_ep, bad, stop_ep, bad_stop = 1e9, None, 0, None, None
        for ep, vl in meas:
            if legacy:
                improved = vl < best_val - val_imp
                bad = 0 if improved else bad + 1
                stop = (not improved) and bad >= patience
            else:
                bad, improved, stop = early_stop_step(vl, best_val, bad,
                                                      val_every, patience, val_imp)
            if improved:
                best_val, best_ep = vl, ep
            if stop and stop_ep is None:
                stop_ep, bad_stop = ep, bad
                break
        return best_val, best_ep, stop_ep, bad_stop

    # --- olculmus kosu: en iyi val nerede? --------------------------------
    def test_measured_run_best_at_epoch_4(self):
        best_val, best_ep, _, _ = self._run(self._grid(self._VAL10, 2))
        self.assertEqual(best_ep, 4)
        self.assertAlmostEqual(best_val, 0.5492, places=6)

    def test_measured_run_v2_stops_legacy_does_not(self):
        # v2: kotu olcum 2 epoch biriktirir -> ep 6,8,10 -> bad 2,4,6 -> ep 10
        # v1: kotu olcum 1 biriktirir   -> ep 6,8,10 -> bad 1,2,3 -> HIC olusmaz
        meas = self._grid(self._VAL10, 2)
        _, _, stop_v2, _ = self._run(meas, val_every=2)
        _, _, stop_v1, _ = self._run(meas, val_every=2, legacy=True)
        self.assertEqual(stop_v2, 10)
        self.assertIsNone(stop_v1)

    def test_measured_run_with_70_epoch_budget_v2_saves_6_epochs(self):
        # Bu kosunun devami (val artmaya devam ediyor) + 70 epoch butcesi:
        # v2 ep 10'da durur, v1 ancak ep 16'da dururdu -> 6 epoch (~44 dk).
        seq = list(self._VAL10) + [0.580 + 0.001 * i for i in range(60)]
        meas = self._grid(seq, 2)
        _, _, stop_v2, _ = self._run(meas, val_every=2)
        _, _, stop_v1, _ = self._run(meas, val_every=2, legacy=True)
        self.assertEqual(stop_v2, 10)
        self.assertEqual(stop_v1, 16)
        self.assertEqual(stop_v1 - stop_v2, 6)

    # --- ASIL DEGISMEZ: epoch cinsinden tolerans --------------------------
    def test_epoch_tolerance_is_patience_rounded_up_to_measurement_grid(self):
        # v2'de sayac EPOCH biriktirir: durdurma aninda
        #     patience <= bad < patience + val_every
        # (sayac yalnizca val_every katlarina oturabildigi icin tam patience'a
        # ulasamayabilir; ve=4, patience=6 -> 8'de durur, 12'de degil.)
        # v1'de sayac OLCUM biriktirdigi icin epoch toleransi
        # patience*val_every idi: val_every 2 -> 12, 3 -> 18, 4 -> 24 epoch.
        for ve in (1, 2, 3, 4, 5, 6):
            _, _, stop_ep, bad_stop = self._run(self._grid(self._DENSE, ve),
                                                val_every=ve)
            self.assertIsNotNone(stop_ep, 've=%d durdurmadi' % ve)
            self.assertGreaterEqual(bad_stop, 6, 've=%d cok erken durdu' % ve)
            self.assertLess(bad_stop, 6 + ve,
                            've=%d: bad=%d, toleransin cok uzerinde' % (ve, bad_stop))

    def test_exact_patience_when_val_every_divides_patience(self):
        # 6'nin bolenleri olan val_every'lerde sayac tam 6'ya oturur.
        for ve in (1, 2, 3, 6):
            _, _, _, bad_stop = self._run(self._grid(self._DENSE, ve),
                                         val_every=ve)
            self.assertEqual(bad_stop, 6, 've=%d bad=%s' % (ve, bad_stop))

    def test_legacy_tolerance_grows_with_val_every(self):
        # Hatanin sayisal ifadesi: v1'de 6 OLCUMun kac epoch tutugu
        # val_every'ye bagliydi. Olcum sayisi sabit -> epoch toleransi KATLANIR.
        # 60 epoch yeter ki en sik izarada (ve=6) da 10 olcum olsun.
        long_seq = ([1.0, 0.70, 0.60, 0.5492]
                    + [0.5492 + 0.01 * i for i in range(1, 57)])
        for ve in (1, 2, 3, 4, 6):
            _, _, stop_ep, bad_stop = self._run(self._grid(long_seq, ve),
                                                val_every=ve, legacy=True)
            self.assertEqual(bad_stop, 6, 've=%d legacy bad=%s' % (ve, bad_stop))
            # 6 OLCUM = 6*ve EPOCH bekleniyordu (hata)
            _, best_ep, _, _ = self._run(self._grid(long_seq, ve), val_every=ve)
            self.assertLessEqual(stop_ep - best_ep, 6 * ve)
        # v2'de hepsi 6 epoch'a sabit
        for ve in (1, 2, 3, 4, 6):
            _, best_ep, stop_ep, bad_stop = self._run(self._grid(long_seq, ve),
                                                      val_every=ve)
            self.assertGreaterEqual(bad_stop, 6)
            self.assertLess(bad_stop, 6 + ve)
            self.assertLessEqual(stop_ep - best_ep, 6 + ve - 1)

    def test_stop_never_exceeds_tolerance_by_more_than_grid_rounding(self):
        for ve in (1, 2, 3, 4, 5, 6):
            _, best_ep, stop_ep, _ = self._run(self._grid(self._DENSE, ve),
                                               val_every=ve)
            self.assertIsNotNone(stop_ep)
            self.assertLessEqual(stop_ep - best_ep, 6 + ve - 1,
                                 've=%d toleransi asirdi' % ve)

    def test_val_every_two_exact_grid(self):
        # olcum ep 1,2,4,6,8,10 -> kotu olcum 6,8,10 -> bad 2,4,6 -> ep 10
        _, best_ep, stop_ep, bad = self._run(self._grid(self._VAL10, 2),
                                             val_every=2)
        self.assertEqual((best_ep, stop_ep, bad), (4, 10, 6))

    def test_val_every_one_exact_grid(self):
        # kotu olcum ep 5..10 -> bad 1..6 -> ep 10
        _, best_ep, stop_ep, bad = self._run(self._grid(self._DENSE, 1),
                                             val_every=1)
        self.assertEqual((best_ep, stop_ep, bad), (4, 10, 6))

    def test_val_every_three_grid(self):
        # izar 1,3,6,9,12. _DENSE ep6'ya kadar AZALIYOR (0.5692), o yuzden
        # izarinun en iyisi ep6; kotu olcum 9 (bad 3), 12 (bad 6) -> ep 12.
        _, best_ep, stop_ep, bad = self._run(self._grid(self._DENSE, 3),
                                             val_every=3)
        self.assertEqual((best_ep, stop_ep, bad), (6, 12, 6))

    def test_val_every_four_rounds_up_to_eight(self):
        # izar 1,4,8,12. en iyi ep4; kotu olcum 8 (bad 4), 12 (bad 8>=6) -> ep 12
        _, best_ep, stop_ep, bad = self._run(self._grid(self._DENSE, 4),
                                             val_every=4)
        self.assertEqual((best_ep, stop_ep, bad), (4, 12, 8))

    # --- sayac davranisi -------------------------------------------------
    def test_improvement_resets_counter(self):
        bad, improved, stop = early_stop_step(0.9, 1.0, 4, 2, 6)
        self.assertTrue(improved)
        self.assertEqual(bad, 0)
        self.assertFalse(stop)

    def test_equal_val_is_not_improvement(self):
        # eski rekorla ayni -> VAL_IMP kadar iyi gelistirme yok
        bad, improved, stop = early_stop_step(1.0, 1.0, 4, 2, 6)
        self.assertFalse(improved)
        self.assertEqual(bad, 6)
        self.assertTrue(stop)

    def test_bad_measurement_accumulates_by_val_every(self):
        bad, improved, stop = early_stop_step(1.0, 0.5, 2, 2, 6)
        self.assertFalse(improved)
        self.assertEqual(bad, 4)
        self.assertFalse(stop)

    def test_stops_at_patience_boundary(self):
        bad, improved, stop = early_stop_step(1.0, 0.5, 4, 2, 6)
        self.assertEqual(bad, 6)
        self.assertTrue(stop)

    def test_val_imp_guards_against_noise(self):
        bad, improved, _ = early_stop_step(0.9999, 1.0, 0, 1, 6, 5e-4)
        self.assertFalse(improved)
        self.assertEqual(bad, 1)
        bad, improved, _ = early_stop_step(0.9990, 1.0, 0, 1, 6, 5e-4)
        self.assertTrue(improved)
        self.assertEqual(bad, 0)

    def test_monotonic_decline_never_stops(self):
        vals = [1.0 / (i + 1) for i in range(40)]
        _, _, stop_ep, _ = self._run(self._grid(vals, 1), val_every=1)
        self.assertIsNone(stop_ep)

    def test_perfect_plateau_stops_at_patience(self):
        # val_every=1, patience=6: ilk olcum ep1 rekora girer, ep2..ep7 kotu
        _, best_ep, stop_ep, bad = self._run(self._grid([0.5] * 30, 1),
                                             val_every=1)
        self.assertEqual((best_ep, stop_ep, bad), (1, 7, 6))

    def test_patience_zero_stops_at_first_non_improvement(self):
        # patience=0: ilk KOTU olcumde dur. _VAL10'da ilk 3 olcum (ep 1,2,4)
        # rekora girdigi icin durus ep 6.
        _, best_ep, stop_ep, bad = self._run(self._grid(self._VAL10, 2),
                                             patience=0)
        self.assertEqual((best_ep, stop_ep, bad), (4, 6, 2))

    # --- effective_stop_epoch: butce yeterliligi -------------------------
    def test_effective_stop_epoch_measured_case(self):
        # epochs=10, patience=6, val_every=2 -> 3 kotu olcum = 6 epoch
        last, needed = effective_stop_epoch(6, 2, 10)
        self.assertEqual(needed, 6)
        self.assertEqual(last, 6)

    def test_effective_stop_epoch_cannot_exceed_budget(self):
        # epochs=4, patience=6, val_every=2 -> gerek 6 > 4 -> TETIKLENEMEZ
        last, needed = effective_stop_epoch(6, 2, 4)
        self.assertEqual(needed, 6)
        self.assertEqual(last, 4)

    def test_effective_stop_epoch_rounds_up_to_measurement_grid(self):
        # patience=5, val_every=2 -> ceil(5/2)=3 olcum -> 6 epoch
        _, needed = effective_stop_epoch(5, 2, 100)
        self.assertEqual(needed, 6)

    def test_effective_stop_epoch_guards_bad_input(self):
        self.assertEqual(effective_stop_epoch(6, 0, 100)[1], 6)   # val_every=0
        self.assertEqual(effective_stop_epoch(-1, 1, 100)[1], 0)  # patience<0

    def test_realistic_70_epoch_budget_can_stop(self):
        # kaggle_start.sh varsayilani: epochs=70, patience=6, val_every=2
        last, needed = effective_stop_epoch(6, 2, 70)
        self.assertEqual(needed, 6)
        self.assertLess(needed, 70)          # butce YETERLI
        self.assertEqual(last, 6)

    # --- CONFIG uyarisinin esikleri ---------------------------------------
    # main() min_stoppable_epoch(patience, val_every) esigine bakar:
    #   [esik > epochs -> KRITIK, esik == epochs -> UYARI, aksi halde sessiz]
    def test_budget_warning_thresholds(self):
        # epochs=4,5 -> en kencar durus 6 -> HIC TETIKLENEMEZ
        for ep in (4, 5):
            self.assertGreater(min_stoppable_epoch(6, 2), ep,
                               'epochs=%d KRITIK bekleniyordu' % ep)
        # epochs=6 -> tam son epochta, hicbir epoch tasarruf etmez -> UYARI
        self.assertEqual(min_stoppable_epoch(6, 2), 6)
        # epochs>=7 -> sessiz (en az 1 epoch yedek kalir)
        for ep in (7, 10, 70):
            self.assertLess(min_stoppable_epoch(6, 2), ep,
                            'epochs=%d sessiz olmaliydi' % ep)

    def test_config_tolerance_is_grid_rounded_not_patience(self):
        """CONFIG satiri toleransi patience DEGIL, gercek izarada biriken
        epoch sayisini yazmali. val_every patience'i bolmedigi icin farkli
        olur: val_every=4, patience=6 -> 2 olcum x 4 = 8 epoch (6 degil)."""
        for p, ve, beklenen_olcum, beklenen_ep in [
                (6, 2, 3, 6), (6, 3, 2, 6), (6, 4, 2, 8), (6, 5, 2, 10),
                (6, 6, 1, 6), (5, 2, 3, 6), (5, 4, 2, 8), (7, 2, 4, 8),
                (3, 2, 2, 4), (1, 2, 1, 2)]:
            n_meas = max(1, -(-p // ve))
            self.assertEqual(n_meas, beklenen_olcum, 'p=%d ve=%d' % (p, ve))
            self.assertEqual(n_meas * ve, beklenen_ep, 'p=%d ve=%d' % (p, ve))
            # effective_stop_epoch ile de ayni (val_every>=2'de)
            self.assertEqual(effective_stop_epoch(p, ve, 999)[1], beklenen_ep,
                             'p=%d ve=%d' % (p, ve))

    def test_budget_warning_scales_with_val_every(self):
        # val_every buyudukce son durdurma epoch'u da buyer
        self.assertEqual(effective_stop_epoch(6, 1, 100)[1], 6)
        self.assertEqual(effective_stop_epoch(6, 2, 100)[1], 6)
        self.assertEqual(effective_stop_epoch(6, 3, 100)[1], 6)
        self.assertEqual(effective_stop_epoch(6, 4, 100)[1], 8)
        self.assertEqual(effective_stop_epoch(6, 6, 100)[1], 6)

    @staticmethod
    def _brute_min_stoppable(patience, val_every, ust=40):
        """Sablonu oynatip tetiklenebilirligin en kencar epoch'unu bulur.

        Senaryo: ilk olcum kayit acar, sonraki HEP kotudur. Yani durus
        ancak ve sadece butce yeterliyse olusur.
        """
        for n in range(1, ust + 1):
            best, bad = 1e9, 0
            for ep in range(1, n + 1):
                if not (ep % val_every == 0 or ep == 1):
                    continue
                vl = 1.0 if ep == 1 else 2.0
                bad, imp, stop = early_stop_step(vl, best, bad,
                                                val_every, patience, 5e-4)
                if imp:
                    best = vl
                if stop:
                    return ep
        return None

    def test_min_stoppable_matches_brute_force(self):
        """min_stoppable_epoch() gercek oynatmadan TURETILMIS sayiyi verir."""
        for ve in (1, 2, 3, 4, 5, 6, 7, 8, 12):
            for p in (1, 2, 3, 6, 7, 12):
                beklenen = self._brute_min_stoppable(p, ve)
                gercek = min_stoppable_epoch(p, ve)
                self.assertEqual(
                    gercek, beklenen,
                    'patience=%d val_every=%d: formul %d, oynatma %d'
                    % (p, ve, gercek, beklenen))

    def test_patience_two_stops_on_first_val_rise(self):
        """KULLANICI TERCİHI: --patience 2 + --val-every 2 -> val TEK SEFER
        yukselince dur. 6 -> 2 degisikliginin regresyon kilidi.

        Yan etki: gurultu toleransi kalkar; VAL_IMP (5e-4) tek kalan
        korumadir.
        """
        p, ve = 2, 2
        # olculmus egri: tepe 4, ilk yukselme 6
        best_val, best_ep, stop_ep, bad = self._run(
            self._grid(self._VAL10, ve), patience=p, val_every=ve)
        self.assertEqual(best_ep, 4)
        self.assertEqual(stop_ep, 6, 'ilk yukselmede durmaliydi')
        self.assertEqual(bad, ve)
        # 1 kotu olcum yeterli: tepeyi 1 (en kencar) yap
        _, bep, sep, _ = self._run(self._grid(self._VAL10, ve),
                                   patience=p, val_every=ve, val_imp=0.5)
        self.assertEqual(bep, 1)
        self.assertEqual(sep, 2, 'en kencar durus 2. epoch')
        # ama val gercekten dusmeye devam ederse hic durmaz
        # (_DENSE tanimi gereği 4. epoch'tan sonra YUKSELIR; o yuzden
        #  burada gercek monotonik dusus serisi kuruyoruz)
        lusen = [1.0 / (1 + i) for i in range(24)]
        _, _, sep_d, _ = self._run(self._grid(lusen, ve),
                                   patience=p, val_every=ve)
        self.assertIsNone(sep_d, 'monotonik dususte durmamaliydi')
        # eski ayar 3 kotu olcum bekliyordu, artik 1
        _, _, sep6, bad6 = self._run(self._grid(self._VAL10, ve),
                                     patience=6, val_every=ve)
        self.assertEqual(sep6, 10)
        self.assertEqual(bad6, 6)

    def test_min_stoppable_differs_from_tolerans_at_val_every_1(self):
        """val_every=1'de iki kavram AYRILIR. Regresyon gerekcesi: eskiden
        esik toleranstan turetiliyordu ve epochs=6/patience=6/val_every=1
        kombinasyonu 'SON epochta tetiklenir' diye UYARI aliyordu, oysa en
        kencar durus 7. epoch -> asla tetiklenemez. min_stoppable_epoch
        ayirir.
        """
        for p in range(1, 13):
            _, toler = effective_stop_epoch(p, 1, 100)
            self.assertEqual(min_stoppable_epoch(p, 1), 1 + p)
            self.assertEqual(min_stoppable_epoch(p, 1), toler + 1)
        # val_every >= 2'de ikisi ayni
        for ve in (2, 3, 4, 5, 6, 8, 12):
            for p in range(1, 13):
                _, toler = effective_stop_epoch(p, ve, 100)
                self.assertEqual(min_stoppable_epoch(p, ve), toler,
                                 'p=%d ve=%d' % (p, ve))

    def test_min_stoppable_handles_zero_patience(self):
        """patience<=0: ilk KOTU olcumde durulur. effective_stop_epoch 0
        dondurur; mutlak epoch olarak 0 yanlis olurdu (1. epoch her zaman
        kayit acar, duramaz)."""
        self.assertEqual(effective_stop_epoch(0, 2, 100)[1], 0)
        self.assertEqual(min_stoppable_epoch(0, 2), 2)   # ilk kotu olcum 2
        self.assertEqual(min_stoppable_epoch(0, 3), 3)   # ilk kotu olcum 3
        self.assertEqual(min_stoppable_epoch(0, 1), 2)   # ilk kotu olcum 2
        self.assertEqual(min_stoppable_epoch(-1, 2), 2)  # negatif = 0 gibi

    def test_budget_gate_agrees_with_tetiklenebilirlik(self):
        """Esik, tetiklenebilirligin gercek siniriyla tam uyumlu olmali:
        altinda KRITIK, taminda UYARI, ustunda sessiz."""
        for ve in (1, 2, 3, 4, 6):
            for p in (0, 1, 2, 6, 7):
                esik = min_stoppable_epoch(p, ve)
                for epochs in range(1, esik + 4):
                    beklenen = 'KRITIK' if epochs < esik else (
                        'UYARI' if epochs == esik else None)
                    # train_llm.py icindeki dallanmayi birebir taklit et
                    if esik > epochs:
                        seviye = 'KRITIK'
                    elif esik == epochs:
                        seviye = 'UYARI'
                    else:
                        seviye = None
                    self.assertEqual(
                        seviye, beklenen,
                        'p=%d ve=%d epochs=%d esik=%d' % (p, ve, epochs, esik))
                    # kaba kuvvet: KRITIK dedigimiz butcede GERCEKTEN
                    # durulamamali, UYARI/sessiz dedigimizde durabilmeli
                    durus = self._brute_min_stoppable(p, ve, ust=epochs)
                    self.assertEqual(
                        durus is not None, beklenen != 'KRITIK',
                        'p=%d ve=%d epochs=%d: kaba kuvvet durus=%s, esik=%s'
                        % (p, ve, epochs, durus, beklenen))
                # esik butcesinde gercekten durus olusmali (tam tolerans)
                self.assertEqual(
                    self._brute_min_stoppable(p, ve, ust=esik), esik,
                    'p=%d ve=%d: esik %d' % (p, ve, esik))

    def test_old_default_250_exceeds_session_budget(self):
        # Belgeleyici olcum: 7.31 dk/epoch x 250 = 30.4 saat. 9h oturuma
        # sigmazdi; bu yuzden varsayilan 70 yapildi.
        self.assertGreater(250 * 7.31 / 60.0, 30.0)
        self.assertLess(70 * 7.31 / 60.0, 9.0)


if __name__ == '__main__':
    unittest.main()
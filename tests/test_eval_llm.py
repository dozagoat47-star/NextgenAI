"""eval_llm metrikleri ve ornekleme raporu icin regresyon testleri.

Saf metrikler (bleu, rep_rate, topic_overlap...) model gerektirmez; rapor
hatti stub modelle dogrulanir. Egitilmis llm_model.json varsa kucuk bir
entegrasyon testi de kosulur (model yoksa atlanir).
"""
import io
import json
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from eval_llm import (METRIC_VERSION, STOPWORDS, _measure, aggregate_report, bleu,
                      compare_reports, distinct_ratio, gold_recall, prec1,
                      rep_rate, sample_report, tokenize, topic_overlap)


class TestBleu(unittest.TestCase):
    def test_exact_copy_is_one(self):
        toks = tokenize('merhaba bugun hava cok guzel')
        self.assertEqual(bleu(toks, toks), 1.0)

    def test_disjoint_is_near_zero(self):
        ref = tokenize('merhaba bugun hava cok guzel')
        cand = tokenize('xcds qwrt zxpl mnok iuvb')
        self.assertLess(bleu(ref, cand), 0.3)

    def test_empty_candidate_is_zero(self):
        self.assertEqual(bleu(tokenize('hava'), []), 0.0)
        self.assertEqual(bleu([], []), 0.0)

    def test_partial_copy_between(self):
        ref = tokenize('hava yagmurlu olacak ve ruzgarli gececek')
        cand = tokenize('hava yagmurlu olacak')
        b = bleu(ref, cand)
        self.assertGreater(b, 0.0)
        self.assertLess(b, 1.0)

    def test_brevity_penalty_short_candidate(self):
        ref = tokenize('bugun hava cok guzel ve gunesli')
        cand = tokenize('evet')
        self.assertLess(bleu(ref, cand), 0.2)


class TestLexicalMetrics(unittest.TestCase):
    def test_prec1_full_match(self):
        self.assertEqual(prec1(tokenize('hava gunesli'), tokenize('hava gunesli')), 1.0)

    def test_prec1_no_match(self):
        self.assertEqual(prec1(tokenize('hava'), tokenize('muzik')), 0.0)

    def test_rep_rate_fluent_zero(self):
        self.assertEqual(rep_rate(tokenize('bugun hava cok guzel oldu'), 2), 0.0)

    def test_rep_rate_repetitive_positive(self):
        self.assertGreater(rep_rate(tokenize('evet evet evet evet'), 2), 0.0)

    def test_distinct_degenerate(self):
        self.assertEqual(distinct_ratio(tokenize('aaa aaa aaa aaa')), 0.25)

    def test_distinct_all_unique(self):
        self.assertEqual(distinct_ratio(tokenize('bir iki uc dort')), 1.0)

    def test_topic_overlap_stopwords_ignored(self):
        cand = tokenize('evet hava bugun cok guzel')
        query = tokenize('hava nasil bugun guzel mi')
        ov = topic_overlap(cand, query, STOPWORDS)
        self.assertAlmostEqual(ov, 1.0)

    def test_topic_overlap_zero(self):
        cand = tokenize('muzik dinlemeyi seviyorum')
        query = tokenize('hava nasil olacak')
        self.assertEqual(topic_overlap(cand, query, STOPWORDS), 0.0)


class _StubModel:
    def __init__(self):
        self.calls = []

    def sample(self, context, temperature=0.7, top_k=10, knowledge=None,
               rep_penalty=0.3):
        self.calls.append(knowledge)
        if knowledge:
            return 'evet gunes doguyor ve hava acik'
        return 'evet tabii ki yardimci olurum'


class TestSampleReport(unittest.TestCase):
    def setUp(self):
        self.stub = _StubModel()

    def test_rows_have_all_metric_keys(self):
        items = [('hava nasil', 'bugun hava cok guzel olacak')]
        rows = sample_report(self.stub, items, seed=1)
        self.assertEqual(len(rows), 1)
        for k in ('copy_bleu', 'copy_prec1', 'rep2', 'distinct1', 'fluency',
                  'topic_q', 'topic_mean', 'gold_recall', 'length_ratio',
                  'avg_word_len', 'gen_index'):
            self.assertIn(k, rows[0])

    def test_knowledge_is_passed_down(self):
        items = [('hava nasil', 'gunes doguyor', 'gunes bugun dogudan dogar')]
        rows = sample_report(self.stub, items, seed=1)
        self.assertIsNotNone(self.stub.calls[0])
        self.assertTrue(rows[0]['has_knowledge'])
        self.assertIsNotNone(rows[0]['topic_k'])

    def test_aggregate_shape(self):
        items = [('hava nasil', 'bugun guzel')] * 3
        rows = sample_report(self.stub, items, seed=1)
        agg = aggregate_report(rows)
        self.assertEqual(agg['n_samples'], 3)
        for k in ('copy_bleu', 'gen_index', 'fluency', 'topic_q'):
            self.assertIn(k + '_std', agg)
            self.assertGreaterEqual(agg[k], 0.0)
            self.assertLessEqual(agg[k], 1.0 + 1e-9)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_report(self):
        from llm import LLM
        m = LLM(['<PAD>', '<BOS>', '<SEP>', '<EOS>', 'a', 'b', 'c', ' '],
                d_model=32, num_blocks=2, num_heads=2,
                max_ctx_len=40, max_seq_len=80, seed=7)
        items = [('abc', 'cba aaa')]
        r1 = sample_report(m, items, seed=42)
        r2 = sample_report(m, items, seed=42)
        self.assertEqual(r1[0]['generated'], r2[0]['generated'])

    def test_stub_measure_manual(self):
        row = _measure('hava cok guzel', 'hava cok guzel', 'hava cok guzel',
                       None, STOPWORDS)
        self.assertAlmostEqual(row['copy_bleu'], 1.0)
        self.assertAlmostEqual(row['topic_mean'], 1.0)


# --- qa_score v2: ana eksen duzeltmesi -------------------------------------
# v1'de qa_score'un %50'si topic_overlap(uretilen, SORGU) idi. Olculmus
# hata: iyi bir cevap soruyu TEKRARLAMAZ, dogru cevaplar bu eksende ~0 alir,
# soru kelimelerini yankilayan bozuk cevaplar yuksek alir. Orinek (rapordan):
#   S: hayvan turleri nelerdir
#   U/G: kediler balik, tavuk ve kuru mama yerler   (BIREBIR KOPYA)
#   v1: copy_bleu=1.000, topic_q=0.000 -> qa=0.489
# v2'de ana eksen gold_recall(uretilen, ALTIN).
Q_FAUNA = 'hayvan turleri nelerdir'
G_FAUNA = 'kediler balik, tavuk ve kuru mama yerler sevimli ve bagimsiz hayvanlar'
KB_FAUNA = ('Evcil kediler balik, tavuk ve kuru mama ile beslenir. '
            'Kediler bagimsiz ve sevimli hayvanlardir.')


class TestGoldRecall(unittest.TestCase):
    """gold_recall: ALTIN cevabin icerik kelimelerinin kaci uretimde gecti."""

    def test_exact_copy_is_one(self):
        t = tokenize(G_FAUNA)
        self.assertAlmostEqual(gold_recall(t, t, STOPWORDS), 1.0)

    def test_disjoint_is_zero(self):
        c = tokenize('istanbul bogazici uzanir martiniler sinirlarinda yasar')
        self.assertEqual(gold_recall(c, tokenize(G_FAUNA), STOPWORDS), 0.0)

    def test_partial_coverage(self):
        c = tokenize('kediler balik yerler')
        # gold content: {kediler, balik, tavuk, kuru, mama, yerler, sevimli,
        #                 bagimsiz, hayvanlar} = 9
        self.assertAlmostEqual(gold_recall(c, tokenize(G_FAUNA), STOPWORDS),
                               3 / 9.0)

    def test_stopwords_ignored(self):
        c = tokenize('bir iki ve ile kediler balik')
        self.assertAlmostEqual(gold_recall(c, tokenize('kediler balik'), STOPWORDS),
                               1.0)

    def test_empty_gold_is_zero(self):
        self.assertEqual(gold_recall(tokenize('herhangi'), [], STOPWORDS), 0.0)

    def test_empty_candidate_is_zero(self):
        self.assertEqual(gold_recall([], tokenize(G_FAUNA), STOPWORDS), 0.0)


class TestQaScoreV2(unittest.TestCase):
    """Ana eksen ALTIN kapsamasi; soru yankilamasi ODULLENDIRILMEZ."""

    def _m(self, gen, knowledge=KB_FAUNA):
        return _measure(Q_FAUNA, G_FAUNA, gen, knowledge, STOPWORDS)

    def test_exact_copy_scores_high(self):
        r = self._m(G_FAUNA)
        self.assertAlmostEqual(r['gold_recall'], 1.0)
        # v1'de burasi 0.489 idi
        self.assertGreater(r['qa_score'], 0.85)

    def test_query_echo_does_not_score_high(self):
        # Soruyu kelimesi kelimesine yankilayan cevap: v1'in SEVDIGI davranis.
        r = self._m(Q_FAUNA)
        self.assertAlmostEqual(r['topic_q'], 1.0)      # tq ust-duzey yakaliyor
        self.assertEqual(r['gold_recall'], 0.0)        # ...ama icerik tasimiyor
        self.assertLess(r['qa_score'], 0.35)

    def test_echo_is_worse_than_irrelevant_but_fluent(self):
        # v1'de yankilayan (0.527) ALAKASIZDAN (0.314) iyiydi -> eksen ters
        # calisiyordu. v2'de alakasiz kazandirmali.
        echo = self._m(Q_FAUNA)['qa_score']
        off = self._m('istanbul bogazici uzanir martiniler sinirlarinda yasar'
                      )['qa_score']
        self.assertLess(echo, off)

    def test_irrelevant_answer_beats_echo(self):
        off = self._m('kediler cok sevimli hayvanlardir ve bagimsiz yasarlar'
                      )['qa_score']
        echo = self._m(Q_FAUNA)['qa_score']
        self.assertGreater(off, echo)

    def test_ordering_copy_gt_partial_gt_off_gt_empty(self):
        partial = self._m('kediler balik yerler')['qa_score']
        off = self._m('istanbul bogazici uzanir')['qa_score']
        empty = self._m('')['qa_score']
        self.assertGreater(partial, off)
        self.assertGreater(off, empty)
        # Bos uretim: tum agirlikli bilesenler 0, acik -0.5 cezasi.
        self.assertAlmostEqual(empty, 0.25 * 0.5 - 0.5)

    def test_tmean_uses_knowledge_when_present(self):
        r = self._m('kediler balik, tavuk ve kuru mama yerler')
        # gr>0 ve tk>0 -> tmean ikisinin ortasi
        self.assertGreater(r['topic_k'], 0.0)
        self.assertAlmostEqual(r['topic_mean'],
                               (r['gold_recall'] + r['topic_k']) / 2.0)

    def test_tmean_falls_back_to_gold_recall_without_knowledge(self):
        r = _measure(Q_FAUNA, G_FAUNA, 'kediler balik yerler', None, STOPWORDS)
        self.assertIsNone(r['topic_k'])
        self.assertAlmostEqual(r['topic_mean'], r['gold_recall'])


class TestMetricVersionGuard(unittest.TestCase):
    """Farkli sema surumleri karsilastirilamaz (olcut ayni degil)."""

    def _write(self, path, version, items):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'model': 'x', 'report': {'metric_version': version,
                                                'qa_score': 0.5, 'qa_score_std': 0.1},
                       'items': items}, f)

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.a = os.path.join(self.tmp, 'a.json')
        self.b = os.path.join(self.tmp, 'b.json')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mixed_versions_rejected(self):
        self._write(self.a, 1, [{'q': 's', 'qa_score': 0.5}])
        self._write(self.b, 2, [{'q': 's', 'qa_score': 0.9}])
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = compare_reports(self.a, self.b)
        finally:
            sys.stdout = old
        self.assertEqual(rc, 1)
        self.assertIn('semasi farkli', buf.getvalue())

    def test_same_version_accepted(self):
        it = [{'q': 's1', 'qa_score': 0.5}, {'q': 's2', 'qa_score': 0.7}]
        self._write(self.a, 2, it)
        self._write(self.b, 2, it)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = compare_reports(self.a, self.b)
        finally:
            sys.stdout = old
        self.assertEqual(rc, 0)
        self.assertIn('PAIRED', buf.getvalue())

    def test_current_version_is_two(self):
        self.assertEqual(METRIC_VERSION, 2)

    def test_aggregate_carries_version(self):
        rows = sample_report(_StubModel(), [('hava nasil', 'bugun guzel')],
                             seed=1)
        self.assertEqual(aggregate_report(rows)['metric_version'],
                         METRIC_VERSION)


_MODEL_PATH = os.path.join(BASE, 'model', 'llm_model.json')


@unittest.skipUnless(os.path.exists(_MODEL_PATH),
                     'egitilmis llm_model.json yok - entegrasyon atlaniyor')
class TestRealModelEval(unittest.TestCase):
    def test_real_model_smoke_report(self):
        from llm import load_llm
        model = load_llm()
        self.assertIsNotNone(model)
        items = [('merhaba nasilsin', 'iyiyim tesekkur ederim sen nasilsin'),
                 ('hava nasil olacak', 'yarin yagmur bekleniyor')]
        rows = sample_report(model, items, seed=3)
        agg = aggregate_report(rows)
        self.assertEqual(agg['n_samples'], 2)
        self.assertGreaterEqual(agg['gen_index'], 0.0)
        self.assertLessEqual(agg['gen_index'], 1.0)


class TestDecodingAyariDenetimi(unittest.TestCase):
    """Farkli DECODING AYARI olan iki rapor "model farki" gibi okunamaz.

    OLCULEN HATA (29.09): eval varsayilani knowledge_bias=0.0, URETIM 1.2
    (brain.py:764). 29.09 kod ayari taramasinda ayni model, ayni sorular
    icin kb=0 -> kopya 0.226, kb=1.2 -> 0.215 cikti; bu bir model
    degisikligi degil ayar etkisidir, ama compare_reports "A daha iyi"
    diye okunabilirdi.

    Karsilastirma ENGELLENMEZ (ayar secimi de bu yolla yapiliyor) ama
    farkin AYARDAN geldigi yazilir.
    """

    def _rapor(self, tmp, ad, cfg, deger=0.5):
        yol = os.path.join(tmp, ad)
        with io.open(yol, 'w', encoding='utf-8') as f:
            json.dump({
                'model': 'd=384 blok=6',
                'config': cfg,
                'report': {'metric_version': METRIC_VERSION, 'copy_bleu': deger,
                           'copy_bleu_std': 0.1, 'qa_score': deger,
                           'qa_score_std': 0.1},
                'items': [{'q': 's1', 'copy_bleu': deger, 'qa_score': deger},
                          {'q': 's2', 'copy_bleu': deger, 'qa_score': deger}],
            }, f, ensure_ascii=False)
        return yol

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix='ayarlı_')
        self.cfg = {'temperature': 0.7, 'top_k': 10, 'rep_penalty': 0.4,
                    'knowledge_bias': 1.2, 'max_len': None, 'rag': True,
                    'natural': 5, 'seed': 7}

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _karsilastir_ve_yakala(self, a, b):
        import io as _io
        import contextlib
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            compare_reports(a, b, 'copy_bleu')
        return buf.getvalue()

    def test_ayni_ayarda_uyari_yok(self):
        a = self._rapor(self.tmp, 'a.json', dict(self.cfg))
        b = self._rapor(self.tmp, 'b.json', dict(self.cfg), 0.4)
        cikti = self._karsilastir_ve_yakala(a, b)
        self.assertNotIn('DECODING AYARLARI FARKLI', cikti)

    def test_farkli_ayarda_uyari_var(self):
        a = self._rapor(self.tmp, 'a.json', dict(self.cfg))
        bozuk = dict(self.cfg, knowledge_bias=0.0)
        b = self._rapor(self.tmp, 'b.json', bozuk, 0.6)
        cikti = self._karsilastir_ve_yakala(a, b)
        self.assertIn('DECODING AYARLARI FARKLI', cikti)
        self.assertIn('knowledge_bias', cikti)

    def test_anlamli_fark_ayar_kaynakli_diyi_soyleniyor(self):
        """Anlamli fark ciktiginda sonuc 'model secimi' diye sunulmamali."""
        a = self._rapor(self.tmp, 'a.json', dict(self.cfg), 0.5)
        b = self._rapor(self.tmp, 'b.json', dict(self.cfg), 0.9)
        # ayni ayar, ama sorularin degerleri farkli -> anlamli fark uretilmez.
        # Bu yuzden ayari da degistirip degeri de degistiriyoruz.
        bozuk = dict(self.cfg, top_k=40)
        b = self._rapor(self.tmp, 'b2.json', bozuk, 0.9)
        cikti = self._karsilastir_ve_yakala(a, b)
        self.assertIn('DECODING AYARLARI FARKLI', cikti)
        if 'AYARDAN' in cikti or 'AYARINDAN' in cikti:
            self.assertIn('model secimi degil', cikti)


class TestButceBoslukHatasi(unittest.TestCase):
    """29.09: eval_llm degerlendirme kumesini BOS birakti, metrikler 0.000.

    MAX_PAIRS ham modul sabiti 0'dir ('0 = otomatik', coz_max_pairs). 0'i
    load_pairs'a vermek 0 cift dondurur -> val kumesi bos -> tum skorlar
    sessizce 0.000 basilir. Gozle gorunmez: rapor yine uretilir.

    Kanit (bu hatadan sonra yapilan ilk olcum):
        degerlendirme seti: 0 cift (rag=False)
        ALTIN KAPSAMA  (gold_recall)   : 0.000
        QA skoru     (0-1, alaka-ust) : 0.000
    """

    def test_hamsabit_kullanilmiyor(self):
        """_load_items icinde MAX_PAIRS sabiti GECMEMELI."""
        import eval_llm
        import inspect
        kaynak = inspect.getsource(eval_llm._load_items)
        self.assertNotIn('max_pairs=MAX_PAIRS', kaynak,
                         'ham MAX_PAIRS (0) kullanildi -> bos degerlendirme')
        self.assertIn('coz_max_pairs', kaynak,
                      'egitimle ayni butce cozumlemesi kullanilmalı')

    def test_butce_cozumlemesi_egitimle_ayni(self):
        from train_llm import MAX_PAIRS, coz_max_pairs
        self.assertEqual(MAX_PAIRS, 0,
                         'bu test MAX_PAIRS=0 varsayimina dayanir')
        self.assertGreater(coz_max_pairs(yaz=False), 0,
                           'coz_max_pairs bos butce dondurdu')

    def test_bos_kume_sessizce_gecmiyor(self):
        """Bos val kumesi hata olmalı, 0.000 metrik degil."""
        import eval_llm
        import inspect
        kaynak = inspect.getsource(eval_llm._load_items)
        self.assertIn('raise SystemExit', kaynak,
                      'bos degerlendirme kumesi sessizce 0 basiyor')

    def test_kaynak_kodu_bos_kume_kontrolu_iceriyor(self):
        with io.open(os.path.join(BASE, 'eval_llm.py'), encoding='utf-8') as f:
            kaynak = ' '.join(f.read().split())
        self.assertIn('degerlendirme kumesi BOS', kaynak)


if __name__ == '__main__':
    unittest.main()
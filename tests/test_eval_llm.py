"""eval_llm metrikleri ve ornekleme raporu icin regresyon testleri.

Saf metrikler (bleu, rep_rate, topic_overlap...) model gerektirmez; rapor
hatti stub modelle dogrulanir. Egitilmis llm_model.json varsa kucuk bir
entegrasyon testi de kosulur (model yoksa atlanir).
"""
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from eval_llm import (STOPWORDS, _measure, aggregate_report, bleu, distinct_ratio,
                      prec1, rep_rate, sample_report, tokenize, topic_overlap)


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
                  'topic_q', 'topic_mean', 'length_ratio', 'avg_word_len',
                  'gen_index'):
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


if __name__ == '__main__':
    unittest.main()
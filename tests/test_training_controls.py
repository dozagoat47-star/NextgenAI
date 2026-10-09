import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import train_llm


class _Tokenizer:
    bos_id, sep_id, eos_id = 1, 2, 3

    def encode(self, _text):
        return [4]

    def vocab(self):
        return {str(i): i for i in range(5)}


class TestPairBudgetArgument(unittest.TestCase):
    def test_prepare_data_limits_balanced_pairs(self):
        knowledge = [('bilgi sorusu %d' % i, 'bilgi yaniti %d' % i)
                     for i in range(10)]
        conversation = [('sohbet sorusu %d' % i, 'sohbet yaniti %d' % i)
                        for i in range(10)]
        knowledge_contexts = {ctx for ctx, _ in knowledge}

        with mock.patch('train_llm.coz_max_pairs', return_value=20), \
                mock.patch('train_llm.load_pairs', return_value=knowledge), \
                mock.patch('train_llm.load_chatgrow_pairs',
                           return_value=conversation), \
                mock.patch('train_llm._knowledge_contexts',
                           return_value=knowledge_contexts):
            data = train_llm.prepare_data(
                False, NATURAL=0, tokenizer=_Tokenizer(), max_ctx_len=32,
                max_seq_len=32, chatgrow_path=[], limit_pairs=4,
                intents_path='unused.json')
            natural_data = train_llm.prepare_data(
                False, NATURAL=5, tokenizer=_Tokenizer(), max_ctx_len=32,
                max_seq_len=32, chatgrow_path=[], limit_pairs=4,
                intents_path='unused.json')

        self.assertEqual(len(data['tr']) + len(data['va']), 4)
        self.assertEqual(data['base_data_fingerprint'],
                         natural_data['base_data_fingerprint'])

    def test_cli_forwards_limit_pairs_to_prepare_data(self):
        result = {'vocab': [], 'tokenizer': None, 'tr': [], 'va': []}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(train_llm, 'BASE', tmp), \
                    mock.patch.object(train_llm, 'SAVE_DIR', tmp), \
                    mock.patch.object(train_llm, 'load_tokenizer',
                                      return_value=None), \
                    mock.patch.object(train_llm, 'prepare_data',
                                      return_value=result) as prepare, \
                    mock.patch.object(sys, 'argv', [
                        'train_llm.py', '--dry-run', '--limit-pairs', '4']):
                self.assertEqual(train_llm.main(), 0)

        self.assertEqual(prepare.call_args.kwargs['limit_pairs'], 4)

    def test_fixed_step_run_requires_fresh_start(self):
        with mock.patch.object(sys, 'argv', [
                'train_llm.py', '--max-steps', '3']):
            with self.assertRaises(SystemExit) as exc:
                train_llm.main()

        self.assertEqual(exc.exception.code, 2)

    @unittest.skipIf(train_llm.torch is None, 'PyTorch is not installed')
    def test_fixed_step_run_performs_exact_update_count(self):
        train = [
            (np.array([1, 4, 2, 5, 3, 0, 0, 0]),
             np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.float32))
            for _ in range(3)
        ]
        val = [
            (np.array([1, 4, 2, 6, 3, 0, 0, 0]),
             np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.float32))
        ]
        data = {'vocab': [str(i) for i in range(8)],
                'tokenizer': None, 'tr': train, 'va': val}

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = os.path.join(tmp, 'export')
            argv = [
                'train_llm.py', '--fresh', '--max-steps', '2',
                '--batch-size', '1', '--d-model', '8', '--num-blocks', '1',
                '--num-heads', '2', '--ff-mult', '2', '--max-ctx-len', '2',
                '--max-seq-len', '8', '--export-dir', export_dir,
            ]
            with mock.patch.object(train_llm, 'BASE', tmp), \
                    mock.patch.object(train_llm, 'SAVE_DIR', tmp), \
                    mock.patch.object(train_llm, 'load_tokenizer',
                                      return_value=None), \
                    mock.patch.object(train_llm, 'prepare_data',
                                      return_value=data), \
                    mock.patch.object(sys, 'argv', argv):
                self.assertEqual(train_llm.main(), 0)

            checkpoint = train_llm.torch.load(
                os.path.join(tmp, 'llm_ckpt.pt'),
                map_location='cpu', weights_only=True)
            exported_weights_exist = os.path.isfile(
                os.path.join(export_dir, 'llm_model_weights.npz'))
            metadata_path = os.path.join(export_dir, 'training_metadata.json')
            with open(metadata_path, encoding='utf-8') as metadata_file:
                metadata = train_llm.json.load(metadata_file)

        self.assertEqual(checkpoint['step'], 2)
        self.assertTrue(exported_weights_exist)
        self.assertEqual(metadata['optimizer_updates'], 2)
        self.assertEqual(metadata['max_steps'], 2)


if __name__ == '__main__':
    unittest.main()

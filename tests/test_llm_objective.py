"""Regression tests for causal next-token training targets."""
import os
import sys
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import LLM, encode_llm
from train_llm import _encode_training_pair, llm_loss, masked_acc

requires_torch = unittest.skipIf(
    torch is None, 'PyTorch kurulu degil => objective testi skip')

VOCAB = ['<PAD>', '<BOS>', '<SEP>', '<EOS>', 'm', 'e', 'r', 'h', 'a',
         'b', ' ', 'n', 's', 'i', 'l', 'y', 'q', 'u', 'g', 'z', 'd', 'o']


class _StubTokenizer:
    bos_id = 1
    sep_id = 2
    eos_id = 3
    tokens = {'q': 4, 'a': 5, 'b': 6}

    def encode(self, text):
        return [self.tokens[char] for char in text]


class TestTrainingPairMask(unittest.TestCase):
    def test_mask_supervises_separator_to_first_response_token(self):
        ids, mask = _encode_training_pair(_StubTokenizer(), 'q', 'ab')
        self.assertEqual(ids, [1, 4, 2, 5, 6, 3])
        self.assertEqual(mask, [0, 0, 1, 1, 1, 0])


@requires_torch
class TestAutoregressiveObjective(unittest.TestCase):
    def test_loss_penalizes_a_wrong_first_response_token(self):
        ids, mask = _encode_training_pair(_StubTokenizer(), 'q', 'ab')
        targets = torch.tensor([ids])
        correct_logits = torch.zeros(1, len(ids), len(VOCAB))
        incorrect_logits = correct_logits.clone()
        for position in (2, 3, 4):
            correct_logits[0, position, ids[position + 1]] = 12.0
            incorrect_logits[0, position, ids[position + 1]] = 12.0
        incorrect_logits[0, 2, ids[3]] = 0.0
        incorrect_logits[0, 2, ids[3] + 1] = 12.0

        response_mask = torch.tensor([mask], dtype=torch.float32)
        self.assertLess(
            llm_loss(correct_logits, targets, response_mask).item(),
            llm_loss(incorrect_logits, targets, response_mask).item())

    def test_loss_and_accuracy_score_the_next_response_token(self):
        model = LLM(VOCAB, d_model=8, num_blocks=1, num_heads=2,
                    max_ctx_len=10, max_seq_len=24, seed=7)
        encoded = [encode_llm(model, 'merhaba', 'iyiyim'),
                   encode_llm(model, 'nasilsin', 'merhaba')]
        targets = torch.from_numpy(np.stack([seq for seq, _ in encoded]))
        response_mask = torch.from_numpy(
            np.stack([mask for _, mask in encoded]))
        logits = torch.zeros(2, 24, len(VOCAB))

        for batch_index in range(targets.size(0)):
            for position in range(targets.size(1) - 1):
                next_token = int(targets[batch_index, position + 1])
                logits[batch_index, position, next_token] = 12.0

        self.assertLess(llm_loss(logits, targets, response_mask).item(), 0.01)
        self.assertEqual(masked_acc(logits, targets, response_mask), 1.0)


if __name__ == '__main__':
    unittest.main()

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
from train_llm import llm_loss, masked_acc

requires_torch = unittest.skipIf(
    torch is None, 'PyTorch kurulu degil => objective testi skip')

VOCAB = ['<PAD>', '<BOS>', '<SEP>', '<EOS>', 'm', 'e', 'r', 'h', 'a',
         'b', ' ', 'n', 's', 'i', 'l', 'y', 'q', 'u', 'g', 'z', 'd', 'o']


@requires_torch
class TestAutoregressiveObjective(unittest.TestCase):
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

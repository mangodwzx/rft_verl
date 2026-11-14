"""Gradient invariants for the real losses, including distributed microbatches."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest


HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "verl"))


@unittest.skipUnless(HAS_STACK, "Requires the pinned torch/verl environment")
class AlgorithmGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from math_rft import algorithms
        cls.torch = torch
        cls.algorithms = algorithms

    def config(self, *, batch=6, dp=1, scale=4096):
        return SimpleNamespace(global_batch_info={"global_batch_size": batch, "dp_size": dp,
                                                  "loss_scale_factor": scale},
                               clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)

    def batch(self):
        t = self.torch
        lengths = t.tensor([2, 5, 3, 4, 1, 3])
        mask = (t.arange(7)[None, :] < lengths[:, None]).to(t.float64)
        # Mixed-success, all-success and all-failure groups. Padding is excluded
        # by position, while the last real (EOS) position remains trainable.
        scores = t.tensor([1., 0., 1., 1., 0., 0.], dtype=t.float64)
        rewards = t.zeros_like(mask)
        rewards[t.arange(6), lengths - 1] = scores
        weights, _ = self.algorithms.question_success_weights(
            rewards, mask, index=["mixed", "mixed", "success", "success", "failure", "failure"])
        return mask, weights

    def test_success_weight_and_global_denominator_include_failed_responses(self):
        t = self.torch
        mask, weights = self.batch()
        logp = t.full_like(mask, -2, requires_grad=True)
        loss, _ = self.algorithms.rft_nll(None, logp, weights, mask, config=self.config())
        loss.backward()
        expected = -t.tensor([2., 0., 1., 1., 0., 0.])[:, None] * mask / (6 * 4096)
        t.testing.assert_close(logp.grad, expected)
        self.assertNotEqual(logp.grad[0, 1].item(), 0)  # true EOS
        self.assertEqual(logp.grad[0, 2].item(), 0)  # padding, even if token ID equals EOS
        self.assertEqual(logp.grad[1].abs().sum().item(), 0)

    def test_dynamic_padding_microbatch_and_rank_averaging_preserve_gradient(self):
        t = self.torch
        mask, weights = self.batch()
        full = t.full_like(mask, -2, requires_grad=True)
        loss, _ = self.algorithms.rft_nll(None, full, weights, mask, config=self.config())
        loss.backward()
        micro = t.full_like(mask, -2, requires_grad=True)
        # Two DP ranks, each processing 3 independent, differently padded microbatches.
        for index in range(6):
            width = int(mask[index].sum()) + index % 2
            part, _ = self.algorithms.rft_nll(None, micro[index:index+1, :width],
                                             weights[index:index+1, :width], mask[index:index+1, :width],
                                             config=self.config(dp=2))
            (part / 2).backward()  # FSDP averages rank gradients.
        t.testing.assert_close(micro.grad, full.grad)

    def test_legacy_reduction_is_explicit_and_still_reproducible(self):
        t = self.torch
        mask, weights = self.batch()
        logp = t.full_like(mask, -2, requires_grad=True)
        loss, _ = self.algorithms.rft_nll(None, logp, weights, mask,
                                         loss_agg_mode="seq-mean-token-mean", config=self.config(scale=None))
        loss.backward()
        t.testing.assert_close(logp.grad, -weights / mask.sum(-1, keepdim=True) / 6)

    def test_fixed_scale_cannot_fall_back_to_microbatch_width(self):
        t = self.torch
        mask, weights = self.batch()
        logp = t.full_like(mask, -2, requires_grad=True)
        for scale in (None, 0, -1, float("nan"), float("inf"), True):
            for function in (self.algorithms.rft_nll, self.algorithms.grpo_clip):
                with self.subTest(scale=scale, function=function.__name__), self.assertRaises(ValueError):
                    function(logp.detach(), logp, weights, mask, config=self.config(scale=scale))

    def test_grpo_uses_same_fixed_scale_for_positive_and_negative_advantages(self):
        t = self.torch
        mask, _ = self.batch()
        advantages = t.tensor([1., -1., 0., 0., 0., 0.])[:, None] * mask
        logp = t.full_like(mask, -2, requires_grad=True)
        loss, metrics = self.algorithms.grpo_clip(logp.detach(), logp, advantages, mask, config=self.config())
        loss.backward()
        t.testing.assert_close(logp.grad, -advantages / (6 * 4096))
        self.assertEqual(metrics["pg_clipfrac"], 0)


if __name__ == "__main__":
    unittest.main()

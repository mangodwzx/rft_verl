"""verl extensions for online question-balanced RFT and clipped GRPO.

Imported through ``actor_rollout_ref.model.external_lib`` in the controller and
each Ray worker. Never compute success counts again inside a shuffled minibatch.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from verl.trainer.ppo.core_algos import agg_loss, register_adv_est, register_policy_loss

from math_rft.entropy_control import install_optional_entropy


def _groups(token_level_rewards, response_mask, index):
    if index is None or len(index) != token_level_rewards.shape[0]:
        raise ValueError("The full rollout batch must contain one uid per response")
    scores = token_level_rewards.float().sum(-1)
    if not torch.isfinite(scores).all() or not ((scores == 0) | (scores == 1)).all():
        raise ValueError("These estimators require strictly binary, non-KL-shaped rewards")
    if torch.any(response_mask.sum(-1) == 0):
        raise ValueError("A generated response has no scored tokens; keep EOS in response_mask")
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[str(uid)].append(row)
    sizes = {len(rows) for rows in groups.values()}
    if len(sizes) != 1 or next(iter(sizes)) < 2:
        raise ValueError("Expected complete, equally sized rollout groups with m >= 2")
    return scores, groups


@register_adv_est("math_rft_question")
def question_success_weights(token_level_rewards, response_mask, index=None, config=None, **kwargs):
    """Return m*r/C; averaging all B*m trajectories gives (1/B)sum_i mean_success_i.

    Failed groups contribute zero but remain in the denominator. Successful
    groups (including C=m) are retained. The tensor is frozen before shuffling.
    """
    del config, kwargs
    with torch.no_grad():
        scores, groups = _groups(token_level_rewards, response_mask, index)
        weights = torch.zeros_like(scores)
        for rows in groups.values():
            ids = torch.tensor(rows, device=scores.device, dtype=torch.long)
            successes = scores[ids].sum()
            weights[ids] = scores[ids] * len(rows) / successes.clamp_min(1)
        weights = weights.unsqueeze(-1) * response_mask
    return weights, weights.clone()


@register_policy_loss("math_rft_nll")
def rft_nll(old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="seq-mean-token-mean", config=None,
            rollout_is_weights=None, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
    """Actual weighted supervised NLL; no PPO ratio or clipping is used.

    Do not mask unsuccessful responses out of ``response_mask``: that changes
    the global denominator. ``advantages`` already contains detached m*r/C.
    """
    del old_log_prob, kwargs
    if config is None or not config.global_batch_info.get("global_batch_size"):
        raise ValueError("RFT requires verl's original global rollout batch denominator")
    if rollout_is_weights is not None:
        raise ValueError("Rollout IS/rejection correction must be disabled for this RFT definition")
    if loss_agg_mode != "seq-mean-token-mean":
        raise ValueError("Main RFT averages tokens per response, then averages the original rollout batch")
    weights = advantages.detach()
    if torch.any(weights < 0):
        raise ValueError("RFT success weights cannot be negative")
    loss = agg_loss(loss_mat=-log_prob * weights, loss_mask=response_mask,
                    loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    return loss, {"rft/mean_weight": weights.sum().detach().item() / response_mask.sum().item()}


@register_policy_loss("math_grpo_clip")
def grpo_clip(old_log_prob, log_prob, advantages, response_mask,
              loss_agg_mode="seq-mean-token-mean", config=None,
              rollout_is_weights=None, **kwargs):
    """Standard two-sided PPO clipped objective, without upstream dual clipping."""
    del kwargs
    if config is None:
        raise ValueError("Actor config is required")
    if rollout_is_weights is not None:
        raise ValueError("This controlled comparison disables rollout correction")
    low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    log_ratio = (log_prob - old_log_prob).clamp(-20, 20)
    ratio = log_ratio.exp()
    losses = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1-low, 1+high))
    loss = agg_loss(loss_mat=losses, loss_mask=response_mask,
                    loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    denom = response_mask.sum().clamp_min(1)
    return loss, {
        "pg_clipfrac": (((ratio < 1-low) | (ratio > 1+high)) * response_mask).sum().detach().item() / denom.item(),
        "ppo_kl": ((-log_ratio * response_mask).sum() / denom).detach().item(),
    }


def _install_qwen35_text_only_patch():
    """Freeze unused vision tower before FSDP wrapping at the pinned verl API.

    The current unified FSDP engine does not consume actor.freeze_vision_tower.
    This small, version-guarded hook makes that policy explicit and reports the
    actual trainable parameter count. No weights are removed from checkpoints.
    """
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original = FSDPEngine._build_module
    if getattr(original, "_math_rft_patched", False):
        return

    def build_module(engine):
        module = original(engine)
        if getattr(module.config, "model_type", "") in {"qwen3_5", "qwen3_5_moe"}:
            vision = getattr(getattr(module, "model", None), "visual", None)
            if vision is None:
                vision = getattr(module, "visual", None)
            if vision is None:
                raise RuntimeError("Qwen3.5 vision tower path changed; audit the pinned architecture")
            vision.requires_grad_(False)
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in module.parameters())
        print(f"MATH_RFT_PARAMETERS trainable={trainable} total={total} frozen={total-trainable}", flush=True)
        return module

    build_module._math_rft_patched = True
    FSDPEngine._build_module = build_module


_install_qwen35_text_only_patch()
# external_lib is loaded by the parent, Ray controller and model workers.
# The controller imports it before constructing its trainer. Entropy-on keeps
# the upstream behavior; entropy-off removes only the optional statistics.
install_optional_entropy()

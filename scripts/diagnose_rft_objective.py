"""Exactly enumerate on-policy update gradients for two complete responses.

The first token selects a short or long branch, with p(short)=sigmoid(theta).
All remaining tokens, including the final EOS, are deterministic. Thus complete
response probabilities are p and 1-p, while their lengths can differ. The EOS
position is included in every loss mask; no answer token is substituted.

For all-correct responses, the legacy RFT expected loss gradient is
    p*(1-p)*(1/long_length - 1/short_length).
Gradient descent therefore increases the short-branch probability. A fixed
constant S and sequence-summed NLL instead give E[-grad log pi / S] = 0.
The expectation is over frozen rollout probabilities: this is an expected
training update, NOT differentiation through the sampling distribution.

More generally, with binary rewards and q=P_pi(correct)>0, retaining the
question-balanced m*r/C weights gives the fixed-scale expected loss gradient
    -[1-(1-q)**m] / (S*q) * grad q.
Conditioned on C>0, the mean of the successful samples has expectation under
pi(response | correct). The score-function identity then gives grad q / q.
Thus negative failure weights are not required to remove this length drift.

The finite-batch token-mean alternative is included because its random token
denominator remains correlated with the samples and need not remove the drift.
All binary reward cases are enumerated. This proves a mechanism in a toy policy,
not that it is the only cause of the observed language-model training collapse.

Run with the standard library, or add --verify-registered-loss to independently
check the calculations with PyTorch autograd and the actual registered losses.
The latter imports the pinned training environment but starts no models or GPUs.
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace


OBJECTIVES = (
    "rft_per_response", "rft_fixed", "rft_batch_token_mean",
    "grpo_per_response", "grpo_fixed",
)
REWARD_CASES = {"both_correct": (1, 1), "short_correct": (1, 0),
                "long_correct": (0, 1), "both_incorrect": (0, 0)}


def validate(probability: float, lengths: tuple[int, int], group_size: int, fixed_scale: float) -> None:
    if not 0 < probability < 1:
        raise ValueError("Short-branch probability must be strictly between 0 and 1")
    if any(type(length) is not int or length < 2 for length in lengths):
        raise ValueError("Each response needs its branch token and EOS (length >= 2)")
    if type(group_size) is not int or group_size < 2:
        raise ValueError("Group size must be an integer >= 2")
    if not math.isfinite(fixed_scale) or fixed_scale <= 0:
        raise ValueError("The predeclared fixed scale must be positive and finite")


def batch_gradient(count_short: int, probability: float, lengths: tuple[int, int],
                   rewards: tuple[int, int], group_size: int, fixed_scale: float,
                   objective: str) -> float:
    """Gradient of the loss on one frozen sampled group, evaluated on-policy."""
    counts = (count_short, group_size - count_short)
    score_gradients = (1 - probability, -probability)  # grad_theta log pi(branch)
    successes = sum(count * reward for count, reward in zip(counts, rewards, strict=True))
    if objective.startswith("rft_"):
        weights = tuple(group_size * reward / max(1, successes) for reward in rewards)
    elif objective.startswith("grpo_"):
        mean = successes / group_size
        std = math.sqrt(sum(count * (reward - mean) ** 2
                            for count, reward in zip(counts, rewards, strict=True)) / (group_size - 1))
        weights = tuple((reward - mean) / (std + 1e-6) for reward in rewards)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    numerators = [-count * weight * grad
                  for count, weight, grad in zip(counts, weights, score_gradients, strict=True)]
    if objective.endswith("_per_response"):
        return math.fsum(value / length for value, length in zip(numerators, lengths, strict=True)) / group_size
    if objective.endswith("_fixed"):
        return math.fsum(numerators) / (group_size * fixed_scale)
    if objective == "rft_batch_token_mean":
        return math.fsum(numerators) / sum(count * length for count, length in zip(counts, lengths, strict=True))
    raise ValueError(f"Unknown objective: {objective}")


def expected_gradient(probability: float = 0.5, lengths: tuple[int, int] = (2, 8),
                      rewards: tuple[int, int] = (1, 1), group_size: int = 8,
                      fixed_scale: float = 4096, objective: str = "rft_per_response") -> float:
    validate(probability, lengths, group_size, fixed_scale)
    if objective not in OBJECTIVES or rewards not in REWARD_CASES.values():
        raise ValueError("Use a listed objective and binary rewards for both branches")
    return math.fsum(
        math.comb(group_size, count) * probability ** count * (1 - probability) ** (group_size - count)
        * batch_gradient(count, probability, lengths, rewards, group_size, fixed_scale, objective)
        for count in range(group_size + 1)
    )


def verify_registered_losses(probability: float = 0.5, lengths: tuple[int, int] = (2, 8),
                             group_size: int = 8, fixed_scale: float = 4096) -> dict:
    """Compare exact formulas with real autograd for every group/reward case.

    Legacy and fixed RFT/GRPO call their registered policy-loss functions, with
    advantages from their actual registered estimators. The diagnostic token-
    mean candidate calls the same upstream agg_loss, since it is not a supported
    RFT training definition. All sampled probabilities are ordinary floats.
    """
    import torch
    import math_rft.algorithms  # noqa: F401 -- registers the audited losses
    from verl.trainer.ppo.core_algos import agg_loss, get_adv_estimator_fn, get_policy_loss_fn

    validate(probability, lengths, group_size, fixed_scale)
    totals = {case: dict.fromkeys(OBJECTIVES, 0.0) for case in REWARD_CASES}
    maximum_batch_error = 0.0
    for case, rewards in REWARD_CASES.items():
        for count in range(group_size + 1):
            branches = [0] * count + [1] * (group_size - count)
            for objective in OBJECTIVES:
                theta = torch.tensor(math.log(probability / (1 - probability)), dtype=torch.float64,
                                     requires_grad=True)
                branch_log_probs = (torch.nn.functional.logsigmoid(theta),
                                    torch.nn.functional.logsigmoid(-theta))
                # Continuation and EOS tokens have probability 1; padding is masked.
                log_probs = torch.stack([torch.stack([branch_log_probs[branch]]
                                        + [theta * 0] * (max(lengths) - 1)) for branch in branches])
                mask = torch.tensor([[float(token < lengths[branch]) for token in range(max(lengths))]
                                     for branch in branches], dtype=torch.float64)
                token_rewards = torch.zeros_like(mask)
                for row, branch in enumerate(branches):
                    token_rewards[row, lengths[branch] - 1] = rewards[branch]
                is_rft = objective.startswith("rft_")
                estimator = get_adv_estimator_fn("math_rft_question" if is_rft else "grpo")
                advantages, _ = estimator(token_rewards, mask, index=["same_question"] * group_size)
                reduction = ("seq-mean-token-sum-norm" if objective.endswith("_fixed")
                             else "seq-mean-token-mean")
                info = dict(dp_size=1, global_batch_size=group_size,
                            batch_num_tokens=int(mask.sum().item()), loss_scale_factor=fixed_scale)
                config = SimpleNamespace(global_batch_info=info, loss_scale_factor=fixed_scale,
                                         clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
                if objective == "rft_batch_token_mean":
                    loss = agg_loss(-log_probs * advantages, mask, "token-mean", **info)
                else:
                    loss, _ = get_policy_loss_fn("math_rft_nll" if is_rft else "math_grpo_clip")(
                        old_log_prob=log_probs.detach(), log_prob=log_probs, advantages=advantages,
                        response_mask=mask, loss_agg_mode=reduction, config=config)
                actual = torch.autograd.grad(loss, theta)[0].item()
                expected = batch_gradient(count, probability, lengths, rewards, group_size, fixed_scale, objective)
                error = abs(actual - expected)
                maximum_batch_error = max(maximum_batch_error, error)
                if error > 1e-7:
                    raise AssertionError(f"{case}/{objective}/short_count={count}: {actual} != {expected}")
                mass = math.comb(group_size, count) * probability ** count * (1 - probability) ** (group_size - count)
                totals[case][objective] += mass * actual
    return {"all_batches_match": True, "maximum_batch_gradient_error": maximum_batch_error,
            "expected_loss_gradients": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-short", type=float, default=0.5)
    parser.add_argument("--short-length", type=int, default=2)
    parser.add_argument("--long-length", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--fixed-scale", type=float, default=4096)
    parser.add_argument("--verify-registered-loss", action="store_true")
    args = parser.parse_args()
    lengths = (args.short_length, args.long_length)
    try:
        validate(args.prob_short, lengths, args.group_size, args.fixed_scale)
    except ValueError as exc:
        parser.error(str(exc))
    kwargs = dict(probability=args.prob_short, lengths=lengths, group_size=args.group_size,
                  fixed_scale=args.fixed_scale)
    result = {
        "policy": {**kwargs, "lengths_include_eos": True, "sample_probabilities_are_detached": True},
        "reading": "Negative loss gradient increases short-answer probability under gradient descent.",
        "expected_loss_gradients": {
            case: {objective: expected_gradient(rewards=rewards, objective=objective, **kwargs)
                   for objective in OBJECTIVES} for case, rewards in REWARD_CASES.items()},
        "all_correct_legacy_analytic_gradient":
            args.prob_short * (1 - args.prob_short) * (1 / lengths[1] - 1 / lengths[0]),
        "limitations": ["Exact toy-policy update expectation, not a causal diagnosis of the entire real run.",
                        "Finite individual batches can have a nonzero fixed-scale update even when all rewards are 1.",
                        "Changing the fixed scale changes gradient magnitude; it must stay fixed across steps."],
    }
    if args.verify_registered_loss:
        result["registered_loss_check"] = verify_registered_losses(**kwargs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

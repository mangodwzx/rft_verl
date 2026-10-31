"""Optional entropy reporting for the pinned verl trainer.

The pinned trainer honors ``actor.calculate_entropy`` during updates, but forces
entropy on while recomputing old log probabilities and then unconditionally
consumes that metric in ``fit``. There is no callback for either operation.

Adapt only those two small sections at runtime, leaving the vendored checkout
clean. Full method fingerprints and exact replacement checks make this fail
closed on upstream changes. The original methods are compiled with their own
globals; the loss, optimizer, old log probabilities and rollout are untouched.
This adapter covers the non-fused output path used by this project's profiles.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap


_SOURCE_SHA256 = {
    "_compute_old_log_prob": "37936dd4c6debf300c4f0ef85919990ae627b874548de109403303237fec4ae2",
    "fit": "b5678ef1a7c07ab61e2b2cd37233a9c980c6dd71eeeee9d1a02a6f829d5ffd32",
}


def entropy_requested(actor_config) -> bool:
    """Match verl's update-stage rule, including nonzero entropy regularization."""
    return bool(actor_config.calculate_entropy or actor_config.entropy_coeff != 0.0)


def old_log_prob_metrics(old_log_prob, batch, actor_config, mfu, aggregate):
    """Consume an optional real entropy tensor; never report a fabricated zero."""
    metrics = {"perf/mfu/actor_infer": mfu}
    if entropy_requested(actor_config):
        # A requested but absent tensor is an error, rather than a hidden worker failure.
        entropy = old_log_prob.batch.pop("entropys")
        value = aggregate(
            loss_mat=entropy,
            loss_mask=batch.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        metrics["actor/entropy"] = value.detach().item()
    elif "entropys" in old_log_prob.batch:
        raise RuntimeError("Entropy was returned while entropy reporting was disabled")
    return metrics


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("Pinned verl entropy adapter no longer matches the upstream source")
    return source.replace(old, new, 1)


def adapt_source(name: str, source: str) -> str:
    """Return an audited method with optional entropy, or refuse unknown source."""
    source = textwrap.dedent(source)
    if name not in _SOURCE_SHA256 or hashlib.sha256(source.encode()).hexdigest() != _SOURCE_SHA256[name]:
        raise RuntimeError(f"Unaudited verl method for entropy adapter: {name}")
    if name == "_compute_old_log_prob":
        source = _replace_once(
            source,
            "    tu.assign_non_tensor(\n",
            "    calculate_entropy = _math_rft_entropy_requested(self.config.actor_rollout_ref.actor)\n"
            "    tu.assign_non_tensor(\n",
        )
        source = _replace_once(source, "calculate_entropy=True,", "calculate_entropy=calculate_entropy,")
        source = _replace_once(
            source,
            '    entropy = tu.get(output, "entropy")\n',
            '    entropy = tu.get(output, "entropy") if calculate_entropy else None\n',
        )
        source = _replace_once(
            source,
            "    entropy = no_padding_2_padding(entropy, batch_td)\n",
            "    if calculate_entropy:\n"
            "        entropy = no_padding_2_padding(entropy, batch_td)\n",
        )
        return _replace_once(
            source,
            '    result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}\n',
            '    result = {"old_log_probs": log_probs.float()}\n'
            '    if calculate_entropy:\n'
            '        result["entropys"] = entropy.float()\n',
        )
    old = '''                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
'''
    new = '''                        metrics.update(_math_rft_old_log_prob_metrics(
                            old_log_prob, batch, self.config.actor_rollout_ref.actor,
                            old_log_prob_mfu, agg_loss,
                        ))
'''
    return _replace_once(source, old, new)


def install_optional_entropy() -> None:
    """Install once per controller process, before constructing RayPPOTrainer."""
    from math_rft.compatibility import check_verl_revision
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer, "_math_rft_optional_entropy", False):
        return
    check_verl_revision()
    adapted = {}
    for name in _SOURCE_SHA256:
        original = getattr(RayPPOTrainer, name)
        source = adapt_source(name, inspect.getsource(original))
        namespace = dict(original.__globals__)
        namespace.update(
            _math_rft_entropy_requested=entropy_requested,
            _math_rft_old_log_prob_metrics=old_log_prob_metrics,
        )
        # Only fingerprinted, installed upstream source is compiled, never user input.
        filename = f"{__file__}:adapted_{name}"
        exec(compile(source, filename, "exec"), namespace)
        adapted[name] = functools.wraps(original)(namespace[name])
    # Validate/compile both methods before changing either method on the class.
    for name, method in adapted.items():
        setattr(RayPPOTrainer, name, method)
    RayPPOTrainer._math_rft_optional_entropy = True

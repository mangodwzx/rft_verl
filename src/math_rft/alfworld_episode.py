"""Shared raw-text ALFWorld protocol for training and independent evaluation.

Only assistant action tokens receive loss. Observations are appended as context
without retokenizing any sampled prefix. Importing this module starts no runtime.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Awaitable, Callable

PROMPT_VERSION = "alfworld_raw_actions_v1"
GRADER_VERSION = "alfworld_textworld_success_v1"
STOP_STRINGS = ["\n"]
INSTRUCTION = (
    "You are an agent in a text household environment. Complete the task by interacting with the environment.\n"
    "Output exactly one environment command on one line at each Action: prompt. "
    "Do not write explanations or invent observations. Use object and location names from the observations.\n"
    "Commands include: look; inventory; go to LOCATION; open OBJECT; close OBJECT; "
    "take OBJECT from LOCATION; put OBJECT in/on LOCATION; clean OBJECT with TOOL; "
    "heat OBJECT with TOOL; cool OBJECT with TOOL; use OBJECT.\n"
    "The environment will report the result after each action."
)


def render_prompt(initial_observation: str) -> str:
    if not isinstance(initial_observation, str) or not initial_observation.strip():
        raise ValueError("ALFWorld initial observation must be a nonempty string")
    return f"{INSTRUCTION}\n\nObservation: {initial_observation}\nAction:"


def prompt_fingerprint() -> str:
    return hashlib.sha256(render_prompt("<INITIAL_OBSERVATION>").encode()).hexdigest()


def turn_seed(episode_seed: int, turn_index: int) -> int:
    """Stable independent request seeds, separate from the fixed task state."""
    value = f"alfworld:{episode_seed}:{turn_index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


@dataclass
class GeneratedAction:
    token_ids: list[int]
    log_probs: list[float] | None = None
    finish_reason: str = "stop"


@dataclass
class EpisodeResult:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    score: int
    turns: list[dict[str, Any]]
    termination: str
    text: str
    metrics: dict[str, Any]


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


async def _environment_call(function, *args):
    """Finish a running backend operation before cancellation can close it."""
    operation = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        # Cancelling to_thread cannot stop the backend thread. Wait for it to
        # release its state before run_episode's finally closes the backend.
        try:
            await operation
        except Exception:
            pass  # cancellation remains the primary failure
        raise


async def run_episode(
    task: dict[str, Any],
    tokenizer: Any,
    generate: Callable[[list[int], int, int | None], Awaitable[GeneratedAction]],
    *,
    max_steps: int = 50,
    max_action_tokens: int = 128,
    max_response_tokens: int = 4096,
    max_prompt_tokens: int = 4096,
    seed: int | None = 0,
    env_factory: Callable[..., Any] | None = None,
) -> EpisodeResult:
    """Run one reset-to-terminal attempt; environment/inference errors propagate.

    ``generate`` must preserve the actual sampled token IDs (including the stop
    newline/EOS), stop at the first newline and obey ``max_tokens``. The response
    budget includes actions and inserted observations. An observation that does
    not fit ends the attempt, with no silent history truncation. Terminal state
    observations are logged but need not be fed back to the model.
    """
    for name, value in (("max_steps", max_steps), ("max_action_tokens", max_action_tokens),
                        ("max_response_tokens", max_response_tokens), ("max_prompt_tokens", max_prompt_tokens)):
        _positive_integer(name, value)
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None for the training engine RNG")
    if not isinstance(task, dict):
        raise TypeError("ALFWorld task must be a dictionary")
    if env_factory is None:
        from .alfworld_env import AlfWorldEnvironment
        env_factory = AlfWorldEnvironment
    # The adapter's constructor only holds configuration; reset opens resources.
    # Synchronous construction prevents cancellation losing a newly created env.
    env = env_factory(task, max_steps=max_steps, seed=0)
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    has_logprobs: bool | None = None
    turns: list[dict[str, Any]] = []
    score, termination = 0, "max_steps"
    generation_time, environment_time = 0.0, 0.0
    started = time.monotonic()
    try:
        reset_started = time.monotonic()
        observation = await _environment_call(env.reset)
        environment_time += time.monotonic() - reset_started
        expected = task.get("initial_observation")
        if expected is not None and observation != expected:
            raise ValueError("ALFWorld reset observation differs from the prepared task")
        prompt = render_prompt(observation)
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if not prompt_ids or len(prompt_ids) > max_prompt_tokens:
            raise ValueError(f"ALFWorld prompt has {len(prompt_ids)} tokens, limit={max_prompt_tokens}")
        for turn_index in range(max_steps):
            remaining = max_response_tokens - len(response_ids)
            if remaining < 1:
                termination = "response_token_limit"
                break
            token_limit = min(max_action_tokens, remaining)
            generation_started = time.monotonic()
            action_seed = turn_seed(seed, turn_index) if seed is not None else None
            generated = await generate(prompt_ids + response_ids, token_limit, action_seed)
            generation_time += time.monotonic() - generation_started
            if not isinstance(generated, GeneratedAction):
                raise TypeError("The ALFWorld generator must return GeneratedAction")
            ids = list(generated.token_ids)
            if not ids or len(ids) > token_limit:
                raise RuntimeError("ALFWorld generator returned no tokens or exceeded its token budget")
            if generated.finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"ALFWorld generation did not complete: {generated.finish_reason}")
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("ALFWorld generator returned invalid token IDs")
            current_has_logprobs = generated.log_probs is not None
            if has_logprobs is not None and has_logprobs != current_has_logprobs:
                raise RuntimeError("ALFWorld generation log-probability availability changed within an episode")
            has_logprobs = current_has_logprobs
            if has_logprobs:
                if (len(generated.log_probs) != len(ids)
                        or any(not math.isfinite(value) for value in generated.log_probs)):
                    raise ValueError("ALFWorld action log probabilities must be finite and align with sampled tokens")
                response_logprobs.extend(generated.log_probs)
            response_ids.extend(ids)
            response_mask.extend([1] * len(ids))
            text = tokenizer.decode(ids, skip_special_tokens=True)
            action = text.strip()
            record: dict[str, Any] = {"index": turn_index, "action": action, "text": text,
                                      "token_ids": ids, "seed": action_seed,
                                      "finish_reason": generated.finish_reason}
            turns.append(record)
            eos = tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            complete_action = "\n" in text or ids[-1] in eos_ids
            # The training server coalesces stop/length into one status. Use
            # sampled boundaries for both paths, including a newline/EOS that
            # lands exactly on the requested last token.
            if not complete_action:
                termination = "response_token_limit" if token_limit == remaining else "action_token_limit"
                break
            if not action or "\n" in action or "\r" in action:
                termination = "invalid_action"
                break
            step_started = time.monotonic()
            observation, success, done, info = await _environment_call(env.step, action)
            environment_time += time.monotonic() - step_started
            if (not isinstance(observation, str) or type(success) is not bool
                    or type(done) is not bool or not isinstance(info, dict)):
                raise TypeError("ALFWorld step must return (str, bool, bool, dict)")
            record.update({"observation": observation, "success": success, "done": done, "info": info})
            if success:
                score, termination = 1, "success"
                break
            if done:
                termination = "max_steps" if info.get("truncated", False) else "environment_done"
                break
            if turn_index + 1 == max_steps:
                termination = "max_steps"
                break
            suffix = f"\nObservation: {observation}\nAction:"
            observation_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
            # Leave at least one position for the next model action. Never train
            # a cropped action or change the task by truncating an observation.
            if len(response_ids) + len(observation_ids) >= max_response_tokens:
                termination = "response_token_limit"
                break
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if has_logprobs:
                response_logprobs.extend([0.0] * len(observation_ids))
        return EpisodeResult(
            prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask,
            response_logprobs=response_logprobs if has_logprobs else None,
            score=score, turns=turns, termination=termination,
            text=tokenizer.decode(response_ids, skip_special_tokens=True),
            metrics={"generate_sequences": generation_time, "tool_calls": environment_time,
                     "wall_seconds": time.monotonic() - started, "steps": len(turns),
                     "environment_steps": sum("observation" in turn for turn in turns),
                     "actor_tokens": sum(response_mask), "observation_tokens": response_mask.count(0),
                     "environment_seed": 0},
        )
    finally:
        await _environment_call(env.close)

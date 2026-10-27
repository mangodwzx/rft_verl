"""Shared binary mathematical correctness reward for training and evaluation.

Only the final boxed answer is judged. Infrastructure failures raise; they are
never silently converted into an incorrect model answer. Math-Verify uses
SIGALRM, so call this module from the main thread of Linux worker processes.
"""

from __future__ import annotations

import inspect
import atexit
import asyncio
import importlib
import multiprocessing
import re
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from functools import partial
from typing import Any

GRADER_VERSION = "boxed_math_verify_v1"
_SCORING_POOL: ProcessPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


class GraderError(RuntimeError):
    """Missing/incompatible grader, invalid reference, or unsafe execution mode."""


def extract_boxed(text: str) -> str | None:
    """Extract the last box with balanced braces, including nested fractions.

    A malformed last box is invalid, even if an earlier box is well formed.
    Escaped literal braces do not change nesting depth.
    """
    starts = list(re.finditer(r"\\(?:boxed|fbox)\b", text))
    if not starts:
        return None
    opening = starts[-1].end()
    while opening < len(text) and text[opening].isspace():
        opening += 1
    if opening >= len(text) or text[opening] != "{":
        return None
    start = opening + 1
    depth = 1
    for index in range(start, len(text)):
        character = text[index]
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return None


@lru_cache(maxsize=1)
def _backend() -> tuple[Any, Any, Any, Any]:
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        from math_verify.errors import TimeoutException
    except ImportError as exc:
        raise GraderError("Install the pinned math-verify dependency before scoring.") from exc
    if "raise_on_error" not in inspect.signature(parse).parameters or "raise_on_error" not in inspect.signature(verify).parameters:
        raise GraderError("This Math-Verify version lacks raise_on_error; install the project's pinned version.")
    return parse, verify, LatexExtractionConfig, TimeoutException


def _check_execution_context(timeout_seconds: int) -> None:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive integer")
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise GraderError("Bounded Math-Verify scoring requires a Linux main thread; use process workers, not threads.")


@lru_cache(maxsize=32768)
def _parse_gold(answer: str, timeout_seconds: int) -> Any:
    parse, _, config_type, _ = _backend()
    try:
        gold = parse("$" + answer + "$", extraction_config=[config_type()], fallback_mode="first_match", extraction_mode="first_match", parsing_timeout=timeout_seconds, raise_on_error=True)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise GraderError("The reference answer could not be parsed within the time limit.") from exc
    if not gold:
        raise GraderError("The reference answer produced no Math-Verify extraction.")
    return gold


def score_answer(solution_str: str, ground_truth: str, *, timeout_seconds: int = 5) -> dict[str, Any]:
    """Return score/status/extracted_answer; valid model failures have score zero."""
    if not isinstance(solution_str, str):
        raise TypeError("solution_str must be a string")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise GraderError("ground_truth must be a non-empty string")
    _check_execution_context(timeout_seconds)
    parse, verify, config_type, timeout_type = _backend()
    # Validate references even when a model has not emitted any box.
    gold = _parse_gold(ground_truth.strip(), timeout_seconds)
    extracted = extract_boxed(solution_str)
    result: dict[str, Any] = {"score": 0.0, "status": "missing_box", "extracted_answer": extracted or "", "grader": GRADER_VERSION}
    if extracted is None:
        return result
    if not extracted:
        result["status"] = "empty_answer"
        return result
    try:
        prediction = parse("$" + extracted + "$", extraction_config=[config_type()], fallback_mode="first_match", extraction_mode="first_match", parsing_timeout=timeout_seconds, raise_on_error=True)
        if not prediction:
            result["status"] = "parse_failure"
            return result
        correct = bool(verify(gold, prediction, strict=True, timeout_seconds=timeout_seconds, raise_on_error=True))
    except timeout_type:
        result["status"] = "timeout"
        return result
    except (ValueError, SyntaxError, ArithmeticError, RecursionError):
        result["status"] = "parse_failure"
        return result
    result["score"] = float(correct)
    result["status"] = "correct" if correct else "incorrect"
    return result


def _shutdown_scoring_pool() -> None:
    global _SCORING_POOL
    if _SCORING_POOL is not None:
        _SCORING_POOL.shutdown(wait=True, cancel_futures=True)
        _SCORING_POOL = None


def _get_scoring_pool() -> ProcessPoolExecutor:
    """One lazy spawned grader process per Ray reward actor process.

    SIGALRM cannot run inside verl's thread executor or a Ray asyncio thread.
    A spawned worker runs score_answer in its own main thread, without inheriting
    the actor's CUDA state. Importing this module creates no child process.
    """
    global _SCORING_POOL
    with _POOL_LOCK:
        if _SCORING_POOL is None:
            _SCORING_POOL = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            atexit.register(_shutdown_scoring_pool)
        return _SCORING_POOL


async def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Async verl adapter; the canonical scorer executes in a process main thread.

    verl may load this file under a generated module name. Import the installed
    package explicitly so multiprocessing can pickle the worker callable and
    resolve it by a stable module path. Offline evaluation calls score_answer
    directly and does not create this pool.
    """
    del data_source, extra_info
    canonical = importlib.import_module("math_rft.reward")
    score = partial(canonical.score_answer, solution_str, ground_truth,
                    timeout_seconds=kwargs.get("timeout_seconds", 5))
    return await asyncio.get_running_loop().run_in_executor(canonical._get_scoring_pool(), score)

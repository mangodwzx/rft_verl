"""Persistent, spawn-isolated ALFWorld environments with one shared caller LLM.

Only environment operations cross the process boundary. The parent retains
tokenization, seeded generation, episode histories, and ledger commits. Each
worker owns its TextWorld state and serialization lock; no CUDA engine is
constructed in a worker.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import threading
from typing import Any, Callable


_ENVIRONMENTS: dict[int, Any] = {}
_FACTORY: Callable[..., Any] | None = None


def _initialize(factory: Callable[..., Any] | None) -> None:
    global _FACTORY
    # Environment workers never need a GPU. Use spawn, rather than inheriting
    # the parent's initialized CUDA runtime or TextWorld native state.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if factory is None:
        from .alfworld_env import AlfWorldEnvironment
        factory = AlfWorldEnvironment
    _FACTORY = factory


def _dispatch(operation: str, identity: int, *arguments: Any) -> Any:
    if operation == "reset":
        if identity not in _ENVIRONMENTS:
            task, max_steps, seed = arguments
            if _FACTORY is None:
                raise RuntimeError("Environment worker was not initialized")
            _ENVIRONMENTS[identity] = _FACTORY(task, max_steps=max_steps, seed=seed)
        return _ENVIRONMENTS[identity].reset()
    if operation == "step":
        if identity not in _ENVIRONMENTS:
            raise RuntimeError("Call reset before step")
        return _ENVIRONMENTS[identity].step(*arguments)
    if operation == "close":
        env = _ENVIRONMENTS.pop(identity, None)
        if env is not None:
            env.close()
        return None
    raise ValueError(f"Unknown environment operation: {operation}")


def _close_all() -> None:
    errors = []
    for identity in list(_ENVIRONMENTS):
        try:
            _dispatch("close", identity)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


class _EnvironmentProxy:
    """Synchronous environment API, called through run_episode's to_thread."""

    def __init__(self, pool: EnvironmentPool, worker: int, identity: int,
                 task: dict[str, Any], max_steps: int, seed: int):
        self._pool, self._worker, self._identity = pool, worker, identity
        self._task, self._max_steps, self._seed = dict(task), max_steps, seed
        self._lock = threading.Lock()
        self._closed = False

    def reset(self) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("Environment lease is closed")
            return self._pool._submit(self._worker, "reset", self._identity,
                                      self._task, self._max_steps, self._seed).result()

    def step(self, action: str) -> tuple[str, bool, bool, dict[str, Any]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Environment lease is closed")
            return self._pool._submit(self._worker, "step", self._identity, action).result()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Pool shutdown closes all outstanding worker-side leases. A late
            # episode finally block must not submit work to a closed executor.
            future = self._pool._submit(self._worker, "close", self._identity, allow_closed=True)
            if future is not None:
                future.result()


class EnvironmentPool:
    """Lease environments round-robin across persistent CPU worker processes.

    ``factory`` is compatible with ``run_episode(env_factory=pool.factory)``.
    A lease always returns to the same worker; different leases never share
    game state. ``env_factory`` must be importable/picklable under spawn and is
    intended for tests; production uses the unchanged AlfWorldEnvironment.
    """

    def __init__(self, workers: int, *, env_factory: Callable[..., Any] | None = None):
        if type(workers) is not int or workers < 1:
            raise ValueError("Environment workers must be a positive integer")
        self.workers = workers
        self._lock = threading.Lock()
        self._closed = False
        self._next_identity = 0
        context = multiprocessing.get_context("spawn")
        self._executors = [ProcessPoolExecutor(max_workers=1, mp_context=context,
                                              initializer=_initialize, initargs=(env_factory,))
                           for _ in range(workers)]

    def factory(self, task: dict[str, Any], *, max_steps: int = 50, seed: int = 0) -> _EnvironmentProxy:
        with self._lock:
            if self._closed:
                raise RuntimeError("Environment pool is closed")
            identity = self._next_identity
            self._next_identity += 1
            return _EnvironmentProxy(self, identity % self.workers, identity, task, max_steps, seed)

    def _submit(self, worker: int, operation: str, identity: int, *arguments: Any,
                allow_closed: bool = False):
        with self._lock:
            if self._closed:
                if allow_closed:
                    return None
                raise RuntimeError("Environment pool is closed")
            return self._executors[worker].submit(_dispatch, operation, identity, *arguments)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        errors = []
        # Submit cleanup to every worker before waiting, so outstanding work
        # drains concurrently. Even a failed worker cannot skip peer shutdown.
        pending = []
        for executor in self._executors:
            try:
                pending.append(executor.submit(_close_all))
            except Exception as exc:
                errors.append(exc)
        for future in pending:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=True)
        if errors:
            raise errors[0]

    def __enter__(self) -> EnvironmentPool:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

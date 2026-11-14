"""Independent, verifiable ALFWorld text episodes (no model/runtime imports).

The adapter uses the official TextWorld PDDL backend and ALFWorld's demangler.
It deliberately does not install an expert wrapper or expose hidden state to the
policy. All attempts use the same game with domain randomization disabled.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys
import threading
from typing import Any

from .io_utils import canonical_json, sha256_file, sha256_text

ENVIRONMENT_VERSION = "alfworld_text_v1"
_ENVIRONMENT_LOCK = threading.RLock()
_IDLE_BACKENDS: list[Any] = []
_ALL_BACKENDS: list[Any] = []


def validate_task(task: dict[str, Any]) -> None:
    """Verify frozen local task assets without importing ALFWorld/TextWorld."""
    if not isinstance(task, dict):
        raise ValueError("ALFWorld task metadata must be a dictionary")
    for filename, checksum in (("game_file", "game_sha256"), ("trajectory_file", "trajectory_sha256")):
        value, digest = task.get(filename), task.get(checksum)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"ALFWorld {filename} must be an absolute local path")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"ALFWorld task is missing {checksum}")
        if sha256_file(value) != digest:
            raise ValueError(f"ALFWorld asset changed since preparation: {value}")
    observation = task.get("initial_observation")
    if observation is not None and (not isinstance(observation, str) or not observation.strip()):
        raise ValueError("ALFWorld initial_observation must be nonempty text")


def environment_identity() -> dict[str, Any]:
    """Record versions and executable package assets; fail before rollout if absent."""
    packages = {}
    for name in ("alfworld", "textworld", "fast-downward-textworld"):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise ImportError(
                f"ALFWorld needs {name}; install this project's alfworld extra in the training environment"
            ) from exc
        files = {}
        for entry in distribution.files or []:
            if Path(str(entry)).suffix == ".pyc" or any(part.endswith(".dist-info") for part in entry.parts):
                continue
            path = distribution.locate_file(entry)
            if path.is_file():
                files[str(entry)] = sha256_file(path)
        packages[name] = {"version": distribution.version, "files_sha256": files}
    return {"adapter": ENVIRONMENT_VERSION, "domain_randomization": False,
            "packages": packages, "packages_sha256": sha256_text(canonical_json(packages))}


def _start_environment(game_file: str):
    try:
        import textworld
        from textworld.envs.pddl.pddl import PddlEnv
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    except ImportError as exc:
        raise ImportError("Install the project's alfworld extra to execute ALFWorld text games") from exc
    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    # PddlEnv.close() is a no-op in TextWorld 1.7; repeatedly making backends
    # leaks 34 MiB native mappings per episode. Its planner's dlclose path is
    # unsafe on this stack. Keep a pool bounded by peak simultaneous leases,
    # reloading/resetting game state on every lease, as the official loader does.
    # Never block waiting for a pool slot inside asyncio.to_thread: that can
    # starve queued step/close calls and deadlock all active episodes.
    if _IDLE_BACKENDS:
        env = _IDLE_BACKENDS.pop()
    else:
        env = AlfredInfos(AlfredDemangler(PddlEnv(infos), shuffle=False))
        _ALL_BACKENDS.append(env)  # OS releases native mappings at worker exit.
    try:
        env.load(game_file)
        return _BackendLease(env)
    except BaseException:
        # Quarantine partially loaded state: retain its mapping, never reuse it.
        raise


class _BackendLease:
    def __init__(self, backend):
        self.backend, self.broken = backend, False

    def seed(self, seed):
        return self.backend.seed(seed)

    def reset(self):
        try:
            return self.backend.reset()
        except BaseException:
            self.broken = True
            raise

    def step(self, action):
        try:
            return self.backend.step(action)
        except BaseException:
            self.broken = True
            raise

    def close(self):
        with _ENVIRONMENT_LOCK:
            backend, self.backend = self.backend, None
            if backend is not None and not self.broken:
                _IDLE_BACKENDS.append(backend)


def _feedback(state) -> str:
    value = state.get("feedback")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("ALFWorld returned missing/empty observation text")
    return value


class AlfWorldEnvironment:
    """One isolated environment; success is ``won``, never the step limit.

    ``seed`` is recorded/passed to the backend; fixed PDDL tasks with demangler
    shuffling disabled are deterministic. Every reset exclusively leases a
    backend and reloads the original game, so task state/history cannot carry
    over between attempts. Native resources are reused after close.
    """

    def __init__(self, task: dict[str, Any], *, max_steps: int = 50, seed: int = 0):
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("environment seed must be a nonnegative integer")
        self.task = dict(task)
        self.max_steps, self.seed = max_steps, seed
        self._env = None
        self._state = None
        self.steps = 0
        self.done = False

    def reset(self) -> str:
        self.close()
        validate_task(self.task)
        try:
            # Fast Downward's Python translator mutates global options/sys.argv
            # and temporarily captures stdout. Its initialization must not race
            # across async rollout threads. Loaded episode libraries are separate.
            with _ENVIRONMENT_LOCK:
                previous_argv = sys.argv
                try:
                    self._env = _start_environment(self.task["game_file"])
                    self._env.seed(self.seed)
                    self._state = self._env.reset()
                finally:
                    sys.argv = previous_argv
            observation = _feedback(self._state)
            if self._state.get("won"):
                raise ValueError("ALFWorld task is already solved at reset")
            expected = self.task.get("initial_observation")
            if expected is not None and observation != expected:
                raise ValueError("ALFWorld initial observation differs from prepared task; recheck environment versions")
            self.steps, self.done = 0, False
            return observation
        except BaseException:
            self.close()
            raise

    def step(self, action: str) -> tuple[str, bool, bool, dict[str, Any]]:
        if self._env is None or self._state is None:
            raise RuntimeError("Call reset before step")
        if self.done:
            raise RuntimeError("Episode has ended; reset before another step")
        if not isinstance(action, str) or not action.strip() or "\n" in action or "\r" in action:
            raise ValueError("An ALFWorld action must be one nonempty command line")
        commands = self._state.get("admissible_commands")
        invalid = commands is not None and action.strip() not in commands
        # The TextWorld grammar parser is also process-global and is used during
        # every step, so steps must share initialization's serialization lock.
        with _ENVIRONMENT_LOCK:
            self._state, _, terminated = self._env.step(action.strip())
        self.steps += 1
        observation = _feedback(self._state)
        if "won" not in self._state:
            raise RuntimeError("ALFWorld backend omitted its terminal success flag")
        success = bool(self._state["won"])
        terminated = bool(terminated or success)
        truncated = not terminated and self.steps >= self.max_steps
        self.done = terminated or truncated
        return observation, success, self.done, {
            "steps": self.steps, "won": success, "terminated": terminated,
            "truncated": truncated, "invalid_action": bool(invalid),
        }

    def close(self) -> None:
        env, self._env = self._env, None
        self._state = None
        if env is not None:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

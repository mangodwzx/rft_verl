"""Controlled online RFT / GRPO training on the pinned verl FSDP2 backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from math_rft.compatibility import VERL_COMMIT, check_verl_revision, environment_versions
from math_rft.prompts import RAW_CHAT_TEMPLATE, prompt_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "rft": ("math_rft_question", "math_rft_nll", "seq-mean-token-sum-norm"),
    "grpo": ("grpo", "math_grpo_clip", "seq-mean-token-sum-norm"),
}
STAGES = ("pilot", "full", "deepmath_8192", "deepmath_16384")
PROFILES = ("default", "lowmem")
BENCHMARKS = ("math", "alfworld")
LOSS_NORMALIZATIONS = ("fixed", "per_response")
DEFAULT_LOSS_SCALE = 4096


def loss_reduction(normalization: str, scale: int | None) -> tuple[str, int | None]:
    """Choose a run-wide constant, never a sampled or padded response length."""
    if normalization == "per_response":
        if scale is not None:
            raise ValueError("--loss-scale-factor applies only to --loss-normalization fixed")
        return "seq-mean-token-mean", None
    if normalization != "fixed":
        raise ValueError(f"Unknown loss normalization: {normalization}")
    scale = DEFAULT_LOSS_SCALE if scale is None else scale
    if type(scale) is not int or scale <= 0:
        raise ValueError("--loss-scale-factor must be a positive integer")
    return "seq-mean-token-sum-norm", scale


def default_output_dir(profile: str, normalization: str, *, pipeline: bool = False,
                       benchmark: str = "math") -> str:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    directory = "runs" + ("_alfworld" if benchmark == "alfworld" else "")
    if profile == "lowmem":
        directory += "_lowmem"
    if normalization == "fixed":
        directory += "_fixed"
    if pipeline and profile == "lowmem":
        directory += "_pipeline"
    return directory


def benchmark_prompt_fingerprint(benchmark: str) -> str:
    if benchmark == "alfworld":
        from math_rft.alfworld_episode import prompt_fingerprint as alfworld_prompt_fingerprint

        return alfworld_prompt_fingerprint()
    return prompt_fingerprint()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = [path for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "configs")
             for path in directory.rglob("*")
             if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".jinja"}]
    paths.append(PROJECT_ROOT / "pyproject.toml")
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sorted(paths) if path.is_file()}


def _require_hf_weights(model_path: Path):
    indexes = list(model_path.glob("*.index.json"))
    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No model weights in {model_path}; config/tokenizer files alone cannot be trained")
    for index_path in indexes:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        for relative in set(payload.get("weight_map", {}).values()):
            if not (model_path / relative).is_file():
                raise FileNotFoundError(f"Missing model shard {relative} referenced by {index_path}")


def _resolve_model(model: str, revision: str | None, local_model: str | None):
    models = json.loads((PROJECT_ROOT / "configs/models.json").read_text(encoding="utf-8"))
    if model not in models:
        raise ValueError(f"Unknown model {model}; choose {list(models)}")
    repo_id = models[model]["repo_id"]
    if local_model:
        model_path = Path(local_model).expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"No Hugging Face config.json in {model_path}")
        _require_hf_weights(model_path)
        files = sorted(path for path in model_path.rglob("*") if path.is_file())
        # Hash weights too: local checkpoints must not silently change between runs.
        hashes = {str(path.relative_to(model_path)): _sha256(path) for path in files}
        return str(model_path), {"repo_id": repo_id, "local_path": str(model_path), "files_sha256": hashes}
    from math_rft.model_cache import hub_snapshot

    if revision is not None and not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("--model-revision must be an immutable 40-character commit SHA")
    model_path, resolved = hub_snapshot(repo_id, revision)
    return model_path, {"repo_id": repo_id, "revision": resolved, "snapshot_path": model_path}


def compose_config(args, model_path: str):
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(verl.__file__).resolve().parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(PROJECT_ROOT / "configs/verl_base.yaml"))
    benchmark = getattr(args, "benchmark", "math")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if benchmark == "alfworld":
        if args.stage not in ("pilot", "full"):
            raise ValueError("ALFWorld supports pilot/full stages only")
        config = OmegaConf.merge(config, OmegaConf.load(PROJECT_ROOT / "configs/verl_alfworld.yaml"))
    profile = getattr(args, "profile", "default")
    if profile not in PROFILES:
        raise ValueError(f"Unknown training profile: {profile}")
    if profile != "default":
        config = OmegaConf.merge(config, OmegaConf.load(PROJECT_ROOT / f"configs/verl_{profile}.yaml"))
    data_dir = Path(args.data_dir or ("data_alfworld" if benchmark == "alfworld" else "data")).expanduser().resolve()
    normalization = args.loss_normalization
    reduction, scale = loss_reduction(normalization, args.loss_scale_factor)
    output_dir = args.output_dir or default_output_dir(profile, normalization, benchmark=benchmark)
    run_dir = Path(output_dir).expanduser().resolve() / args.model / args.method / args.stage / f"seed_{args.seed}"
    advantage, loss, _ = METHODS[args.method]
    values = {
        "data.train_files": [str(data_dir / f"{args.stage}.parquet")],
        "data.val_files": [str(data_dir / "dev.parquet")],
        "data.seed": args.seed,
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.model.custom_chat_template": RAW_CHAT_TEMPLATE,
        "actor_rollout_ref.actor.data_loader_seed": args.seed,
        "actor_rollout_ref.actor.fsdp_config.seed": args.seed,
        "actor_rollout_ref.rollout.seed": args.seed,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(PROJECT_ROOT / "configs/verl_raw_agent.yaml"),
        "actor_rollout_ref.actor.policy_loss.loss_mode": loss,
        "actor_rollout_ref.actor.loss_agg_mode": reduction,
        "actor_rollout_ref.actor.loss_scale_factor": scale,
        "algorithm.adv_estimator": advantage,
        "trainer.experiment_name": f"{args.model}_{args.method}_{args.stage}_s{args.seed}" + (
            f"_fixed_s{scale}" if normalization == "fixed" else ""),
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.validation_data_dir": str(run_dir / "dev_rollouts"),
        "reward.custom_reward_function.path": str(Path(__file__).with_name("reward.py")),
    }
    if benchmark == "alfworld":
        values.update({
            "alfworld.max_steps": args.max_steps,
            "alfworld.max_action_tokens": args.max_action_tokens,
            "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(PROJECT_ROOT / "configs/verl_alfworld_agent.yaml"),
            "reward.custom_reward_function.path": str(Path(__file__).with_name("alfworld_reward.py")),
            "trainer.project_name": "alfworld_rft",
        })
        values["trainer.experiment_name"] = "alfworld_" + values["trainer.experiment_name"]
    for key, value in values.items():
        OmegaConf.update(config, key, value, merge=False)
    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Expected --set KEY=VALUE, got {override}")
        key, raw_value = override.split("=", 1)
        sentinel = object()
        if OmegaConf.select(config, key, default=sentinel) is sentinel:
            raise ValueError(f"Unknown config key {key}; check the pinned verl configuration")
        value = OmegaConf.from_dotlist([f"value={raw_value}"]).value
        OmegaConf.update(config, key, value, merge=False)
    if args.resume_from:
        config.trainer.resume_mode = "resume_path"
        config.trainer.resume_from_path = str(Path(args.resume_from).expanduser().resolve())
    OmegaConf.resolve(config)
    _validate_config(config, args)
    return config, run_dir


def _validate_config(config, args):
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if config.trainer.use_v1:
        raise ValueError("This audited extension uses trainer.use_v1=false at the pinned commit")
    if config.trainer.val_only or config.trainer.save_freq <= 0:
        raise ValueError("Training requires updates and positive checkpoint frequency")
    if "hf_model" not in actor.checkpoint.save_contents or actor.checkpoint.async_save:
        raise ValueError("Save synchronous full Hugging Face weights at each checkpoint")
    if actor.ppo_mini_batch_size != config.data.train_batch_size or actor.ppo_epochs != 1:
        raise ValueError("One update per fresh prompt batch is required; minibatch size must match train batch")
    if config.data.gen_batch_size not in (None, config.data.train_batch_size):
        raise ValueError("Generation batch size must match train_batch_size for one update per fresh prompt batch")
    if rollout.n < 2 or not rollout.do_sample or rollout.temperature != 1.0:
        raise ValueError("Use m>=2, stochastic sampling, temperature=1.0 for the defined objective")
    if rollout.top_p != 1.0 or rollout.top_k != -1:
        raise ValueError("Main protocol requires untruncated sampling (top_p=1, top_k=-1)")
    if rollout.ignore_eos or rollout.engine_kwargs.vllm.generation_config != "vllm":
        raise ValueError("Use EOS-only raw sampling without model-side generation_config defaults")
    if config.algorithm.use_kl_in_reward or config.algorithm.filter_groups.enable:
        raise ValueError("Reward shaping and task/group filtering are excluded from the controlled protocol")
    if actor.entropy_coeff != 0 or actor.use_kl_loss:
        raise ValueError("Use zero entropy/KL regularization for the controlled baseline")
    if actor.optim.weight_decay != 0:
        raise ValueError("The main experiment uses zero weight decay for both methods")
    if not config.algorithm.norm_adv_by_std_in_grpo:
        raise ValueError("Main GRPO requires group standard deviation normalization")
    if actor.strategy != "fsdp2" or actor.fsdp_config.strategy != "fsdp2":
        raise ValueError("Only the audited FSDP2 route is supported")
    if config.actor_rollout_ref.model.lora_rank or config.actor_rollout_ref.model.lora.rank:
        raise ValueError("The registered experiment uses full language-model parameter training")
    if config.actor_rollout_ref.model.external_lib != "math_rft.algorithms":
        raise ValueError("The algorithm module must be imported by every Ray worker")
    if config.actor_rollout_ref.model.custom_chat_template != RAW_CHAT_TEMPLATE:
        raise ValueError("Use the same raw-completion prompt in training and evaluation")
    benchmark = getattr(args, "benchmark", "math")
    expected_loop = "alfworld" if benchmark == "alfworld" else "raw_math"
    if rollout.agent.default_agent_loop != expected_loop:
        raise ValueError(f"Use the {expected_loop} rollout path for benchmark {benchmark}")
    if benchmark == "alfworld":
        if rollout.full_determinism:
            raise ValueError("ALFWorld requires full_determinism=false: the pinned worker does not expose per-attempt seeds")
        if not rollout.multi_turn.enable:
            raise ValueError("ALFWorld requires multi_turn.enable=true")
        for name in ("max_steps", "max_action_tokens"):
            value = config.alfworld[name]
            if type(value) is not int or value <= 0 or value != getattr(args, name):
                raise ValueError(f"Set a positive --{name.replace('_', '-')} instead of overriding alfworld.{name}")
        if config.alfworld.max_action_tokens > config.data.max_response_length:
            raise ValueError("ALFWorld action budget must fit inside the episode response budget")
        expected_reward = str(Path(__file__).with_name("alfworld_reward.py"))
        if config.reward.custom_reward_function.path != expected_reward:
            raise ValueError("ALFWorld requires environment episode rewards")
    if config.data.filter_overlong_prompts or config.data.truncation != "error":
        raise ValueError("Overlong prompts must fail instead of silently changing the training population")
    correction = config.algorithm.rollout_correction
    if correction.rollout_is is not None or correction.rollout_rs is not None or correction.bypass_mode:
        raise ValueError("Rollout correction changes the objective and must remain disabled")
    reduction, scale = loss_reduction(args.loss_normalization, args.loss_scale_factor)
    expected = (*METHODS[args.method][:2], reduction)
    if (config.algorithm.adv_estimator, actor.policy_loss.loss_mode, actor.loss_agg_mode) != expected:
        raise ValueError("Algorithm/loss/reduction must agree with --method; use an explicit method")
    if actor.loss_scale_factor != scale:
        raise ValueError("Loss scale must agree with --loss-normalization and --loss-scale-factor")
    if rollout.max_model_len < config.data.max_prompt_length + config.data.max_response_length:
        raise ValueError("rollout.max_model_len must cover prompt + response budgets")
    world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
    if world_size <= 0 or config.data.train_batch_size <= 0:
        raise ValueError("GPU world size and training batch size must be positive")
    if (config.data.train_batch_size * rollout.n) % world_size:
        raise ValueError("Global rollout batch must be divisible by the FSDP world size")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="math")
    parser.add_argument("--max-steps", type=int, default=50, help="ALFWorld actions per episode")
    parser.add_argument("--max-action-tokens", type=int, default=128, help="ALFWorld generated tokens per action")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=STAGES, default="pilot")
    parser.add_argument("--profile", choices=PROFILES, default="default",
                        help="lowmem reduces execution memory while preserving training budgets and objectives")
    parser.add_argument("--loss-normalization", choices=LOSS_NORMALIZATIONS, default="fixed",
                        help="fixed sums response tokens then divides by a run-wide constant; per_response reproduces the legacy length-biased objective")
    parser.add_argument("--loss-scale-factor", type=int,
                        help="Positive constant for fixed normalization (default 4096), shared across both methods")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", help="Default: data for math, data_alfworld for ALFWorld")
    parser.add_argument("--output-dir", help="Default: runs_fixed / runs_lowmem_fixed; legacy per_response: runs / runs_lowmem")
    parser.add_argument("--model-revision")
    parser.add_argument("--local-model", help="Optional identical starting HF checkpoint for both methods")
    parser.add_argument("--resume-from", help="Exact verl global_step_N checkpoint directory")
    parser.add_argument("--print-config", action="store_true", help="Compose only; no model/data download, Ray, or training")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main():
    cli = parser()
    args = cli.parse_args()
    if args.seed < 0:
        cli.error("--seed must be nonnegative")
    upstream = check_verl_revision()
    models = json.loads((PROJECT_ROOT / "configs/models.json").read_text(encoding="utf-8"))
    if args.model not in models:
        cli.error(f"--model must be one of {', '.join(models)}")
    if args.print_config:
        from omegaconf import OmegaConf
        config, _ = compose_config(args, args.local_model or models[args.model]["repo_id"])
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    if sys.platform != "linux":
        cli.error("Run verl training on Linux with CUDA; this repository can be edited statically on Windows")
    model_path, model_provenance = _resolve_model(args.model, args.model_revision, args.local_model)
    config, run_dir = compose_config(args, model_path)
    data_paths = [Path(path) for path in [*config.data.train_files, *config.data.val_files]]
    for path in data_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prepare the data first: {path}")
    import pyarrow.parquet as pq
    train_rows = sum(pq.read_metadata(path).num_rows for path in config.data.train_files)
    if config.data.train_max_samples >= 0:
        train_rows = min(train_rows, config.data.train_max_samples)
    batches_per_epoch = train_rows // (config.data.gen_batch_size or config.data.train_batch_size)
    available_steps = batches_per_epoch * config.trainer.total_epochs
    expected_steps = config.trainer.total_training_steps or available_steps
    if expected_steps <= 0 or expected_steps > available_steps:
        raise ValueError("Training steps must be positive and fit within the configured complete-batch epochs")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "method": args.method, "stage": args.stage, "seed": args.seed,
        "profile": args.profile,
        "loss_normalization": args.loss_normalization,
        "loss_scale_factor": config.actor_rollout_ref.actor.loss_scale_factor,
        "upstream": upstream, "verl_commit": VERL_COMMIT,
        "model_provenance": model_provenance,
        "prompt_sha256": benchmark_prompt_fingerprint(args.benchmark),
        "data_sha256": {str(path): _sha256(path) for path in data_paths},
        "environment": environment_versions(),
        "source_sha256": _source_hashes(),
        "expected_steps": expected_steps,
        "argv": sys.argv,
    }
    manifest["benchmark"] = args.benchmark
    if args.benchmark == "alfworld":
        from math_rft.alfworld_env import environment_identity, validate_task

        dataset_manifest = Path(config.data.train_files[0]).parent / "manifest.json"
        prepared = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if prepared.get("benchmark") != "alfworld":
            raise ValueError("ALFWorld training requires an ALFWorld dataset manifest")
        if prepared.get("prompt_sha256") != manifest["prompt_sha256"]:
            raise ValueError("ALFWorld prepared prompt protocol differs from the current runner")
        if prepared.get("environment") != environment_identity():
            raise ValueError("ALFWorld runtime differs from the prepared dataset environment")
        for path in data_paths:
            is_training = str(path) in config.data.train_files
            expected_split = args.stage if is_training else "dev"
            expected_source = "train" if is_training else "valid_seen"
            if path.stem != expected_split:
                raise ValueError(f"ALFWorld requires {expected_split}.parquet here; held-out test tasks cannot train/validate")
            split = prepared.get("splits", {}).get(path.stem, {})
            if split.get("parquet_sha256") != _sha256(path):
                raise ValueError(f"ALFWorld prepared data changed: {path}")
            for row in pq.read_table(path, columns=["extra_info"]).to_pylist():
                info = row["extra_info"]
                task = info["alfworld_task"]
                if info["split"] != expected_split or task["source_split"] != expected_source:
                    raise ValueError("ALFWorld training/development row comes from an incorrect source split")
                validate_task(task)
        manifest["data_sha256"][str(dataset_manifest)] = _sha256(dataset_manifest)
        manifest["alfworld"] = {"max_steps": config.alfworld.max_steps,
                               "max_action_tokens": config.alfworld.max_action_tokens}
    from omegaconf import OmegaConf
    config_text = OmegaConf.to_yaml(config, resolve=True)
    manifest["resolved_config_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not args.resume_from:
        raise FileExistsError(f"Run already exists: {run_dir}; choose a new --output-dir or explicit --resume-from")
    if (run_dir / "training_complete.json").exists():
        raise FileExistsError("This run already has a successful completion marker; use a new output directory")
    if args.resume_from:
        if not manifest_path.is_file():
            raise FileNotFoundError("Resume requires the original run directory and run_manifest.json")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("benchmark", "math") != args.benchmark:
            raise ValueError("Resume identity mismatch for benchmark")
        if prior.get("profile", "default") != args.profile:
            raise ValueError("Resume identity mismatch for profile")
        if (prior.get("loss_normalization", "per_response") != args.loss_normalization
                or prior.get("loss_scale_factor") != manifest["loss_scale_factor"]):
            raise ValueError("Resume identity mismatch for training objective; start a separate run")
        for key in ("model", "method", "stage", "seed", "prompt_sha256", "data_sha256", "model_provenance",
                    "upstream", "environment", "source_sha256"):
            if prior[key] != manifest[key]:
                raise ValueError(f"Resume identity mismatch for {key}")
        resume_path = Path(config.trainer.resume_from_path).resolve()
        checkpoint_root = (run_dir / "checkpoints").resolve()
        if (not resume_path.is_dir() or not resume_path.is_relative_to(checkpoint_root)
                or not re.fullmatch(r"global_step_[0-9]+", resume_path.name)
                or not (resume_path / "actor").is_dir()):
            raise ValueError("Resume must point to an existing global_step_N checkpoint inside this original run")
        initial_config_path = run_dir / "resolved_config.yaml"
        initial_text = initial_config_path.read_text(encoding="utf-8")
        if hashlib.sha256(initial_text.encode()).hexdigest() != prior["resolved_config_sha256"]:
            raise ValueError("The initial resolved_config.yaml no longer matches its recorded hash")
        initial_config = OmegaConf.to_container(OmegaConf.create(initial_text), resolve=True)
        resumed_config = OmegaConf.to_container(config, resolve=True)
        for compared in (initial_config, resumed_config):
            compared["trainer"].pop("resume_mode", None)
            compared["trainer"].pop("resume_from_path", None)
        if initial_config != resumed_config:
            raise ValueError("Resume configuration changed; only resume_mode and resume_from_path may differ")
        # Retain the initial provenance rather than silently replacing it.
        manifest_path = run_dir / f"resume_manifest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / ("resume_config.yaml" if args.resume_from else "resolved_config.yaml")).write_text(config_text, encoding="utf-8")
    # Explicit parent import and external_lib together cover parent/controller/actors.
    import math_rft.algorithms  # noqa: F401
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import TaskRunner
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=TaskRunner)
    checkpoint_root = Path(config.trainer.default_local_dir)
    latest_path = checkpoint_root / "latest_checkpointed_iteration.txt"
    global_step = int(latest_path.read_text(encoding="utf-8").strip())
    if global_step != expected_steps:
        raise RuntimeError(f"Training returned without the planned final checkpoint: {global_step} != {expected_steps}")
    final_checkpoint = checkpoint_root / f"global_step_{global_step}"
    final_hf_model = final_checkpoint / "actor/huggingface"
    if not (final_hf_model / "config.json").is_file():
        raise FileNotFoundError(f"Final HF export is incomplete: {final_hf_model}")
    _require_hf_weights(final_hf_model)
    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_checkpoint": str(final_checkpoint),
        "final_hf_model": str(final_hf_model),
        "global_step": global_step,
        "resolved_config_sha256": _sha256(run_dir / "resolved_config.yaml"),
        "source_sha256": manifest["source_sha256"],
        "model": args.model, "method": args.method, "stage": args.stage, "seed": args.seed,
        "profile": args.profile,
        "loss_normalization": args.loss_normalization,
        "loss_scale_factor": config.actor_rollout_ref.actor.loss_scale_factor,
        "benchmark": args.benchmark,
    }
    if args.benchmark == "alfworld":
        completion["alfworld"] = manifest["alfworld"]
    temporary = run_dir / "training_complete.json.tmp"
    temporary.write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "training_complete.json")


if __name__ == "__main__":
    main()

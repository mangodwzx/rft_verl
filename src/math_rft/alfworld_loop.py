"""Pinned verl adapter for ALFWorld reset-to-terminal rollouts."""
from __future__ import annotations

from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from .alfworld_episode import GeneratedAction, render_prompt, run_episode


class AlfWorldAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, priority=0, **kwargs):
        task = dict(kwargs["extra_info"]["alfworld_task"])
        messages = list(kwargs["raw_prompt"])
        expected = [{"role": "user", "content": render_prompt(task["initial_observation"])}]
        if messages != expected:
            raise ValueError("ALFWorld raw_prompt must match its prepared initial observation and prompt protocol")
        settings = self.config.alfworld

        async def generate(prompt_ids, max_tokens, turn_seed):
            parameters = dict(sampling_params)
            parameters.update(max_tokens=max_tokens, repetition_penalty=1.0, presence_penalty=0.0,
                              frequency_penalty=0.0, ignore_eos=False, stop=["\n"],
                              include_stop_str_in_output=True, skip_special_tokens=False)
            # The worker does not forward rollout_n into run(). Using task IDs
            # as request seeds would make all m trajectories identical. Preserve
            # the training engine RNG unless a caller supplies an episode seed.
            if turn_seed is not None:
                parameters["seed"] = turn_seed
            output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids,
                sampling_params=parameters, priority=int(priority),
            )
            if output.stop_reason == "aborted":
                raise RuntimeError("ALFWorld generation was aborted")
            # The pinned server maps both normal stops and length limits to
            # 'completed'. run_episode detects a truncated action from the
            # absence of the requested newline/EOS boundary in sampled IDs.
            return GeneratedAction(token_ids=list(output.token_ids), log_probs=output.log_probs)

        result = await run_episode(
            task, self.tokenizer, generate,
            max_steps=int(settings.max_steps), max_action_tokens=int(settings.max_action_tokens),
            max_prompt_tokens=int(self.rollout_config.prompt_length),
            max_response_tokens=int(self.rollout_config.response_length),
            seed=int(sampling_params["seed"]) if sampling_params.get("seed") is not None else None,
        )
        return AgentLoopOutput(
            prompt_ids=result.prompt_ids, response_ids=result.response_ids,
            response_mask=result.response_mask, response_logprobs=result.response_logprobs,
            reward_score=float(result.score), multi_modal_data={},
            num_turns=1 + len(result.turns) + result.metrics["environment_steps"],
            metrics={name: result.metrics[name] for name in ("generate_sequences", "tool_calls")},
            extra_fields={"turn_scores": [], "tool_rewards": [], "alfworld_success": result.score,
                          "alfworld_termination": result.termination, "alfworld_turns": result.turns,
                          "alfworld_metrics": result.metrics},
        )

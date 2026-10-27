"""Exact raw-completion rollout path for Base models, without chat boundaries."""
from __future__ import annotations

import asyncio
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer


class RawMathAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, priority=0, **kwargs):
        messages = list(kwargs["raw_prompt"])
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("RawMathAgentLoop expects exactly one user message containing the raw text prompt")
        prompt = messages[0]["content"]
        loop = asyncio.get_running_loop()
        prompt_ids = await loop.run_in_executor(
            None, lambda: self.tokenizer.encode(prompt, add_special_tokens=False)
        )
        if not prompt_ids or len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError(f"Prompt has {len(prompt_ids)} tokens; configured maximum is {self.rollout_config.prompt_length}")
        metrics = {}
        parameters = dict(sampling_params)
        parameters.update(repetition_penalty=1.0, presence_penalty=0.0,
                          frequency_penalty=0.0, ignore_eos=False, stop=[])
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=parameters,
                priority=int(priority),
            )
        response_ids = list(output.token_ids)
        if not response_ids or len(response_ids) > self.rollout_config.response_length:
            raise ValueError("Rollout returned an empty response or exceeded its configured response budget")
        log_probs = output.log_probs
        if log_probs is not None and len(log_probs) != len(response_ids):
            raise ValueError("Rollout log-probability length does not match response tokens")
        extra_fields = dict(output.extra_fields or {})
        extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=log_probs,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

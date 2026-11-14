"""ALFWorld rewards are produced by the live environment, never parsed from text."""
from __future__ import annotations

from math_rft.alfworld_episode import GRADER_VERSION  # noqa: F401


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, solution_str, ground_truth, extra_info, kwargs
    raise RuntimeError(
        "ALFWorld requires AlfWorldAgentLoop's live environment reward_score. "
        "A transcript or model claim of success cannot be graded as a completed task."
    )

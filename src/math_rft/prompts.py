"""One raw-completion prompt shared by all Base models and all algorithms."""

from __future__ import annotations

from .io_utils import sha256_text

PROMPT_VERSION = "raw_math_fewshot_v1"

# Hand-written demonstrations, not selected from any benchmark split.
DEMONSTRATIONS = (
    ("Solve 3x + 7 = 22 for x.", r"Subtracting 7 gives 3x = 15. Dividing by 3 gives x = 5. The final answer is \boxed{5}."),
    ("A rectangle has side lengths 4 and 9. What is its area?", r"Its area is the product of its side lengths: 4 \times 9 = 36. The final answer is \boxed{36}."),
    ("What is the probability of rolling an even number on a fair six-sided die?", r"The even outcomes are 2, 4, and 6, so there are 3 favorable outcomes among 6 equally likely outcomes. The final answer is \boxed{\frac{1}{2}}."),
)

INSTRUCTION = r"Solve the following mathematics problem. Show your reasoning, and put your final answer in \boxed{...}."
# EOS-only stopping matches the verl rollout path and preserves sampled tokens
# exactly for gradient diagnostics. Do not add an evaluation-only stop string.
STOP_STRINGS: list[str] = []

# The prepared parquet contains exactly one user message whose content is the
# already-rendered raw prompt. No ChatML, thinking tokens, or assistant prefix.
RAW_CHAT_TEMPLATE = "{{ messages[0]['content'] }}"


def render_prompt(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    examples = "\n\n".join(f"Problem: {question}\nSolution: {solution}" for question, solution in DEMONSTRATIONS)
    return f"{INSTRUCTION}\n\n{examples}\n\nProblem: {problem.strip()}\nSolution:"


def prompt_fingerprint() -> str:
    return sha256_text(render_prompt("<PROBLEM>"))

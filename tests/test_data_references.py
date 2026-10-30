"""Reference-only handling of historical MATH formatting and empty-box defects."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from math_rft.data import math_reference_answer, normalize_row, problem_key
from math_rft.io_utils import sha256_text
from math_rft.reward import extract_boxed


class ReferenceAnswerTests(unittest.TestCase):
    def test_unknown_training_level_is_preserved_without_guessing(self):
        row = {"problem": "Find x.", "solution": r"$\boxed{2}$", "type": "Algebra", "level": "Level ?"}
        self.assertIsNone(normalize_row(row, "math", "train")["level"])
        with self.assertRaisesRegex(ValueError, "Invalid MATH difficulty"):
            normalize_row(row, "math", "test")
        with self.assertRaisesRegex(ValueError, "Invalid MATH difficulty"):
            normalize_row({**row, "level": "broken"}, "math", "train")

    def test_unbraced_digit_reference_does_not_relax_grading(self):
        solution = r"The largest root is $\boxed 2$."
        answer, adjustment = math_reference_answer("Find the largest root.", solution)
        self.assertEqual(answer, "2")
        self.assertEqual(adjustment["kind"], "unbraced_digit_box")
        self.assertIsNone(extract_boxed(solution))

    def test_standard_nested_answer_is_unchanged(self):
        answer, adjustment = math_reference_answer("A problem", r"Thus $\boxed{\frac{1}{2}}$.")
        self.assertEqual(answer, r"\frac{1}{2}")
        self.assertIsNone(adjustment)

    def test_missing_or_malformed_final_answers_are_not_guessed(self):
        for solution in (r"Answer is 2.", r"$\boxed{}$", r"$\boxed 2 or 3$",
                         r"$\boxed{1}$ then $\boxed{", r"$\boxed 12$"):
            with self.subTest(solution=solution):
                self.assertEqual(math_reference_answer("Unknown question", solution), (None, None))

    def test_audited_correction_requires_both_fingerprints(self):
        problem, solution = "Synthetic audited problem", r"There are $\boxed{}$ primes."
        correction = {"problem_sha256": problem_key(problem), "solution_sha256": sha256_text(solution),
                      "answer": "0", "reason": "Synthetic regression fixture"}
        with patch("math_rft.data._reference_corrections", return_value={problem_key(problem): correction}):
            answer, adjustment = math_reference_answer(problem, solution)
            self.assertEqual(answer, "0")
            self.assertEqual(adjustment["kind"], "audited_empty_box")
            self.assertEqual(math_reference_answer(problem + " changed", solution), (None, None))
            self.assertEqual(math_reference_answer(problem, solution + " changed"), (None, None))

    def test_normalization_records_adjustment_only_for_math_train(self):
        row = {"problem": "Find x.", "solution": r"$\boxed 9$", "type": "Algebra", "level": "Level 3"}
        normalized = normalize_row(row, "math", "train")
        self.assertEqual(normalized["answer"], "9")
        self.assertIsNotNone(normalized["reference_answer_adjustment"])
        with self.assertRaisesRegex(ValueError, "No reference answer"):
            normalize_row(row, "math", "test")
        normalized = normalize_row({**row, "answer": "9"}, "math", "train")
        self.assertIsNone(normalized["reference_answer_adjustment"])


if __name__ == "__main__":
    unittest.main()

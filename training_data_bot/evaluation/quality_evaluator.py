"""Quality evaluator for generated training data."""

import re
import uuid
from typing import Any

from training_data_bot.core.logging import get_logger

logger = get_logger("quality_evaluator")


class QualityEvaluator:
    """Evaluator for checking quality of generated training examples."""

    # Minimum requirements for different task types
    MIN_REQUIREMENTS = {
        "qa_generation": {
            "min_question_length": 10,
            "min_answer_length": 5,
        },
        "summarization": {
            "min_summary_length": 20,
        },
        "classification": {
            "min_text_length": 5,
        },
        "instruction_response": {
            "min_instruction_length": 10,
            "min_response_length": 5,
        },
    }

    def __init__(
        self,
        min_quality_score: float = 0.5,
        enable_filters: bool = True,
    ) -> None:
        """
        Initialize the quality evaluator.

        Args:
            min_quality_score: Minimum quality score to pass (0-1)
            enable_filters: Whether to enable filtering
        """
        self.min_quality_score = min_quality_score
        self.enable_filters = enable_filters

    def evaluate(self, examples: list[dict]) -> dict[str, Any]:
        """
        Evaluate the quality of training examples.

        Args:
            examples: List of training examples to evaluate

        Returns:
            Dictionary with evaluation results
        """
        if not examples:
            return {
                "passed": True,
                "overall_score": 1.0,
                "example_count": 0,
                "issues": [],
                "warnings": [],
                "passed_examples": {},
            }

        # Assign IDs to examples if they don't have one
        for i, example in enumerate(examples):
            if "id" not in example:
                example["id"] = str(uuid.uuid4())

        # Evaluate each example
        passed_examples = {}
        issues = []
        warnings = []
        total_score = 0.0

        for example in examples:
            example_id = example.get("id", "")
            task_type = example.get("task_type", "unknown")

            # Check basic requirements
            passed, example_issues, example_score = self._check_basic_requirements(
                example, task_type
            )

            if passed and self.enable_filters:
                # Check for common issues
                is_valid, validation_issues = self._validate_example(example, task_type)
                if not is_valid:
                    passed = False
                    issues.extend(validation_issues)

            passed_examples[example_id] = passed
            total_score += example_score

            if not passed:
                issues.append(f"Example {example_id} failed quality checks for task {task_type}")

        # Calculate overall score
        overall_score = total_score / len(examples) if examples else 1.0

        # Determine if evaluation passed
        passed = (
            overall_score >= self.min_quality_score
            and len(issues) == 0
        )

        return {
            "passed": passed,
            "overall_score": overall_score,
            "example_count": len(examples),
            "passed_count": sum(1 for v in passed_examples.values() if v),
            "failed_count": sum(1 for v in passed_examples.values() if not v),
            "issues": issues[:10],  # Limit to first 10 issues
            "warnings": warnings[:10],
            "passed_examples": passed_examples,
        }

    def _check_basic_requirements(
        self, example: dict, task_type: str
    ) -> tuple[bool, list[str], float]:
        """
        Check basic requirements for an example.

        Args:
            example: The example to check
            task_type: Type of task

        Returns:
            Tuple of (passed, issues, score)
        """
        requirements = self.MIN_REQUIREMENTS.get(task_type, {})
        issues = []
        score = 1.0

        if task_type == "qa_generation":
            question = example.get("question", "")
            answer = example.get("answer", "")

            if len(question) < requirements.get("min_question_length", 10):
                issues.append("Question too short")
                score -= 0.2

            if len(answer) < requirements.get("min_answer_length", 5):
                issues.append("Answer too short")
                score -= 0.2

            if not question or not answer:
                issues.append("Missing question or answer")
                score -= 0.3

        elif task_type == "summarization":
            summary = example.get("summary", "")

            if len(summary) < requirements.get("min_summary_length", 20):
                issues.append("Summary too short")
                score -= 0.3

            if not summary:
                issues.append("Missing summary")
                score -= 0.3

        elif task_type == "classification":
            text = example.get("text", "")
            label = example.get("label")

            if len(text) < requirements.get("min_text_length", 5):
                issues.append("Text too short")
                score -= 0.2

            if not label:
                issues.append("Missing label")
                score -= 0.2

        elif task_type == "instruction_response":
            instruction = example.get("instruction", "")
            response = example.get("response", "")

            if len(instruction) < requirements.get("min_instruction_length", 10):
                issues.append("Instruction too short")
                score -= 0.2

            if len(response) < requirements.get("min_response_length", 5):
                issues.append("Response too short")
                score -= 0.2

            if not instruction or not response:
                issues.append("Missing instruction or response")
                score -= 0.3

        passed = len(issues) == 0 and score >= self.min_quality_score
        score = max(0.0, score)  # Ensure non-negative

        return passed, issues, score

    def _validate_example(
        self, example: dict, task_type: str
    ) -> tuple[bool, list[str]]:
        """
        Validate an example for common issues.

        Args:
            example: The example to validate
            task_type: Type of task

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        # Check for placeholder text
        placeholder_patterns = [
            r"^the text contains",
            r"^information about",
            r"^placeholder",
            r"^lorem ipsum",
        ]

        if task_type == "qa_generation":
            answer = example.get("answer", "").lower()
            for pattern in placeholder_patterns:
                if re.match(pattern, answer):
                    issues.append("Answer appears to be placeholder text")
                    break

        # Check for empty strings
        for key, value in example.items():
            if isinstance(value, str) and not value.strip():
                issues.append(f"Empty value for field: {key}")

        # Check for very long strings (likely errors)
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 10000:
                issues.append(f"Extremely long value for field: {key}")

        is_valid = len(issues) == 0

        return is_valid, issues

    def filter_examples(self, examples: list[dict]) -> list[dict]:
        """
        Filter examples based on quality checks.

        Args:
            examples: List of examples to filter

        Returns:
            List of examples that passed quality checks
        """
        result = self.evaluate(examples)
        passed_examples = result.get("passed_examples", {})

        filtered = [
            ex for ex in examples
            if passed_examples.get(ex.get("id", ""), False)
        ]

        return filtered

"""Task manager for handling different task types and configurations."""

import asyncio
from typing import Any

from training_data_bot.core.logging import get_logger
from training_data_bot.tasks.task_generator import TaskGenerator

logger = get_logger("task_manager")


class TaskManager:
    """Manager for handling different task types and configurations."""

    SUPPORTED_TASKS = [
        "qa_generation",
        "summarization",
        "classification",
        "instruction_response",
    ]

    TASK_CONFIGS = {
        "qa_generation": {
            "description": "Generate question-answer pairs from text",
            "default_params": {
                "num_pairs": 3,
            },
            "required_params": [],
        },
        "summarization": {
            "description": "Generate summaries of text",
            "default_params": {
                "num_summaries": 1,
                "length": "medium",
            },
            "required_params": [],
        },
        "classification": {
            "description": "Classify text into categories",
            "default_params": {
                "categories": ["informative", "question", "instruction", "other"],
            },
            "required_params": [],
        },
        "instruction_response": {
            "description": "Generate instruction-response pairs",
            "default_params": {
                "num_examples": 2,
            },
            "required_params": [],
        },
    }

    def __init__(self) -> None:
        self.logger = logger

    def get_supported_tasks(self) -> list[str]:
        """Return list of supported task types."""
        return self.SUPPORTED_TASKS

    def get_task_config(self, task_type: str) -> dict[str, Any] | None:
        """Get configuration for a specific task type."""
        return self.TASK_CONFIGS.get(task_type)

    def validate_task(
        self, task_type: str, params: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """
        Validate task type and parameters.

        Args:
            task_type: Type of task to validate
            params: Parameters for the task

        Returns:
            Tuple of (is_valid, error_message)
        """
        if task_type not in self.SUPPORTED_TASKS:
            return False, f"Unsupported task type: {task_type}"

        config = self.TASK_CONFIGS.get(task_type, {})

        # Check required parameters
        required = config.get("required_params", [])
        if params:
            for param in required:
                if param not in params:
                    return False, f"Missing required parameter: {param}"

        return True, ""

    async def execute_task(
        self,
        task_type: str,
        text: str,
        params: dict[str, Any] | None = None,
        generator: TaskGenerator | None = None,
    ) -> dict[str, Any]:
        """
        Execute a task with the given text and parameters.

        Args:
            task_type: Type of task to execute
            text: Input text
            params: Task parameters
            generator: Task generator instance

        Returns:
            Dictionary with task results
        """
        # Validate task
        is_valid, error = self.validate_task(task_type, params)
        if not is_valid:
            return {
                "status": "error",
                "task_type": task_type,
                "error": error,
                "examples": [],
            }

        # Merge params with defaults
        config = self.TASK_CONFIGS.get(task_type, {})
        default_params = config.get("default_params", {}).copy()
        if params:
            default_params.update(params)

        # Use provided generator or create new one
        if generator is None:
            generator = TaskGenerator()

        try:
            examples = await generator.generate_task(text, task_type, **default_params)

            return {
                "status": "success",
                "task_type": task_type,
                "examples": examples,
                "example_count": len(examples),
            }

        except Exception as e:
            self.logger.error(f"Error executing task {task_type}: {e}")
            return {
                "status": "error",
                "task_type": task_type,
                "error": str(e),
                "examples": [],
            }

    async def execute_multiple_tasks(
        self,
        task_types: list[str],
        text: str,
        params: dict[str, Any] | None = None,
        generator: TaskGenerator | None = None,
    ) -> dict[str, Any]:
        """
        Execute multiple tasks on the same text.

        Args:
            task_types: List of task types to execute
            text: Input text
            params: Parameters for all tasks
            generator: Task generator instance

        Returns:
            Dictionary with results for all tasks
        """
        if generator is None:
            generator = TaskGenerator()

        # Create tasks for parallel execution
        tasks = [
            self.execute_task(task_type, text, params, generator)
            for task_type in task_types
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_type": task_types[i],
                    "status": "error",
                    "error": str(result),
                    "examples": [],
                })
            else:
                processed_results.append(result)

        # Calculate totals
        total_examples = sum(
            r.get("example_count", 0) for r in processed_results
        )

        return {
            "status": "success",
            "task_results": processed_results,
            "total_examples": total_examples,
            "task_count": len(task_types),
        }

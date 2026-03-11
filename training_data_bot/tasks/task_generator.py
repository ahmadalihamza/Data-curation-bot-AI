"""LLM-powered task generator for creating training examples."""

import json
from typing import Any

from openai import AsyncOpenAI

from training_data_bot.core.config import get_settings
from training_data_bot.core.logging import get_logger

settings = get_settings()
logger = get_logger("task_generator")


class TaskGenerator:
    """Generator for creating training data tasks using LLM."""

    # Groq API base URL
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """
        Initialize the task generator.

        Args:
            api_key: Groq API key (defaults to settings.LLM_API_KEY)
            model: Model to use (defaults to settings.LLM_MODEL)
        """
        self.api_key = api_key or settings.LLM_API_KEY
        self.model = model or settings.LLM_MODEL
        self.provider = settings.LLM_PROVIDER.lower()

        if self.api_key == "replace_me":
            logger.warning("LLM_API_KEY not set - using placeholder mode")

        self.client = None
        if self.api_key and self.api_key != "replace_me":
            if self.provider == "groq":
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.GROQ_BASE_URL,
                )
            else:
                self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate_qa_pairs(
        self, text: str, num_pairs: int = 3
    ) -> list[dict[str, str]]:
        """
        Generate question-answer pairs from text.

        Args:
            text: Input text to generate QA pairs from
            num_pairs: Number of QA pairs to generate

        Returns:
            List of QA pair dictionaries with 'question' and 'answer' keys
        """
        if not self.client:
            return self._generate_placeholder_qa(num_pairs)

        prompt = f"""Based on the following text, generate {num_pairs} question-answer pairs that would be useful for training a language model. 
The questions should be clear and specific. The answers should be complete and based only on information in the text.

Text:
{text[:2000]}

Generate {num_pairs} QA pairs in JSON format:
[{{"question": "...", "answer": "..."}}]"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates training data."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if content:
                try:
                    result = json.loads(content)
                    if isinstance(result, list):
                        return result
                    elif isinstance(result, dict) and "qa_pairs" in result:
                        return result["qa_pairs"]
                    elif isinstance(result, dict) and "pairs" in result:
                        return result["pairs"]
                except json.JSONDecodeError:
                    pass

            return self._generate_placeholder_qa(num_pairs)

        except Exception as e:
            logger.error(f"Error generating QA pairs: {e}")
            return self._generate_placeholder_qa(num_pairs)

    async def generate_summaries(
        self, text: str, num_summaries: int = 1
    ) -> list[dict[str, str]]:
        """
        Generate summaries from text.

        Args:
            text: Input text to summarize
            num_summaries: Number of summaries to generate

        Returns:
            List of summary dictionaries with 'summary' and optional 'length' keys
        """
        if not self.client:
            return [{"summary": text[:200], "length": "short"}]

        prompt = f"""Generate a concise summary of the following text. 
The summary should capture the main points and key information.

Text:
{text[:2000]}

Generate the summary in JSON format:
{{"summary": "...", "length": "short/medium/long"}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
            )

            content = response.choices[0].message.content
            if content:
                try:
                    result = json.loads(content)
                    if isinstance(result, dict):
                        return [result]
                except json.JSONDecodeError:
                    pass

            return [{"summary": text[:200], "length": "short"}]

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return [{"summary": text[:200], "length": "short"}]

    async def generate_classification_examples(
        self, text: str, categories: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Generate classification examples from text.

        Args:
            text: Input text to generate classification examples from
            categories: List of categories to classify text into

        Returns:
            List of classification examples with 'text', 'label', and 'confidence' keys
        """
        if categories is None:
            categories = ["informative", "question", "instruction", "other"]

        if not self.client:
            return [
                {"text": text[:100], "label": categories[0], "confidence": 0.8}
            ]

        categories_str = ", ".join(categories)

        prompt = f"""Analyze the following text and generate classification examples.
Classify the text into one of these categories: {categories_str}

Text:
{text[:2000]}

Generate classification examples in JSON format as a list:
[{{"text": "...", "label": "...", "confidence": 0.0-1.0}}]"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates classification data."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
            )

            content = response.choices[0].message.content
            if content:
                try:
                    result = json.loads(content)
                    if isinstance(result, list):
                        return result
                    elif isinstance(result, dict) and "examples" in result:
                        return result["examples"]
                except json.JSONDecodeError:
                    pass

            return [{"text": text[:100], "label": categories[0], "confidence": 0.8}]

        except Exception as e:
            logger.error(f"Error generating classification examples: {e}")
            return [{"text": text[:100], "label": categories[0], "confidence": 0.8}]

    async def generate_instruction_response(
        self, text: str, num_examples: int = 2
    ) -> list[dict[str, str]]:
        """
        Generate instruction-response pairs from text.

        Args:
            text: Input text to generate instruction-response pairs from
            num_examples: Number of examples to generate

        Returns:
            List of instruction-response dictionaries with 'instruction' and 'response' keys
        """
        if not self.client:
            return self._generate_placeholder_instruction_response(num_examples)

        prompt = f"""Based on the following text, generate {num_examples} instruction-response pairs for training a language model.
The instructions should be natural questions or commands. The responses should be informative and based on the text.

Text:
{text[:2000]}

Generate {num_examples} instruction-response pairs in JSON format:
[{{"instruction": "...", "response": "..."}}]"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates instruction-response training data."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if content:
                try:
                    result = json.loads(content)
                    if isinstance(result, list):
                        return result
                    elif isinstance(result, dict) and "pairs" in result:
                        return result["pairs"]
                except json.JSONDecodeError:
                    pass

            return self._generate_placeholder_instruction_response(num_examples)

        except Exception as e:
            logger.error(f"Error generating instruction-response pairs: {e}")
            return self._generate_placeholder_instruction_response(num_examples)

    async def generate_task(
        self, text: str, task_type: str, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Generate training examples for a specific task type.

        Args:
            text: Input text
            task_type: Type of task ('qa_generation', 'summarization', 'classification', 'instruction_response')
            **kwargs: Additional parameters specific to each task type

        Returns:
            List of generated examples
        """
        if task_type == "qa_generation":
            num_pairs = kwargs.get("num_pairs", 3)
            return await self.generate_qa_pairs(text, num_pairs)
        elif task_type == "summarization":
            num_summaries = kwargs.get("num_summaries", 1)
            return await self.generate_summaries(text, num_summaries)
        elif task_type == "classification":
            categories = kwargs.get("categories")
            return await self.generate_classification_examples(text, categories)
        elif task_type == "instruction_response":
            num_examples = kwargs.get("num_examples", 2)
            return await self.generate_instruction_response(text, num_examples)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return []

    def _generate_placeholder_qa(self, num_pairs: int) -> list[dict[str, str]]:
        """Generate placeholder QA pairs when API is not available."""
        return [
            {
                "question": f"What is the main topic discussed in the text?",
                "answer": "The text contains information about the topic.",
            }
            for _ in range(num_pairs)
        ]

    def _generate_placeholder_instruction_response(
        self, num_examples: int
    ) -> list[dict[str, str]]:
        """Generate placeholder instruction-response pairs when API is not available."""
        return [
            {
                "instruction": "Can you provide more information about this?",
                "response": "The text contains relevant information on this topic.",
            }
            for _ in range(num_examples)
        ]

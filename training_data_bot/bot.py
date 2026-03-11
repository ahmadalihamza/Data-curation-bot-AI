"""Main orchestrator for the Training Data Bot."""

import asyncio
import uuid
from datetime import datetime
from typing import Any

from training_data_bot.core.config import get_settings
from training_data_bot.core.logging import get_logger
from training_data_bot.evaluation.quality_evaluator import QualityEvaluator
from training_data_bot.preprocessing.text_preprocessor import TextPreprocessor
from training_data_bot.sources.unified import UnifiedLoader
from training_data_bot.storage.dataset_exporter import DatasetExporter
from training_data_bot.tasks.task_manager import TaskManager

settings = get_settings()
logger = get_logger("training_data_bot")


class TrainingDataBot:
    """
    Main orchestrator class that coordinates all components:
    - loading documents
    - preprocessing text
    - generating tasks/examples
    - evaluating quality
    - exporting datasets
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.logger = logger
        self.config = config or {}
        self.documents: dict[str, Any] = {}
        self.datasets: dict[str, Any] = {}
        self.jobs: dict[str, Any] = {}

        # Initialize components
        self.loader = UnifiedLoader()
        self.preprocessor = TextPreprocessor()
        self.task_manager = TaskManager()
        self.quality_evaluator = QualityEvaluator()
        self.exporter = DatasetExporter()

        # Processing settings
        self.chunk_size = self.config.get("chunk_size", settings.CHUNK_SIZE)
        self.chunk_overlap = self.config.get("chunk_overlap", settings.CHUNK_OVERLAP)
        self.max_workers = self.config.get("max_workers", settings.MAX_WORKERS)

        self.logger.info("TrainingDataBot initialized successfully.")

    async def __aenter__(self) -> "TrainingDataBot":
        self.logger.info("Entering TrainingDataBot context.")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup()

    async def load_documents(self, sources: list[str] | str) -> list[dict[str, Any]]:
        """
        Load documents from sources.

        Args:
            sources: Single source or list of sources (file paths or URLs)

        Returns:
            List of loaded documents with content and metadata
        """
        if isinstance(sources, str):
            sources = [sources]

        self.logger.info(f"Loading {len(sources)} documents...")

        results = await self.loader.load_batch(sources, max_concurrent=self.max_workers)

        # Store documents with IDs
        documents = []
        for result in results:
            doc_id = str(uuid.uuid4())
            result["id"] = doc_id

            if result.get("status") == "success":
                self.documents[doc_id] = result
                self.logger.info(f"Loaded document: {result.get('source', 'unknown')}")
            else:
                self.logger.warning(f"Failed to load: {result.get('source', 'unknown')}")

            documents.append(result)

        self.logger.info(f"Loaded {len(documents)} documents.")
        return documents

    async def load_directory(
        self, directory: str, extensions: list[str] | None = None, recursive: bool = True
    ) -> list[dict[str, Any]]:
        """
        Load all files in a directory.

        Args:
            directory: Path to directory
            extensions: List of file extensions to include
            recursive: Whether to search recursively

        Returns:
            List of loaded documents
        """
        self.logger.info(f"Loading directory: {directory}")
        results = await self.loader.load_directory(directory, extensions, recursive)

        # Store documents with IDs
        documents = []
        for result in results:
            doc_id = str(uuid.uuid4())
            result["id"] = doc_id

            if result.get("status") == "success":
                self.documents[doc_id] = result
            documents.append(result)

        return documents

    async def process_documents(
        self,
        documents: list[dict[str, Any]] | None = None,
        task_types: list[str] | None = None,
        quality_filter: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Process documents and generate training examples.

        Args:
            documents: List of documents to process (uses loaded documents if None)
            task_types: List of task types to generate
            quality_filter: Whether to filter by quality
            **kwargs: Additional parameters for task generation

        Returns:
            Dictionary with processing results
        """
        if documents is None:
            documents = list(self.documents.values())

        if task_types is None:
            task_types = ["qa_generation"]

        if not documents:
            return {
                "status": "error",
                "error": "No documents to process",
                "documents_received": 0,
                "task_types": task_types,
            }

        # Create a job for tracking
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "document_count": len(documents),
            "task_types": task_types,
        }
        self.jobs[job_id] = job

        try:
            self.logger.info(f"Processing {len(documents)} documents with tasks: {task_types}")

            all_examples = []
            results_by_document = []

            for doc in documents:
                if doc.get("status") != "success":
                    continue

                content = doc.get("content", "")
                if not content:
                    continue

                # Chunk text if needed
                chunks = self.preprocessor.chunk_text(
                    content,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )

                doc_examples = []
                for chunk in chunks:
                    # Generate examples for each task type
                    task_result = await self.task_manager.execute_multiple_tasks(
                        task_types, chunk, params=kwargs
                    )

                    if task_result.get("status") == "success":
                        for task_result_item in task_result.get("task_results", []):
                            examples = task_result_item.get("examples", [])
                            for example in examples:
                                example["source"] = doc.get("source")
                                example["task_type"] = task_result_item.get("task_type")
                                example["chunk_preview"] = chunk[:100]
                            doc_examples.extend(examples)

                results_by_document.append({
                    "document": doc.get("source"),
                    "chunks": len(chunks),
                    "examples": len(doc_examples),
                })

                all_examples.extend(doc_examples)

            # Evaluate quality if enabled
            if quality_filter and all_examples:
                quality_result = await self.evaluate_dataset(all_examples)
                filtered_examples = [
                    ex for ex in all_examples
                    if quality_result.get("passed_examples", {}).get(ex.get("id", ""), True)
                ]

                job["quality_filtered"] = True
                job["original_count"] = len(all_examples)
                job["filtered_count"] = len(filtered_examples)
                all_examples = filtered_examples
            else:
                job["quality_filtered"] = False

            # Create dataset
            dataset_id = str(uuid.uuid4())
            dataset = {
                "id": dataset_id,
                "examples": all_examples,
                "created_at": datetime.utcnow().isoformat(),
                "task_types": task_types,
                "document_count": len(documents),
            }
            self.datasets[dataset_id] = dataset

            # Update job status
            job["status"] = "completed"
            job["completed_at"] = datetime.utcnow().isoformat()
            job["dataset_id"] = dataset_id
            job["example_count"] = len(all_examples)

            return {
                "status": "success",
                "job_id": job_id,
                "dataset_id": dataset_id,
                "dataset": dataset,  # Include dataset directly for UI
                "documents_received": len(documents),
                "task_types": task_types,
                "quality_filter": quality_filter,
                "example_count": len(all_examples),
                "results_by_document": results_by_document,
            }

        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            job["status"] = "failed"
            job["error"] = str(e)
            return {
                "status": "error",
                "error": str(e),
                "job_id": job_id,
                "documents_received": len(documents),
                "task_types": task_types,
                "quality_filter": quality_filter,
            }

    async def evaluate_dataset(self, dataset: list[dict] | dict) -> dict[str, Any]:
        """
        Evaluate the quality of a dataset.

        Args:
            dataset: Dataset to evaluate (list of examples or dataset dict)

        Returns:
            Dictionary with evaluation results
        """
        # Extract examples from dataset if it's a dict
        if isinstance(dataset, dict):
            examples = dataset.get("examples", [])
        else:
            examples = dataset

        self.logger.info(f"Evaluating {len(examples)} examples...")

        result = self.quality_evaluator.evaluate(examples)

        return result

    async def export_dataset(
        self,
        dataset: list[dict] | dict | str,
        output_path: str,
        format: str = "jsonl",
    ) -> dict[str, Any]:
        """
        Export a dataset to a file.

        Args:
            dataset: Dataset to export (list, dict, or dataset ID)
            output_path: Path to save the exported file
            format: Export format ('json', 'jsonl', 'csv')

        Returns:
            Dictionary with export results
        """
        # Handle dataset ID
        if isinstance(dataset, str):
            dataset = self.datasets.get(dataset)
            if not dataset:
                return {
                    "status": "error",
                    "error": f"Dataset not found: {dataset}",
                }

        # Extract examples if it's a dict
        if isinstance(dataset, dict):
            examples = dataset.get("examples", [])
        else:
            examples = dataset

        self.logger.info(f"Exporting {len(examples)} examples to {output_path}...")

        try:
            result = self.exporter.export(examples, output_path, format)

            return {
                "status": "success",
                "output_path": result.get("output_path"),
                "format": format,
                "example_count": len(examples),
            }

        except Exception as e:
            self.logger.error(f"Error exporting dataset: {e}")
            return {
                "status": "error",
                "error": str(e),
                "output_path": output_path,
                "format": format,
            }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get job status by ID."""
        return self.jobs.get(job_id)

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """Get dataset by ID."""
        return self.datasets.get(dataset_id)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document by ID."""
        return self.documents.get(document_id)

    def list_documents(self) -> list[dict[str, Any]]:
        """List all loaded documents."""
        return [
            {
                "id": doc_id,
                "source": doc.get("source"),
                "doc_type": doc.get("doc_type"),
                "status": doc.get("status"),
                "char_count": doc.get("metadata", {}).get("char_count", 0),
            }
            for doc_id, doc in self.documents.items()
        ]

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all created datasets."""
        return [
            {
                "id": dataset_id,
                "example_count": len(dataset.get("examples", [])),
                "created_at": dataset.get("created_at"),
                "task_types": dataset.get("task_types", []),
            }
            for dataset_id, dataset in self.datasets.items()
        ]

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs."""
        return [
            {
                "id": job_id,
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "completed_at": job.get("completed_at"),
            }
            for job_id, job in self.jobs.items()
        ]

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        await self.loader.close()
        self.logger.info("Cleanup complete.")

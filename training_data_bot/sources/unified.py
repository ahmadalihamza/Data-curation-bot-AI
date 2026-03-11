"""Unified document loader that routes to appropriate loader based on source type."""

import asyncio
from pathlib import Path
from typing import Any

from training_data_bot.sources.document_loader import DocumentLoader
from training_data_bot.sources.pdf_loader import PDFLoader
from training_data_bot.sources.web_loader import WebLoader


class UnifiedLoader:
    """Unified loader that handles multiple document types."""

    def __init__(self) -> None:
        self.document_loader = DocumentLoader()
        self.pdf_loader = PDFLoader()
        self.web_loader = WebLoader()

    async def load_single(self, source: str) -> dict[str, Any]:
        """
        Load a single source.

        Args:
            source: File path or URL

        Returns:
            Dictionary with loaded content and metadata
        """
        # Check if it's a URL
        if self._is_url(source):
            return await self.web_loader.load_single(source)

        # Check file extension
        suffix = Path(source).suffix.lower()

        if suffix == ".pdf":
            return await self.pdf_loader.load_single(source)

        # Default to document loader
        return await self.document_loader.load_single(source)

    async def load_batch(
        self, sources: list[str], max_concurrent: int = 4
    ) -> list[dict[str, Any]]:
        """
        Load multiple sources concurrently.

        Args:
            sources: List of file paths or URLs
            max_concurrent: Maximum number of concurrent loads

        Returns:
            List of loaded documents
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def load_with_limit(source: str) -> dict[str, Any]:
            async with semaphore:
                return await self.load_single(source)

        tasks = [load_with_limit(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "source": sources[i],
                    "doc_type": "unknown",
                    "content": "",
                    "status": "error",
                    "error": str(result),
                })
            else:
                processed_results.append(result)

        return processed_results

    async def load_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Load all files in a directory.

        Args:
            directory: Path to directory
            extensions: List of extensions to include (e.g., ['.txt', '.pdf'])
            recursive: Whether to search recursively

        Returns:
            List of loaded documents
        """
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            return []

        # Default extensions if not specified
        if extensions is None:
            extensions = [".txt", ".md", ".pdf", ".docx", ".json", ".xml", ".csv"]

        # Collect files
        files = []
        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            files.extend(dir_path.glob(f"{pattern}{ext}"))

        # Convert to string paths
        sources = [str(f) for f in files]

        if not sources:
            return []

        return await self.load_batch(sources)

    def detect_type(self, source: str) -> str:
        """
        Detect the type of a source without loading it.

        Args:
            source: File path or URL

        Returns:
            Type string: 'url', 'pdf', 'document', or 'unknown'
        """
        if self._is_url(source):
            return "url"

        suffix = Path(source).suffix.lower()

        if suffix == ".pdf":
            return "pdf"

        if suffix in {".txt", ".md", ".json", ".xml", ".csv", ".log", ".docx"}:
            return "document"

        return "unknown"

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        return source.startswith(("http://", "https://"))

    async def close(self) -> None:
        """Close any resources used by loaders."""
        await self.web_loader.close()

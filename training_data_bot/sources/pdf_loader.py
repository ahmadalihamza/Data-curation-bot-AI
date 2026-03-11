"""PDF loader using PyMuPDF (fitz)."""

import os
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from training_data_bot.sources.base import BaseLoader


class PDFLoader(BaseLoader):
    """Loader for PDF documents using PyMuPDF."""

    async def load_single(self, source: str) -> dict[str, Any]:
        """
        Load a single PDF file.

        Args:
            source: Path to the PDF file

        Returns:
            Dictionary containing source, content, and metadata
        """
        file_path = Path(source)

        if not file_path.exists():
            self.logger.warning(f"PDF file not found: {source}")
            return {
                "source": source,
                "doc_type": "pdf",
                "content": "",
                "status": "error",
                "error": f"File not found: {source}",
            }

        if file_path.suffix.lower() != ".pdf":
            self.logger.warning(f"File is not a PDF: {source}")
            return {
                "source": source,
                "doc_type": "pdf",
                "content": "",
                "status": "error",
                "error": f"Not a PDF file: {source}",
            }

        try:
            return await self._load_pdf(file_path, source)
        except Exception as e:
            self.logger.error(f"Error reading PDF {source}: {e}")
            return {
                "source": source,
                "doc_type": "pdf",
                "content": "",
                "status": "error",
                "error": str(e),
            }

    async def _load_pdf(self, file_path: Path, source: str) -> dict[str, Any]:
        """Load PDF content and metadata using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)

            # Extract text from all pages
            content_parts = []
            page_texts = []

            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text("text")
                text = text.strip()

                if text:
                    page_texts.append(text)
                    content_parts.append(text)

            content = "\n\n".join(content_parts)

            # Get metadata
            metadata = doc.metadata
            file_stat = os.stat(file_path)

            # Get page dimensions for first page
            first_page = doc[0]
            first_page_rect = first_page.rect

            result = {
                "source": source,
                "doc_type": "pdf",
                "content": content,
                "status": "success",
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_stat.st_size,
                    "extension": ".pdf",
                    "page_count": page_count,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                    "first_page_width": first_page_rect.width,
                    "first_page_height": first_page_rect.height,
                    "char_count": len(content),
                    "has_text": len(content) > 0,
                },
            }

            # Close the document
            doc.close()

            return result

        except Exception as e:
            self.logger.error(f"Error loading PDF {source}: {e}")
            return {
                "source": source,
                "doc_type": "pdf",
                "content": "",
                "status": "error",
                "error": str(e),
            }

    async def load_page(self, source: str, page_num: int) -> dict[str, Any]:
        """
        Load a specific page from a PDF.

        Args:
            source: Path to the PDF file
            page_num: Page number (0-indexed)

        Returns:
            Dictionary containing page content and metadata
        """
        file_path = Path(source)

        if not file_path.exists():
            return {
                "source": source,
                "doc_type": "pdf",
                "content": "",
                "status": "error",
                "error": f"File not found: {source}",
            }

        try:
            doc = fitz.open(file_path)

            if page_num < 0 or page_num >= len(doc):
                doc.close()
                return {
                    "source": source,
                    "doc_type": "pdf",
                    "content": "",
                    "status": "error",
                    "error": f"Page number out of range: {page_num}",
                }

            page = doc[page_num]
            text = page.get_text("text").strip()

            result = {
                "source": source,
                "doc_type": "pdf",
                "content": text,
                "status": "success",
                "metadata": {
                    "page_num": page_num,
                    "total_pages": len(doc),
                    "page_width": page.rect.width,
                    "page_height": page.rect.height,
                    "char_count": len(text),
                },
            }

            doc.close()
            return result

        except Exception as e:
            self.logger.error(f"Error loading PDF page {source}:{page_num}: {e}")
            return {
                "source": source,
                "doc_type": "pdf",
                "content": "",
                "status": "error",
                "error": str(e),
            }

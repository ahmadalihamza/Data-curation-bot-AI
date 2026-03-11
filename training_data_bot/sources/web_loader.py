"""Web content loader using httpx and BeautifulSoup."""

import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from training_data_bot.sources.base import BaseLoader


class WebLoader(BaseLoader):
    """Loader for web content."""

    # Tags to remove from HTML
    REMOVE_TAGS = {
        "script",
        "style",
        "noscript",
        "iframe",
        "nav",
        "header",
        "footer",
        "aside",
    }

    # Common patterns for noise
    NOISE_PATTERNS = [
        re.compile(r"cookie\s*notice", re.IGNORECASE),
        re.compile(r"privacy\s*policy", re.IGNORECASE),
        re.compile(r"terms\s*of\s*service", re.IGNORECASE),
        re.compile(r"subscribe\s*to\s*newsletter", re.IGNORECASE),
        re.compile(r"advertisement", re.IGNORECASE),
    ]

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__()
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Training-Data-Bot/1.0 (document loader)",
                },
            )
        return self.client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def load_single(self, source: str) -> dict[str, Any]:
        """
        Load content from a web URL.

        Args:
            source: URL to fetch

        Returns:
            Dictionary containing source, content, and metadata
        """
        if not source.startswith(("http://", "https://")):
            self.logger.warning(f"Invalid URL: {source}")
            return {
                "source": source,
                "doc_type": "url",
                "content": "",
                "status": "error",
                "error": "Invalid URL - must start with http:// or https://",
            }

        try:
            client = await self._get_client()
            response = await client.get(source)

            if response.status_code != 200:
                self.logger.warning(f"HTTP {response.status_code} for {source}")
                return {
                    "source": source,
                    "doc_type": "url",
                    "content": "",
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "metadata": {
                        "status_code": response.status_code,
                        "final_url": str(response.url),
                    },
                }

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type:
                return await self._parse_html(response, source)
            elif "text/plain" in content_type:
                return {
                    "source": source,
                    "doc_type": "url",
                    "content": response.text,
                    "status": "success",
                    "metadata": {
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "final_url": str(response.url),
                        "char_count": len(response.text),
                    },
                }
            else:
                self.logger.warning(f"Unsupported content type: {content_type}")
                return {
                    "source": source,
                    "doc_type": "url",
                    "content": "",
                    "status": "error",
                    "error": f"Unsupported content type: {content_type}",
                }

        except httpx.TimeoutException:
            self.logger.error(f"Timeout fetching {source}")
            return {
                "source": source,
                "doc_type": "url",
                "content": "",
                "status": "error",
                "error": "Request timed out",
            }
        except httpx.RequestError as e:
            self.logger.error(f"Request error for {source}: {e}")
            return {
                "source": source,
                "doc_type": "url",
                "content": "",
                "status": "error",
                "error": str(e),
            }
        except Exception as e:
            self.logger.error(f"Error fetching {source}: {e}")
            return {
                "source": source,
                "doc_type": "url",
                "content": "",
                "status": "error",
                "error": str(e),
            }

    async def _parse_html(
        self, response: httpx.Response, source: str
    ) -> dict[str, Any]:
        """Parse HTML content and extract clean text."""
        try:
            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted tags
            for tag in soup.find_all(self.REMOVE_TAGS):
                tag.decompose()

            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith("<!--")):
                comment.extract()

            # Try to find main content areas
            content = self._extract_main_content(soup)

            # Clean up whitespace
            content = self._normalize_whitespace(content)

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string or ""
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            return {
                "source": source,
                "doc_type": "url",
                "content": content,
                "status": "success",
                "metadata": {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "final_url": str(response.url),
                    "title": title,
                    "char_count": len(content),
                    "word_count": len(content.split()),
                },
            }

        except Exception as e:
            self.logger.error(f"Error parsing HTML from {source}: {e}")
            return {
                "source": source,
                "doc_type": "url",
                "content": "",
                "status": "error",
                "error": f"HTML parsing error: {str(e)}",
            }

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML, trying common content selectors."""
        # Try common main content areas
        content_selectors = [
            "article",
            "main",
            "[role='main']",
            ".content",
            ".main-content",
            ".post-content",
            ".article-content",
            "#content",
            "#main",
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem and content_elem.get_text(strip=True):
                break

        if content_elem is None:
            # Fall back to body
            content_elem = soup.body if soup.body else soup

        # Extract text
        text = content_elem.get_text(separator="\n", strip=True)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split("\n")
        cleaned_lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in lines]

        # Remove empty lines at start and end
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

class TextPreprocessor:
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> list[str]:
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])

            if end >= len(text):
                break

            start = end - chunk_overlap

        return chunks
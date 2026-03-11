"""Dataset exporter for saving training data in various formats."""

import csv
import json
import zipfile
from io import StringIO
from pathlib import Path
from typing import Any

from training_data_bot.core.logging import get_logger

logger = get_logger("dataset_exporter")


class DatasetExporter:
    """Exporter for saving training data in various formats."""

    def __init__(self) -> None:
        self.logger = logger

    def export(
        self,
        examples: list[dict],
        output_path: str,
        format: str = "jsonl",
    ) -> dict[str, Any]:
        """
        Export examples to a file.

        Args:
            examples: List of examples to export
            output_path: Path to save the file
            format: Export format ('json', 'jsonl', 'csv', 'zip')

        Returns:
            Dictionary with export results
        """
        format = format.lower()

        try:
            if format == "json":
                return self._export_json(examples, output_path)
            elif format == "jsonl":
                return self._export_jsonl(examples, output_path)
            elif format == "csv":
                return self._export_csv(examples, output_path)
            elif format == "zip":
                return self._export_zip(examples, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {e}")
            raise

    def _export_json(self, examples: list[dict], output_path: str) -> dict[str, Any]:
        """Export examples as JSON."""
        path = Path(output_path)
        
        # Add .json extension if not present
        if path.suffix != ".json":
            path = path.with_suffix(".json")

        # Prepare data with metadata
        export_data = {
            "metadata": {
                "example_count": len(examples),
                "export_format": "json",
            },
            "examples": examples,
        }

        path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "output_path": str(path),
            "example_count": len(examples),
            "format": "json",
        }

    def _export_jsonl(self, examples: list[dict], output_path: str) -> dict[str, Any]:
        """Export examples as JSONL (JSON Lines)."""
        path = Path(output_path)
        
        # Add .jsonl extension if not present
        if path.suffix not in (".jsonl", ".jsonl"):
            path = path.with_suffix(".jsonl")

        lines = []
        for example in examples:
            lines.append(json.dumps(example, ensure_ascii=False))

        path.write_text("\n".join(lines), encoding="utf-8")

        return {
            "output_path": str(path),
            "example_count": len(examples),
            "format": "jsonl",
        }

    def _export_csv(self, examples: list[dict], output_path: str) -> dict[str, Any]:
        """Export examples as CSV."""
        path = Path(output_path)
        
        # Add .csv extension if not present
        if path.suffix != ".csv":
            path = path.with_suffix(".csv")

        if not examples:
            # Create empty CSV with headers
            path.write_text("", encoding="utf-8")
            return {
                "output_path": str(path),
                "example_count": 0,
                "format": "csv",
            }

        # Get all unique keys from examples
        fieldnames = set()
        for example in examples:
            fieldnames.update(example.keys())

        # Sort fieldnames for consistent output
        fieldnames = sorted(fieldnames)

        # Write CSV
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(examples)

        return {
            "output_path": str(path),
            "example_count": len(examples),
            "format": "csv",
        }

    def _export_zip(self, examples: list[dict], output_path: str) -> dict[str, Any]:
        """Export examples as a ZIP archive with multiple formats."""
        path = Path(output_path)
        
        # Add .zip extension if not present
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add JSON file
            json_data = json.dumps({"examples": examples}, indent=2, ensure_ascii=False)
            zf.writestr("data.json", json_data)

            # Add JSONL file
            jsonl_lines = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
            zf.writestr("data.jsonl", jsonl_lines)

            # Add CSV file
            if examples:
                fieldnames = sorted(set().union(*(ex.keys() for ex in examples)))
                output = StringIO()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(examples)
                zf.writestr("data.csv", output.getvalue())

            # Add metadata file
            metadata = {
                "example_count": len(examples),
                "export_format": "zip",
                "files": ["data.json", "data.jsonl", "data.csv"],
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        return {
            "output_path": str(path),
            "example_count": len(examples),
            "format": "zip",
        }

    def export_json(self, examples: list[dict], output_path: str) -> str:
        """Export examples as JSON (legacy method)."""
        result = self._export_json(examples, output_path)
        return result["output_path"]

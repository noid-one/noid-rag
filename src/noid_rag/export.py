"""Export chunks and search results to JSON, JSONL, CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any


def export(data: list[Any], output: str | Path, fmt: str | None = None) -> str:
    """Export data to file. Auto-detects format from extension if fmt is None.

    Returns the output path as string.
    """
    output = Path(output)
    if fmt is None:
        fmt = _detect_format(output)

    content = format_data(data, fmt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return str(output)


def format_data(data: list[Any], fmt: str) -> str:
    """Format data as string in the given format."""
    records = [_to_dict(item) for item in data]

    if fmt == "json":
        return json.dumps(records, indent=2, default=str)
    elif fmt == "jsonl":
        return "\n".join(json.dumps(r, default=str) for r in records) + "\n"
    elif fmt == "csv":
        return _to_csv(records)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use json, jsonl, or csv.")


def _detect_format(path: Path) -> str:
    """Detect format from file extension."""
    ext = path.suffix.lower()
    mapping = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}
    if ext not in mapping:
        raise ValueError(f"Cannot detect format from extension '{ext}'. Use json, jsonl, or csv.")
    return mapping[ext]


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass or dict to dict."""
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        # Remove private/internal fields
        d.pop("_docling_doc", None)
        # Remove embedding vectors (too large for export)
        d.pop("embedding", None)
        return d
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _to_csv(records: list[dict[str, Any]]) -> str:
    """Convert records to CSV, flattening nested metadata."""
    if not records:
        return ""

    # Flatten metadata into top-level keys
    flat_records = []
    for r in records:
        flat = {}
        for k, v in r.items():
            if k == "metadata" and isinstance(v, dict):
                for mk, mv in v.items():
                    flat[f"metadata.{mk}"] = mv
            else:
                flat[k] = v if not isinstance(v, (list, dict)) else json.dumps(v, default=str)
        flat_records.append(flat)

    # Collect all keys preserving insertion order
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in flat_records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat_records)
    return output.getvalue()

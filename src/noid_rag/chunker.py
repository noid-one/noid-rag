"""Document chunking via Docling HybridChunker."""

from __future__ import annotations

from noid_rag.config import ChunkerConfig
from noid_rag.models import Chunk, Document


def chunk(doc: Document, config: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk a document using Docling's HybridChunker.

    Requires doc._docling_doc to be set (preserved by parser).
    Falls back to fixed-size chunking if DoclingDocument is not available.
    """
    config = config or ChunkerConfig()

    if config.method == "hybrid" and doc._docling_doc is not None:
        return _hybrid_chunk(doc, config)
    return _fixed_chunk(doc, config)


def _hybrid_chunk(doc: Document, config: ChunkerConfig) -> list[Chunk]:
    """Structure-aware + token-aware chunking via Docling."""
    from docling_core.transforms.chunker import HybridChunker

    chunker = HybridChunker(
        tokenizer=config.tokenizer,
        max_tokens=config.max_tokens,
    )

    chunks = []
    for chunk_obj in chunker.chunk(doc._docling_doc):
        text = chunk_obj.text
        meta = {**doc.metadata}
        # Add heading info if available
        if hasattr(chunk_obj, "meta") and chunk_obj.meta:
            if hasattr(chunk_obj.meta, "headings") and chunk_obj.meta.headings:
                meta["headings"] = chunk_obj.meta.headings
            if hasattr(chunk_obj.meta, "page") and chunk_obj.meta.page is not None:
                meta["page"] = chunk_obj.meta.page

        chunks.append(
            Chunk(
                text=text,
                document_id=doc.id,
                metadata=meta,
            )
        )

    return chunks


def _fixed_chunk(doc: Document, config: ChunkerConfig) -> list[Chunk]:
    """Simple fixed-size character chunking with overlap."""
    # Approximate chars per token
    chars_per_token = 4
    chunk_size = config.max_tokens * chars_per_token
    overlap = min(config.overlap * chars_per_token, chunk_size - 1)
    step = max(chunk_size - overlap, 1)  # always advance by at least 1 char

    text = doc.content
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    document_id=doc.id,
                    metadata={**doc.metadata, "chunk_method": "fixed"},
                )
            )

        start += step

    return chunks

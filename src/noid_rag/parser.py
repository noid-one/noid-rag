"""Document parsing via Docling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from noid_rag.config import ParserConfig
from noid_rag.models import Document

logger = logging.getLogger(__name__)


def parse(source: str | Path, config: ParserConfig | None = None) -> Document:
    """Parse a document using Docling's DocumentConverter.

    Returns a Document with markdown content and preserved DoclingDocument.
    """
    config = config or ParserConfig()
    source = Path(source)

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Configure PDF format option with OCR settings
    pdf_format_option = PdfFormatOption()
    if config.ocr_enabled:
        if config.ocr_engine == "easyocr":
            from docling.datamodel.pipeline_options import EasyOcrOptions

            pdf_format_option.pipeline_options.ocr_options = EasyOcrOptions()
        elif config.ocr_engine == "tesseract":
            from docling.datamodel.pipeline_options import TesseractOcrOptions

            pdf_format_option.pipeline_options.ocr_options = TesseractOcrOptions()
        # else: "auto" — keep Docling's default OcrAutoOptions
    else:
        pdf_format_option.pipeline_options.do_ocr = False

    format_options = {
        InputFormat.PDF: pdf_format_option,
    }

    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(str(source))

    docling_doc = result.document
    md_content = docling_doc.export_to_markdown()

    # Enforce max_pages by truncating at Docling's page-break markers
    if config.max_pages > 0:
        marker = "<!-- PageBreak -->"
        if marker in md_content:
            parts = md_content.split(marker)
            if len(parts) > config.max_pages:
                md_content = marker.join(parts[: config.max_pages])
        else:
            logger.warning(
                "max_pages=%d set but no page-break markers found in parsed output",
                config.max_pages,
            )

    metadata: dict[str, Any] = {
        "source_type": source.suffix.lstrip(".").lower(),
        "filename": source.name,
    }

    # Add page count if available
    if hasattr(docling_doc, "pages") and docling_doc.pages:
        page_count = len(docling_doc.pages)
        if config.max_pages > 0 and page_count > config.max_pages:
            page_count = config.max_pages
        metadata["page_count"] = page_count

    return Document(
        source=str(source),
        content=md_content,
        metadata=metadata,
        _docling_doc=docling_doc,
    )

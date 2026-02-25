"""Tests for document parser."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from noid_rag.config import ParserConfig


# Helper to create fresh mocks for the docling module tree.
def _docling_mocks():
    return {
        "docling": MagicMock(),
        "docling.document_converter": MagicMock(),
        "docling.datamodel": MagicMock(),
        "docling.datamodel.pipeline_options": MagicMock(),
        "docling.datamodel.base_models": MagicMock(),
    }


def _setup_converter(modules, *, md_content="# Test\n\nContent", pages=None):
    """Wire up a fake DocumentConverter inside the patched module dict."""
    mock_docling_doc = MagicMock()
    mock_docling_doc.export_to_markdown.return_value = md_content
    mock_docling_doc.pages = pages

    mock_result = MagicMock()
    mock_result.document = mock_docling_doc

    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result

    modules["docling.document_converter"].DocumentConverter.return_value = mock_converter
    return mock_docling_doc, mock_converter


class TestParser:
    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_returns_document(self, tmp_path):
        mock_docling_doc, _ = _setup_converter(
            sys.modules,
            md_content="# Test\n\nContent",
            pages=[MagicMock(), MagicMock()],
        )

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        from noid_rag.parser import parse

        doc = parse(test_file)

        assert doc.content == "# Test\n\nContent"
        assert doc.source == str(test_file)
        assert doc.id.startswith("doc_")

    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_preserves_docling_doc(self, tmp_path):
        mock_docling_doc, _ = _setup_converter(sys.modules, pages=[MagicMock()])

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        from noid_rag.parser import parse

        doc = parse(test_file)
        assert doc._docling_doc is mock_docling_doc

    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_sets_source_type_from_extension(self, tmp_path):
        _setup_converter(sys.modules, md_content="Hello", pages=None)

        test_file = tmp_path / "report.docx"
        test_file.write_bytes(b"fake docx")

        from noid_rag.parser import parse

        doc = parse(test_file)
        assert doc.metadata["source_type"] == "docx"
        assert doc.metadata["filename"] == "report.docx"

    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_sets_page_count(self, tmp_path):
        _setup_converter(
            sys.modules,
            md_content="pages",
            pages=[MagicMock(), MagicMock(), MagicMock()],
        )

        test_file = tmp_path / "big.pdf"
        test_file.write_bytes(b"fake")

        from noid_rag.parser import parse

        doc = parse(test_file)
        assert doc.metadata["page_count"] == 3

    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_no_page_count_when_pages_missing(self, tmp_path):
        _setup_converter(sys.modules, md_content="text", pages=None)

        test_file = tmp_path / "file.html"
        test_file.write_text("<html></html>")

        from noid_rag.parser import parse

        doc = parse(test_file)
        assert "page_count" not in doc.metadata

    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_with_custom_config(self, tmp_path):
        _setup_converter(sys.modules, md_content="Hello", pages=None)

        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body>Hello</body></html>")

        from noid_rag.parser import parse

        config = ParserConfig(ocr_enabled=False, ocr_engine="tesseract")
        doc = parse(test_file, config=config)

        assert doc.metadata["source_type"] == "html"
        assert "page_count" not in doc.metadata

    @patch.dict(sys.modules, _docling_mocks())
    def test_parse_uses_default_config(self, tmp_path):
        """When config is None, a default ParserConfig is used."""
        _setup_converter(sys.modules, md_content="default", pages=None)

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        from noid_rag.parser import parse

        doc = parse(test_file, config=None)
        assert doc.content == "default"


class TestParserOcrConfig:
    """Tests that exercise real docling classes to catch API mismatches."""

    @pytest.mark.parametrize("engine", ["easyocr", "tesseract"])
    def test_ocr_options_set_without_error(self, engine):
        """Constructing the converter with OCR options must not raise."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            TesseractOcrOptions,
        )
        from docling.document_converter import PdfFormatOption

        pdf_format_option = PdfFormatOption()
        if engine == "tesseract":
            pdf_format_option.pipeline_options.ocr_options = TesseractOcrOptions()
        else:
            pdf_format_option.pipeline_options.ocr_options = EasyOcrOptions()

        # Should not raise
        format_options = {InputFormat.PDF: pdf_format_option}
        assert format_options is not None

    def test_ocr_disabled_no_error(self):
        """Constructing with OCR disabled must not raise."""
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        pdf_format_option = PdfFormatOption()
        format_options = {InputFormat.PDF: pdf_format_option}
        assert format_options is not None

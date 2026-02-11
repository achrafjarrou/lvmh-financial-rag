import pytest
from langchain_core.documents import Document

from src.pdf_processor import PDFProcessor
from src.config import Config


def test_pdf_processor_init():
    """
    EN: Basic init test.
    FR: Vérifie que l'init ne casse rien et que les paramètres viennent de Config.
    """
    proc = PDFProcessor()
    assert proc.pdf_path == Config.pdf_path
    assert proc.splitter is not None


def test_process_adds_metadata(monkeypatch):
    """
    EN: Test the full process() pipeline but without reading a real PDF.
    FR: On mock le loader PDF pour garder un test rapide et stable.
    """

    # Fake loader that returns 2 "pages"
    class FakeLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return [
                Document(page_content="Revenue in 2023 was 86,153 million euros.", metadata={"page": 10}),
                Document(page_content="Operating profit increased in 2023.", metadata={"page": 12}),
            ]

    # Patch the PyPDFLoader used inside src.pdf_processor
    import src.pdf_processor as pdf_mod
    monkeypatch.setattr(pdf_mod, "PyPDFLoader", FakeLoader)

    proc = PDFProcessor()
    docs = proc.process()

    assert len(docs) > 0
    assert all(hasattr(d, "page_content") for d in docs)
    assert all(hasattr(d, "metadata") for d in docs)

    # Check metadata enrichment
    assert all("chunk_id" in d.metadata for d in docs)
    assert all("has_numbers" in d.metadata for d in docs)
    assert all("word_count" in d.metadata for d in docs)
    assert all(isinstance(d.metadata["word_count"], int) for d in docs)

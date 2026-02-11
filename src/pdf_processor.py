# src/pdf_processor.py
"""
Galère: PyPDF2 good  ,but  PyPDF  is better 
"""

from typing import List
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config
from loguru import logger

class PDFProcessor:
    def __init__(self):
        self.pdf_path = Config.pdf_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.chunk_size,
            chunk_overlap=Config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"PDFProcessor initialisé: {self.pdf_path}")
    
    def process(self) -> List[Document]:
        """Charge et découpe le PDF"""
        logger.info("Chargement du PDF...")
        # Normalize financial tables spacing
        text = text.replace("EUR millions", "EUR million")
        text = text.replace("€", " EUR ")

        # 1. Charger
        loader = PyPDFLoader(str(self.pdf_path))
        pages = loader.load()
        logger.info(f"✓ {len(pages)} pages chargées")
        
        # 2. Nettoyer - les PDFs c'est souvent sale
        cleaned = []
        for page in pages:
            # Virer espaces multiples
            text = re.sub(r'\s+', ' ', page.page_content)
            # Garder que le printable
            text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
            page.page_content = text.strip()
            cleaned.append(page)
        
        # 3. Découper
        chunks = self.splitter.split_documents(cleaned)
        logger.info(f"✓ {len(chunks)} chunks créés")
        
        # 4. Enrichir métadonnées
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['has_numbers'] = bool(re.search(r'\d+', chunk.page_content))
            chunk.metadata['word_count'] = len(chunk.page_content.split())
        
        return chunks

if __name__ == "__main__":
    proc = PDFProcessor()
    docs = proc.process()
    print(f"\nTest: {len(docs)} chunks")
    print(f"Premier: {docs[0].page_content[:200]}...")



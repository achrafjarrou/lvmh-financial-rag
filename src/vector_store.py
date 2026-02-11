"""
I struggled with persistence at first
"""
from typing import List, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from src.config import Config
from src.pdf_processor import PDFProcessor
from loguru import logger

class VectorStore:
    def __init__(self):
        self.db_path = Config.chroma_dir
        
        # Embeddings
        logger.info(f"Chargement embeddings: {Config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.embedding_model,
            model_kwargs={'device': Config.embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.db: Optional[Chroma] = None
        
        if self.exists():
            self.load()
    
    def exists(self) -> bool:
        """Check si la DB existe"""
        return self.db_path.exists() and list(self.db_path.glob("*.sqlite3"))
    
    def create(self, force=False):
        """Crée la DB depuis le PDF - prend 2-3 minutes"""
        if self.exists() and not force:
            logger.warning("DB existe déjà. Use force=True pour recréer")
            return
        
        logger.info("Création de la base vectorielle...")
        
        # Charger et découper PDF
        processor = PDFProcessor()
        docs = processor.process()
        
        # DB - Chroma 
        logger.info("Embedding en cours (ça prend du temps)...")
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=str(self.db_path),
            collection_name=Config.collection_name
        )
        
        logger.info(f"✓ Base créée: {len(docs)} docs")
    
    def load(self):
        """Charge DB existante"""
        logger.info("Chargement base existante...")
        self.db = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=self.embeddings,
            collection_name=Config.collection_name
        )
        logger.info("✓ Base chargée")
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Recherche par similarité
        Returns: [(doc, score), ...] triés par pertinence
        """
        if not self.db:
            raise ValueError("DB pas chargée!")
        
        k = k or Config.top_k_retrieval
        
        # Recherche
        results = self.db.similarity_search_with_score(query, k=k)
        
        # Convertir distance en similarité: 1/(1+dist)
        results_sim = [(doc, 1.0 / (1.0 + dist)) for doc, dist in results]
        results_sim.sort(key=lambda x: x[1], reverse=True)
        
        return results_sim
    
    def stats(self):
        """Stats de la DB"""
        if not self.db:
            return {"status": "not_loaded"}
        return {
            "total_docs": self.db._collection.count(),
            "db_path": str(self.db_path),
            "model": Config.embedding_model
        }

if __name__ == "__main__":
    store = VectorStore()
    if not store.exists():
        store.create()
    
    results = store.search("chiffre d'affaires 2023", k=3)
    print(f"\nTest recherche:")
    for doc, score in results:
        print(f"Score: {score:.3f} | {doc.page_content[:80]}...")





from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="LVMH Financial RAG API",
    description="API pour interroger le rapport financier LVMH 2023",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init RAG au démarrage
rag = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_rerank: Optional[bool] = True
    use_cache: Optional[bool] = True

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "message": "LVMH RAG API"
    }

@app.post("/query")
def query(req: QueryRequest):
    """
    Interroge le système RAG
    
    Exemple:
POST /query
{
    "question": "Quel est le CA 2023?",
    "top_k": 5
}
    """
    try:
        result = rag.query(
            question=req.question,
            top_k=req.top_k,
            use_rerank=req.use_rerank,
            use_cache=req.use_cache
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Retourne les métriques du système"""
    return rag.get_metrics()

@app.get("/health")
def health():
    """Health check détaillé"""
    try:
        stats = rag.vector_store.stats()
        metrics = rag.get_metrics()
        
        return {
            "status": "healthy",
            "database": stats,
            "metrics": metrics
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
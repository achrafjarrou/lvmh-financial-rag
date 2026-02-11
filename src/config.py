import os
from pathlib import Path
from dotenv import load_dotenv


# =========================
# Chargement .env (debug)
# =========================
print("Début du chargement config...")
print(f"Répertoire courant: {os.getcwd()}")
print(f"Fichier .env existe (cwd): {os.path.exists('.env')}")

# Charger dotenv avec chemin explicite
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

print(f"Chemin .env: {ENV_PATH}")
print(f"Ce fichier existe: {ENV_PATH.exists()}")

# NOTE: override=False pour ne pas écraser des variables déjà présentes
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Afficher variables pertinentes (masquées)
print("\nVariables d'environnement (partielles):")
for key, value in os.environ.items():
    if "GROK" in key or "GROQ" in key:
        masked = (value[:15] + "...") if value else "None"
        print(f"  {key}: {masked}")

# =========================
# Clé API (hors classe)
# =========================
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY manquante !\n"
        f"- Vérifie que le fichier .env existe ici: {ENV_PATH}\n"
        "- Ajoute dedans une ligne:\n"
        "  GROQ_API_KEY=gsk_ton_key_ici\n"
    )


class Config:
    # =========================
    # Racine du projet
    # =========================
    project_root = PROJECT_ROOT

    # Dossiers
    data_dir = project_root / "data"
    db_dir = project_root / "db"

    # Fichier PDF
    pdf_path = data_dir / "financial-documents-lvmh-december-31-2023.pdf"

    # =========================
    # Chroma
    # =========================
    chroma_dir = db_dir / "chroma_lvmh"
    collection_name = "lvmh_collection"

    # =========================
    # Embeddings
    # =========================
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device = "cpu"

    # =========================
    # Chunking
    # =========================
    chunk_size = 1200
    chunk_overlap = 250

    # =========================
    # Retrieval / Rerank
    # =========================
    top_k_retrieval = 10
    top_k_final = 5
    min_score = 0.7

    # =========================
    # LLM (Groq)
    # =========================
    llm_provider = "groq"
    llm_api_key = GROQ_API_KEY

    llm_model = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()

    # Sécurité : fallback si modèle décommissionné
    _DEPRECATED_MODELS = {"llama3-70b-8192"}
    if llm_model in _DEPRECATED_MODELS:
        llm_model = "llama-3.3-70b-versatile"

    llm_temperature = 0.1
    llm_max_tokens = 600

    # Aliases (compatibilité avec le reste du code)
    GROQ_API_KEY = llm_api_key
    LLM_MODEL = llm_model
    LLM_TEMPERATURE = llm_temperature
    LLM_MAX_TOKENS = llm_max_tokens

    # =========================
    # Cache
    # =========================
    enable_cache = True
    cache_ttl = 3600
    cache_max_size = 500

    # =========================
    # Logs
    # =========================
    log_level = "INFO"
    log_file = project_root / "rag.log"

    LOG_LEVEL = log_level
    LOG_FILE = log_file

    # =========================
    # API
    # =========================
    api_host = "0.0.0.0"
    api_port = 8000

    # =========================
    # Validation
    # =========================
    @classmethod
    def validate(cls):
        cls.data_dir.mkdir(exist_ok=True)
        cls.db_dir.mkdir(exist_ok=True)
        cls.chroma_dir.mkdir(parents=True, exist_ok=True)

        if not cls.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF introuvable: {cls.pdf_path}\n"
                "➡️ Mets le PDF dans le dossier data/ ou change Config.pdf_path"
            )

        print("✓ Config validée")


if __name__ == "__main__":
    Config.validate()

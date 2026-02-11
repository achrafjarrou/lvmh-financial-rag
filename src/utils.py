import logging
from pathlib import Path
from loguru import logger
from src.config import Config

def setup_logging():
    """Configure loguru"""
    logger.remove()  # Virer le handler par dÃ©faut
    
    # Console
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=Config.LOG_LEVEL
    )
    
    # Fichier
    logger.add(
        Config.LOG_FILE,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )
    
    logger.info("Logging configurÃ©")

def format_number(num: float) -> str:
    """Formate un nombre avec virgules"""
    return f"{num:,.0f}".replace(",", " ")

if __name__ == "__main__":
    setup_logging()
    logger.info("Test logging")
    logger.debug("Debug message")
    logger.warning("Warning message")


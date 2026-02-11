from loguru import logger
from src.config import Config


def setup_logging():
    """Configure loguru"""
    logger.remove()  # Virer le handler par défaut

    # Console
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=Config.log_level
    )

    # Fichier
    logger.add(
        str(Config.log_file),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("Logging configuré")


def format_number(num: float) -> str:
    """Formate un nombre avec espaces"""
    return f"{num:,.0f}".replace(",", " ")

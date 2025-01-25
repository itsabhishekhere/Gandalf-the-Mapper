"""
Application configuration and settings.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATE_DIR: Path = BASE_DIR / "templates"
    DATA_DIR: Path = BASE_DIR / "gandalf/data"
    
    # File paths
    TAXONOMY_FILE: Path = BASE_DIR / "gandalf/data/Content Taxonomy 3.1.tsv"
    RESULTS_DIR: Path = DATA_DIR / "results"
    
    # Crawler settings
    MAX_URLS: int = int(os.getenv("MAX_URLS", "100"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    MAX_MATCHES: int = int(os.getenv("MAX_MATCHES", "5"))
    
    # Model settings
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME", 
        "sentence-transformers/all-mpnet-base-v2"
    )
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5001"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.STATIC_DIR.mkdir(exist_ok=True)

# Create global config instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the application configuration."""
    return config 
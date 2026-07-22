import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
ASSETS_DIR = BASE_DIR / "app" / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

# File paths
SNEAKERS_FILE = DATA_DIR / "sneakers.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"

# Model settings
MODEL_NAME = "clip-ViT-B-32"

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

import json
import pickle
import urllib.request
from pathlib import Path
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
IMAGES_DIR = BASE_DIR / "app" / "assets" / "images"
SNEAKERS_FILE = DATA_DIR / "sneakers.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"
MODEL_NAME = "clip-ViT-B-32"

# Replacement verified URLs for fallbacks (sneaker-type images from Unsplash)
REPLACEMENTS = {
    "snk-007": "https://images.unsplash.com/photo-1605348532760-6753d2c43329?w=600&q=80",  # Jordan 4 Bred
    "snk-012": "https://images.unsplash.com/photo-1556048219-bb6978360b84?w=600&q=80",  # Air Max Plus
    "snk-015": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=600&q=80",  # Stan Smith
    "snk-021": "https://images.unsplash.com/photo-1509631179647-0177331693ae?w=600&q=80",  # Forum Low
    "snk-033": "https://images.unsplash.com/photo-1565814329452-e1efa11c5b89?w=600&q=80",  # Suede Classic
    "snk-034": "https://images.unsplash.com/photo-1600185365926-3a2ce3cdb9eb?w=600&q=80",  # RS-X
    "snk-038": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=600&q=80",  # Cali Dream
    "snk-049": "https://images.unsplash.com/photo-1571752726703-5e7d1f6a986d?w=600&q=80",  # Gel-Nimbus 9
    "snk-055": "https://images.unsplash.com/photo-1544441893-675973e31985?w=600&q=80",  # Vans Slip-On
    "snk-061": "https://images.unsplash.com/photo-1583744946564-b52d01e7f922?w=600&q=80",  # HOVR Phantom 3
    "snk-065": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&q=80",  # Salomon XT-6
}

def download_image(url, filepath):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp, open(filepath, "wb") as f:
        f.write(resp.read())

def fix():
    print("Re-downloading failed images...")
    for snk_id, url in REPLACEMENTS.items():
        filepath = IMAGES_DIR / f"{snk_id}.jpg"
        try:
            download_image(url, filepath)
            with Image.open(filepath) as img:
                img.convert("RGB").save(filepath, "JPEG", quality=85)
            print(f"  OK: {snk_id}")
        except Exception as e:
            print(f"  FAIL {snk_id}: {e}")

    # Regenerate embeddings for the fixed ones
    print("\nRegenerating embeddings for fixed sneakers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_dict = pickle.load(f)

    for snk_id in REPLACEMENTS.keys():
        img_path = IMAGES_DIR / f"{snk_id}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
            emb = model.encode(image, show_progress_bar=False)
            embeddings_dict[snk_id] = emb
            print(f"  Embedded: {snk_id}")
        except Exception as e:
            print(f"  Embed FAIL {snk_id}: {e}")

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print("\nDone! All images fixed.")

if __name__ == "__main__":
    fix()

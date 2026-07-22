"""
SneakVault — Setup Pipeline (Scraped Superkicks Data Only)
Source: scraped_sneakers.json (Superkicks product listings with direct purchase URLs)
"""

import json
import pickle
from pathlib import Path
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
IMAGES_DIR = BASE_DIR / "app" / "assets" / "images"
SNEAKERS_FILE = DATA_DIR / "sneakers.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"
SCRAPED_FILE = DATA_DIR / "scraped_sneakers.json"
MODEL_NAME = "clip-ViT-B-32"

def setup():
    print("=" * 60)
    print("  SneakVault Setup — Superkicks Real Catalog Only")
    print("=" * 60)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    processed = []
    failed = 0

    print(f"\nLoading scraped data from {SCRAPED_FILE}...")
    if SCRAPED_FILE.exists():
        with open(SCRAPED_FILE, 'r', encoding='utf-8') as f:
            scraped = json.load(f)
        
        valid = 0
        for item in scraped:
            # Verify image file exists
            img_path = IMAGES_DIR / item["image_filename"]
            if img_path.exists():
                # Verify it has a valid source and source_url
                if item.get("source") == "superkicks" and item.get("source_url"):
                    processed.append(item)
                    valid += 1
                else:
                    failed += 1
            else:
                failed += 1
        print(f"      Loaded {valid} Superkicks sneakers ({failed} items filtered out or missing images)")
    else:
        print(f"      Error: Scraped data file not found at {SCRAPED_FILE}!")
        print(f"      Please run python scripts/scrape_shopify.py first.")
        return

    print(f"\nTotal Database Count: {len(processed)} sneakers")

    # ── Save dataset ──
    print(f"Saving dataset to {SNEAKERS_FILE}...")
    with open(SNEAKERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    # ── Generate CLIP embeddings ──
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Generating CLIP embeddings...")
    embeddings_dict = {}
    for snk in processed:
        img_path = IMAGES_DIR / snk["image_filename"]
        try:
            emb = model.encode(Image.open(img_path).convert("RGB"), show_progress_bar=False)
            embeddings_dict[snk["id"]] = emb
        except Exception as e:
            print(f"  Embed FAIL {snk['id']}: {e}")

    print(f"Saving embeddings to {EMBEDDINGS_FILE}...")
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    # ── Summary ──
    brands = {}
    for s in processed:
        brands[s["brand"]] = brands.get(s["brand"], 0) + 1
    print(f"\n{'='*60}")
    print(f"  Setup complete — {len(processed)} Superkicks sneakers ready")
    print(f"{'='*60}")
    print(f"\n  Brand breakdown:")
    for brand, count in sorted(brands.items(), key=lambda x: -x[1]):
        print(f"    {brand}: {count}")

if __name__ == "__main__":
    setup()

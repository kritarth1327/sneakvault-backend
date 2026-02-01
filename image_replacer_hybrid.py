import json
import requests
import time
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# -------------------------
# Device (GPU if available)
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -------------------------
# Load CLIP
# -------------------------
print("Loading CLIP...")
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP ready")

# -------------------------
# Tuning knobs
# -------------------------
MAX_IMAGES = 6          # reduce a bit for speed
SLEEP_TIME = 0.4        # polite but fast
IMPROVEMENT_MARGIN = 0.05  # ⭐ key logic

# -------------------------
# Helpers
# -------------------------
def get_image(url):
    try:
        res = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        img = Image.open(BytesIO(res.content)).convert("RGB")
        return img
    except:
        return None

def image_embedding(img):
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze()

def text_embedding(text):
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze()

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# -------------------------
# Nice Kicks search (priority)
# -------------------------
def search_nicekicks(brand, model):
    query = f"{brand} {model}"
    url = f"https://www.nicekicks.com/?s={query.replace(' ', '+')}"
    images = []

    try:
        res = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(res.text, "html.parser")

        for img in soup.find_all("img"):
            src = img.get("src")
            if src and "wp-content/uploads" in src:
                images.append(src)

    except:
        pass

    return list(set(images))[:MAX_IMAGES]

# -------------------------
# DuckDuckGo fallback
# -------------------------
def search_duckduckgo(query):
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=MAX_IMAGES)
            for r in results:
                urls.append(r["image"])
    except:
        pass
    return urls

# -------------------------
# Load flagged sneakers
# -------------------------
with open("backend/data/sneakers_flagged.json", "r", encoding="utf-8") as f:
    FLAGGED = json.load(f)

UPDATED = []
FAILED = []

print("\n🔄 Hybrid image replacement (Nice Kicks → DuckDuckGo)\n")

# -------------------------
# Main loop
# -------------------------
for s in FLAGGED:
    name = f"{s['brand']} {s['model']}"
    print(f"→ {name}")

    base_score = s.get("image_text_similarity", 0.0)
    best_score = base_score
    best_image = s["image"]

    txt_emb = text_embedding(f"{name} sneaker")

    # 1️⃣ Nice Kicks first
    nk_images = search_nicekicks(s["brand"], s["model"])
    for url in nk_images:
        img = get_image(url)
        if not img:
            continue
        score = cosine(image_embedding(img), txt_emb)
        if score > best_score:
            best_score = score
            best_image = url

    # 2️⃣ DuckDuckGo fallback
    if best_score - base_score < IMPROVEMENT_MARGIN:
        ddg_images = search_duckduckgo(
            f"{name} sneaker official product"
        )
        for url in ddg_images:
            img = get_image(url)
            if not img:
                continue
            score = cosine(image_embedding(img), txt_emb)
            if score > best_score:
                best_score = score
                best_image = url

    # 3️⃣ Decision (RELATIVE improvement)
    if (best_score - base_score) >= IMPROVEMENT_MARGIN:
        s["image"] = best_image
        s["image_text_similarity"] = round(best_score, 3)
        UPDATED.append(s)
        print(f"   ✅ Improved {base_score:.2f} → {best_score:.2f}\n")
    else:
        FAILED.append(s)
        print("   ⚠️ No meaningful improvement\n")

    time.sleep(SLEEP_TIME)

# -------------------------
# Save outputs
# -------------------------
with open("backend/data/sneakers_auto_fixed.json", "w", encoding="utf-8") as f:
    json.dump(UPDATED, f, indent=2)

with open("backend/data/sneakers_still_flagged.json", "w", encoding="utf-8") as f:
    json.dump(FAILED, f, indent=2)

print("\n🎉 Hybrid replacement DONE")
print(f"✅ Auto-fixed: {len(UPDATED)}")
print(f"⚠️ Still flagged: {len(FAILED)}")
    
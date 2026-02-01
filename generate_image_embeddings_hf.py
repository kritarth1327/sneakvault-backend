import json
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# Load CLIP model
# -------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP loaded")    

# -------------------------
# Thresholds (TUNABLE)
# -------------------------
KEEP_THRESHOLD = 0.28     # image matches sneaker text
FLAG_THRESHOLD = 0.20     # maybe correct, needs review

# -------------------------
# Load sneakers
# -------------------------
with open("backend/data/sneakers.json", "r", encoding="utf-8") as f:
    SNEAKERS = json.load(f)

EMBEDDINGS = {}
FLAGGED = []
REMOVED = []

# -------------------------
# Helpers
# -------------------------
def load_image(url):
    try:
        res = requests.get(url, timeout=10)
        img = Image.open(BytesIO(res.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"❌ Image load failed: {url} ({e})")
        return None

def image_embedding(img):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze()

def text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze()

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# -------------------------
# Main pipeline
# -------------------------
print("\n🔍 Verifying images and generating embeddings...\n")

for s in SNEAKERS:
    name = f"{s['brand']} {s['model']}"
    print(f"→ {name}")

    img = load_image(s["image"])
    if img is None:
        REMOVED.append({**s, "reason": "image_load_failed"})
        print("   ❌ Removed (image load failed)\n")
        continue

    img_emb = image_embedding(img)
    txt = f"{s['brand']} {s['model']} sneaker"
    txt_emb = text_embedding(txt)

    similarity = cosine(img_emb, txt_emb)
    similarity = round(similarity, 3)
    s["image_text_similarity"] = similarity

    # Decision
    if similarity >= KEEP_THRESHOLD:
        EMBEDDINGS[s["id"]] = {
            "embedding": img_emb.tolist(),
            "brand": s["brand"],
            "model": s["model"],
            "price_inr": s["price_inr"],
            "image": s["image"],
            "confidence": similarity
        }
        print(f"   ✅ Kept ({similarity})\n")

    elif similarity >= FLAG_THRESHOLD:
        FLAGGED.append(s)
        print(f"   ⚠️ Flagged ({similarity})\n")

    else:
        REMOVED.append({**s, "reason": "low_similarity"})
        print(f"   ❌ Removed ({similarity})\n")

# -------------------------
# Save outputs
# -------------------------
with open("backend/data/image_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(EMBEDDINGS, f, indent=2)

with open("backend/data/sneakers_flagged.json", "w", encoding="utf-8") as f:
    json.dump(FLAGGED, f, indent=2)

with open("backend/data/sneakers_removed.json", "w", encoding="utf-8") as f:
    json.dump(REMOVED, f, indent=2)

print("\n🎉 DONE")
print(f"✅ Clean embeddings generated: {len(EMBEDDINGS)}")
print(f"⚠️ Flagged sneakers (needs image fix): {len(FLAGGED)}")
print(f"❌ Removed sneakers: {len(REMOVED)}")

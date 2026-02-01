import json
import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# Load CLIP
# -------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP loaded")

# -------------------------
# Load data
# -------------------------
with open("backend/data/sneakers.json", "r", encoding="utf-8") as f:
    sneakers = json.load(f)

with open("backend/data/image_embeddings.json", "r", encoding="utf-8") as f:
    image_embeddings = json.load(f)

# -------------------------
# Helpers
# -------------------------
def text_to_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze()

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# -------------------------
# Verification
# -------------------------
VERIFIED = 0
UNVERIFIED = 0

for sneaker in sneakers:
    sid = sneaker["id"]

    if sid not in image_embeddings:
        sneaker["image_verified"] = False
        UNVERIFIED += 1
        continue

    text = f"{sneaker['brand']} {sneaker['model']} sneaker"
    text_emb = text_to_embedding(text)

    image_emb = torch.tensor(image_embeddings[sid]["embedding"])

    similarity = cosine_sim(text_emb, image_emb)

    # Thresholds (tuned for CLIP)
    if similarity >= 0.27:
        sneaker["image_verified"] = True
        VERIFIED += 1
    else:
        sneaker["image_verified"] = False
        UNVERIFIED += 1

print(f"Verified: {VERIFIED}")
print(f"Hidden (unverified): {UNVERIFIED}")

# -------------------------
# Save updated dataset
# -------------------------
with open("backend/data/sneakers_verified.json", "w", encoding="utf-8") as f:
    json.dump(sneakers, f, indent=2)

print("Saved → sneakers_verified.json")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import json
import torch

from transformers import CLIPProcessor, CLIPModel

# -------------------------
# App setup
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load VERIFIED data only
# -------------------------
with open("image_embeddings.json", "r", encoding="utf-8") as f:
    IMAGE_EMBEDDINGS = json.load(f)

# Convert embeddings dict → sneaker list (verified only)
SNEAKERS = [
    {
        "id": sneaker_id,
        "brand": info["brand"],
        "model": info["model"],
        "price_inr": info["price_inr"],
        "image": info["image"],
    }
    for sneaker_id, info in IMAGE_EMBEDDINGS.items()
]

# -------------------------
# Load CLIP once
# -------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP loaded")

# -------------------------
# Constants (TUNABLE)
# -------------------------
MIN_SIMILARITY = 0.25
IMAGE_WEIGHT = 0.7
PRICE_WEIGHT = 0.3

# -------------------------
# Utility functions
# -------------------------
def image_to_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze()

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def price_similarity(query_price, candidate_price):
    if candidate_price >= query_price:
        return 0.0
    diff_ratio = (query_price - candidate_price) / query_price
    return max(0.0, 1 - diff_ratio)

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "SneakVault API Running"}

@app.get("/sneakers/all")
def get_all():
    return SNEAKERS

@app.get("/sneakers/category/{cat}")
def get_category(cat: str):
    cat = cat.lower()

    if cat == "budget":
        filtered = [s for s in SNEAKERS if s["price_inr"] < 5000]
    elif cat == "streetwear":
        filtered = [s for s in SNEAKERS if 5000 <= s["price_inr"] < 10000]
    elif cat == "heat":
        filtered = [s for s in SNEAKERS if 10000 <= s["price_inr"] < 25000]
    else:
        return {"error": "Invalid category"}

    return {
        "category": cat,
        "count": len(filtered),
        "results": filtered
    }

# -------------------------
# AI: Cheaper alternatives
# -------------------------
@app.post("/cheaper-alternatives/image")
async def cheaper_alternatives_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    query_embedding = image_to_embedding(img)
    scored = []

    for sneaker_id, info in IMAGE_EMBEDDINGS.items():
        sneaker_embedding = torch.tensor(info["embedding"])

        image_sim = cosine_sim(query_embedding, sneaker_embedding)

        if image_sim < MIN_SIMILARITY:
            continue

        scored.append({
            "id": sneaker_id,
            "brand": info["brand"],
            "model": info["model"],
            "price_inr": info["price_inr"],
            "image": info["image"],
            "image_similarity": image_sim,
        })

    if not scored:
        return {
            "error": "No confident sneaker match found",
            "confidence": 0.0
        }

    scored.sort(key=lambda x: x["image_similarity"], reverse=True)

    best_match = scored[0]
    best_price = best_match["price_inr"]

    final_ranked = []
    for s in scored:
        price_score = price_similarity(best_price, s["price_inr"])
        final_score = (
            IMAGE_WEIGHT * s["image_similarity"]
            + PRICE_WEIGHT * price_score
        )
        final_ranked.append({
            **s,
            "final_score": round(final_score, 4)
        })

    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)

    best_match = final_ranked[0]

    cheaper_alternatives = [
        s for s in final_ranked[1:]
        if s["price_inr"] < best_match["price_inr"]
    ][:5]

    return {
        "best_match": best_match,
        "cheaper_alternatives": cheaper_alternatives,
        "reason": {
            "confidence": round(best_match["image_similarity"], 2),
            "matched_on": ["shape", "color", "silhouette"]
        }
    }

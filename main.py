from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np

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

# Convert embeddings dict → sneaker list
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
# Constants
# -------------------------
MIN_SIMILARITY = 0.25
IMAGE_WEIGHT = 0.7
PRICE_WEIGHT = 0.3

# -------------------------
# Utility functions
# -------------------------
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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
@app.get("/cheaper-alternatives/{sneaker_id}")
def cheaper_alternatives(sneaker_id: str):

    if sneaker_id not in IMAGE_EMBEDDINGS:
        return {"error": "Sneaker not found"}

    target = IMAGE_EMBEDDINGS[sneaker_id]
    target_embedding = target["embedding"]

    scored = []

    for sid, info in IMAGE_EMBEDDINGS.items():
        if sid == sneaker_id:
            continue

        sim = cosine_sim(target_embedding, info["embedding"])

        if sim < MIN_SIMILARITY:
            continue

        scored.append({
            "id": sid,
            "brand": info["brand"],
            "model": info["model"],
            "price_inr": info["price_inr"],
            "image": info["image"],
            "image_similarity": sim,
        })

    if not scored:
        return {"error": "No similar sneakers found"}

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
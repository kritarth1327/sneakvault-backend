from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import List, Optional
import io
from PIL import Image

from app.models.schemas import Sneaker, RecommendationResponse, Recommendation, SmartBuyResponse
from app.services.sneaker_service import sneaker_service
from app.services.ai_service import ai_service

router = APIRouter()

@router.get("/sneakers", response_model=List[Sneaker])
def get_sneakers(brand: Optional[str] = None, max_price: Optional[int] = Query(None, description="Maximum price filter")):
    """List all sneakers, optionally filtered."""
    return sneaker_service.get_all(brand=brand, max_price=max_price)

@router.get("/sneakers/{sneaker_id}", response_model=Sneaker)
def get_sneaker(sneaker_id: str):
    """Get a single sneaker by ID."""
    sneaker = sneaker_service.get_by_id(sneaker_id)
    if not sneaker:
        raise HTTPException(status_code=404, detail="Sneaker not found")
    return sneaker

@router.post("/recommendations/upload", response_model=SmartBuyResponse)
async def upload_for_recommendations(
    file: UploadFile = File(...),
    max_price: Optional[int] = Query(None, description="Maximum budget in INR"),
):
    """
    Upload a sneaker photo to find CHEAPER visually similar alternatives.
    
    - Identifies what you uploaded (closest match in our database)
    - Returns cheaper alternatives ranked by visual similarity + price savings
    - Optionally filter by max_price budget cap
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    # 1. Generate CLIP embedding for the uploaded image
    query_embedding = ai_service.embed_image(image)

    # 2. Build price lookup from the sneaker database
    all_sneakers = sneaker_service.get_all()
    price_map = {s.id: s.price for s in all_sneakers}

    # 3. Get cheaper alternatives using smart-buy algorithm
    result = ai_service.get_cheaper_alternatives(
        query_embedding=query_embedding,
        sneaker_prices=price_map,
        max_price=max_price,
        top_k=8,
    )

    # 4. Build response
    matched_sneaker = None
    matched_score = None
    if result["matched_id"]:
        matched_sneaker = sneaker_service.get_by_id(result["matched_id"])
        matched_score = result["matched_score"]

    alternatives = []
    for item in result["alternatives"]:
        sneaker = sneaker_service.get_by_id(item["sneaker_id"])
        if sneaker:
            alternatives.append(
                Recommendation(
                    sneaker=sneaker,
                    similarity_score=item["similarity_score"]
                )
            )

    # Calculate savings range
    savings_range = None
    if alternatives and matched_sneaker:
        prices = [a.sneaker.price for a in alternatives]
        min_saving = matched_sneaker.price - max(prices)
        max_saving = matched_sneaker.price - min(prices)
        if min_saving > 0 and max_saving > 0:
            savings_range = f"Save Rs.{min_saving:,} - Rs.{max_saving:,}"

    return SmartBuyResponse(
        matched_sneaker=matched_sneaker,
        matched_score=matched_score,
        alternatives=alternatives,
        savings_range=savings_range,
    )

@router.post("/recommendations/similar", response_model=RecommendationResponse)
async def upload_for_similar(file: UploadFile = File(...)):
    """
    Upload a sneaker photo to find the most visually similar matches.
    (Original behavior — no price filtering, just pure visual similarity.)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    query_embedding = ai_service.embed_image(image)
    similar_items = ai_service.get_similar_sneakers(query_embedding, top_k=5)

    recommendations = []
    for item in similar_items:
        sneaker = sneaker_service.get_by_id(item["sneaker_id"])
        if sneaker:
            recommendations.append(
                Recommendation(
                    sneaker=sneaker,
                    similarity_score=item["similarity_score"]
                )
            )

    return RecommendationResponse(recommendations=recommendations)

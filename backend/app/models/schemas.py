from pydantic import BaseModel
from typing import List, Optional

class SneakerBase(BaseModel):
    id: str
    name: str
    brand: str
    price: int
    image_filename: str
    description: Optional[str] = None
    colorway: Optional[str] = None
    isTrending: Optional[bool] = False
    source: Optional[str] = None
    source_url: Optional[str] = None

class Sneaker(SneakerBase):
    pass

class Recommendation(BaseModel):
    sneaker: Sneaker
    similarity_score: float

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]

class SmartBuyResponse(BaseModel):
    matched_sneaker: Optional[Sneaker] = None
    matched_score: Optional[float] = None
    alternatives: List[Recommendation]
    savings_range: Optional[str] = None

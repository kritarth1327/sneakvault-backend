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
    similarity_score: float  # e.g. 95.5 for 95.5%

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]

class SmartBuyResponse(BaseModel):
    """Response for the smart-buy (cheaper alternatives) feature."""
    matched_sneaker: Optional[Sneaker] = None   # What we think you uploaded
    matched_score: Optional[float] = None        # How confident we are
    alternatives: List[Recommendation]            # Cheaper visually-similar options
    savings_range: Optional[str] = None           # e.g. "Save Rs.3,000 - Rs.12,000"

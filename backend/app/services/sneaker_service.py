import json
from typing import List, Optional
from app.config import SNEAKERS_FILE
from app.models.schemas import Sneaker

class SneakerService:
    def __init__(self):
        self.sneakers = {}
        self.initialize()

    def initialize(self):
        if not SNEAKERS_FILE.exists():
            print(f"Warning: Sneakers database not found at {SNEAKERS_FILE}")
            return
        
        try:
            with open(SNEAKERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    sneaker = Sneaker(**item)
                    self.sneakers[sneaker.id] = sneaker
            print(f"Loaded {len(self.sneakers)} sneakers into memory.")
        except Exception as e:
            print(f"Error loading sneakers DB: {e}")

    def get_all(self, brand: Optional[str] = None, max_price: Optional[int] = None) -> List[Sneaker]:
        results = list(self.sneakers.values())
        if brand:
            results = [s for s in results if s.brand.lower() == brand.lower()]
        if max_price:
            results = [s for s in results if s.price <= max_price]
        return results

    def get_by_id(self, sneaker_id: str) -> Optional[Sneaker]:
        return self.sneakers.get(sneaker_id)

sneaker_service = SneakerService()

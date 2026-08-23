import pickle
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from app.config import MODEL_NAME, EMBEDDINGS_FILE

class AIService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.embeddings_dict = {}
        self.sneaker_ids = []
        self.embedding_matrix = None

    def initialize(self):
        print(f"Loading Model ({MODEL_NAME}) on {self.device}...")
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.load_embeddings()

    def load_embeddings(self):
        if not EMBEDDINGS_FILE.exists():
            print(f"Warning: Embeddings file not found at {EMBEDDINGS_FILE}")
            return
        
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                self.embeddings_dict = pickle.load(f)
            
            if self.embeddings_dict:
                self.sneaker_ids = list(self.embeddings_dict.keys())
                embeddings_list = [self.embeddings_dict[k] for k in self.sneaker_ids]
                self.embedding_matrix = torch.tensor(np.array(embeddings_list), device=self.device)
                print(f"Loaded {len(self.sneaker_ids)} embeddings.")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        if self.model is None:
            self.initialize()
        return self.model.encode(image, convert_to_tensor=True, device=self.device)

    def get_similar_sneakers(self, query_embedding: torch.Tensor, top_k: int = 5):
        if self.embedding_matrix is None or len(self.sneaker_ids) == 0:
            return []

        cos_scores = util.cos_sim(query_embedding, self.embedding_matrix)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.sneaker_ids)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append({
                "sneaker_id": self.sneaker_ids[idx.item()],
                "similarity_score": round(score.item() * 100, 2)
            })
        return results

    def get_cheaper_alternatives(
        self,
        query_embedding: torch.Tensor,
        sneaker_prices: dict,
        max_price: int = None,
        top_k: int = 8,
        similarity_weight: float = 0.6,
        savings_weight: float = 0.4,
    ):
        if self.embedding_matrix is None or len(self.sneaker_ids) == 0:
            return {"matched_id": None, "matched_score": 0, "alternatives": []}

        cos_scores = util.cos_sim(query_embedding, self.embedding_matrix)[0]

        best_idx = torch.argmax(cos_scores).item()
        best_id = self.sneaker_ids[best_idx]
        best_score = cos_scores[best_idx].item()
        best_price = sneaker_prices.get(best_id, 0)

        price_ceiling = max_price if max_price else best_price

        candidates = []
        for i, snk_id in enumerate(self.sneaker_ids):
            if snk_id == best_id:
                continue
            
            price = sneaker_prices.get(snk_id, 0)
            if price <= 0 or price >= price_ceiling:
                continue

            sim_score = cos_scores[i].item()
            if sim_score < 0.15:
                continue

            savings_ratio = (price_ceiling - price) / price_ceiling
            combined = (similarity_weight * sim_score) + (savings_weight * savings_ratio)

            candidates.append({
                "sneaker_id": snk_id,
                "similarity_score": round(sim_score * 100, 2),
                "savings": price_ceiling - price,
                "combined_score": combined,
            })

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        return {
            "matched_id": best_id,
            "matched_score": round(best_score * 100, 2),
            "alternatives": candidates[:top_k],
        }

ai_service = AIService()

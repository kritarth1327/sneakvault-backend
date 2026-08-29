import os
import pickle
import numpy as np
from PIL import Image
import httpx
import onnxruntime as ort
from app.config import ONNX_MODEL_FILE, EMBEDDINGS_FILE

MODEL_CDN_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx"

class AIService:
    def __init__(self):
        self.session = None
        self.embeddings_dict = {}
        self.sneaker_ids = []
        self.embedding_matrix = None

    def _ensure_model_exists(self):
        if ONNX_MODEL_FILE.exists():
            return
        
        print(f"ONNX Model not found locally. Downloading from CDN ({MODEL_CDN_URL})...")
        ONNX_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_file = ONNX_MODEL_FILE.with_suffix(".tmp")
        try:
            with httpx.stream("GET", MODEL_CDN_URL, follow_redirects=True, timeout=120.0) as r:
                r.raise_for_status()
                with open(tmp_file, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
            tmp_file.replace(ONNX_MODEL_FILE)
            print(f"Model downloaded successfully to {ONNX_MODEL_FILE}")
        except Exception as e:
            if tmp_file.exists():
                tmp_file.unlink()
            raise RuntimeError(f"Failed to download ONNX model: {e}")

    def initialize(self):
        self._ensure_model_exists()

        print(f"Loading ONNX Model from {ONNX_MODEL_FILE}...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(ONNX_MODEL_FILE), sess_options, providers=["CPUExecutionProvider"])
        self.load_embeddings()

    def load_embeddings(self):
        if not EMBEDDINGS_FILE.exists():
            print(f"Warning: Embeddings file not found at {EMBEDDINGS_FILE}")
            return

        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                self.embeddings_dict = pickle.load(f)

            if self.embeddings_dict:
                self.sneaker_ids = list(self.embeddings_dict.keys())
                embeddings_list = [self.embeddings_dict[k] for k in self.sneaker_ids]
                raw_matrix = np.array(embeddings_list, dtype=np.float32)
                norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.embedding_matrix = raw_matrix / norms
                print(f"Loaded and normalized {len(self.sneaker_ids)} sneaker embeddings.")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((224, 224), Image.BICUBIC)
        arr = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # HWC to CHW
        return np.expand_dims(arr, axis=0)  # NCHW

    def embed_image(self, image: Image.Image) -> np.ndarray:
        if self.session is None:
            self.initialize()
        pixel_values = self.preprocess_image(image)
        outputs = self.session.run(None, {"pixel_values": pixel_values})
        emb = outputs[0][0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def get_similar_sneakers(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.embedding_matrix is None or len(self.sneaker_ids) == 0:
            return []

        cos_scores = np.dot(self.embedding_matrix, query_embedding)
        top_indices = np.argsort(cos_scores)[::-1][:min(top_k, len(self.sneaker_ids))]

        results = []
        for idx in top_indices:
            results.append({
                "sneaker_id": self.sneaker_ids[idx],
                "similarity_score": round(float(cos_scores[idx]) * 100, 2)
            })
        return results

    def get_cheaper_alternatives(
        self,
        query_embedding: np.ndarray,
        sneaker_prices: dict,
        max_price: int = None,
        top_k: int = 8,
        similarity_weight: float = 0.6,
        savings_weight: float = 0.4,
    ):
        if self.embedding_matrix is None or len(self.sneaker_ids) == 0:
            return {"matched_id": None, "matched_score": 0, "alternatives": []}

        cos_scores = np.dot(self.embedding_matrix, query_embedding)

        best_idx = int(np.argmax(cos_scores))
        best_id = self.sneaker_ids[best_idx]
        best_score = float(cos_scores[best_idx])
        best_price = sneaker_prices.get(best_id, 0)

        price_ceiling = max_price if max_price else best_price

        candidates = []
        for i, snk_id in enumerate(self.sneaker_ids):
            if snk_id == best_id:
                continue

            price = sneaker_prices.get(snk_id, 0)
            if price <= 0 or price >= price_ceiling:
                continue

            sim_score = float(cos_scores[i])
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

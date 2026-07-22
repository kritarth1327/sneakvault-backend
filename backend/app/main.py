from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import IMAGES_DIR
from app.services.ai_service import ai_service
from app.services.sneaker_service import sneaker_service

app = FastAPI(
    title="SneakVault API",
    description="Backend for SneakVault sneaker recommendations.",
    version="1.0.0"
)

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static images directory so the frontend can display them
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Include API routes
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # Warm up AI model and load data into memory
    print("Starting up SneakVault backend...")
    sneaker_service.initialize()
    ai_service.initialize()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

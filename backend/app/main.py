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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    print("Starting up SneakVault backend...")
    sneaker_service.initialize()
    ai_service.initialize()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

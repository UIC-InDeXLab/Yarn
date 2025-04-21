import os
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.services.video_service import VideoService
from app.routers import search
from app.utils.config import API_TITLE, API_DESCRIPTION, API_VERSION, VIDEO_DIRECTORY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize video service on startup
@app.on_event("startup")
async def startup_event():
    # Create videos directory if it doesn't exist
    os.makedirs(VIDEO_DIRECTORY, exist_ok=True)
    
    # Initialize video service
    video_service = VideoService()
    await video_service.index_videos(VIDEO_DIRECTORY)

# Include routers
app.include_router(search.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
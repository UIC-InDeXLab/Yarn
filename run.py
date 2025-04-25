#!/usr/bin/env python3
"""
Run script for the Yarn API server
"""
import logging
import os

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY is not set. API will not be able to use OpenAI services.")

# Create videos directory if it doesn't exist
videos_dir = os.getenv("VIDEO_DIRECTORY", "./videos")
if not os.path.exists(videos_dir):
    logger.info(f"Creating videos directory: {videos_dir}")
    os.makedirs(videos_dir, exist_ok=True)

# Run the application
if __name__ == "__main__":
    logger.info("Starting Yarn API server")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

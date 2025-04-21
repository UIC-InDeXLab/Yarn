import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# API settings
API_VERSION = "0.1.0"
API_TITLE = "Yarn API"
API_DESCRIPTION = "Text to video retrieval engine using GenAI"

# Application settings
VIDEO_DIRECTORY = os.getenv("VIDEO_DIRECTORY", "./videos")
MAX_FRAMES_DEFAULT = int(os.getenv("MAX_FRAMES_DEFAULT", "5"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "3"))

# Debug settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ["true", "1", "yes"]
DEBUG_LOG_DIR = os.getenv("DEBUG_LOG_DIR", "./logs/debug")
DEBUG_FRAMES_DIR = os.getenv("DEBUG_FRAMES_DIR", "./logs/frames")

# Ensure debug directories exist if debug mode is enabled
if DEBUG_MODE:
    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
    os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
    logging.info(f"Debug mode enabled. Logs will be saved to {DEBUG_LOG_DIR}")

# Check for required environment variables
required_env_vars = ["OPENAI_API_KEY"]

for var in required_env_vars:
    if not os.getenv(var):
        logging.warning(f"Environment variable {var} is not set. Some features may not work correctly.")
# Yarn - Text to Video Retrieval Engine

Yarn is a FastAPI backend service that provides text-to-video retrieval capabilities using GenAI technologies. The system indexes video files by extracting key frames and generating embeddings, then matches user text queries to the most relevant videos.

## Features

- Automatic video indexing on startup
- Frame extraction using scene detection
- Embedding generation using CLIP
- Query processing using GPT-4o
- Multiple image generation models:
  - Stable Diffusion via Replicate API (default)
  - DALL-E via OpenAI API
- Two frame generation modes:
  - Independent: each frame generated separately
  - Continuous: frames generated as a coherent sequence
- Dynamic Time Warping (DTW) for comparing query and video embeddings
- REST API for searching videos by text query
- Debug mode for logging queries, frame descriptions, and generated images

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:

```
# OpenAI API key (required for GPT-4o frame descriptions and DALL-E if selected)
OPENAI_API_KEY=your_openai_api_key_here

# Replicate API token (required for Stable Diffusion if selected)
REPLICATE_API_TOKEN=your_replicate_api_token_here

# Video directory and search defaults
VIDEO_DIRECTORY=./videos
MAX_FRAMES_DEFAULT=5
TOP_K_DEFAULT=3

# Debug settings
DEBUG_MODE=false
DEBUG_LOG_DIR=./logs/debug
DEBUG_FRAMES_DIR=./logs/frames
```

## Usage

1. Place your videos in the `videos` directory

2. Start the server:

```bash
python run.py
```

3. The API will be available at http://localhost:8000

4. API documentation will be available at http://localhost:8000/docs

5. Test the API using the provided test client:

```bash
# Using Stable Diffusion (default) and independent frame generation
python test_client.py --query "A person walking on the beach"

# Using DALL-E for image generation
python test_client.py --query "A person walking on the beach" --image-model sd

# Using continuous frame generation mode for more coherent sequences
python test_client.py --query "A person walking on the beach" --frame-mode continuous
```

## Foundation Models

Yarn implements a flexible foundation model framework that allows you to choose different models for various tasks:

### Image Generation Models

- **Stable Diffusion** (default): Uses Replicate API to generate images
  - Independent mode: Uses standard image generation
  - Continuous mode: Uses inpainting to maintain visual consistency

- **DALL-E**: Uses OpenAI API to generate images
  - Independent mode: Uses DALL-E 3 for high-quality images
  - Continuous mode: Uses DALL-E 2's edit capabilities for consistency

### Embedding Models

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model for generating embeddings from images

## API Endpoints

### Search Videos

```
POST /api/search/
```

Request body:
```json
{
  "query": "A person running on the beach at sunset",
  "max_frames": 5,
  "top_k": 3,
  "frame_mode": "independent",
  "image_model": "sd",
  "embedding_model": "clip"
}
```

Response:
```json
[
  {
    "video_id": "beach_sunset.mp4",
    "distance": 0.28,
    "score": 0.78
  },
  {
    "video_id": "exercise.mp4",
    "distance": 0.45,
    "score": 0.69
  },
  {
    "video_id": "nature.mp4",
    "distance": 0.52,
    "score": 0.66
  }
]
```

## Frame Generation Modes

Yarn supports two modes for generating frames from a query:

### Independent Mode (Default)

In this mode, each frame is generated independently based on its description. This gives the model maximum freedom to interpret each description without constraints from previous frames.

### Continuous Mode

In this mode, frames are generated sequentially, with each frame using the previous frame as a reference:

- **Stable Diffusion**: Uses inpainting to maintain consistent elements while updating others
- **DALL-E**: Uses DALL-E 2's image editing capabilities to maintain consistency

This creates a more coherent sequence of frames with consistent characters, settings, colors, and visual style throughout the sequence.

## Debug Mode

Yarn includes a debug mode that logs detailed information about each search query, including:

1. The original query text and parameters
2. Generated frame descriptions
3. Generated images for each frame description
4. Search results
5. Foundation models used

To enable debug mode, set the `DEBUG_MODE` environment variable to `true` in your `.env` file:

```
DEBUG_MODE=true
```

Debug logs will be saved to:
- `./logs/debug/[session_id]/query.json` - Query details
- `./logs/debug/[session_id]/descriptions.json` - Generated frame descriptions
- `./logs/debug/[session_id]/results.json` - Search results
- `./logs/frames/[session_id]/frame_##_*.jpg` - Generated images
- `./logs/frames/[session_id]/metadata.json` - Image metadata

## Extending with New Models

Yarn's architecture is designed to be easily extended with new foundation models:

1. **Add a new model enum** in `app/models/video.py`
2. **Implement the model class** in `app/services/foundation_models.py` by extending the appropriate abstract base class
3. **Register the model** in the `ModelFactory` class

## Requirements

- Python 3.8+
- FastAPI
- OpenCV
- PyTorch
- Transformers (CLIP)
- OpenAI API access
- Replicate API access
- CUDA-capable GPU (recommended but not required)
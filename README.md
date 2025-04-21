# Yarn - Text to Video Retrieval Engine

Yarn is a FastAPI backend service that provides text-to-video retrieval capabilities using GenAI technologies. The system indexes video files by extracting key frames and generating embeddings, then matches user text queries to the most relevant videos.

## Features

- Automatic video indexing on startup
- Frame extraction using scene detection
- Embedding generation using CLIP
- Query processing using GPT-4o
- Image generation using DALL-E
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
OPENAI_API_KEY=your_openai_api_key
VIDEO_DIRECTORY=./videos
MAX_FRAMES_DEFAULT=5
TOP_K_DEFAULT=3
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
python test_client.py --query "A person walking on the beach" --max-frames 5 --top-k 3
```

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
  "top_k": 3
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

## Debug Mode

Yarn includes a debug mode that logs detailed information about each search query, including:

1. The original query text
2. Generated frame descriptions
3. Generated images for each frame description
4. Search results

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

Each search query creates a new session with a unique ID (timestamp), making it easy to correlate logs and images.

## How It Works

1. **Video Indexing**: On startup, the system processes all videos in the specified directory, extracting key frames and generating embeddings.

2. **Query Processing**: When a user submits a text query, the system uses GPT-4o to generate descriptive frames from the query.

3. **Image Generation**: DALL-E generates visual representations based on the frame descriptions.

4. **Embedding Comparison**: The system generates embeddings for the generated images and compares them to the video embeddings using Dynamic Time Warping (DTW).

5. **Result Ranking**: Videos are ranked by similarity score, and the top-k results are returned.

## Requirements

- Python 3.8+
- FastAPI
- OpenCV
- PyTorch
- Transformers (CLIP)
- OpenAI API access
- CUDA-capable GPU (recommended but not required)
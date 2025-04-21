from typing import List, Optional
import numpy as np
from pydantic import BaseModel
from enum import Enum

class VideoFrame:
    def __init__(self, timestamp: float, image: np.ndarray, embedding: Optional[np.ndarray] = None):
        """
        A single frame from a video
        
        Args:
            timestamp: The timestamp of the frame in seconds
            image: The image data as a numpy array
            embedding: The embedding vector for the frame
        """
        self.timestamp = timestamp
        self.image = image
        self.embedding = embedding

class Video:
    def __init__(self, id: str, path: str, frames: List[VideoFrame]):
        """
        A video with extracted frames and embeddings
        
        Args:
            id: Unique identifier for the video
            path: Path to the video file
            frames: List of extracted key frames
        """
        self.id = id
        self.path = path
        self.frames = frames

class FrameGenerationMode(str, Enum):
    """Enum for frame generation modes"""
    INDEPENDENT = "independent"  # Generate each frame independently
    CONTINUOUS = "continuous"    # Generate frames in a continuous sequence (edit-based)

class ImageGenerationModel(str, Enum):
    """Enum for image generation models"""
    DALLE = "dalle"              # OpenAI DALL-E models
    STABLE_DIFFUSION = "sd"      # Stable Diffusion models via Replicate

class EmbeddingModel(str, Enum):
    """Enum for embedding models"""
    CLIP = "clip"                # OpenAI CLIP model

class SearchQuery(BaseModel):
    query: str
    max_frames: int = 5
    top_k: int = 3
    frame_mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    image_model: ImageGenerationModel = ImageGenerationModel.STABLE_DIFFUSION
    embedding_model: EmbeddingModel = EmbeddingModel.CLIP
    
class SearchResult(BaseModel):
    video_id: str
    distance: float
    score: float
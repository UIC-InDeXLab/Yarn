from typing import List, Optional
import numpy as np
from pydantic import BaseModel

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

class SearchQuery(BaseModel):
    query: str
    max_frames: int = 5
    top_k: int = 3
    
class SearchResult(BaseModel):
    video_id: str
    distance: float
    score: float
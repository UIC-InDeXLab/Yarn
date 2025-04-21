from enum import Enum
from typing import List, Optional, Dict, Union

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingModel(str, Enum):
    """Enum for embedding models"""
    CLIP = "vit_large_patch14_clip_336.openai_ft_in12k_in1k"  # OpenAI CLIP model
    DINO = "vit_large_patch14_reg4_dinov2.lvd142m"
    MOBILENET = "mobilenet"  # MobileNet v2 model
    RESNET50 = "resnet50"  # ResNet-50 model
    EFFICIENTNET = "efficientnet_b0"  # EfficientNet B0
    VIT = "vit_base_patch16_224"  # Vision Transformer
    SWIN = "swin_tiny_patch4_window7_224"  # Swin Transformer


class EmbedderConfig(BaseModel):
    """Configuration for an embedder"""
    model: EmbeddingModel
    weight: float = 1.0


class VideoFrame:
    def __init__(self, timestamp: float, image: np.ndarray):
        """
        A single frame from a video
        
        Args:
            timestamp: The timestamp of the frame in seconds
            image: The image data as a numpy array
        """
        self.timestamp = timestamp
        self.image = image
        self.embeddings = {}


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
    CONTINUOUS = "continuous"  # Generate frames in a continuous sequence (edit-based)


class ImageGenerationModel(str, Enum):
    """Enum for image generation models"""
    DALLE = "dalle"  # OpenAI DALL-E models
    STABLE_DIFFUSION = "sd"  # Stable Diffusion models via Replicate


class SearchQuery(BaseModel):
    query: str
    max_frames: int = 5
    top_k: int = 3
    frame_mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    image_model: ImageGenerationModel = ImageGenerationModel.STABLE_DIFFUSION
    embedding_models: List[EmbedderConfig] = Field(
        default_factory=lambda: [EmbedderConfig(model=EmbeddingModel.CLIP, weight=1.0)]
    )


class SearchResult(BaseModel):
    video_id: str
    distance: float
    score: float


def aggregate_rankings(rankers_results: List[List[Dict]], weights: List[float], k: int) -> List[Dict]:
    """
    Aggregate rankings from multiple rankers based on rank position
    
    Args:
        rankers_results: List of results from each ranker
        weights: List of weights for each ranker
        k: Number of top results to return
        
    Returns:
        Aggregated ranked results
    """
    scores = {}

    for i, R_i in enumerate(rankers_results):
        for j, result in enumerate(R_i):
            video_id = result["video_id"]
            if video_id not in scores:
                scores[video_id] = {
                    "video_id": video_id,
                    "score": 0,
                    "distance": 0,
                    "original_scores": {}
                }

            # Store the original score and distance
            scores[video_id]["original_scores"][i] = result["score"]

            # Use rank position (j+1) directly instead of score
            # The weight affects the importance of this ranker, not the score itself
            scores[video_id]["score"] += weights[i] * (1.0 / (j + 1))

    # Convert dictionary to list and sort by score
    ranked_results = list(scores.values())
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    # Compute an appropriate distance based on the inverse of the aggregated score
    for result in ranked_results:
        # Normalize the score to range [0, 1]
        max_score = ranked_results[0]["score"] if ranked_results else 1
        normalized_score = result["score"] / max_score if max_score > 0 else 0

        # Convert to a "distance" (smaller is better)
        result["distance"] = 1.0 - normalized_score

    return ranked_results[:k]

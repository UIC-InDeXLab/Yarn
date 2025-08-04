from enum import Enum
from typing import List, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingModel(str, Enum):
    """Enum for embedding models"""
    CLIP = "vit_large_patch14_clip_336.openai_ft_in12k_in1k"  # OpenAI CLIP model
    DINO = "vit_large_patch14_reg4_dinov2.lvd142m"
    # MOBILENET = "mobilenet"  # MobileNet v2 model
    # RESNET50 = "resnet50"  # ResNet-50 model
    # EFFICIENTNET = "efficientnet_b0"  # EfficientNet B0
    # VIT = "vit_base_patch16_224"  # Vision Transformer
    # SWIN = "swin_tiny_patch4_window7_224"  # Swin Transformer


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
    DEFAULT = "default"
    VIDEO = "video"  # Video generation model via Lightricks/LTX-Video


class SearchQuery(BaseModel):
    query: str
    max_frames: int = 5
    num_frames: Optional[int] = Field(
        None,
        description="Number of frames for video generation",
    )
    top_k: int = 3
    frame_mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    image_model: ImageGenerationModel = ImageGenerationModel.DEFAULT
    video_model_id: str = Field(
        "Lightricks/LTX-Video",
        description="Hugging Face model ID for video generation"
    )
    embedding_models: List[EmbedderConfig] = Field(
        default_factory=lambda: [EmbedderConfig(model=EmbeddingModel.CLIP, weight=1.0)]
    )
    lambda_param: float = Field(
        0.5,
        description="Weight for combining contextual and action distances (0.0 to 1.0)"
    )


class SearchResponse(BaseModel):
    results: List[str]
    preview_url: str


def aggregate_rankings(rankers_results: List[List[Dict]], weights: List[float], k: int) -> List[str]:
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
            if result["video_id"] not in scores:
                scores[result["video_id"]] = 0
            scores[result["video_id"]] += weights[i] * (1 / (j + 1))

    ranked_results = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return ranked_results[:k]

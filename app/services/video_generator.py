import logging
import logging
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from diffusers import DiffusionPipeline

from app.services.generator import Generator

logger = logging.getLogger(__name__)


class VideoGenerator(Generator, ABC):
    """Abstract base class for video generators"""

    @abstractmethod
    async def generate_video_frames(self, prompt: str, num_frames: int) -> List[np.ndarray]:
        pass

    async def generate(self, prompt: str, **kwargs) -> List[np.ndarray]:
        num_frames = kwargs.get('num_frames', 10)
        return await self.generate_video_frames(prompt, num_frames)


class LTXVideoGenerator(VideoGenerator):
    """Video generator using LTX-Video model from Hugging Face"""

    def __init__(self, model_id: str = "Lightricks/LTX-Video"):
        # Initialize device and pipeline with specified Hugging Face model
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_ACCESS_TOKEN")
        # Load the video pipeline
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                token=hf_token,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            logger.error(f"Failed to load {self.model_id} pipeline: {e}")
            raise
        # Move or offload model
        if self.device == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe.to(self.device)
        else:
            self.pipe.to(self.device)
        logger.info(f"Initialized LTX-Video generator pipeline: {self.model_id}")

    async def generate_video_frames(self, prompt: str, num_frames: int) -> List[np.ndarray]:
        def _gen():
            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
            # Generate video with the specified number of frames
            video = self.pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=50,
                negative_prompt=negative_prompt,
            )
            return video.frames[0]

        # Run generation in thread to avoid blocking
        video_frames = _gen()
        logger.info(f"Generated video with {len(video_frames)} frames for prompt: {prompt[:50]}")
        return [np.array(img) for img in video_frames]

# ----------------------------------------------------------------------------
# CogVideoX Generator Support
# ----------------------------------------------------------------------------
class CogVideoXGenerator(VideoGenerator):
    """Video generator using THUDM/CogVideoX-5b model from Hugging Face"""

    def __init__(self, model_id: str = "THUDM/CogVideoX-5b"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_ACCESS_TOKEN")
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                token=hf_token,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            logger.error(f"Failed to load {self.model_id} pipeline: {e}")
            raise
        if self.device == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe.to(self.device)
        else:
            self.pipe.to(self.device)
        logger.info(f"Initialized CogVideoX generator pipeline: {self.model_id}")

    async def generate_video_frames(self, prompt: str, num_frames: int) -> List[np.ndarray]:
        """Generate video frames using the CogVideoX pipeline"""
        def _gen():
            # Generate video; CovVideoX pipelines return output with .frames or .videos
            output = self.pipe(
                prompt=prompt,
                num_frames=49,
                num_inference_steps=50,
            )
            # Extract frames list from output
            if hasattr(output, "frames"):
                # .frames is a list of videos, take first video
                return output.frames[0]
            if hasattr(output, "videos"):
                # .videos is a list of videos, take first video
                return output.videos[0]
            raise RuntimeError(f"Unexpected output format from model {self.model_id}")

        video_frames = _gen()
        logger.info(f"Generated video with {len(video_frames)} frames for prompt: {prompt[:50]}")

        res = []
        for i, frame in enumerate(video_frames):
            if i % 6 == 0:
                res.append(frame)

        return [np.array(img) for img in res]


def get_video_generator(model_id: str) -> VideoGenerator:
    """
    Factory for video generators based on model ID.
    """
    # Choose specific generator based on model ID prefix
    if model_id and model_id.startswith("THUDM/CogVideoX"):
        return CogVideoXGenerator(model_id=model_id)
    # Default to LTX-Video generator
    return LTXVideoGenerator(model_id=model_id)

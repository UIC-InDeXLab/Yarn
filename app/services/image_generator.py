import asyncio
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxPipeline

from app.models.video import FrameGenerationMode

logger = logging.getLogger(__name__)


class ImageGenerator(ABC):
    @abstractmethod
    async def generate_image(self, prompt: str) -> Tuple[np.ndarray, Any]:
        pass

    @abstractmethod
    async def edit_image(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Any]:
        pass

    @abstractmethod
    async def generate_batch(
        self,
        prompts: List[str],
        mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT,
        width: int = 1024,
        height: int = 1024
    ) -> List[np.ndarray]:
        pass


class StableDiffusionGenerator(ImageGenerator):
    """Local Stable Diffusion generator using diffusers pipelines"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize pipelines with CPU offloading for memory efficiency
        self.txt2img_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            token=os.getenv("HF_ACCESS_TOKEN"),
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        self.img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            token=os.getenv("HF_ACCESS_TOKEN"),
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        
        # Enable CPU offloading for both pipelines to reduce memory usage
        if self.device == "cuda":
            self.txt2img_pipe.enable_model_cpu_offload()
            self.img2img_pipe.enable_model_cpu_offload()
        else:
            self.txt2img_pipe.to(self.device)
            self.img2img_pipe.to(self.device)
        self.generator = torch.Generator(self.device).manual_seed(42)
        logger.info("Initialized local Stable Diffusion generator via diffusers")

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> Tuple[np.ndarray, Any]:
        """
        Generate an image using the local Stable Diffusion (txt2img pipeline)
        
        Args:
            prompt: The text prompt for image generation
            width: Output image width
            height: Output image height
            
        Returns:
            Tuple of (image array, None)
        """
        def _generate():
            # Simple generation without fallback resizing
            return self.txt2img_pipe(
                prompt,
                width=width,
                height=height,
                guidance_scale=7.5,
                num_inference_steps=4,
                generator=self.generator
            ).images[0]

        image = await asyncio.to_thread(_generate)
        image_array = np.array(image)
        logger.info(f"Generated image for prompt: {prompt[:50]}...")
        return image_array, None

    async def edit_image(self, image: np.ndarray, prompt: str, width: int = 1024, height: int = 1024) -> Tuple[np.ndarray, Any]:
        """
        Edit an image using the local Stable Diffusion (img2img pipeline)
        
        Args:
            image: Source image as numpy array
            prompt: The text prompt for image editing
            width: Output image width
            height: Output image height
            
        Returns:
            Tuple of (edited image array, None)
        """
        def _edit():
            pil_image = Image.fromarray(image).convert("RGB")
            return self.img2img_pipe(
                prompt=prompt,
                image=pil_image,
                width=width,
                height=height,
                strength=0.93,
                guidance_scale=11.5,
                num_inference_steps=40,
                generator=self.generator
            ).images[0]

        edited = await asyncio.to_thread(_edit)
        edited_array = np.array(edited)
        logger.info(f"Edited image for prompt: {prompt[:50]}...")
        return edited_array, None

    async def generate_batch(
        self,
        prompts: List[str],
        mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT,
        width: int = 1024,
        height: int = 1024
    ) -> List[np.ndarray]:
        """
        Generate a batch of images. In independent mode each image is generated
        from scratch with txt2img; in continuous mode each next image uses the
        previous one as reference with img2img.
        """
        if not prompts:
            return []

        images: List[np.ndarray] = []
        for idx, prompt in enumerate(prompts):
            try:
                if idx == 0 or mode == FrameGenerationMode.INDEPENDENT:
                    img, _ = await self.generate_image(prompt, width, height)
                else:
                    img, _ = await self.edit_image(images[-1], prompt, width, height)
                images.append(img)
            except Exception as e:
                logger.error(f"Error processing prompt '{prompt[:30]}...': {e}")
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                images.append(blank)
            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return images
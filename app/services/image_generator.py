import asyncio
import os
import logging
from abc import abstractmethod, ABC
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

from app.models.video import FrameGenerationMode
from app.services.generator import Generator

logger = logging.getLogger(__name__)


class ImageGenerator(Generator, ABC):
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
        mode: FrameGenerationMode = FrameGenerationMode.CONTINUOUS,
        width: int = 1024,
        height: int = 1024
    ) -> List[np.ndarray]:
        pass
    async def generate(self, prompts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Generate content based on prompts. Delegates to generate_batch for image models.
        """
        mode = kwargs.get('mode', FrameGenerationMode.INDEPENDENT)
        width = kwargs.get('width', 1024)
        height = kwargs.get('height', 1024)
        return await self.generate_batch(prompts, mode, width, height)


class Pix2PixImageGenerator(ImageGenerator):
    """Image generator using InstructPix2Pix."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.generator = torch.Generator(self.device).manual_seed(42)
        logger.info("Initialized InstructPix2Pix generator.")

    async def generate_image(self, prompt: str, width: int = 512, height: int = 512) -> np.ndarray:
        # Pix2Pix is primarily for editing, so we generate a blank image first
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return await self.edit_image(image, prompt, width, height)

    async def edit_image(self, image: np.ndarray, prompt: str, width: int = 512, height: int = 512) -> np.ndarray:
        def _edit():
            pil_image = Image.fromarray(image).convert("RGB")
            edited_image = self.pipe(
                prompt,
                image=pil_image,
                num_inference_steps=10,
                image_guidance_scale=1,
                guidance_scale=7.5,
                generator=self.generator,
            ).images[0]
            return edited_image

        edited = await asyncio.to_thread(_edit)
        return np.array(edited)

    async def generate_batch(
        self,
        prompts: List[str],
        mode: FrameGenerationMode = FrameGenerationMode.CONTINUOUS,
        width: int = 512,
        height: int = 512
    ) -> List[np.ndarray]:
        if not prompts:
            return []

        images: List[np.ndarray] = []
        for idx, prompt in enumerate(prompts):
            try:
                if idx == 0:
                    # Generate a blank image for the first frame
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                    img = await self.edit_image(img, prompt, width, height)
                else:
                    img = await self.edit_image(images[-1], prompt, width, height)
                images.append(img)
            except Exception as e:
                logger.error(f"Error processing prompt '{prompt[:30]}...': {e}")
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                images.append(blank)
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return images

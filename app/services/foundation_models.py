import base64
import io
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Any, Tuple

import numpy as np
import replicate
import requests
from PIL import Image
from openai import OpenAI

from app.models.video import ImageGenerationModel, FrameGenerationMode
from app.utils.config import OPENAI_API_KEY, REPLICATE_API_TOKEN

logger = logging.getLogger(__name__)


class ImageGenerator(ABC):
    """Abstract base class for image generators"""

    @abstractmethod
    async def generate_image(self, prompt: str) -> Tuple[np.ndarray, Any]:
        """Generate an image based on a text prompt"""
        pass

    @abstractmethod
    async def edit_image(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Any]:
        """Edit an existing image based on a text prompt"""
        pass

    @abstractmethod
    async def generate_batch(
            self,
            prompts: List[str],
            mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    ) -> List[np.ndarray]:
        """Generate a batch of images"""
        pass


class DalleImageGenerator(ImageGenerator):
    """OpenAI DALL-E image generator"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Initialized OpenAI DALL-E image generator")

    async def generate_image(self, prompt: str) -> Tuple[np.ndarray, str]:
        """
        Generate an image using DALL-E 3
        
        Args:
            prompt: Text description for image generation
            
        Returns:
            Tuple of (generated image as numpy array, base64 string of image)
        """
        try:
            # Generate image with DALL-E 3
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json"
            )

            # Decode base64 image
            image_data = base64.b64decode(response.data[0].b64_json)

            # Convert to PIL Image and then to numpy array
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_array = np.array(image)

            logger.info(f"Generated DALL-E image for prompt: {prompt[:50]}...")
            return image_array, response.data[0].b64_json

        except Exception as e:
            logger.error(f"Error generating DALL-E image: {str(e)}")
            # Return a dummy image
            return np.zeros((1024, 1024, 3), dtype=np.uint8), ""

    async def edit_image(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, str]:
        """
        Edit an image using DALL-E 2
        
        Args:
            image: The source image as numpy array
            prompt: Text description for image editing
            
        Returns:
            Tuple of (edited image as numpy array, base64 string of image)
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image).convert("RGB")

            # DALL-E 2 requires dimensions to be square and no larger than 1024x1024
            # Resize to 512x512 which is safer for the API
            pil_image = pil_image.resize((512, 512))

            # Save image to a temporary file
            temp_image_path = "/tmp/temp_image_for_dalle.png"
            mask_path = "/tmp/temp_mask_for_dalle.png"

            # Save as PNG with proper format
            pil_image.save(temp_image_path, format="PNG")

            # Create a transparent mask (all areas open for editing)
            mask = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            mask.save(mask_path, format="PNG")

            # Open file handles for the OpenAI API
            with open(temp_image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
                # Edit image with DALL-E 2
                edited_response = self.client.images.edit(
                    model="dall-e-2",
                    image=image_file,
                    mask=mask_file,
                    prompt=prompt,
                    n=1,
                    size="512x512",
                    response_format="b64_json"
                )

            # Decode base64 image
            edited_image_data = base64.b64decode(edited_response.data[0].b64_json)

            # Convert to PIL Image and then to numpy array
            edited_image = Image.open(io.BytesIO(edited_image_data)).convert("RGB")

            # Resize back to 1024x1024 to match DALL-E 3 output
            edited_image = edited_image.resize((1024, 1024))
            edited_image_array = np.array(edited_image)

            # Clean up temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)

            logger.info(f"Edited DALL-E image with prompt: {prompt[:50]}...")
            return edited_image_array, edited_response.data[0].b64_json

        except Exception as e:
            logger.error(f"Error editing DALL-E image: {str(e)}")
            logger.error(f"Exception details: {e}")

            # Clean up any temporary files that might exist
            for path in ["/tmp/temp_image_for_dalle.png", "/tmp/temp_mask_for_dalle.png"]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

            # As fallback, generate a new image instead of editing
            logger.info(f"Falling back to generating new image for prompt: {prompt[:50]}...")
            return await self.generate_image(prompt)

    async def generate_batch(
            self,
            prompts: List[str],
            mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    ) -> List[np.ndarray]:
        """
        Generate a batch of images using DALL-E
        
        Args:
            prompts: List of text prompts
            mode: Frame generation mode (independent or continuous)
            
        Returns:
            List of generated images as numpy arrays
        """
        if not prompts:
            return []

        if mode == FrameGenerationMode.INDEPENDENT:
            return await self._generate_independent_batch(prompts)
        else:
            return await self._generate_continuous_batch(prompts)

    async def _generate_independent_batch(self, prompts: List[str]) -> List[np.ndarray]:
        """Generate independent images for each prompt"""
        generated_images = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Generating independent DALL-E frame {i + 1}/{len(prompts)}")
            frame, _ = await self.generate_image(prompt)
            generated_images.append(frame)

        return generated_images

    async def _generate_continuous_batch(self, prompts: List[str]) -> List[np.ndarray]:
        """Generate continuous sequence of images"""
        if not prompts:
            return []

        generated_images = []

        # Generate first frame independently
        first_frame, _ = await self.generate_image(prompts[0])
        generated_images.append(first_frame)

        # Generate subsequent frames by editing the previous ones
        for i in range(1, len(prompts)):
            try:
                logger.info(f"Generating continuous DALL-E frame {i + 1}/{len(prompts)} using edit mode")
                # Generate next frame by editing the previous frame
                next_frame, _ = await self.edit_image(
                    generated_images[-1],
                    prompts[i]
                )
                generated_images.append(next_frame)
            except Exception as e:
                logger.error(f"Error in continuous DALL-E generation for frame {i + 1}: {str(e)}")
                # Fall back to independent generation
                logger.info(f"Falling back to independent DALL-E generation for frame {i + 1}")
                fallback_frame, _ = await self.generate_image(prompts[i])
                generated_images.append(fallback_frame)

        return generated_images


class StableDiffusionGenerator(ImageGenerator):
    """Stable Diffusion image generator via Replicate"""

    def __init__(self):
        # Set the Replicate API token
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        self.client = replicate

        # Stable Diffusion model for image generation
        self.sd_model = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"

        # Stable Diffusion model for inpainting
        self.inpaint_model = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
        logger.info("Initialized Stable Diffusion generator via Replicate")

    async def generate_image(self, prompt: str) -> Tuple[np.ndarray, Any]:
        """
        Generate an image using Stable Diffusion
        
        Args:
            prompt: Text description for image generation
            
        Returns:
            Tuple of (generated image as numpy array, model output)
        """
        try:
            # Run Stable Diffusion inference
            output = self.client.run(
                self.sd_model,
                input={
                    "prompt": prompt,
                    "width": 1024,
                    "height": 1024,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50
                }
            )

            # The output is a list of image URLs
            if output and len(output) > 0:
                # Download the image
                response = requests.get(output[0])
                if response.status_code == 200:
                    # Convert to PIL Image and then to numpy array
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    image_array = np.array(image)

                    logger.info(f"Generated Stable Diffusion image for prompt: {prompt[:50]}...")
                    return image_array, output[0]

            raise Exception("No image generated by Stable Diffusion")

        except Exception as e:
            logger.error(f"Error generating Stable Diffusion image: {str(e)}")
            # Return a dummy image
            return np.zeros((1024, 1024, 3), dtype=np.uint8), None

    async def edit_image(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Any]:
        """
        Edit an image using Stable Diffusion inpainting
        
        Args:
            image: The source image as numpy array
            prompt: Text description for image editing
            
        Returns:
            Tuple of (edited image as numpy array, model output)
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image).convert("RGB")

            # Save image to a temporary file
            temp_image_path = "/tmp/temp_image_for_sd.png"

            # Save as PNG
            pil_image.save(temp_image_path, format="PNG")

            # Convert images to base64
            with open(temp_image_path, "rb") as img_file:
                image_base64 = f"data:application/octet-stream;base64,{base64.b64encode(img_file.read()).decode("utf-8")}"

            # Run Stable Diffusion inpainting
            output = self.client.run(
                self.inpaint_model,
                input={
                    "prompt": prompt,
                    "image": image_base64,
                    "num_outputs": 1,
                }
            )

            # Clean up temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            # The output is a list of image URLs
            if output and len(output) > 0:
                # Download the image
                response = requests.get(output[0])
                if response.status_code == 200:
                    # Convert to PIL Image and then to numpy array
                    edited_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    edited_image_array = np.array(edited_image)

                    logger.info(f"Edited Stable Diffusion image with prompt: {prompt[:50]}...")
                    return edited_image_array, output[0]

            raise Exception("No image generated by Stable Diffusion inpainting")

        except Exception as e:
            logger.error(f"Error editing Stable Diffusion image: {str(e)}")
            logger.error(f"Exception details: {e}")

            # Clean up any temporary files that might exist
            for path in ["/tmp/temp_image_for_sd.png", "/tmp/temp_mask_for_sd.png"]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

            # As fallback, generate a new image instead of editing
            logger.info(f"Falling back to generating new image for prompt: {prompt[:50]}...")
            return await self.generate_image(prompt)

    async def generate_batch(
            self,
            prompts: List[str],
            mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    ) -> List[np.ndarray]:
        """
        Generate a batch of images using Stable Diffusion
        
        Args:
            prompts: List of text prompts
            mode: Frame generation mode (independent or continuous)
            
        Returns:
            List of generated images as numpy arrays
        """
        if not prompts:
            return []

        if mode == FrameGenerationMode.INDEPENDENT:
            return await self._generate_independent_batch(prompts)
        else:
            return await self._generate_continuous_batch(prompts)

    async def _generate_independent_batch(self, prompts: List[str]) -> List[np.ndarray]:
        """Generate independent images for each prompt"""
        generated_images = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Generating independent Stable Diffusion frame {i + 1}/{len(prompts)}")
            frame, _ = await self.generate_image(prompt)
            generated_images.append(frame)

        return generated_images

    async def _generate_continuous_batch(self, prompts: List[str]) -> List[np.ndarray]:
        """Generate continuous sequence of images"""
        if not prompts:
            return []

        generated_images = []

        # Generate first frame independently
        first_frame, _ = await self.generate_image(prompts[0])
        generated_images.append(first_frame)

        # Generate subsequent frames by editing the previous ones
        for i in range(1, len(prompts)):
            try:
                logger.info(f"Generating continuous Stable Diffusion frame {i + 1}/{len(prompts)} using inpainting")
                # Generate next frame by editing the previous frame
                next_frame, _ = await self.edit_image(
                    generated_images[-1],
                    prompts[i]
                )
                generated_images.append(next_frame)
            except Exception as e:
                logger.error(f"Error in continuous Stable Diffusion generation for frame {i + 1}: {str(e)}")
                # Fall back to independent generation
                logger.info(f"Falling back to independent Stable Diffusion generation for frame {i + 1}")
                fallback_frame, _ = await self.generate_image(prompts[i])
                generated_images.append(fallback_frame)

        return generated_images


class ModelFactory:
    """Factory for creating foundation models"""

    @staticmethod
    def create_image_generator(model_type: ImageGenerationModel) -> ImageGenerator:
        """Create an image generator based on the model type"""
        if model_type == ImageGenerationModel.DALLE:
            return DalleImageGenerator()
        elif model_type == ImageGenerationModel.STABLE_DIFFUSION:
            return StableDiffusionGenerator()
        else:
            logger.warning(f"Unknown image generation model: {model_type}, using Stable Diffusion")
            return StableDiffusionGenerator()

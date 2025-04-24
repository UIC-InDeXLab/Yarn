import asyncio
import logging
from abc import abstractmethod, ABC
from typing import List, Any, Tuple

import numpy as np
import timm
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxPipeline
from timm.data import resolve_model_data_config, create_transform
from transformers import CLIPModel, CLIPProcessor

from app.models.video import FrameGenerationMode, ImageGenerationModel

logger = logging.getLogger(__name__)


class ImageGenerator:
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
            mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    ) -> List[np.ndarray]:
        pass


class StableDiffusionGenerator(ImageGenerator):
    """Local Stable Diffusion generator using diffusers pipelines"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.txt2img_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.generator = torch.Generator(self.device).manual_seed(42)
        logger.info("Initialized local Stable Diffusion generator via diffusers")

    async def generate_image(self, prompt: str) -> Tuple[np.ndarray, Any]:
        """
        Generate an image using the local Stable Diffusion (txt2img pipeline)
        """

        def _generate():
            image = self.txt2img_pipe(
                prompt,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=self.generator
            ).images[0]
            return image

        image = await asyncio.to_thread(_generate)
        image_array = np.array(image)
        logger.info(f"Generated local image for prompt: {prompt[:50]}...")
        # Returning None as output info since image is generated locally.
        return image_array, None

    async def edit_image(self, image: np.ndarray, prompt: str) -> Tuple[np.ndarray, Any]:
        """
        Edit an image using the local Stable Diffusion (img2img pipeline)
        """

        def _edit():
            pil_image = Image.fromarray(image).convert("RGB")
            edited_image = self.img2img_pipe(
                prompt=prompt,
                image=pil_image,
                strength=0.75,
                guidance_scale=7.5,
                generator=self.generator
            ).images[0]
            return edited_image

        edited = await asyncio.to_thread(_edit)
        edited_array = np.array(edited)
        logger.info(f"Edited local image with prompt: {prompt[:50]}...")
        return edited_array, None

    async def generate_batch(
            self,
            prompts: List[str],
            mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT
    ) -> List[np.ndarray]:
        """
        Generate a batch of images. In independent mode each image is generated
        from scratch with txt2img; in continuous mode each next image uses the
        previous one as reference with img2img.
        """
        if not prompts:
            return []

        if mode == FrameGenerationMode.INDEPENDENT:
            images = []
            for prompt in prompts:
                img, _ = await self.generate_image(prompt)
                images.append(img)
            return images
        else:
            def _generate_continuous():
                images = []
                # Generate the first image from text
                image = self.txt2img_pipe(
                    prompts[0],
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    generator=self.generator
                ).images[0]
                images.append(np.array(image))
                # Generate subsequent images from the previous one
                for prompt in prompts[1:]:
                    image = image.resize((512, 512))
                    image = self.img2img_pipe(
                        prompt=prompt,
                        image=image,
                        strength=0.75,
                        guidance_scale=7.5,
                        generator=self.generator
                    ).images[0]
                    images.append(np.array(image))
                return images

            images = await asyncio.to_thread(_generate_continuous)
            return images


class Embedder(ABC):
    """Abstract base class for image embedders"""

    @abstractmethod
    async def embed_images(self, images: List[Any]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of images

        Args:
            images: List of images (can be PIL Images, numpy arrays or file paths)

        Returns:
            List of embedding vectors
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of this embedder"""
        pass

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings generated by this embedder"""
        pass


class ClipEmbedder(Embedder):
    """CLIP-based image embedder"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP embedder

        Args:
            model_name: The CLIP model to use
        """
        self._name = "clip"
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model.to(self.device)
            self._embedding_dim = self.clip_model.get_image_features(
                **self.clip_processor(images=[Image.new("RGB", (224, 224))], return_tensors="pt").to("cpu")).shape[1]
            logger.info(f"CLIP model loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            self.clip_model = None
            self.clip_processor = None
            self._embedding_dim = 512  # Default CLIP dimension

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def embed_images(self, images: List[Any]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of images using CLIP

        Args:
            images: List of images (can be PIL Images, numpy arrays or file paths)

        Returns:
            List of embedding vectors
        """
        if not self.clip_model or not self.clip_processor:
            logger.error("CLIP model not initialized")
            return [np.zeros(self.embedding_dim) for _ in images]  # Return dummy embeddings

        embeddings = []
        batch_size = 16  # Process images in batches to avoid OOM

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            processed_batch = []

            for img in batch:
                if isinstance(img, str):  # File path
                    try:
                        pil_img = Image.open(img).convert("RGB")
                        processed_batch.append(pil_img)
                    except Exception as e:
                        logger.error(f"Error loading image from {img}: {str(e)}")
                        processed_batch.append(Image.new("RGB", (224, 224)))
                elif isinstance(img, np.ndarray):  # Numpy array
                    try:
                        pil_img = Image.fromarray(img).convert("RGB")
                        processed_batch.append(pil_img)
                    except Exception as e:
                        logger.error(f"Error converting numpy array to PIL Image: {str(e)}")
                        processed_batch.append(Image.new("RGB", (224, 224)))
                else:  # Assume it's already a PIL Image or compatible
                    processed_batch.append(img)

            try:
                with torch.no_grad():
                    inputs = self.clip_processor(images=processed_batch, return_tensors="pt").to(self.device)
                    image_features = self.clip_model.get_image_features(**inputs)
                    # Normalize embeddings
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    # Convert to numpy and move to CPU
                    batch_embeddings = image_features.cpu().numpy()
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                # Add zero embeddings for the failed batch
                dummy_embedding = np.zeros(self.embedding_dim)
                embeddings.extend([dummy_embedding] * len(processed_batch))

        return embeddings


class TimmEmbedder(Embedder):
    """General embedder using timm models"""

    def __init__(self, model_name: str):
        """
        Initialize embedder using any timm model

        Args:
            model_name: Name of the model in timm library
        """
        self._name = model_name
        self._embedding_dim = 1000  # Default, will be updated when model is loaded
        try:
            # Import timm here to avoid dependency issue if not available

            # Load the model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
            # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            #     self.model = nn.DataParallel(model)
            # else:
            #     self.model = model
            self.model.eval()

            # Get preprocessing transform
            self.preprocess = self.get_preprocess()

            # Update embedding dimension
            self._embedding_dim = self.model.num_features

            self.model.to(self.device)
            logger.info(f"Timm model '{model_name}' loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading timm model '{model_name}': {str(e)}")
            self.model = None
            self.preprocess = None

    def get_preprocess(self):
        # Unwrap the model if wrapped in DataParallel
        model_for_config = self.model.module if hasattr(self.model, 'module') else self.model
        data_config = resolve_model_data_config(model_for_config)
        return create_transform(**data_config, is_training=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def embed_images(self, images: List[Any]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of images using the timm model

        Args:
            images: List of images (can be PIL Images, numpy arrays or file paths)

        Returns:
            List of embedding vectors
        """
        if not self.model or not self.preprocess:
            logger.error(f"Model {self.name} not initialized")
            return [np.zeros(self.embedding_dim) for _ in images]  # Return dummy embeddings

        embeddings = []
        batch_size = 16  # Process images in batches to avoid OOM

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            processed_batch = []

            for img in batch:
                try:
                    # Convert to PIL Image if needed
                    if isinstance(img, str):  # File path
                        pil_img = Image.open(img).convert("RGB")
                    elif isinstance(img, np.ndarray):  # Numpy array
                        pil_img = Image.fromarray(img).convert("RGB")
                    else:  # Assume it's already a PIL Image or compatible
                        pil_img = img

                    # Apply preprocessing from timm
                    tensor = self.preprocess(pil_img).unsqueeze(0)
                    processed_batch.append(tensor)
                except Exception as e:
                    logger.error(f"Error processing image for {self.name}: {str(e)}")
                    # Add a dummy tensor (using correct input shape)
                    processed_batch.append(torch.zeros(1, 3, 224, 224))

            try:
                with torch.no_grad():
                    # Concatenate and process batch
                    batch_tensor = torch.cat(processed_batch, dim=0).to(self.device)
                    # Get features directly from the model
                    features = self.model(batch_tensor)

                    # Normalize the features for similarity comparison
                    features = features / features.norm(dim=1, keepdim=True)

                    # Convert to numpy and move to CPU
                    batch_embeddings = features.cpu().numpy()
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating {self.name} embeddings: {str(e)}")
                # Add zero embeddings for the failed batch
                dummy_embedding = np.zeros(self.embedding_dim)
                embeddings.extend([dummy_embedding] * len(processed_batch))

        return embeddings


class ModelFactory:
    """Factory for creating foundation models"""

    @staticmethod
    def create_image_generator(model_type: ImageGenerationModel) -> ImageGenerator:
        """Create an image generator based on the model type"""
        if model_type == ImageGenerationModel.STABLE_DIFFUSION:
            return StableDiffusionGenerator()
        else:
            logger.warning(f"Unknown image generation model: {model_type}, using Stable Diffusion")
            return StableDiffusionGenerator()

    @staticmethod
    def create_embedder(model_type) -> Embedder:
        """
        Create an embedder based on the model type
        The model_type can be an EmbeddingModel enum or a string
        """
        # Check if it's a string (model name directly)
        if isinstance(model_type, str):
            if model_type == "clip":
                return ClipEmbedder()
            else:
                return TimmEmbedder(model_type)

        # Handle as an enum
        model_value = getattr(model_type, "value", str(model_type))

        # Check special cases
        if model_value == "clip":
            return ClipEmbedder()
        else:
            # Assume it's a timm model name
            return TimmEmbedder(model_value)

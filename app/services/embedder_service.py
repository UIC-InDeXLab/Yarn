import logging
from typing import List, Any, Dict, Optional

import numpy as np

from app.models.video import FrameGenerationMode, ImageGenerationModel, EmbeddingModel, EmbedderConfig
from app.services.model_factory import ModelFactory
from app.services.image_generator import ImageGenerator
from app.services.embedders import Embedder
from app.utils.config import DEBUG_MODE, OPENAI_API_KEY
from app.utils.debug import DebugLogger

logger = logging.getLogger(__name__)


class EmbedderService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbedderService, cls).__new__(cls)
            cls._instance._init_services()
        return cls._instance

    def _init_services(self):
        """Initialize predefined embedder models"""
        self.embedders = {
            EmbeddingModel.CLIP.value: ModelFactory.create_embedder(EmbeddingModel.CLIP),
            EmbeddingModel.DINO.value: ModelFactory.create_embedder(EmbeddingModel.DINO),
            # EmbeddingModel.RESNET50.value: ModelFactory.create_embedder(EmbeddingModel.RESNET50),
            # EmbeddingModel.EFFICIENTNET.value: ModelFactory.create_embedder(EmbeddingModel.EFFICIENTNET),
            # EmbeddingModel.VIT.value: ModelFactory.create_embedder(EmbeddingModel.VIT),
            # EmbeddingModel.SWIN.value: ModelFactory.create_embedder(EmbeddingModel.SWIN),
            # EmbeddingModel.MOBILENET.value: ModelFactory.create_embedder(EmbeddingModel.MOBILENET),
        }

    def get_image_generator(self, model_type: ImageGenerationModel) -> ImageGenerator:
        """Get an image generator based on the model type"""
        return ModelFactory.create_image_generator(model_type)

    def get_embedder(self, model_type: EmbeddingModel) -> Embedder:
        """Get an embedder based on the model type"""
        model_name = model_type.value
        if model_name not in self.embedders:
            logger.warning(f"Embedder {model_name} not found in predefined models. Using CLIP instead.")
            return self.embedders[EmbeddingModel.CLIP.value]
        return self.embedders[model_name]

    async def embed_images(self, images: List[Any], model_type: Optional[EmbeddingModel] = None) -> List[np.ndarray]:
        """
        Generate embeddings for a list of images using a specific embedder
        
        Args:
            images: List of images (can be PIL Images, numpy arrays or file paths)
            model_type: The embedder model to use (defaults to CLIP if not specified)
            
        Returns:
            List of embedding vectors
        """
        if model_type is None:
            model_type = EmbeddingModel.CLIP

        embedder = self.get_embedder(model_type)
        return await embedder.embed_images(images)

    async def embed_images_with_multiple_models(
            self,
            images: List[Any],
            embedder_configs: List[EmbedderConfig]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Generate embeddings for a list of images using multiple embedders
        
        Args:
            images: List of images (can be PIL Images, numpy arrays or file paths)
            embedder_configs: List of embedder configurations to use
            
        Returns:
            Dictionary mapping embedder name to list of embedding vectors
        """
        results = {}

        for config in embedder_configs:
            embedder = self.get_embedder(config.model)
            embeddings = await embedder.embed_images(images)
            results[config.model.value] = embeddings

        return results

    async def generate_frame_descriptions(self, query: str, max_frames: int = 5, min_frames: int = 3,
                                          session_id: str = "") -> List[str]:
        """
        Generate frame descriptions from a text query using GPT-4o
        
        Args:
            query: The text query to generate frame descriptions from
            max_frames: Maximum number of frame descriptions to generate
            session_id: Debug session ID (for logging)
            
        Returns:
            List of frame descriptions
        """
        from openai import OpenAI

        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return [f"Frame description {i + 1} for {query}" for i in range(max_frames)]

        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            system_prompt = (
                "You are an expert at creating concise, visual descriptions for frames that will be used "
                "to find matching videos. Focus only on key visual elements directly related to the query. "
                "Do not add unnecessary details or interpret beyond what's explicitly in the query."
            )

            user_prompt = (
                f"Convert the following query into a sequence of {max_frames} or fewer visual frames, at least include {min_frames} frames. "
                f"Each frame should capture essential visual elements only. "
                f"Be precise and focus solely on elements explicitly mentioned in the query. "
                f"In each frame, you should describe everything in detail, since frames are being generated independently, you should remention everything that you have mentioned earlier if you need it, do not assume that you have information about previous frames"
                f"Note that, for some actions, you might need more than one frame to describe the action. do not hesitate to add more frames to show the actions."
                f"Avoid adding invented details that aren't directly implied by the query.\n\n"
                f"Query: {query}\n\n"
                f"Format your response as a numbered list of frame descriptions (like 1. 2. 3. etc.), with one description per frame."
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()

            # Parse the response to extract frame descriptions
            frame_descriptions = []
            current_description = ""

            for line in response_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check if line starts with a frame number
                if line[0].isdigit() and "." in line[:5]:
                    if current_description:
                        frame_descriptions.append(current_description.strip())
                    current_description = line.split(".", 1)[1].strip()
                else:
                    current_description += " " + line

            # Add the last description
            if current_description:
                frame_descriptions.append(current_description.strip())

            logger.info(f"Generated {len(frame_descriptions)} frame descriptions from query: {query}")

            # Log frame descriptions in debug mode
            if DEBUG_MODE and session_id:
                DebugLogger.log_frame_descriptions(session_id, frame_descriptions)

            return frame_descriptions

        except Exception as e:
            logger.error(f"Error generating frame descriptions: {str(e)}")
            return [f"Frame description {i + 1} for {query}" for i in range(max_frames)]

    async def generate_images_from_descriptions(
            self,
            descriptions: List[str],
            mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT,
            model_type: ImageGenerationModel = ImageGenerationModel.DEFAULT,
            session_id: str = ""
    ) -> List[np.ndarray]:
        """
        Generate images from frame descriptions
        
        Args:
            descriptions: List of frame descriptions
            mode: Frame generation mode (independent or continuous)
            model_type: Image generation model to use
            session_id: Debug session ID (for logging)
            
        Returns:
            List of generated images as numpy arrays
        """
        if not descriptions:
            logger.warning("No descriptions provided for image generation")
            return []

        # Get the appropriate image generator
        image_generator = self.get_image_generator(model_type)

        # Generate images
        images = await image_generator.generate_batch(descriptions, mode)

        # Save generated images in debug mode
        if DEBUG_MODE and session_id and images:
            DebugLogger.save_generated_frames(session_id, images)

        return images

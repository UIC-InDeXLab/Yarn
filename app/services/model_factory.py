import logging

from app.models.video import ImageGenerationModel
from app.services.image_generator import ImageGenerator, StableDiffusionGenerator
from app.services.embedders import Embedder, ClipEmbedder, TimmEmbedder

logger = logging.getLogger(__name__)


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
            return TimmEmbedder(model_value)
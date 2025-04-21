import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from app.models.video import FrameGenerationMode, ImageGenerationModel, EmbeddingModel
from app.utils.config import DEBUG_MODE, OPENAI_API_KEY
from app.utils.debug import DebugLogger
from app.services.foundation_models import ModelFactory, ImageGenerator

logger = logging.getLogger(__name__)

class EmbedderService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbedderService, cls).__new__(cls)
            cls._instance._init_services()
        return cls._instance
    
    def _init_services(self):
        """Initialize models"""
        # Initialize CLIP model for embeddings
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model.to(self.device)
            logger.info(f"CLIP model loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            self.clip_model = None
            self.clip_processor = None
    
    def get_image_generator(self, model_type: ImageGenerationModel) -> ImageGenerator:
        """Get an image generator based on the model type"""
        return ModelFactory.create_image_generator(model_type)
    
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
            return [np.zeros(512) for _ in images]  # Return dummy embeddings
        
        embeddings = []
        batch_size = 16  # Process images in batches to avoid OOM
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
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
                dummy_embedding = np.zeros(512)
                embeddings.extend([dummy_embedding] * len(processed_batch))
        
        return embeddings
    
    async def generate_frame_descriptions(self, query: str, max_frames: int = 5, session_id: str = "") -> List[str]:
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
            return [f"Frame description {i+1} for {query}" for i in range(max_frames)]
        
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            system_prompt = (
                "You are an expert at creating concise, visual descriptions for frames that will be used "
                "to find matching videos. Focus only on key visual elements directly related to the query. "
                "Do not add unnecessary details or interpret beyond what's explicitly in the query."
            )
            
            user_prompt = (
                f"Convert the following query into a sequence of {max_frames} or fewer visual frames. "
                f"Each frame should capture essential visual elements only. "
                f"Be precise and focus solely on elements explicitly mentioned in the query. "
                f"Avoid adding invented details that aren't directly implied by the query.\n\n"
                f"Query: {query}\n\n"
                f"Format your response as a numbered list of frame descriptions, with one description per frame."
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
            return [f"Frame description {i+1} for {query}" for i in range(max_frames)]
    
    async def generate_images_from_descriptions(
        self, 
        descriptions: List[str], 
        mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT,
        model_type: ImageGenerationModel = ImageGenerationModel.STABLE_DIFFUSION,
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
            DebugLogger.save_generated_frames(session_id, images, descriptions, model_type.value)
            
        return images
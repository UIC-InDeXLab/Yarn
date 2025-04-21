import logging
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import io
import os
from openai import OpenAI
import torch
from transformers import CLIPProcessor, CLIPModel

from app.utils.config import DEBUG_MODE
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
        """Initialize the embedding model and OpenAI client"""
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
        
        # Initialize OpenAI client
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.openai_client = None
    
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
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return [f"Frame description {i+1} for {query}" for i in range(max_frames)]
        
        try:
            system_prompt = (
                "You are an expert storyboard artist. Your task is to convert a text query into a "
                "sequence of visual frames that capture the essence of the query. "
                "Each frame should be described in detail so it can be visualized clearly."
            )
            
            user_prompt = (
                f"Convert the following query into a sequence of {max_frames} or fewer visual frames "
                f"that best represent this concept. For each frame, provide a detailed description that "
                f"could be used to generate an image. Focus on visual elements, composition, lighting, "
                f"and mood. The descriptions should be cohesive and flow naturally from one to the next.\n\n"
                f"Query: {query}\n\n"
                f"Format your response as a numbered list of frame descriptions, with one description per frame. "
                f"Do not include any explanations or notes - just the numbered frames and their descriptions."
            )
            
            response = self.openai_client.chat.completions.create(
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
    
    async def generate_images_from_descriptions(self, descriptions: List[str], session_id: str = "") -> List[np.ndarray]:
        """
        Generate images from frame descriptions using DALL-E
        
        Args:
            descriptions: List of frame descriptions
            session_id: Debug session ID (for logging)
            
        Returns:
            List of generated images as numpy arrays
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            # Return dummy images
            return [np.zeros((512, 512, 3), dtype=np.uint8) for _ in descriptions]
        
        generated_images = []
        
        for desc in descriptions:
            try:
                response = self.openai_client.images.generate(
                    model="dall-e-3",
                    prompt=desc,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                    response_format="b64_json"
                )
                
                # Decode base64 image
                import base64
                image_data = base64.b64decode(response.data[0].b64_json)
                
                # Convert to PIL Image and then to numpy array
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_array = np.array(image)
                
                generated_images.append(image_array)
                logger.info(f"Generated image for description: {desc[:50]}...")
                
            except Exception as e:
                logger.error(f"Error generating image for description: {desc[:50]}...: {str(e)}")
                # Add a dummy image
                generated_images.append(np.zeros((1024, 1024, 3), dtype=np.uint8))
        
        # Save generated images in debug mode
        if DEBUG_MODE and session_id and descriptions:
            DebugLogger.save_generated_frames(session_id, generated_images, descriptions)
            
        return generated_images
import datetime
import json
import logging
import os
import time
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from app.utils.config import DEBUG_MODE, DEBUG_LOG_DIR, DEBUG_FRAMES_DIR

logger = logging.getLogger(__name__)


class DebugLogger:
    """Debug logger for saving query details and generated frames"""

    @staticmethod
    def get_session_dir(session_id: str) -> str:
        """Get the directory for a debug session"""
        return os.path.join(DEBUG_LOG_DIR, session_id)

    @staticmethod
    def get_frames_dir(session_id: str) -> str:
        """Get the frames directory for a debug session"""
        return os.path.join(DEBUG_FRAMES_DIR, session_id)

    @staticmethod
    def log_search_query(
            query: str,
            max_frames: int,
            min_frames: int,
            top_k: int,
            frame_mode: str = "independent",
            image_model: str = "sd"
    ) -> str:
        """
        Log a search query and return a unique session ID
        
        Args:
            query: The text query
            max_frames: Maximum number of frames
            top_k: Number of results to return
            frame_mode: Frame generation mode
            image_model: Image generation model
            
        Returns:
            Session ID (timestamp)
        """
        if not DEBUG_MODE:
            return ""

        # Create a unique session ID based on timestamp
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create session directory
        session_dir = DebugLogger.get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save query info
        query_info = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "query": query,
            "max_frames": max_frames,
            "min_frames": min_frames,
            "top_k": top_k,
            "frame_mode": frame_mode,
            "image_model": image_model
        }

        with open(os.path.join(session_dir, "query.json"), 'w') as f:
            json.dump(query_info, f, indent=2)

        logger.info(f"Debug: Logged query to {session_dir}/query.json")
        return session_id

    @staticmethod
    def log_frame_descriptions(session_id: str, descriptions: List[str]) -> None:
        """
        Log generated frame descriptions
        
        Args:
            session_id: The session ID
            descriptions: List of frame descriptions
        """
        if not DEBUG_MODE or not session_id:
            return

        session_dir = DebugLogger.get_session_dir(session_id)

        # Save descriptions
        descriptions_data = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "count": len(descriptions),
            "descriptions": descriptions
        }

        with open(os.path.join(session_dir, "descriptions.json"), 'w') as f:
            json.dump(descriptions_data, f, indent=2)

        logger.info(f"Debug: Logged {len(descriptions)} frame descriptions to {session_dir}/descriptions.json")

    @staticmethod
    def save_generated_frames(
            session_id: str,
            images: List[np.ndarray],
            descriptions: List[str],
            model_name: str = "unknown"
    ) -> None:
        """
        Save generated frames as images
        
        Args:
            session_id: The session ID
            images: List of generated images as numpy arrays
            descriptions: List of corresponding descriptions
            model_name: Name of the model used for generation
        """
        if not DEBUG_MODE or not session_id:
            return

        # Create frames directory
        frames_dir = DebugLogger.get_frames_dir(session_id)
        os.makedirs(frames_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "count": len(images),
            "model": model_name,
            "files": []
        }

        # Save each image
        for i, (img_array, desc) in enumerate(zip(images, descriptions)):
            # Create a truncated description for filename (first 30 chars)
            desc_short = "".join(c if c.isalnum() else "_" for c in desc[:30])
            filename = f"frame_{i + 1:02d}_{model_name}_{desc_short}.jpg"
            file_path = os.path.join(frames_dir, filename)

            # Convert numpy array to PIL Image and save
            try:
                img = Image.fromarray(img_array)
                img.save(file_path)
                metadata["files"].append({
                    "index": i,
                    "filename": filename,
                    "description": desc,
                    "model": model_name
                })
            except Exception as e:
                logger.error(f"Debug: Failed to save frame image: {str(e)}")

        # Save metadata
        with open(os.path.join(frames_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Debug: Saved {len(images)} generated frames to {frames_dir}")

    @staticmethod
    def log_results(session_id: str, results: List[str]) -> None:
        """
        Log search results
        
        Args:
            session_id: The session ID
            results: List of search results
        """
        if not DEBUG_MODE or not session_id:
            return

        session_dir = DebugLogger.get_session_dir(session_id)

        # Save results
        results_data = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "count": len(results),
            "results": results
        }

        with open(os.path.join(session_dir, "results.json"), 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Debug: Logged {len(results)} search results to {session_dir}/results.json")

import os
import glob
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from app.services.embedder_service import EmbedderService
from app.models.video import Video, VideoFrame
from app.utils.config import DEBUG_MODE
from app.utils.debug import DebugLogger

logger = logging.getLogger(__name__)

class VideoService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoService, cls).__new__(cls)
            cls._instance.videos = {}
            cls._instance.embedder = EmbedderService()
            cls._instance.indexed = False
        return cls._instance
    
    async def index_videos(self, video_directory: str) -> None:
        """
        Index all videos in the specified directory by extracting key frames 
        and generating embeddings
        """
        if self.indexed:
            logger.info("Videos are already indexed")
            return
            
        logger.info(f"Indexing videos from {video_directory}")
        
        # Get all video files
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_directory, ext)))
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Process each video
        for video_path in video_files:
            try:
                video_id = os.path.basename(video_path)
                frames = await self._extract_key_frames(video_path)
                
                if frames:
                    # Get embeddings for all frames
                    frame_embeddings = await self.embedder.embed_images([frame.image for frame in frames])
                    
                    # Assign embeddings to frames
                    for i, frame in enumerate(frames):
                        frame.embedding = frame_embeddings[i]
                    
                    # Store the video
                    self.videos[video_id] = Video(
                        id=video_id,
                        path=video_path,
                        frames=frames
                    )
                    
                    logger.info(f"Indexed video: {video_id} with {len(frames)} frames")
                else:
                    logger.warning(f"No frames extracted from video: {video_path}")
                    
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {str(e)}")
        
        self.indexed = True
        logger.info(f"Finished indexing {len(self.videos)} videos")
    
    async def _extract_key_frames(self, video_path: str, max_frames: int = 30) -> List[VideoFrame]:
        """
        Extract key frames from a video using scene change detection
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # If video is very short, just extract evenly spaced frames
            if total_frames < max_frames * 2:
                frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            else:
                # Use scene detection for longer videos
                prev_frame = None
                frame_diffs = []
                frame_indices = []
                
                # Sample frames to calculate differences
                sample_rate = max(1, total_frames // (max_frames * 10))
                
                for i in range(0, total_frames, sample_rate):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    if prev_frame is not None:
                        diff = cv2.absdiff(prev_frame, gray)
                        diff_score = np.sum(diff)
                        frame_diffs.append((i, diff_score))
                    
                    prev_frame = gray
                
                # Sort by difference score and get top frames
                if frame_diffs:
                    sorted_diffs = sorted(frame_diffs, key=lambda x: x[1], reverse=True)
                    frame_indices = [idx for idx, _ in sorted_diffs[:max_frames]]
                    frame_indices.sort()  # Sort chronologically
            
            # Extract the selected frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB for compatibility with embedder
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = idx / fps if fps > 0 else 0
                    
                    frames.append(VideoFrame(
                        timestamp=timestamp,
                        image=rgb_frame,
                        embedding=None
                    ))
            
            cap.release()
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
        
        return frames
    
    async def search_videos(self, query: str, max_frames: int = 5, top_k: int = 3) -> List[Dict]:
        """
        Search for videos matching the query
        """
        if not self.indexed or not self.videos:
            logger.warning("No videos indexed, cannot perform search")
            return []
        
        # Create debug session if debug mode is enabled
        session_id = ""
        if DEBUG_MODE:
            session_id = DebugLogger.log_search_query(query, max_frames, top_k)
        
        # Generate frame descriptions from the query
        frame_descriptions = await self.embedder.generate_frame_descriptions(query, max_frames, session_id)
        
        # Generate images from frame descriptions
        generated_images = await self.embedder.generate_images_from_descriptions(frame_descriptions, session_id)
        
        # Get embeddings for generated images
        generated_embeddings = await self.embedder.embed_images(generated_images)
        
        # Find the best matching videos using DTW
        results = []
        
        for video_id, video in self.videos.items():
            if not video.frames:
                continue
                
            # Get the embeddings sequence for this video
            video_embeddings = [frame.embedding for frame in video.frames if frame.embedding is not None]
            
            if not video_embeddings:
                continue
            
            # Calculate DTW distance
            distance, _ = fastdtw(np.array(generated_embeddings), np.array(video_embeddings), dist=euclidean)
            
            # Normalize by the number of frames
            normalized_distance = distance / (len(video_embeddings) + len(generated_embeddings))
            
            results.append({
                "video_id": video_id,
                "distance": normalized_distance,
                "score": 1.0 / (1.0 + normalized_distance)  # Convert distance to similarity score
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top-k results
        top_results = results[:top_k]
        
        # Log results in debug mode
        if DEBUG_MODE and session_id:
            DebugLogger.log_results(session_id, top_results)
        
        return top_results
import glob
import logging
import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from app.models.video import Video, VideoFrame, FrameGenerationMode, ImageGenerationModel, EmbeddingModel, EmbedderConfig, aggregate_rankings
from app.services.embedder_service import EmbedderService
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
                    # Get embeddings for all frames using all available embedders
                    embedder_configs = [
                        EmbedderConfig(model=EmbeddingModel.CLIP, weight=1.0),
                        EmbedderConfig(model=EmbeddingModel.DINO, weight=1.0)
                    ]
                    
                    embeddings_dict = await self.embedder.embed_images_with_multiple_models(
                        [frame.image for frame in frames],
                        embedder_configs
                    )

                    # Assign embeddings to frames
                    for i, frame in enumerate(frames):
                        for embedder_name, embeddings in embeddings_dict.items():
                            if i < len(embeddings):
                                frame.embeddings[embedder_name] = embeddings[i]

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
                        image=rgb_frame
                    ))

            cap.release()
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")

        return frames

    async def search_videos(
            self,
            query: str,
            max_frames: int = 5,
            top_k: int = 3,
            frame_mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT,
            image_model: ImageGenerationModel = ImageGenerationModel.STABLE_DIFFUSION,
            embedding_models: List[EmbedderConfig] = None
    ) -> List[Dict]:
        """
        Search for videos matching the query
        
        Args:
            query: The text query to search for
            max_frames: Maximum number of frames to generate
            top_k: Number of results to return
            frame_mode: Mode for frame generation (independent or continuous)
            image_model: Model for image generation
            embedding_models: List of embedder configurations to use
            
        Returns:
            List of matching videos with scores
        """
        if not self.indexed or not self.videos:
            logger.warning("No videos indexed, cannot perform search")
            return []
            
        # Use default embedder if none specified
        if embedding_models is None or len(embedding_models) == 0:
            embedding_models = [EmbedderConfig(model=EmbeddingModel.CLIP, weight=1.0)]

        # Create debug session if debug mode is enabled
        session_id = ""
        if DEBUG_MODE:
            session_id = DebugLogger.log_search_query(
                query,
                max_frames,
                top_k,
                frame_mode=frame_mode.value,
                image_model=image_model.value
            )

        # Generate frame descriptions from the query
        frame_descriptions = await self.embedder.generate_frame_descriptions(query, max_frames, session_id)

        # Generate images from frame descriptions using the specified model and mode
        generated_images = await self.embedder.generate_images_from_descriptions(
            frame_descriptions,
            mode=frame_mode,
            model_type=image_model,
            session_id=session_id
        )

        # Get embeddings for generated images using all specified embedders
        generated_embeddings_dict = await self.embedder.embed_images_with_multiple_models(
            generated_images,
            embedding_models
        )

        # Find the best matching videos using each embedder
        all_ranker_results = []
        weights = []
        
        for config in embedding_models:
            embedder_name = config.model.value
            weight = config.weight
            
            if embedder_name not in generated_embeddings_dict:
                logger.warning(f"No embeddings generated for {embedder_name}, skipping")
                continue
                
            generated_embeddings = generated_embeddings_dict[embedder_name]
            
            # Run search with this embedder
            results = await self._search_with_embedder(
                video_frames_dict=self.videos,
                generated_embeddings=generated_embeddings,
                embedder_name=embedder_name
            )
            
            all_ranker_results.append(results)
            weights.append(weight)

        # If we have multiple rankers, aggregate their results
        if len(all_ranker_results) > 1:
            aggregated_results = aggregate_rankings(all_ranker_results, weights, top_k)
            
            # Log results in debug mode
            if DEBUG_MODE and session_id:
                DebugLogger.log_results(session_id, aggregated_results)
                
            return aggregated_results
        elif len(all_ranker_results) == 1:
            # Just return the results from the single embedder
            results = all_ranker_results[0]
            top_results = results[:top_k]
            
            # Log results in debug mode
            if DEBUG_MODE and session_id:
                DebugLogger.log_results(session_id, top_results)
                
            return top_results
        else:
            logger.warning("No valid embedders found for search")
            return []
            
    async def _search_with_embedder(
            self,
            video_frames_dict: Dict[str, Video],
            generated_embeddings: List[np.ndarray],
            embedder_name: str
    ) -> List[Dict]:
        """
        Search for videos using a specific embedder
        
        Args:
            video_frames_dict: Dictionary of videos to search
            generated_embeddings: Embeddings of the generated frames
            embedder_name: Name of the embedder to use
            
        Returns:
            List of video results sorted by similarity score
        """
        results = []

        for video_id, video in video_frames_dict.items():
            # Get the embeddings sequence for this video using the specified embedder
            video_embeddings = [frame.embeddings.get(embedder_name) for frame in video.frames 
                               if embedder_name in frame.embeddings]
                    
            if not video_embeddings:
                continue

            # Calculate DTW distance
            try:
                distance, _ = fastdtw(np.array(generated_embeddings), np.array(video_embeddings), dist=euclidean)

                # Normalize by the number of frames
                normalized_distance = distance / (len(video_embeddings) + len(generated_embeddings))

                results.append({
                    "video_id": video_id,
                    "distance": normalized_distance,
                    "score": 1.0 / (1.0 + normalized_distance),  # Convert distance to similarity score
                    "embedder": embedder_name
                })
            except Exception as e:
                logger.error(f"Error calculating DTW for video {video_id}: {str(e)}")

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results

import glob
import logging
import os
from typing import List, Dict

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from app.models.video import Video, VideoFrame, FrameGenerationMode, ImageGenerationModel, EmbeddingModel, \
    EmbedderConfig, aggregate_rankings
from app.services.distance_service import contextual_distance, velocity_aware_dtw
from app.services.embedder_service import EmbedderService
from app.services.video_generator import get_video_generator
from app.utils.config import DEBUG_MODE
from app.utils.debug import DebugLogger


logger = logging.getLogger(__name__)


class VideoService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoService, cls).__new__(cls)
            # In-memory store of videos and embeddings
            cls._instance.videos = {}
            # In-memory store for search results
            cls._instance.search_results = {}
            # Embedder service for indexing new content
            cls._instance.embedder = EmbedderService()
            # Directory for persistent index cache
            import os
            cls._instance.index_dir = os.getenv("VIDEO_INDEX_DIR", "./video_index_old")
            os.makedirs(cls._instance.index_dir, exist_ok=True)
            # Load existing index cache if available
            cls._instance.indexed = False
            cls._instance._load_index()
        return cls._instance

    def _load_index(self) -> None:
        """
        Load video embeddings and frame metadata from disk cache
        """
        import pickle
        try:
            loaded = 0
            for fname in os.listdir(self.index_dir):
                if not fname.endswith('.pkl'):
                    continue
                path = os.path.join(self.index_dir, fname)
                with open(path, 'rb') as pf:
                    data = pickle.load(pf)
                vid = data.get('id') or os.path.splitext(fname)[0]
                vpath = data.get('path', '')
                timestamps = data.get('timestamps', [])
                embeddings_dict = data.get('embeddings', {})
                # Reconstruct VideoFrame list
                frames = []
                for idx, ts in enumerate(timestamps):
                    frame = VideoFrame(timestamp=ts, image=None)
                    frame.embeddings = {}
                    for name, arr in embeddings_dict.items():
                        # arr is ndarray of shape (n_frames, dim)
                        frame.embeddings[name] = arr[idx]
                    frames.append(frame)
                # Recreate Video
                self.videos[vid] = Video(id=vid, path=vpath, frames=frames)
                loaded += 1
            if loaded:
                self.indexed = True
                logger.info(f"Loaded {loaded} videos from index cache")
        except Exception as e:
            logger.error(f"Error loading video index cache: {e}")

    def _save_index(self) -> None:
        """
        Save video embeddings and frame metadata to disk cache
        """
        import pickle
        try:
            for vid, video in self.videos.items():
                # Prepare metadata
                embeddings_dict = {}
                if video.frames and video.frames[0].embeddings:
                    for name in video.frames[0].embeddings.keys():
                        # Stack embeddings for each frame
                        arr = np.stack([frame.embeddings.get(name) for frame in video.frames], axis=0)
                        embeddings_dict[name] = arr
                data = {
                    'id': video.id,
                    'path': video.path,
                    'embeddings': embeddings_dict
                }
                cache_path = os.path.join(self.index_dir, f"{vid}.pkl")
                with open(cache_path, 'wb') as pf:
                    pickle.dump(data, pf)
            logger.info(f"Saved {len(self.videos)} videos to index cache")
        except Exception as e:
            logger.error(f"Error saving video index cache: {e}")

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
                # frames = self._extract_frames(video_path)

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
        # Persist index cache to disk
        self._save_index()

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

    def _extract_frames(self, video_path, frame_rate=1.0, size=224):
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps < 1:
            images = np.zeros([3, size, size], dtype=np.float32)
            logger.error("Problem reading video file: ", video_path)
        else:
            total_duration = (frameCount + fps - 1) // fps
            start_sec, end_sec = 0, total_duration
            interval = fps / frame_rate
            frames_idx = np.floor(np.arange(start_sec * fps, end_sec * fps, interval))
            images = []

            for i, idx in enumerate(frames_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)

        cap.release()

        # video_frames = torch.tensor(images)
        return [VideoFrame(timestamp=i * fps, image=img) for i, img in enumerate(images)]

    def get_results_by_session_id(self, session_id: str) -> Dict:
        """Get search results by session ID"""
        return self.search_results.get(session_id)

    def get_video_by_id(self, video_id: str) -> Video:
        """Get a video by its ID"""
        return self.videos.get(video_id)

    async def search_videos(
            self,
            query: str,
            max_frames: int = 5,
            top_k: int = 3,
            frame_mode: FrameGenerationMode = FrameGenerationMode.INDEPENDENT,
            image_model: ImageGenerationModel = ImageGenerationModel.DEFAULT,
            embedding_models: List[EmbedderConfig] = None,
            num_frames: int = None,
            video_model_id: str = None,
            lambda_param: float = 0.5
    ) -> Dict:
        """
        Search for videos matching the query

        Args:
            query: The text query to search for
            max_frames: Maximum number of frames to generate for image models
            num_frames: Number of frames to generate for video models (between 5 and 15)
            top_k: Number of results to return
            frame_mode: Mode for frame generation (independent or continuous; ignored for video)
            image_model: Generation model to use
            video_model_id: Hugging Face model ID for video generation
            embedding_models: List of embedder configurations to use
            lambda_param: Weight for combining contextual and action distances

        Returns:
            Dictionary with results and a preview URL
        """
        if not self.indexed or not self.videos:
            logger.warning("No videos indexed, cannot perform search")
            return {"results": [], "preview_url": ""}

        # Use default embedder if none specified
        if embedding_models is None or len(embedding_models) == 0:
            embedding_models = [EmbedderConfig(model=EmbeddingModel.CLIP, weight=1.0)]

        # Create debug session if debug mode is enabled
        session_id = ""
        if DEBUG_MODE:
            # Use num_frames in logs for video models, otherwise use max_frames
            log_min = num_frames if image_model == ImageGenerationModel.VIDEO else max_frames
            session_id = DebugLogger.log_search_query(
                query,
                max_frames,
                log_min,
                top_k,
                frame_mode=frame_mode.value,
                image_model=image_model.value
            )

        # Generate synthetic frames: separate logic for image vs video models
        if image_model == ImageGenerationModel.VIDEO:
            n = num_frames or max_frames
            model_id = video_model_id or None
            video_gen = get_video_generator(model_id=model_id)
            generated_images = await video_gen.generate(query, num_frames=n)
        else:
            # Generate frame descriptions for image models
            frame_descriptions = await self.embedder.generate_frame_descriptions(
                query,
                max_frames=max_frames,
                min_frames=3,
                session_id=session_id
            )
            # Generate images from frame descriptions
            generated_images = await self.embedder.generate_images_from_descriptions(
                frame_descriptions,
                mode=frame_mode,
                model_type=image_model,
                session_id=session_id
            )

        if DEBUG_MODE and session_id and generated_images:
            DebugLogger.save_generated_frames(session_id, generated_images)

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
                embedder_name=embedder_name,
                lambda_param=lambda_param
            )

            all_ranker_results.append(results)
            weights.append(weight)

        # If we have multiple rankers, aggregate their results
        aggregated_results = aggregate_rankings(all_ranker_results, weights, top_k)

        # Store results for preview
        if session_id:
            self.search_results[session_id] = {"query": query, "results": aggregated_results}

        # Log results in debug mode
        if DEBUG_MODE and session_id:
            DebugLogger.log_results(session_id, aggregated_results)

        return {
            "results": aggregated_results,
            "preview_url": f"/api/preview/{session_id}" if session_id else ""
        }

    async def _search_with_embedder(
            self,
            video_frames_dict: Dict[str, Video],
            generated_embeddings: np.ndarray,
            embedder_name: str,
            lambda_param: float = 0.5,
            beta: float = 1.0
    ) -> List[Dict]:
        """
        Search for videos using a specific embedder and combined distance metric.

        Args:
            video_frames_dict: Dictionary of videos to search.
            generated_embeddings: Embeddings of the generated frames.
            embedder_name: Name of the embedder to use.
            lambda_param: Weight for combining contextual and action distances.
            beta: Weight for the velocity component in action distance.

        Returns:
            List of video results sorted by similarity score.
        """
        results = []
        metric = 'cosine' if embedder_name == EmbeddingModel.CLIP.value else 'sqeuclidean'

        # Normalize generated_embeddings to be a numpy array if it's a list
        if isinstance(generated_embeddings, list):
            generated_embeddings = np.array(generated_embeddings)

        for video_id, video in video_frames_dict.items():
            video_embeddings = [frame.embeddings.get(embedder_name) for frame in video.frames
                                if embedder_name in frame.embeddings]

            if not video_embeddings:
                continue

            video_embeddings = np.array(video_embeddings)

            try:
                # Calculate contextual distance
                d_ctx = contextual_distance(generated_embeddings, video_embeddings, metric=metric)

                # Calculate action distance
                d_act = velocity_aware_dtw(generated_embeddings, video_embeddings, beta=beta, metric=metric)

                # Normalize distances (simple normalization for now)
                # A proper normalization would require statistics from a validation set
                d_ctx_norm = d_ctx / (len(generated_embeddings) * len(video_embeddings))
                d_act_norm = d_act / (len(generated_embeddings) * len(video_embeddings))

                # Combined distance
                combined_distance = (1 - lambda_param) * d_ctx_norm + lambda_param * d_act_norm

                results.append({
                    "video_id": video_id,
                    "distance": combined_distance,
                    "score": 1.0 / (1.0 + combined_distance),
                    "embedder": embedder_name
                })

            except Exception as e:
                logger.error(f"Error calculating distance for video {video_id}: {str(e)}")

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

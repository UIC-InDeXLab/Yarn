from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.models.video import SearchQuery, ImageGenerationModel, SearchResponse
from app.services.video_service import VideoService

router = APIRouter(
    prefix="/api/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


async def get_video_service():
    """Dependency to get the video service instance"""
    return VideoService()


@router.post("/", response_model=SearchResponse)
async def search_videos(
        query: SearchQuery,
        video_service: VideoService = Depends(get_video_service)
):
    """
    Search for videos matching the query text
    
    Args:
        query: Search query with optional parameters:
            - max_frames: Maximum number of frames to generate (default: 5)
            - top_k: Number of results to return (default: 3)
            - frame_mode: Frame generation mode (independent or continuous)
            - image_model: Image generation model (default)
            - embedding_models: List of embedding models with weights
        video_service: Video service instance
        
    Returns:
        Dictionary with a list of matching videos and a preview URL
    """
    if not query.query or len(query.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    # Validate video-specific parameters
    if query.image_model == ImageGenerationModel.VIDEO:
        if query.num_frames is None:
            raise HTTPException(status_code=400, detail="num_frames must be specified for video generation")

    results = await video_service.search_videos(
        query=query.query,
        max_frames=query.max_frames,
        top_k=query.top_k,
        frame_mode=query.frame_mode,
        image_model=query.image_model,
        embedding_models=query.embedding_models,
        num_frames=query.num_frames,
        video_model_id=query.video_model_id,
        lambda_param=query.lambda_param
    )

    return results

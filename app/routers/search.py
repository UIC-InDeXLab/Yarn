from fastapi import APIRouter, Depends, HTTPException
from typing import List

from app.models.video import SearchQuery, SearchResult
from app.services.video_service import VideoService

router = APIRouter(
    prefix="/api/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)

async def get_video_service():
    """Dependency to get the video service instance"""
    return VideoService()

@router.post("/", response_model=List[SearchResult])
async def search_videos(
    query: SearchQuery,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Search for videos matching the query text
    
    Args:
        query: Search query with optional parameters max_frames and top_k
        video_service: Video service instance
        
    Returns:
        List of matching videos with similarity scores
    """
    if not query.query or len(query.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    if query.max_frames < 1 or query.max_frames > 20:
        raise HTTPException(status_code=400, detail="max_frames must be between 1 and 20")
        
    if query.top_k < 1 or query.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    
    results = await video_service.search_videos(
        query.query, 
        max_frames=query.max_frames,
        top_k=query.top_k
    )
    
    return results
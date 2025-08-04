from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
import os
from app.services.video_service import VideoService
from app.utils.video_utils import convert_to_mp4

router = APIRouter(
    prefix="/api/preview",
    tags=["preview"],
    responses={404: {"description": "Not found"}},
)

templates = Jinja2Templates(directory="app/templates")


async def get_video_service():
    """Dependency to get the video service instance"""
    return VideoService()


@router.get("/{session_id}")
async def get_preview(
    request: Request,
    session_id: str,
    video_service: VideoService = Depends(get_video_service),
):
    """
    Display the search results in an HTML page.
    """
    results = video_service.get_results_by_session_id(session_id)
    if not results:
        return templates.TemplateResponse("preview.html", {"request": request, "query": "No results found", "results": []})

    query = results["query"]
    videos = []
    for video_id in results["results"]:
        video = video_service.get_video_by_id(video_id)
        if video:
            mp4_path = convert_to_mp4(video.path)
            if mp4_path:
                video_filename = os.path.basename(mp4_path)
                videos.append({"video_id": video_id, "video_path": f"/videos/{video_filename}"})

    return templates.TemplateResponse("preview.html", {"request": request, "query": query, "results": videos})

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def convert_to_mp4(video_path: str) -> str:
    """
    Converts a video to MP4 format if it's not already an MP4.
    Returns the path to the MP4 video.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    video_dir, video_filename = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_filename)

    if video_ext.lower() == ".mp4":
        return video_path

    mp4_filename = f"{video_name}.mp4"
    mp4_path = os.path.join(video_dir, mp4_filename)

    if os.path.exists(mp4_path):
        return mp4_path

    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", mp4_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully converted {video_path} to {mp4_path}")
        return mp4_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {video_path} to MP4: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please ensure it is installed and in your PATH.")
        return None

#!/usr/bin/env python3
"""
Download script for MSR-VTT dataset
"""
import os
import json
import csv
import logging
import argparse
import urllib.request
import zipfile
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_msrvtt(output_dir, video_dir, annotations_only=False):
    """Download the MSR-VTT dataset
    
    Args:
        output_dir: Directory to save the annotations
        video_dir: Directory to save the videos
        annotations_only: If True, only download annotations, not videos
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    # Download annotation data
    annotation_url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/msrvtt_data.zip"
    annotation_zip = os.path.join(output_dir, "msrvtt_data.zip")
    
    if not os.path.exists(annotation_zip):
        logger.info(f"Downloading annotations from {annotation_url}")
        download_url(annotation_url, annotation_zip)
    
    # Extract annotations
    if not os.path.exists(os.path.join(output_dir, "train_val_annotation")):  
        logger.info(f"Extracting annotations from {annotation_zip}")
        with zipfile.ZipFile(annotation_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    # Process annotations
    process_annotations(output_dir, video_dir)
    
    # Download videos if requested
    if not annotations_only:
        download_videos(output_dir, video_dir)

def process_annotations(annotation_dir, video_dir):
    """Process the MSR-VTT annotations to create a CSV file"""
    # Load the annotations
    train_val_path = os.path.join(annotation_dir, "train_val_annotation", "train_val_videodatainfo.json")
    test_path = os.path.join(annotation_dir, "test_videodatainfo.json")
    
    videos = []
    
    # Process train/val annotations
    if os.path.exists(train_val_path):
        with open(train_val_path, 'r') as f:
            data = json.load(f)
        
        # Extract video info
        for video in data.get('videos', []):
            video_id = video.get('video_id')
            if not video_id:
                continue
                
            category = video.get('category', '')
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            videos.append({
                'video_id': video_id,
                'category': category,
                'path': video_path,
                'descriptions': []
            })
        
        # Extract descriptions
        video_dict = {v['video_id']: v for v in videos}
        
        for sentence in data.get('sentences', []):
            video_id = sentence.get('video_id')
            if video_id in video_dict:
                video_dict[video_id]['descriptions'].append(sentence.get('caption', ''))
    
    # Process test annotations if available
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            data = json.load(f)
        
        for video in data.get('videos', []):
            video_id = video.get('video_id')
            if not video_id:
                continue
                
            category = video.get('category', '')
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            if not any(v['video_id'] == video_id for v in videos):
                videos.append({
                    'video_id': video_id,
                    'category': category,
                    'path': video_path,
                    'descriptions': []
                })
        
        video_dict = {v['video_id']: v for v in videos}
        
        for sentence in data.get('sentences', []):
            video_id = sentence.get('video_id')
            if video_id in video_dict:
                video_dict[video_id]['descriptions'].append(sentence.get('caption', ''))
    
    # Write to CSV
    output_csv = os.path.join(annotation_dir, "msrvtt_videos.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'category', 'path', 'description'])
        
        for video in videos:
            # Join all descriptions with a pipe
            description = ' | '.join(video['descriptions'])
            writer.writerow([video['video_id'], video['category'], video['path'], description])
    
    logger.info(f"Created CSV file with {len(videos)} videos at {output_csv}")
    return output_csv

def download_videos(annotation_dir, video_dir):
    """Download the MSR-VTT videos"""
    # For MSR-VTT, we would normally download the videos from the official source
    # Since the original download might require permissions, this is a placeholder
    # In a real implementation, you would handle the actual video download here
    
    # For example, you might download from the official link:
    video_url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/MSRVTT_videos.zip"
    video_zip = os.path.join(annotation_dir, "MSRVTT_videos.zip")
    
    # Note: The actual MSR-VTT download might be behind a form or require permissions
    # This is simplified for illustration purposes
    try:
        if not os.path.exists(video_zip):
            logger.info(f"Downloading videos from {video_url}")
            download_url(video_url, video_zip)
        
        # Extract videos
        logger.info(f"Extracting videos from {video_zip}")
        with zipfile.ZipFile(video_zip, 'r') as zip_ref:
            zip_ref.extractall(video_dir)
            
        logger.info(f"Videos extracted to {video_dir}")
    except Exception as e:
        logger.error(f"Failed to download or extract videos: {str(e)}")
        logger.info("You may need to manually download the videos from the official source")
        logger.info("https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/")

def main():
    parser = argparse.ArgumentParser(description='Download MSR-VTT dataset')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory for annotations')
    parser.add_argument('--video-dir', type=str, default='./videos', help='Output directory for videos')
    parser.add_argument('--annotations-only', action='store_true', help='Only download annotations, not videos')
    
    args = parser.parse_args()
    
    download_msrvtt(
        output_dir=args.output_dir,
        video_dir=args.video_dir,
        annotations_only=args.annotations_only
    )

if __name__ == "__main__":
    main()

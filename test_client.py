#!/usr/bin/env python3
"""
Test client for Yarn API
"""
import os
import json
import argparse
import requests
from pprint import pprint

def search_videos(query, max_frames=5, top_k=3, api_url="http://localhost:8000"):
    """
    Search for videos using the Yarn API
    
    Args:
        query: Text query to search for
        max_frames: Maximum number of frames to generate
        top_k: Number of results to return
        api_url: Base URL of the Yarn API
        
    Returns:
        List of search results
    """
    endpoint = f"{api_url}/api/search/"
    
    payload = {
        "query": query,
        "max_frames": max_frames,
        "top_k": top_k
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(endpoint, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def main():
    parser = argparse.ArgumentParser(description="Test client for Yarn API")
    parser.add_argument("--query", type=str, required=True, help="Text query to search for")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum number of frames to generate")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="Base URL of the Yarn API")
    parser.add_argument("--output", type=str, help="Output file to save results (JSON format)")
    
    args = parser.parse_args()
    
    print(f"Searching for: {args.query}")
    results = search_videos(
        query=args.query,
        max_frames=args.max_frames,
        top_k=args.top_k,
        api_url=args.api_url
    )
    
    if results:
        print("\nSearch Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Video: {result['video_id']} (Score: {result['score']:.4f})")
        
        # Save to output file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        print("No results found")

if __name__ == "__main__":
    main()
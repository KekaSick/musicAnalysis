import os
import pandas as pd
import logging
import subprocess
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    """Remove invalid characters from filename, and truncate if too long."""
    sanitized = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in filename).rstrip()
    sanitized = sanitized.replace(" ", "_")
    max_len = 100  # Max length for a filename component
    return sanitized[:max_len]

def download_track(track_info, output_dir):
    """Download a track using yt-dlp."""
    try:
        # Create a sanitized filename with cluster information
        filename = f"{track_info['algorithm']}_{track_info['metric_type']}_{track_info['cluster']}_{sanitize_filename(track_info['track_name'])}_{sanitize_filename(track_info['track_genre'])}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return output_path
            
        search_query = f"{track_info['track_name']} {track_info['artists']}"
        logger.info(f"Downloading: {track_info['track_name']} - {track_info['artists']} (Genre: {track_info['track_genre']})")
        
        # Use yt-dlp for downloading
        command = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--output', output_path,
            f'ytsearch1:{search_query}'
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error downloading '{search_query}': {result.stderr}")
            return None
            
        logger.info(f"Successfully downloaded: {filename}")
        return output_path
        
    except Exception as e:
        logger.error(f"Exception while downloading '{track_info['track_name']}': {str(e)}")
        return None

def download_representative_tracks():
    """Download all representative tracks from clusters."""
    # Create output directory
    output_dir = os.path.join("plots", "sampled_dataset_PE_C", "clusters", "representative_tracks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the representative tracks CSV
    csv_path = os.path.join("plots", "sampled_dataset_PE_C", "clusters", "representative_tracks.csv")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} representative tracks from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading representative tracks CSV: {e}")
        return
    
    # Download each track
    successful = 0
    failed = []
    
    for _, track in tqdm(df.iterrows(), total=len(df), desc="Downloading tracks"):
        track_info = track.to_dict()
        result = download_track(track_info, output_dir)
        
        if result:
            successful += 1
        else:
            failed.append(track_info)
    
    # Print summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"Total tracks: {len(df)}")
    logger.info(f"Successfully downloaded: {successful}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.info("\nFailed tracks:")
        for track in failed:
            logger.info(f"  - {track['track_name']} ({track['algorithm']} {track['metric_type']} cluster {track['cluster']})")

if __name__ == "__main__":
    download_representative_tracks() 
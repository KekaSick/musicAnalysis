import os
import subprocess
import time
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import librosa
from scipy.io import wavfile
import antropy as ent
import ordpy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DATASET_PATH = os.path.join("data", "csv_spotify", "csv", "sampled_dataset.csv")
AUDIO_OUTPUT_DIR = os.path.join("data", "music", "audio")
RESULTS_PATH = os.path.join("data", "results", "entropy_complexity.csv")

def sanitize_filename(filename):
    """Remove invalid characters from filename, and truncate if too long."""
    sanitized = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in filename).rstrip()
    sanitized = sanitized.replace(" ", "_")
    max_len = 100  # Max length for a filename component
    return sanitized[:max_len]

def ordpy_process_file(file_path, dim_size=3, hop_size=1):
    # Параметры для фреймирования аудио и STFT
    N_FFT = 2048
    HOP_LENGTH_STFT = 512

    # Загрузка файла
    try:
        data, sr = librosa.load(file_path, sr=None, mono=True)
        logger.info(f"Processing WAV '{file_path}' (sr={sr})")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return np.nan, np.nan

    spectral_entropy_ts = None

    try:
        # Спектральная энтропия (antropy)
        logger.debug(f"Attempting antropy.spectral_entropy for {file_path}")
        audio_frames = librosa.util.frame(data, frame_length=N_FFT, hop_length=HOP_LENGTH_STFT, axis=0)
        spectral_entropy_ts = np.apply_along_axis(
            lambda audio_segment: ent.spectral_entropy(
                audio_segment, 
                sf=sr, 
                method="fft", 
                normalize=True
            ) if np.any(audio_segment) else 0.0,
            axis=0,
            arr=audio_frames
        )
        logger.info(f"Successfully used antropy.spectral_entropy for {file_path}")

    except Exception as e_antropy:
        logger.warning(f"antropy.spectral_entropy failed for {file_path}: {e_antropy}")
        try:
            # Фоллбэк на ручной расчет
            S_complex = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LENGTH_STFT)
            S_mag = np.abs(S_complex)**2
            
            epsilon = np.finfo(float).eps
            prob_dist = S_mag / (np.sum(S_mag, axis=0, keepdims=True) + epsilon)
            prob_dist = np.nan_to_num(prob_dist, nan=0.0, posinf=0.0, neginf=0.0)
            
            entropy_values_bits = -np.sum(prob_dist * np.log2(prob_dist + epsilon), axis=0)
            
            num_freq_bins = S_mag.shape[0]
            if num_freq_bins > 1:
                spectral_entropy_ts = entropy_values_bits / (np.log2(num_freq_bins) + epsilon)
            else:
                spectral_entropy_ts = np.zeros_like(entropy_values_bits)
            
            logger.info(f"Used manual STFT-based calculation for {file_path}")
        except Exception as e_manual:
            logger.error(f"Manual calculation failed for {file_path}: {e_manual}")
            return np.nan, np.nan

    if spectral_entropy_ts is None or len(spectral_entropy_ts) <= dim_size:
        logger.warning(f"Invalid spectral entropy for {file_path}")
        return np.nan, np.nan

    try:
        H_norm, C = ordpy.complexity_entropy(
            spectral_entropy_ts,
            dx=dim_size,
            taux=hop_size
        )
        return H_norm, C
    except Exception as e:
        logger.error(f"ordpy.complexity_entropy failed for {file_path}: {e}")
        return np.nan, np.nan

def download_and_process_track(track_info):
    """Download audio, process it, and clean up."""
    try:
        # Create filename
        track_name = track_info['name']
        artist_name = track_info['artist']
        genre = track_info['genre']
        
        s_genre = sanitize_filename(genre)
        s_track_name = sanitize_filename(track_name)
        s_artist_name = sanitize_filename(artist_name)
        
        output_filename = f"{s_genre}_{s_track_name}_{s_artist_name}.%(ext)s"
        output_template = os.path.join(AUDIO_OUTPUT_DIR, output_filename)
        
        search_query = f"{track_name} {artist_name}"
        
        # Download command
        cmd = [
            'yt-dlp',
            '--format', 'bestaudio/best',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--output', output_template,
            '--default-search', 'ytsearch1:',
            '--no-playlist',
            '--match-filter', '!is_live',
            '--postprocessor-args', 'ffmpeg:-ar 44100',
            '--sleep-interval', '1',
            '--max-sleep-interval', '3',
            search_query
        ]
        
        # Download
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error(f"Failed to download {track_name}: {result.stderr}")
            return None
        
        # Find the downloaded file
        wav_file = None
        for file in os.listdir(AUDIO_OUTPUT_DIR):
            if file.startswith(f"{s_genre}_{s_track_name}_{s_artist_name}") and file.endswith('.wav'):
                wav_file = os.path.join(AUDIO_OUTPUT_DIR, file)
                break
        
        if not wav_file:
            logger.error(f"Cannot find downloaded WAV file for {track_name}")
            return None
        
        # Process the file
        H_norm, C = ordpy_process_file(wav_file)
        
        # Delete the WAV file
        os.remove(wav_file)
        logger.info(f"Deleted {wav_file}")
        
        # Return results
        return {
            'name': track_name,
            'artist': artist_name,
            'genre': genre,
            'entropy': H_norm,
            'complexity': C
        }
        
    except Exception as e:
        logger.error(f"Error processing {track_info.get('name', 'unknown track')}: {e}")
        return None

def main():
    # Create output directories
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    # Read the dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Loaded {len(df)} tracks from dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    results = []
    failed_tracks = []

    # Process tracks with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_track = {
            executor.submit(download_and_process_track, track): track 
            for track in df.to_dict('records')
        }
        
        for future in as_completed(future_to_track):
            track = future_to_track[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"Successfully processed: {result['name']}")
                else:
                    failed_tracks.append(track)
                    logger.warning(f"Failed to process: {track['name']}")
            except Exception as e:
                failed_tracks.append(track)
                logger.error(f"Error processing {track['name']}: {e}")
            
            time.sleep(0.5)  # Be nice to YouTube

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Saved results to {RESULTS_PATH}")

    # Print summary
    logger.info("\n--- Processing Summary ---")
    logger.info(f"Successfully processed: {len(results)} tracks")
    logger.info(f"Failed to process: {len(failed_tracks)} tracks")
    if failed_tracks:
        logger.info("Failed tracks:")
        for track in failed_tracks:
            logger.info(f"  - {track['name']} by {track['artist']} ({track['genre']})")

if __name__ == "__main__":
    main() 
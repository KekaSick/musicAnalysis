import os
import shutil
from pathlib import Path

# Path to the scales directory
scales_dir = Path('data/scales')

# List of all scale folders
scale_folders = [d for d in scales_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

for scale_folder in scale_folders:
    print(f"Processing {scale_folder.name} scale...")
    
    # Get all wav files in the scale folder
    wav_files = [f for f in scale_folder.glob('**/*.wav')]  # Using ** to search in subdirectories too
    
    # Create a set of unique octaves from the filenames
    octaves = set()
    for wav_file in wav_files:
        # Extract octave from filename (format: Note-Octave-Scale-Variation.wav)
        octave = wav_file.name.split('-')[1]
        octaves.add(octave)
    
    # Create folders for each octave
    for octave in octaves:
        octave_folder = scale_folder / f"octave_{octave}"
        octave_folder.mkdir(exist_ok=True)
        
        # Move all files for this octave to its folder
        for wav_file in wav_files:
            if f"-{octave}-" in wav_file.name:
                # Create destination path
                dest = octave_folder / wav_file.name
                # Move the file
                try:
                    shutil.move(str(wav_file), str(dest))
                except shutil.Error:
                    # If file already exists in destination, skip it
                    print(f"Skipping {wav_file.name} - already exists in destination")

print("Octave organization complete!") 
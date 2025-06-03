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
    wav_files = [f for f in scale_folder.glob('*.wav')]
    
    # Create a set of unique notes from the filenames
    notes = set()
    for wav_file in wav_files:
        # Extract note from filename (format: Note-Octave-Scale-Variation.wav)
        note = wav_file.name.split('-')[0]
        notes.add(note)
    
    # Create folders for each note
    for note in notes:
        note_folder = scale_folder / note
        note_folder.mkdir(exist_ok=True)
        
        # Move all files for this note to its folder
        for wav_file in wav_files:
            if wav_file.name.startswith(f"{note}-"):
                # Create destination path
                dest = note_folder / wav_file.name
                # Move the file
                shutil.move(str(wav_file), str(dest))

print("Organization complete!") 
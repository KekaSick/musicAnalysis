import os
import shutil

# This should be the same as DEFAULT_AUDIO_OUTPUT_DIR in download_script.py
# It's the directory where the download_script.py saves the genre-specific music files.
SOURCE_MUSIC_DIRECTORY = os.path.join("data", "music", "audio")

def organize_files():
    """
    Scans the SOURCE_MUSIC_DIRECTORY for audio files and moves them
    into subdirectories named after their genre, extracted from the filename.
    """
    if not os.path.isdir(SOURCE_MUSIC_DIRECTORY):
        print(f"Error: Source directory '{SOURCE_MUSIC_DIRECTORY}' not found.")
        print("Please ensure tracks have been downloaded by download_script.py first,")
        print("and that SOURCE_MUSIC_DIRECTORY is set correctly.")
        return

    print(f"Scanning directory: {SOURCE_MUSIC_DIRECTORY} for audio files...")
    files_moved_count = 0
    files_skipped_count = 0

    for filename in os.listdir(SOURCE_MUSIC_DIRECTORY):
        file_path = os.path.join(SOURCE_MUSIC_DIRECTORY, filename)

        # Process only files, skip directories or other file types if any
        if os.path.isfile(file_path):
            try:
                # Assuming filename format: "genre_rank_track_artist.extension"
                # Example: "Pop_01_SomeSong_SomeArtist.wav"
                # We extract the first part as the genre.
                parts = filename.split('_')
                if len(parts) > 1:
                    genre = parts[0]
                    
                    # Create the genre-specific directory if it doesn't already exist
                    # This directory will be inside SOURCE_MUSIC_DIRECTORY
                    genre_specific_dir = os.path.join(SOURCE_MUSIC_DIRECTORY, genre)
                    os.makedirs(genre_specific_dir, exist_ok=True)
                    
                    # Define the full destination path for the file
                    destination_file_path = os.path.join(genre_specific_dir, filename)
                    
                    # Move the file
                    shutil.move(file_path, destination_file_path)
                    print(f"Moved: '{filename}' to '{genre_specific_dir}'")
                    files_moved_count += 1
                else:
                    # This case handles files that don't match the expected naming convention
                    print(f"Skipped: '{filename}' (could not determine genre from filename).")
                    files_skipped_count += 1
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
                files_skipped_count += 1
        else:
            # If it's a directory (potentially already a genre folder), skip it.
            if os.path.isdir(file_path):
                print(f"Skipped: '{filename}' (already a directory).")
            else:
                print(f"Skipped: '{filename}' (not a file).")


    print(f"\n--- Organization Summary ---")
    print(f"Successfully moved: {files_moved_count} files.")
    if files_skipped_count > 0:
        print(f"Skipped or failed to move: {files_skipped_count} files.")
    print("Finished organizing files.")

if __name__ == "__main__":
    organize_files() 
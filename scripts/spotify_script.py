import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import json
from dotenv import load_dotenv
import time
import traceback
import numpy as np
from collections import defaultdict
import logging # Add logging
import functools # Added for lru_cache


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Spotify API Authentication
client_id = ''
client_secret = ''

GLOBAL_TOP_PLAYLIST_ID = '37i9dQZEVXbMDoHDwVN2tF' # Spotify's "Top 50 - Global" playlist ID

if not client_id or not client_secret:
    logging.error("Missing Spotify API credentials!")
    logging.error("Please make sure to set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file")
    exit(1)

# Configure Spotipy with retries for common transient errors and rate limits
# Note: For 429 with very long Retry-After, custom handling is still better.

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret),
    retries=0,  # <-- Turn it OFF, let your wrapper handle retries
    status_forcelist=(429, 500, 502, 503, 504), # Status codes to retry on (as a tuple)
    backoff_factor=0.5 # Delay factor
)

# Define Genres
genres = ["pop", "rock", "hip-hop", "electronic", "jazz", "classical", "metal", "blues", "reggae", "country"]
base_directory = os.path.join("data", "spotify")
json_directory = os.path.join(base_directory, "json")

# Create the json directory structure
try:
    os.makedirs(json_directory, exist_ok=True)
    logging.info(f"Using directory for metadata: {json_directory}")
except OSError as e:
    logging.error(f"Error creating directory {json_directory}: {e}")
    exit(1)

# Helper function to handle Spotify API calls with rate limit and error handling
def make_spotify_api_call(api_call_lambda, call_description):
    """Makes a Spotify API call with robust error handling and rate limit respect."""
    max_retries = 5
    current_retry = 0
    # PROACTIVE_DELAY_SECONDS = 0.5  # Proactive delay removed as per user undo

    while current_retry < max_retries:
        try:
            # time.sleep(PROACTIVE_DELAY_SECONDS) # Proactive delay removed
            return api_call_lambda()
        except spotipy.SpotifyException as e:
            logging.error(f"Spotify API error during '{call_description}': {e.http_status} - {e.msg}")
            if e.http_status == 429:
                retry_after = e.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after) + 1 # Add a small buffer
                    if wait_time > 3600: # If > 1 hour, log and return sentinel
                        logging.error(f"Spotify API Quota hit. Rate limit requires waiting for {wait_time} seconds for '{call_description}'. Skipping this call.")
                        return "RATE_LIMIT_SKIP" # Return sentinel for very long waits
                    logging.warning(f"Rate limited. Waiting for {wait_time} seconds before retrying '{call_description}' (Attempt {current_retry + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    current_retry += 1
                else:
                    # No Retry-After header, use exponential backoff
                    wait_time = (2 ** current_retry) * 5 
                    logging.warning(f"Rate limited (no Retry-After header). Waiting {wait_time}s before retrying '{call_description}' (Attempt {current_retry + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    current_retry += 1
            elif e.http_status in [500, 502, 503, 504]: # Server-side errors
                wait_time = (2 ** current_retry) * 5 # Exponential backoff
                logging.warning(f"Server error ({e.http_status}). Waiting {wait_time}s before retrying '{call_description}' (Attempt {current_retry + 1}/{max_retries})...")
                time.sleep(wait_time)
                current_retry += 1
            else:
                logging.error(f"Non-retryable Spotify API error for '{call_description}': {e}")
                return None # Or re-raise if it should halt execution
        except Exception as e:
            logging.error(f"General error during '{call_description}': {e}")
            traceback.print_exc()
            # For general errors, maybe retry once or twice with a short delay
            if current_retry < 2: # Limit retries for general errors
                wait_time = 10
                logging.warning(f"Waiting {wait_time}s before retrying '{call_description}' due to general error (Attempt {current_retry + 1}/2)...")
                time.sleep(wait_time)
                current_retry += 1
            else:
                return None # Or re-raise
    logging.error(f"Max retries reached for '{call_description}'. Giving up.")
    return None

# Helper function for chunking iterables
def _chunked(iterable, n):
    """Divide an iterable into chunks of size n."""
    if n < 1:
        raise ValueError("Chunk size n must be 1 or greater.")
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

@functools.lru_cache(maxsize=None) # Added lru_cache decorator
def fetch_playlist_tracks_details(playlist_id, limit=50):
    """Fetches track details (name, artist, id) from a specific playlist."""
    tracks_details = []
    try:
        # Initial call to get the first page of playlist items
        results = make_spotify_api_call(
            lambda: sp.playlist_items(
                playlist_id,
                fields='items.track(id,name,artists,popularity),next',
                limit=min(limit if limit > 0 else 100, 100),
                additional_types=('track',)
            ),
            f"fetching initial page of playlist {playlist_id}"
        )
        if results == "RATE_LIMIT_SKIP": return [] # Handle sentinel

        while results and len(tracks_details) < (limit if limit > 0 else float('inf')):
            if results.get('items'):
                for item_wrapper in results['items']:
                    if len(tracks_details) >= (limit if limit > 0 else float('inf')):
                        break
                    item = item_wrapper.get('track')
                    # Ensure track and essential fields exist, and it's actually a track (not None or an episode if filters fail)
                    if item and item.get('id') and item.get('name') and item.get('artists') and item.get('type') == 'track':
                        tracks_details.append({
                            'name': item['name'],
                            'artist': item['artists'][0]['name'], # Primary artist
                            'artist_id': item['artists'][0]['id'], # Primary artist ID for genre check
                            'spotify_id': item['id'],
                            'popularity': item.get('popularity') # Get popularity, might be None
                        })
            
            if results.get('next'):
                # print(f"      Fetching next page for playlist {playlist_id}...") # Optional: for debugging pagination
                results = make_spotify_api_call(lambda: sp.next(results), f"fetching next page of playlist {playlist_id}")
                if results == "RATE_LIMIT_SKIP" or not results: # Handle sentinel and API call failure
                    break
            else:
                break # No more pages

        return tracks_details
    except spotipy.SpotifyException as e: # This outer try-except might become less necessary or just for initial setup errors
        logging.error(f"  Spotify API error fetching playlist {playlist_id}: {e}")
        # The make_spotify_api_call should handle retries for 429s now.
        # if e.http_status == 429: 
        #     logging.warning("  Rate limited. Waiting for 30 seconds...") # Old handling
        #     time.sleep(30)
        #     return fetch_playlist_tracks_details(playlist_id, limit) # Retry
        return []
    except Exception as e:
        logging.error(f"  General error fetching playlist {playlist_id}: {e}")
        return []

def get_top_tracks_for_genre(genre, limit=50, max_playlists_per_query=20, max_tracks_from_search_fallback=100):
    """Get top tracks for a genre using comprehensive search, pagination, artist genre filtering, and popularity backfilling."""
    all_tracks_details = []
    seen_track_signatures = set() # For robust duplicate checking (id, or (name, artist))
    
    # Target at least limit*2 tracks before filtering and final sort, but cap for performance.
    initial_fetch_target = max(limit * 2, 120)  # Reduced from 5x and 250
    logging.info(f"Fetching for genre: {genre}. Target pool size: {initial_fetch_target}. Final limit: {limit}.")

    # 1. Playlist Search (Prioritizing Spotify-owned, then general)
    playlist_search_configs = [
        {'query_template': '"This Is {genre}"', 'owner_priority': 'spotify', 'message': '"This Is..." playlists by Spotify'},
        {'query_template': '"{genre} Top Hits"', 'owner_priority': 'spotify', 'message': '"Top Hits" playlists by Spotify'},
        {'query_template': '"All Out {genre}"', 'owner_priority': 'spotify', 'message': '"All Out..." playlists by Spotify'},
        {'query_template': '"{genre}"', 'owner_priority': None, 'message': 'Broad search for playlists with "{genre}" in name'},
        # Add other targeted queries if needed, ensuring they don't overlap too much with broad search.
    ]

    processed_playlist_ids = set()

    for config in playlist_search_configs:
        if len(all_tracks_details) >= initial_fetch_target: break
        
        query = config['query_template'].format(genre=genre)
        logging.info(f"  Starting playlist search: {config['message']} (Query: '{query}')")
        
        current_offset = 0
        playlists_processed_this_query = 0
        
        while playlists_processed_this_query < max_playlists_per_query and len(all_tracks_details) < initial_fetch_target:
            try:
                playlist_search_results = make_spotify_api_call(
                    lambda: sp.search(q=query, type='playlist', limit=50, offset=current_offset, market='US'),
                    f"playlist search for '{query}' at offset {current_offset}"
                )
                if playlist_search_results == "RATE_LIMIT_SKIP": break # Handle sentinel, break from this query's pagination
                
                # The following check for `playlist_search_results` is crucial because make_spotify_api_call can return None
                if not playlist_search_results or not playlist_search_results.get('playlists') or not playlist_search_results['playlists'].get('items'):
                    logging.info(f"    No more playlists found for '{query}' at offset {current_offset} or API call failed.")
                    break 

                playlists_on_page = playlist_search_results['playlists']['items']
                
                # Prioritize by owner if specified
                if config['owner_priority'] == 'spotify':
                    spotify_owned = []
                    other_playlists = []
                    for p in playlists_on_page:
                        if p and p.get('owner') and p['owner'].get('id') == 'spotify':
                            spotify_owned.append(p)
                        elif p: # Ensure p is not None before adding to other_playlists
                            other_playlists.append(p)
                    ordered_playlists_to_check = spotify_owned + other_playlists
                else:
                    ordered_playlists_to_check = [p for p in playlists_on_page if p] # Filter out None items

                for playlist in ordered_playlists_to_check:
                    if not playlist or not playlist.get('id'): # Check if playlist or playlist ID is None
                        logging.info("    Skipping a playlist item because it or its ID is None.")
                        continue
                    if len(all_tracks_details) >= initial_fetch_target: break
                    if playlist['id'] in processed_playlist_ids: continue # Avoid re-processing

                    logging.info(f"    Checking playlist: '{playlist['name']}' (Owner: {playlist['owner']['display_name']}, ID: {playlist['id']})")
                    processed_playlist_ids.add(playlist['id'])
                    playlists_processed_this_query += 1

                    tracks_needed_from_playlist = initial_fetch_target - len(all_tracks_details)
                    if tracks_needed_from_playlist <=0: break

                    playlist_tracks = fetch_playlist_tracks_details(playlist['id'], limit=tracks_needed_from_playlist)
                    
                    added_count = 0
                    for track in playlist_tracks:
                        sig_id = track['spotify_id']
                        sig_name_artist = (track['name'].lower(), track['artist'].lower())
                        
                        if sig_id not in seen_track_signatures and sig_name_artist not in seen_track_signatures:
                            all_tracks_details.append(track)
                            seen_track_signatures.add(sig_id)
                            seen_track_signatures.add(sig_name_artist)
                            added_count += 1
                            if len(all_tracks_details) >= initial_fetch_target: break
                    logging.info(f"      Added {added_count} new tracks from '{playlist['name']}'. Pool size: {len(all_tracks_details)}/{initial_fetch_target}")
                    if len(all_tracks_details) >= initial_fetch_target: break
                
                current_offset += len(playlists_on_page) # Increment offset for next page
                 # Check if playlist_search_results and its nested keys exist before accessing 'next'
                if not playlist_search_results or not playlist_search_results.get('playlists') or not playlist_search_results['playlists'].get('next'):
                    logging.info(f"    No more pages of playlists for '{query}'.")
                    break
                # time.sleep(0.5) # Delay between paginated playlist search calls - handled by helper or not strictly needed if respecting Retry-After

            except spotipy.SpotifyException as e: # This might catch errors not handled by the helper, or if helper returns None and it's not checked
                if e.http_status == 429: logging.warning(f"    Rate limit on playlist search. Waiting 60s.") # Fallback, though helper should handle
                else: logging.error(f"    Spotify API error (playlist search for '{query}'): {e}")
                break # Break from this query's pagination on error
            except Exception as e:
                logging.error(f"    General error (playlist search for '{query}'): {e}")
                traceback.print_exc()
                break
        logging.info(f"    Finished with query '{query}'. Processed {playlists_processed_this_query} playlists.")


    # 2. Fallback: Direct Track Search if pool is not full
    if len(all_tracks_details) < initial_fetch_target:
        logging.info(f"  Pool size {len(all_tracks_details)}/{initial_fetch_target}. Falling back to direct genre track search.")
        current_offset = 0
        tracks_fetched_in_fallback = 0
        while len(all_tracks_details) < initial_fetch_target and tracks_fetched_in_fallback < max_tracks_from_search_fallback:
            needed_fallback = initial_fetch_target - len(all_tracks_details)
            fetch_limit_for_call = min(needed_fallback, 50) # Max 50 per call for track search
            if fetch_limit_for_call <= 0: break

            try:
                results = make_spotify_api_call(
                    lambda: sp.search(q=f'genre:"{genre}"', type='track', limit=fetch_limit_for_call, offset=current_offset, market='US'),
                    f"direct track search for genre '{genre}' at offset {current_offset}"
                )
                if results == "RATE_LIMIT_SKIP": break # Handle sentinel
                
                if not results or not results.get('tracks') or not results['tracks'].get('items'):
                    logging.info(f"    No more tracks found in direct search for genre '{genre}' at offset {current_offset} or API call failed.")
                    break
                
                page_tracks = results['tracks']['items']
                tracks_fetched_in_fallback += len(page_tracks)
                added_count = 0
                for item in page_tracks:
                    if item and item.get('id') and item.get('name') and item.get('artists'):
                        sig_id = item['id']
                        primary_artist_name = item['artists'][0]['name'] if item['artists'] else 'Unknown Artist'
                        primary_artist_id = item['artists'][0]['id'] if item['artists'] else None
                        sig_name_artist = (item['name'].lower(), primary_artist_name.lower())

                        if sig_id not in seen_track_signatures and sig_name_artist not in seen_track_signatures:
                            all_tracks_details.append({
                                'name': item['name'], 'artist': primary_artist_name, 'artist_id': primary_artist_id,
                                'spotify_id': sig_id, 'popularity': item.get('popularity')
                            })
                            seen_track_signatures.add(sig_id)
                            seen_track_signatures.add(sig_name_artist)
                            added_count +=1
                            if len(all_tracks_details) >= initial_fetch_target: break
                logging.info(f"    Added {added_count} new tracks from direct search (offset {current_offset}). Pool: {len(all_tracks_details)}")
                if len(all_tracks_details) >= initial_fetch_target: break

                current_offset += len(page_tracks)
                if not results or not results.get('tracks') or not results['tracks'].get('next'): break
                # time.sleep(0.3) # Handled by helper
            except spotipy.SpotifyException as e: # Fallback error handling
                if e.http_status == 429: logging.warning(f"    Rate limit on track search. Waiting 60s.")
                else: logging.error(f"    Spotify API error (direct track search): {e}")
                break 
            except Exception as e:
                logging.error(f"    General error (direct track search): {e}")
                traceback.print_exc()
                break
    
    logging.info(f"  Initial track pool collection complete. Size: {len(all_tracks_details)} tracks.")

    # 3. Back-fill missing popularity scores
    logging.info("  Back-filling missing popularity scores...")
    tracks_missing_popularity = [t for t in all_tracks_details if t.get('popularity') is None]
    if tracks_missing_popularity:
        missing_ids = [t['spotify_id'] for t in tracks_missing_popularity]
        popularity_map = {}
        for id_chunk in _chunked(missing_ids, 50): # sp.tracks can take up to 50 IDs
            try:
                # track_results = sp.tracks(id_chunk)
                track_results = make_spotify_api_call(
                    lambda: sp.tracks(id_chunk),
                    f"fetching track details for popularity backfill (IDs: {id_chunk[:2]}...)"
                )
                if track_results == "RATE_LIMIT_SKIP": continue # Handle sentinel, continue to next chunk
                
                if track_results and track_results.get('tracks'):
                    for track_obj in track_results['tracks']:
                        if track_obj: # Ensure track object is not None
                            popularity_map[track_obj['id']] = track_obj.get('popularity')
                # time.sleep(0.2) # Handled by helper
            except spotipy.SpotifyException as e: # Fallback, ideally helper handles
                logging.error(f"    Spotify API error fetching track details for popularity: {e}")
            except Exception as e:
                logging.error(f"    General error fetching track details for popularity: {e}")

        updated_count = 0
        for track_detail in all_tracks_details:
            if track_detail.get('popularity') is None and track_detail['spotify_id'] in popularity_map:
                track_detail['popularity'] = popularity_map[track_detail['spotify_id']]
                if track_detail['popularity'] is not None : updated_count +=1
        logging.info(f"    Updated popularity for {updated_count} tracks. {len([t for t in all_tracks_details if t.get('popularity') is None])} still missing.")
    else:
        logging.info("    No tracks initially missing popularity scores.")

    # Default any remaining None popularities to 0 for sorting
    for t in all_tracks_details:
        if t.get('popularity') is None:
            t['popularity'] = 0

    # Store pre-filter pool for potential fill-in
    pre_genre_filter_pool = [t.copy() for t in all_tracks_details] # Deep copy might be safer if items are complex

    # 4. Artist Genre Verification
    logging.info("  Performing artist genre verification...")
    artist_ids_to_check = list(set(t['artist_id'] for t in all_tracks_details if t.get('artist_id')))
    artist_genre_map = {}
    if artist_ids_to_check:
        for id_chunk in _chunked(artist_ids_to_check, 50): # sp.artists can take up to 50 IDs
            try:
                # artist_results = sp.artists(id_chunk)
                artist_results = make_spotify_api_call(
                    lambda: sp.artists(id_chunk),
                    f"fetching artist details for genre check (IDs: {id_chunk[:2]}...)"
                )
                if artist_results == "RATE_LIMIT_SKIP": continue # Handle sentinel, continue to next chunk
                
                if artist_results and artist_results.get('artists'):
                    for artist_obj in artist_results['artists']:
                        if artist_obj: # Ensure artist object is not None
                            artist_genre_map[artist_obj['id']] = [g.lower() for g in artist_obj.get('genres', [])]
                # time.sleep(0.2) # Handled by helper
            except spotipy.SpotifyException as e: # Fallback
                logging.error(f"    Spotify API error fetching artist details for genre check: {e}")
            except Exception as e:
                logging.error(f"    General error fetching artist details for genre check: {e}")
    
    genre_specific_tracks = []
    removed_by_genre_filter = 0
    for track in all_tracks_details:
        artist_id = track.get('artist_id')
        if artist_id and artist_id in artist_genre_map:
            if genre.lower() in artist_genre_map[artist_id]:
                genre_specific_tracks.append(track)
            else:
                # print(f"    Filtering out track '{track['name']}' by '{track['artist']}' - artist genres {artist_genre_map[artist_id]} do not match target '{genre}'.")
                removed_by_genre_filter +=1
        elif not artist_id : # No artist ID, can't verify, err on side of caution or keep if desired. For now, remove.
             removed_by_genre_filter +=1
        else: # Artist ID was there, but not found in map (API error or new artist), keep for now? Or remove? Let's keep.
            genre_specific_tracks.append(track) # Keep if artist lookup failed but track seems relevant

    logging.info(f"    Retained {len(genre_specific_tracks)} tracks after artist genre filtering. {removed_by_genre_filter} removed.")
    all_tracks_details = genre_specific_tracks # Replace with filtered list

    # Mark artist-verified tracks
    for track in all_tracks_details:
        track['genre_verification'] = 'artist_verified'

    # 4.5. Fill if necessary from pre-genre-filter pool, prioritizing popularity
    if len(all_tracks_details) < limit:
        needed_supplement = limit - len(all_tracks_details)
        logging.info(f"  Genre-verified list has {len(all_tracks_details)} tracks. Need to supplement {needed_supplement} more.")

        # Get IDs of tracks already selected to avoid adding them again
        selected_ids = {t['spotify_id'] for t in all_tracks_details}
        
        # Filter pre_genre_filter_pool to get candidates not already selected
        supplement_candidates = [t for t in pre_genre_filter_pool if t['spotify_id'] not in selected_ids]
        
        # Sort these candidates by popularity
        supplement_candidates.sort(key=lambda x: x.get('popularity', 0), reverse=True)
        
        # Take the top needed_supplement tracks
        tracks_to_add = supplement_candidates[:needed_supplement]
        for track in tracks_to_add:
            track['genre_verification'] = 'popularity_fill' # Mark as fill-in
        
        all_tracks_details.extend(tracks_to_add)
        logging.info(f"    Added {len(tracks_to_add)} tracks as popularity fill. Pool size now: {len(all_tracks_details)}.")

    if not all_tracks_details:
        logging.info(f"  No tracks found for genre '{genre}' after all filtering and filling steps.")
        return []

    # 5. Final Sort by Popularity
    all_tracks_details.sort(key=lambda x: x.get('popularity', 0), reverse=True)
    
    final_tracks = all_tracks_details[:limit]
    logging.info(f"  Collected {len(all_tracks_details)} unique, genre-verified tracks. Sorted by popularity. Returning top {len(final_tracks)}.")
    
    if len(final_tracks) < limit:
        logging.warning(f"  WARNING: Found only {len(final_tracks)} tracks for {genre}, less than the requested {limit}.")

    return final_tracks

# Fetch tracks for each genre and compile metadata
all_genres_metadata_list = [] # Renamed to avoid conflict if global is separate
genre_stats = defaultdict(dict)  # Store statistics for each genre
processed_genre_names_from_file = set()

metadata_output_file = os.path.join(json_directory, "all_tracks_metadata_v2.json")

# Load existing data if available
if os.path.exists(metadata_output_file):
    try:
        with open(metadata_output_file, 'r', encoding='utf-8') as f:
            all_genres_metadata_list = json.load(f)
            if not isinstance(all_genres_metadata_list, list):
                logging.warning(f"Corrupted data in {metadata_output_file}, expected a list. Starting fresh.")
                all_genres_metadata_list = []
            else:
                loaded_genres = set()
                for record in all_genres_metadata_list:
                    if isinstance(record, dict) and 'genre' in record:
                        loaded_genres.add(record['genre'])
                processed_genre_names_from_file.update(loaded_genres)
                logging.info(f"Loaded {len(all_genres_metadata_list)} records for {len(loaded_genres)} genres from {metadata_output_file}: {', '.join(sorted(list(loaded_genres))) if loaded_genres else 'None'}.")

                # Recalculate stats for loaded genres
                if all_genres_metadata_list:
                    for genre_name_loaded in loaded_genres:
                        genre_specific_tracks_loaded = [t for t in all_genres_metadata_list if t.get('genre') == genre_name_loaded]
                        if genre_specific_tracks_loaded:
                            popularities_loaded = [track.get('popularity', 0) for track in genre_specific_tracks_loaded]
                            if popularities_loaded:
                                genre_stats[genre_name_loaded] = {
                                    'avg_popularity': np.mean(popularities_loaded),
                                    'std_popularity': np.std(popularities_loaded),
                                    'min_popularity': np.min(popularities_loaded),
                                    'max_popularity': np.max(popularities_loaded),
                                    'track_count': len(genre_specific_tracks_loaded)
                                }
    except json.JSONDecodeError:
        logging.warning(f"Could not decode JSON from {metadata_output_file}. Starting fresh.")
        all_genres_metadata_list = []
    except Exception as e:
        logging.error(f"Error loading existing metadata from {metadata_output_file}: {e}. Starting fresh.")
        all_genres_metadata_list = []
else:
    logging.info(f"{metadata_output_file} not found. Starting fresh.")

for genre_index, genre_item in enumerate(genres + ["global_top_50"]): # Add global to the loop
    current_genre_name = genre_item

    if current_genre_name in processed_genre_names_from_file:
        logging.info(f"Skipping {current_genre_name} - already processed and loaded from file.")
        continue # Skip to the next genre

    is_global_fetch = (genre_item == "global_top_50")
    
    logging.info(f"\nProcessing {current_genre_name} ({genre_index+1}/{len(genres)+1})...")

    if is_global_fetch:
        # For global, we use a known playlist ID and a simpler fetch, then backfill popularity
        logging.info(f"  Fetching directly from Global Top 50 playlist ID: {GLOBAL_TOP_PLAYLIST_ID}")
        # Limit for global can be fixed at 50.
        tracks_metadata = fetch_playlist_tracks_details(GLOBAL_TOP_PLAYLIST_ID, limit=50)
        
        # Backfill popularity for global tracks specifically
        if tracks_metadata:
            logging.info("  Back-filling popularity for Global Top 50 tracks...")
            missing_ids_global = [t['spotify_id'] for t in tracks_metadata if t.get('popularity') is None]
            if missing_ids_global:
                pop_map_global = {}
                for id_chunk_g in _chunked(missing_ids_global, 50):
                    try:
                        # track_res_g = sp.tracks(id_chunk_g)
                        track_res_g = make_spotify_api_call(
                            lambda: sp.tracks(id_chunk_g),
                            f"backfilling global popularity (IDs: {id_chunk_g[:2]}...)"
                        )
                        if track_res_g == "RATE_LIMIT_SKIP": break # Handle sentinel, break from this chunk processing for global
                        
                        if track_res_g and track_res_g.get('tracks'):
                            for tr_g in track_res_g['tracks']:
                                if tr_g: pop_map_global[tr_g['id']] = tr_g.get('popularity')
                        # time.sleep(0.2) # Handled by helper
                    except Exception as e_g: logging.error(f"    Error backfilling global pop: {e_g}")
                
                updated_g_count = 0
                for t_g in tracks_metadata:
                    if t_g.get('popularity') is None and t_g['spotify_id'] in pop_map_global:
                        t_g['popularity'] = pop_map_global[t_g['spotify_id']]
                        if t_g['popularity'] is not None: updated_g_count +=1
                logging.info(f"    Updated pop for {updated_g_count} global tracks.")
            # Default remaining None to 0
            for t_g in tracks_metadata:
                if t_g.get('popularity') is None: t_g['popularity'] = 0
            
            # Sort global tracks by popularity
            tracks_metadata.sort(key=lambda x: x.get('popularity', 0), reverse=True)

        # Artist genre check is skipped for global_top_50 as it's inherently cross-genre
    else:
        # Use the full get_top_tracks_for_genre for other genres
        tracks_metadata = get_top_tracks_for_genre(current_genre_name, limit=50) # Standard limit of 50
    
    if tracks_metadata:
        logging.info(f"  Collected {len(tracks_metadata)} final tracks for {current_genre_name}.")
        # Popularity stats calculation (can be reused here)
        popularities = [track.get('popularity', 0) for track in tracks_metadata]
        if popularities:
            avg_popularity = np.mean(popularities)
            std_popularity = np.std(popularities)
            min_pop = np.min(popularities)
            max_pop = np.max(popularities)
            # Store stats in the dictionary
            genre_stats[current_genre_name] = {
                'avg_popularity': avg_popularity,
                'std_popularity': std_popularity,
                'min_popularity': min_pop,
                'max_popularity': max_pop,
                'track_count': len(tracks_metadata)
            }
            logging.info(f"  Popularity Stats for {current_genre_name}: Avg={avg_popularity:.2f}, Std={std_popularity:.2f}, Min={min_pop}, Max={max_pop}")
            if not is_global_fetch and std_popularity > 15 and avg_popularity < 70 : # Adjusted warning thresholds
                logging.warning(f"  NOTE: Stats for {current_genre_name} (StdDev > 15 or AvgPop < 70) might indicate diverse or less popular track pool.")
        else:
            logging.info(f"  No popularity data to calculate stats for {current_genre_name}.")
            genre_stats[current_genre_name] = {
                'avg_popularity': 0,
                'std_popularity': 0,
                'min_popularity': 0,
                'max_popularity': 0,
                'track_count': 0
            }

        for i, track_meta in enumerate(tracks_metadata):
            all_genres_metadata_list.append({
                'genre': current_genre_name, # Use actual genre name or 'global_top_50'
                'rank': i + 1,
                'name': track_meta['name'],
                'artist': track_meta['artist'],
                'artist_id': track_meta.get('artist_id'), # Ensure artist_id is saved
                'spotify_id': track_meta['spotify_id'],
                'popularity': track_meta.get('popularity', 0),
                'genre_verification': track_meta.get('genre_verification', 'N/A') # Add verification status
            })
    else:
        logging.info(f"  No tracks found for {current_genre_name}.")
    
    # Incremental save after processing each genre
    if tracks_metadata: # Only save if new tracks were actually fetched for this genre
        try:
            # We append new tracks for the current genre to the main list
            # The main list was either loaded or started fresh
            # No need to re-add, tracks_metadata becomes part of all_genres_metadata_list via append in the loop
            # The all_genres_metadata_list is now the single source of truth for saving
            with open(metadata_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_genres_metadata_list, f, indent=4, ensure_ascii=False)
            logging.info(f"  Successfully saved/updated metadata for {current_genre_name} to: {metadata_output_file}")
        except IOError as e:
            logging.error(f"  Error saving metadata for {current_genre_name} to file: {e}")

    logging.info(f"  Finished with {current_genre_name}. Waiting for 2 seconds...")
    time.sleep(2)

# Save all metadata (including global if processed this way) to a single JSON file
# This final save is somewhat redundant if incremental saving works, but acts as a final confirmation.
# metadata_output_file = os.path.join(json_directory, "all_tracks_metadata_v2.json") # Defined earlier
try:
    with open(metadata_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_genres_metadata_list, f, indent=4, ensure_ascii=False)
    logging.info(f"\nSuccessfully saved all metadata to: {metadata_output_file}")
except IOError as e:
    logging.error(f"\nError saving combined metadata file: {e}")

logging.info("\nFinished all metadata fetching processes.")

# Print final statistics summary
logging.info("\n=== Final Statistics Summary ===")
logging.info(f"{'Genre':<15} {'Avg Pop':>10} {'Std Dev':>10} {'Min':>6} {'Max':>6} {'Tracks':>8}")
logging.info("-" * 60)
for genre_name, stats in sorted(genre_stats.items()):
    logging.info(f"{genre_name:<15} {stats['avg_popularity']:>10.2f} {stats['std_popularity']:>10.2f} {stats['min_popularity']:>6.0f} {stats['max_popularity']:>6.0f} {stats['track_count']:>8d}")
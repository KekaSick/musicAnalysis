import os
import sys
import subprocess
import pandas as pd
import logging
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import multiprocessing

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Константы
CSV_FILE = 'data/csv_spotify/csv/sampled_dataset_PE_C_fd.csv'
AUDIO_OUTPUT_DIR = 'data/audio'
MAX_WORKERS = multiprocessing.cpu_count() * 2  # Максимальное количество параллельных загрузок
CACHE_FILE = 'download_cache.json'  # Файл для кеша скачанных треков

def sanitize_filename(filename):
    """Очищает имя файла от недопустимых символов."""
    import re
    # Удаляем или заменяем недопустимые символы
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Удаляем лишние пробелы и подчеркивания
    filename = re.sub(r'\s+', ' ', filename).strip()
    filename = re.sub(r'_+', '_', filename)
    # Ограничиваем длину
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def get_track_key(track_info):
    """Создает уникальный ключ для трека на основе названия и исполнителя."""
    track_name = track_info['track_name'].lower().strip()
    artists = track_info['artists'].lower().strip()
    return f"{track_name}_{artists}"

def load_download_cache():
    """Загружает кеш уже скачанных треков."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.info(f"Загружен кеш с {len(cache)} уже скачанными треками")
                return cache
        except Exception as e:
            logger.warning(f"Ошибка загрузки кеша: {e}")
    return {}

def save_download_cache(cache):
    """Сохраняет кеш скачанных треков."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Кеш сохранен с {len(cache)} треками")
    except Exception as e:
        logger.error(f"Ошибка сохранения кеша: {e}")

def check_existing_file(track_info, output_dir):
    """Проверяет существование файла трека и возвращает путь если найден."""
    track_name = track_info['track_name']
    artists = track_info['artists']
    genre = track_info.get('track_genre', 'unknown')
    
    # Создаем безопасное имя файла
    filename = f"{genre}_{sanitize_filename(track_name)}_{sanitize_filename(artists)}.wav"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    return None

def download_track(track_info, output_dir, cache):
    """Скачивает трек в WAV формате с частотой дискретизации 44100 Гц."""
    start_time = time.time()
    try:
        track_name = track_info['track_name']
        artists = track_info['artists']
        genre = track_info.get('track_genre', 'unknown')
        track_key = get_track_key(track_info)
        
        # Проверяем кеш
        if track_key in cache:
            cached_path = cache[track_key]['file_path']
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                elapsed_time = time.time() - start_time
                logger.info(f"Трек найден в кеше: {track_name} - {artists} (время: {elapsed_time:.2f}с)")
                return cached_path
            else:
                # Файл не существует, удаляем из кеша
                del cache[track_key]
        
        # Проверяем существующий файл
        existing_file = check_existing_file(track_info, output_dir)
        if existing_file:
            # Добавляем в кеш
            cache[track_key] = {
                'track_name': track_name,
                'artists': artists,
                'track_genre': genre,
                'file_path': existing_file,
                'downloaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            elapsed_time = time.time() - start_time
            logger.info(f"Файл уже существует: {track_name} - {artists} (время: {elapsed_time:.2f}с)")
            return existing_file
        
        # Создаем безопасное имя файла
        filename = f"{genre}_{sanitize_filename(track_name)}_{sanitize_filename(artists)}.wav"
        output_path = os.path.join(output_dir, filename)
        
        search_query = f"{track_name} {artists}"
        logger.info(f"Скачивание: {track_name} - {artists} (Жанр: {genre})")
        
        # Команда для yt-dlp с настройками WAV 44100 Гц
        command = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',  # Лучшее качество
            '--output', output_path,
            '--no-playlist',
            '--no-warnings',
            '--postprocessor-args', 'ExtractAudio+ffmpeg:-ar 44100',  # Устанавливаем частоту дискретизации 44100 Гц
            f'ytsearch1:{search_query}'
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            elapsed_time = time.time() - start_time
            error_msg = result.stderr.strip()
            
            # Проверяем различные типы ошибок
            if "no such option" in error_msg:
                logger.error(f"Ошибка конфигурации yt-dlp '{search_query}' (время: {elapsed_time:.2f}с): {error_msg}")
            elif "No video found" in error_msg or "No results found" in error_msg:
                logger.error(f"Трек не найден '{search_query}' (время: {elapsed_time:.2f}с): {error_msg}")
            elif "timeout" in error_msg.lower():
                logger.error(f"Таймаут при скачивании '{search_query}' (время: {elapsed_time:.2f}с): {error_msg}")
            else:
                logger.error(f"Ошибка скачивания '{search_query}' (время: {elapsed_time:.2f}с): {error_msg}")
            return None
            
        # Проверяем что файл создался и имеет правильный размер
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Добавляем в кеш
            cache[track_key] = {
                'track_name': track_name,
                'artists': artists,
                'track_genre': genre,
                'file_path': output_path,
                'downloaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            elapsed_time = time.time() - start_time
            logger.info(f"Успешно скачан: {filename} (время: {elapsed_time:.2f}с)")
            return output_path
        else:
            elapsed_time = time.time() - start_time
            logger.error(f"Файл не создался или пустой: {filename} (время: {elapsed_time:.2f}с)")
            return None
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"Таймаут при скачивании '{track_name}' (время: {elapsed_time:.2f}с)")
        return None
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Исключение при скачивании '{track_name}' (время: {elapsed_time:.2f}с): {str(e)}")
        return None

def download_all_tracks():
    """Скачивает все треки из CSV файла."""
    
    overall_start_time = time.time()
    
    # Создаем директорию для аудио файлов
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # Загружаем кеш
    cache = load_download_cache()
    
    # Читаем CSV файл
    try:
        df = pd.read_csv(CSV_FILE)
        logger.info(f"Загружено {len(df)} треков из CSV файла")
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV файла: {e}")
        return
    
    # Конвертируем DataFrame в список словарей (сохраняем порядок из CSV)
    tracks_to_download = df.to_dict('records')
    
    # Фильтруем треки, которые нужно скачать
    tracks_to_process = []
    already_downloaded = []
    
    for track in tracks_to_download:
        track_key = get_track_key(track)
        if track_key in cache:
            cached_path = cache[track_key]['file_path']
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                already_downloaded.append(track)
                continue
        
        # Проверяем существующий файл
        existing_file = check_existing_file(track, AUDIO_OUTPUT_DIR)
        if existing_file:
            # Добавляем в кеш
            cache[track_key] = {
                'track_name': track['track_name'],
                'artists': track['artists'],
                'track_genre': track.get('track_genre', 'unknown'),
                'file_path': existing_file,
                'downloaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            already_downloaded.append(track)
        else:
            tracks_to_process.append(track)
    
    logger.info(f"Уже скачано: {len(already_downloaded)} треков")
    logger.info(f"Требует скачивания: {len(tracks_to_process)} треков")
    
    # Статистика по жанрам
    genre_stats = {}
    for track in tracks_to_download:
        genre = track.get('track_genre', 'unknown')
        if genre not in genre_stats:
            genre_stats[genre] = 0
        genre_stats[genre] += 1
    
    logger.info("Статистика по жанрам:")
    for genre, count in genre_stats.items():
        logger.info(f"  {genre}: {count} треков")
    
    if not tracks_to_process:
        overall_elapsed_time = time.time() - overall_start_time
        logger.info("Все треки уже скачаны!")
        logger.info(f"Общее время выполнения: {overall_elapsed_time:.2f}с")
        return {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_tracks': len(tracks_to_download),
            'successful_downloads': len(already_downloaded),
            'failed_downloads': 0,
            'total_time_seconds': overall_elapsed_time,
            'message': 'All tracks already downloaded'
        }
    
    # Скачиваем треки
    successful_downloads = []
    failed_downloads = []
    download_times = []  # Для статистики времени
    
    logger.info(f"\nНачинаем скачивание {len(tracks_to_process)} треков...")
    
    with tqdm(total=len(tracks_to_process), desc="Скачивание треков") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Создаем задачи для скачивания
            future_to_track = {
                executor.submit(download_track, track, AUDIO_OUTPUT_DIR, cache): track
                for track in tracks_to_process
            }
            
            # Обрабатываем результаты
            for future in as_completed(future_to_track):
                track = future_to_track[future]
                try:
                    result = future.result()
                    if result:
                        successful_downloads.append({
                            'track_name': track['track_name'],
                            'artists': track['artists'],
                            'track_genre': track.get('track_genre', 'unknown'),
                            'file_path': result
                        })
                    else:
                        failed_downloads.append({
                            'track_name': track['track_name'],
                            'artists': track['artists'],
                            'track_genre': track.get('track_genre', 'unknown'),
                            'error': 'Download failed'
                        })
                except Exception as e:
                    failed_downloads.append({
                        'track_name': track['track_name'],
                        'artists': track['artists'],
                        'track_genre': track.get('track_genre', 'unknown'),
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    # Сохраняем обновленный кеш
    save_download_cache(cache)
    
    # Вычисляем общее время
    overall_elapsed_time = time.time() - overall_start_time
    
    # Сохраняем результаты
    results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_tracks': len(tracks_to_download),
        'already_downloaded': len(already_downloaded),
        'new_successful_downloads': len(successful_downloads),
        'failed_downloads': len(failed_downloads),
        'total_time_seconds': overall_elapsed_time,
        'average_time_per_track': overall_elapsed_time / len(tracks_to_process) if tracks_to_process else 0,
        'successful_tracks': successful_downloads,
        'failed_tracks': failed_downloads,
        'genre_statistics': genre_stats
    }
    
    # Сохраняем в JSON
    results_file = f"download_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Выводим итоговую статистику
    logger.info("\n=== Итоговая статистика ===")
    logger.info(f"Всего треков в CSV: {len(tracks_to_download)}")
    logger.info(f"Уже было скачано: {len(already_downloaded)}")
    logger.info(f"Новых успешных скачиваний: {len(successful_downloads)}")
    logger.info(f"Ошибок скачивания: {len(failed_downloads)}")
    logger.info(f"Общее время выполнения: {overall_elapsed_time:.2f}с")
    if tracks_to_process:
        logger.info(f"Среднее время на трек: {overall_elapsed_time / len(tracks_to_process):.2f}с")
    logger.info(f"Результаты сохранены в: {results_file}")
    logger.info(f"Кеш обновлен: {CACHE_FILE}")
    
    if failed_downloads:
        logger.info("\nТреки с ошибками скачивания:")
        for track in failed_downloads[:10]:  # Показываем первые 10
            logger.info(f"  - {track['track_name']} by {track['artists']} ({track['track_genre']}): {track['error']}")
        if len(failed_downloads) > 10:
            logger.info(f"  ... и еще {len(failed_downloads) - 10} треков")
    
    return results

def check_yt_dlp():
    """Проверяет установлен ли yt-dlp."""
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"yt-dlp версия: {result.stdout.strip()}")
            return True
        else:
            logger.error("yt-dlp не найден или не работает")
            return False
    except FileNotFoundError:
        logger.error("yt-dlp не установлен. Установите его командой: pip install yt-dlp")
        return False

if __name__ == "__main__":
    logger.info("Запуск скрипта скачивания треков")
    
    # Проверяем yt-dlp
    if not check_yt_dlp():
        sys.exit(1)
    
    # Проверяем существование CSV файла
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV файл не найден: {CSV_FILE}")
        sys.exit(1)
    
    # Запускаем скачивание
    try:
        results = download_all_tracks()
        logger.info("Скачивание завершено!")
    except KeyboardInterrupt:
        logger.info("Скачивание прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)

import os
import sys
import pandas as pd
import numpy as np
import librosa
import logging
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import warnings

# Игнорируем предупреждения librosa
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Константы
CSV_FILE = 'data/csv_spotify/csv/sampled_dataset_PE_C_fd.csv'
AUDIO_DIR = 'data/audio'
OUTPUT_CSV = 'dataforGithub/features_extracted.csv'
CACHE_FILE = 'feature_extraction_cache.json'
MAX_WORKERS = os.cpu_count() * 2

# Параметры извлечения признаков
SR = 44100  # Частота дискретизации для анализа
N_FFT = 2048
HOP_LENGTH = 512
FRAME_LENGTH = 2048
TEXTURE_WINDOW = 43  # ~1 секунда при hop_length=512

def get_track_key(track_info):
    """Создает уникальный ключ для трека."""
    track_name = track_info['track_name'].lower().strip()
    artists = track_info['artists'].lower().strip()
    return f"{track_name}_{artists}"

def load_extraction_cache():
    """Загружает кеш уже обработанных треков."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.info(f"Загружен кеш с {len(cache)} уже обработанными треками")
                return cache
        except Exception as e:
            logger.warning(f"Ошибка загрузки кеша: {e}")
    return {}

def save_extraction_cache(cache):
    """Сохраняет кеш обработанных треков."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Кеш сохранен с {len(cache)} треками")
    except Exception as e:
        logger.error(f"Ошибка сохранения кеша: {e}")

def find_audio_file(track_info, audio_dir):
    """Находит аудио файл для трека."""
    track_name = track_info['track_name']
    artists = track_info['artists']
    genre = track_info.get('track_genre', 'unknown')
    
    # Создаем ожидаемое имя файла
    import re
    def sanitize_filename(filename):
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        filename = re.sub(r'_+', '_', filename)
        if len(filename) > 100:
            filename = filename[:100]
        return filename
    
    expected_filename = f"{genre}_{sanitize_filename(track_name)}_{sanitize_filename(artists)}.wav"
    expected_path = os.path.join(audio_dir, expected_filename)
    
    if os.path.exists(expected_path) and os.path.getsize(expected_path) > 0:
        return expected_path
    
    # Если не нашли, ищем по частичному совпадению
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            file_lower = file.lower()
            track_lower = track_name.lower()
            artist_lower = artists.lower()
            
            if track_lower in file_lower and artist_lower in file_lower:
                return os.path.join(audio_dir, file)
    
    return None

def rolling_stats(arr, window=TEXTURE_WINDOW):
    """Вычисляет скользящие среднее и дисперсию."""
    if len(arr) == 0:
        return np.array([]), np.array([])
    
    series = pd.Series(arr)
    rolling_mean = series.rolling(window, min_periods=1).mean().values
    rolling_var = series.rolling(window, min_periods=1).var().values
    
    return rolling_mean, rolling_var

def extract_timbral_features(y, sr):
    """Извлекает timbral-texture признаки."""
    try:
        # 1. Spectral centroid (яркость)
        centroids = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        
        # 2. Spectral rolloff (85% энергии)
        rolloffs = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)[0]
        
        # 3. Spectral flux (onset strength)
        flux = librosa.onset.onset_strength(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # 4. Zero-crossing rate (шумность)
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        
        # 5. MFCC (первые 5 коэффициентов)
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=5, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # 6. Low-energy (доля тихих окон)
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        low_energy = np.mean(rms < np.mean(rms))
        
        # Вычисляем статистики по скользящему окну
        cent_mean, cent_var = rolling_stats(centroids)
        roll_mean, roll_var = rolling_stats(rolloffs)
        flux_mean, flux_var = rolling_stats(flux)
        zcr_mean, zcr_var = rolling_stats(zcr)
        
        # MFCC статистики (по каждому коэффициенту)
        mfcc_means = []
        mfcc_vars = []
        for i in range(mfccs.shape[0]):
            mfcc_mean, mfcc_var = rolling_stats(mfccs[i])
            mfcc_means.append(np.mean(mfcc_mean))
            mfcc_vars.append(np.mean(mfcc_var))
        
        # Финальные признаки
        timbral_features = {
            'centroid_mean': np.mean(cent_mean),
            'centroid_var': np.mean(cent_var),
            'rolloff_mean': np.mean(roll_mean),
            'rolloff_var': np.mean(roll_var),
            'flux_mean': np.mean(flux_mean),
            'flux_var': np.mean(flux_var),
            'zcr_mean': np.mean(zcr_mean),
            'zcr_var': np.mean(zcr_var),
            'low_energy': low_energy
        }
        
        # Добавляем MFCC признаки
        for i in range(5):
            timbral_features[f'mfcc{i+1}_mean'] = mfcc_means[i]
            timbral_features[f'mfcc{i+1}_var'] = mfcc_vars[i]
        
        return timbral_features
        
    except Exception as e:
        logger.error(f"Ошибка извлечения timbral признаков: {e}")
        return None

def extract_rhythmic_features(y, sr):
    """Извлекает rhythmic (beat-histogram) признаки."""
    try:
        # 1. Onset envelope
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        
        # 2. Tempogram (автокорреляция огибающей)
        tempogram = librosa.feature.tempogram(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH)
        
        # 3. Beat histogram (сумма по временной оси)
        bh = np.sum(tempogram, axis=1)
        
        # Нормализация
        bh_sum = np.sum(bh)
        if bh_sum > 0:
            bh_norm = bh / bh_sum
        else:
            bh_norm = bh
        
        # Находим пики
        peaks, _ = librosa.effects.hpss(y)
        if len(bh_norm) > 0:
            # Находим два максимальных пика
            peak_indices = np.argsort(bh_norm)[-2:][::-1]
            A0 = bh_norm[peak_indices[0]] if len(peak_indices) > 0 else 0
            A1 = bh_norm[peak_indices[1]] if len(peak_indices) > 1 else 0
            RA = A1 / A0 if A0 > 0 else 0
            
            # Конвертируем индексы в BPM
            # tempogram имеет lags от 1 до max_bpm
            max_bpm = 200
            min_bpm = 40
            lag_to_bpm = lambda lag: max_bpm / (lag + 1)
            
            P0 = lag_to_bpm(peak_indices[0]) if len(peak_indices) > 0 else 0
            P1 = lag_to_bpm(peak_indices[1]) if len(peak_indices) > 1 else 0
        else:
            A0, A1, RA, P0, P1 = 0, 0, 0, 0, 0
        
        rhythmic_features = {
            'beat_hist_A0': A0,
            'beat_hist_A1': A1,
            'beat_hist_RA': RA,
            'beat_hist_P0': P0,
            'beat_hist_P1': P1,
            'beat_hist_SUM': bh_sum
        }
        
        return rhythmic_features
        
    except Exception as e:
        logger.error(f"Ошибка извлечения rhythmic признаков: {e}")
        return None

def extract_harmonic_features(y, sr):
    """Извлекает harmonic/pitch признаки."""
    try:
        # 1. Fundamental frequency tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, 
            frame_length=FRAME_LENGTH, 
            hop_length=HOP_LENGTH
        )
        
        # 2. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
        
        # 3. Unfolded pitch histogram (MIDI ноты)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            midi = librosa.hz_to_midi(valid_f0)
            uph, _ = np.histogram(midi, bins=range(0, 128), density=False)
        else:
            uph = np.zeros(128)
        
        # 4. Folded pitch histogram (chroma)
        fph = np.mean(chroma, axis=1)  # 12-мерный folded
        
        # 5. Признаки
        FA0 = np.max(fph) if len(fph) > 0 else 0  # Сила ведущей pitch-class
        UP0 = np.argmax(uph) if np.sum(uph) > 0 else 0  # Номер MIDI ноты
        FP0 = np.argmax(fph) if len(fph) > 0 else 0  # Номер pitch-class
        
        # IPO1 (простейшая delta)
        if len(fph) > 1:
            fph_shifted = np.roll(fph, 1)
            IPO1 = np.argmax(fph_shifted) - FP0
        else:
            IPO1 = 0
        
        SUM = np.sum(uph)  # Общая энергия pitch detection
        
        harmonic_features = {
            'pitch_FA0': FA0,
            'pitch_UP0': UP0,
            'pitch_FP0': FP0,
            'pitch_IPO1': IPO1,
            'pitch_SUM': SUM
        }
        
        return harmonic_features
        
    except Exception as e:
        logger.error(f"Ошибка извлечения harmonic признаков: {e}")
        return None

def extract_features_from_track(track_info, audio_dir, cache):
    """Извлекает все признаки из одного трека."""
    start_time = time.time()
    
    try:
        track_name = track_info['track_name']
        artists = track_info['artists']
        track_key = get_track_key(track_info)
        
        # Проверяем кеш
        if track_key in cache:
            elapsed_time = time.time() - start_time
            logger.info(f"Трек найден в кеше: {track_name} - {artists} (время: {elapsed_time:.2f}с)")
            return cache[track_key]['features']
        
        # Находим аудио файл
        audio_path = find_audio_file(track_info, audio_dir)
        if not audio_path:
            logger.error(f"Аудио файл не найден: {track_name} - {artists}")
            return None
        
        logger.info(f"Обработка: {track_name} - {artists}")
        
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        
        # Извлекаем признаки
        timbral_features = extract_timbral_features(y, sr)
        rhythmic_features = extract_rhythmic_features(y, sr)
        harmonic_features = extract_harmonic_features(y, sr)
        
        if timbral_features is None or rhythmic_features is None or harmonic_features is None:
            logger.error(f"Ошибка извлечения признаков: {track_name} - {artists}")
            return None
        
        # Объединяем все признаки
        all_features = {**timbral_features, **rhythmic_features, **harmonic_features}
        
        # Добавляем в кеш
        cache[track_key] = {
            'track_name': track_name,
            'artists': artists,
            'features': all_features,
            'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Успешно обработан: {track_name} - {artists} (время: {elapsed_time:.2f}с)")
        
        return all_features
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Исключение при обработке '{track_name}' (время: {elapsed_time:.2f}с): {str(e)}")
        return None

def extract_all_features():
    """Извлекает признаки из всех треков."""
    
    overall_start_time = time.time()
    
    # Загружаем кеш
    cache = load_extraction_cache()
    
    # Читаем CSV файл
    try:
        df = pd.read_csv(CSV_FILE)
        logger.info(f"Загружено {len(df)} треков из CSV файла")
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV файла: {e}")
        return
    
    # Конвертируем в список словарей
    tracks_to_process = df.to_dict('records')
    
    # Фильтруем треки, которые нужно обработать
    tracks_to_extract = []
    already_processed = []
    
    for track in tracks_to_process:
        track_key = get_track_key(track)
        if track_key in cache:
            already_processed.append(track)
        else:
            tracks_to_extract.append(track)
    
    logger.info(f"Уже обработано: {len(already_processed)} треков")
    logger.info(f"Требует обработки: {len(tracks_to_extract)} треков")
    
    if not tracks_to_extract:
        logger.info("Все треки уже обработаны!")
        overall_elapsed_time = time.time() - overall_start_time
        logger.info(f"Общее время выполнения: {overall_elapsed_time:.2f}с")
        return
    
    # Обрабатываем треки
    successful_extractions = []
    failed_extractions = []
    
    logger.info(f"\nНачинаем извлечение признаков из {len(tracks_to_extract)} треков...")
    
    with tqdm(total=len(tracks_to_extract), desc="Извлечение признаков") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Создаем задачи
            future_to_track = {
                executor.submit(extract_features_from_track, track, AUDIO_DIR, cache): track
                for track in tracks_to_extract
            }
            
            # Обрабатываем результаты
            for future in as_completed(future_to_track):
                track = future_to_track[future]
                try:
                    result = future.result()
                    if result:
                        successful_extractions.append({
                            'track_name': track['track_name'],
                            'artists': track['artists'],
                            'track_genre': track.get('track_genre', 'unknown'),
                            'features': result
                        })
                    else:
                        failed_extractions.append({
                            'track_name': track['track_name'],
                            'artists': track['artists'],
                            'track_genre': track.get('track_genre', 'unknown'),
                            'error': 'Feature extraction failed'
                        })
                except Exception as e:
                    failed_extractions.append({
                        'track_name': track['track_name'],
                        'artists': track['artists'],
                        'track_genre': track.get('track_genre', 'unknown'),
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    # Сохраняем обновленный кеш
    save_extraction_cache(cache)
    
    # Создаем итоговый DataFrame, сохраняя все исходные данные
    result_df = df.copy()  # Копируем исходный DataFrame
    
    # Добавляем новые признаки к уже обработанным трекам из кеша
    for track in already_processed:
        track_key = get_track_key(track)
        if track_key in cache:
            features = cache[track_key]['features']
            # Находим индекс трека в DataFrame
            mask = (result_df['track_name'] == track['track_name']) & (result_df['artists'] == track['artists'])
            if mask.any():
                idx = mask.idxmax()
                for feature_name, feature_value in features.items():
                    result_df.at[idx, feature_name] = feature_value
    
    # Добавляем новые обработанные треки
    for extraction in successful_extractions:
        # Находим индекс трека в DataFrame
        mask = (result_df['track_name'] == extraction['track_name']) & (result_df['artists'] == extraction['artists'])
        if mask.any():
            idx = mask.idxmax()
            for feature_name, feature_value in extraction['features'].items():
                result_df.at[idx, feature_name] = feature_value
    
    # Сохраняем результат
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    # Выводим статистику
    overall_elapsed_time = time.time() - overall_start_time
    
    logger.info("\n=== Итоговая статистика ===")
    logger.info(f"Всего треков в CSV: {len(tracks_to_process)}")
    logger.info(f"Уже было обработано: {len(already_processed)}")
    logger.info(f"Новых успешных обработок: {len(successful_extractions)}")
    logger.info(f"Ошибок обработки: {len(failed_extractions)}")
    logger.info(f"Общее время выполнения: {overall_elapsed_time:.2f}с")
    logger.info(f"Результаты сохранены в: {OUTPUT_CSV}")
    logger.info(f"Кеш обновлен: {CACHE_FILE}")
    
    # Информация о признаках
    original_columns = ['track_id', 'artists', 'album_name', 'track_name', 'popularity', 
                       'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
                       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
                       'liveness', 'valence', 'tempo', 'time_signature', 'track_genre',
                       'entropy_amplitude', 'complexity_amplitude', 'entropy_flux', 'complexity_flux',
                       'entropy_harmony', 'complexity_harmony', 'entropy_spectral', 'complexity_spectral',
                       'higuchi_fd', 'box_counting_fd']
    
    feature_columns = [col for col in result_df.columns if col not in original_columns]
    logger.info(f"Извлечено новых признаков: {len(feature_columns)}")
    logger.info(f"Новые признаки: {', '.join(feature_columns)}")
    
    # Статистика по заполненности признаков
    filled_tracks = 0
    for feature in feature_columns:
        filled_count = result_df[feature].notna().sum()
        if filled_count > 0:
            filled_tracks = max(filled_tracks, filled_count)
    
    logger.info(f"Треков с извлеченными признаками: {filled_tracks}")
    
    if failed_extractions:
        logger.info("\nТреки с ошибками обработки:")
        for extraction in failed_extractions[:10]:
            logger.info(f"  - {extraction['track_name']} by {extraction['artists']} ({extraction['track_genre']}): {extraction['error']}")
        if len(failed_extractions) > 10:
            logger.info(f"  ... и еще {len(failed_extractions) - 10} треков")
    
    return result_df

if __name__ == "__main__":
    logger.info("Запуск скрипта извлечения признаков")
    
    # Проверяем существование файлов
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV файл не найден: {CSV_FILE}")
        sys.exit(1)
    
    if not os.path.exists(AUDIO_DIR):
        logger.error(f"Директория с аудио не найдена: {AUDIO_DIR}")
        sys.exit(1)
    
    # Запускаем извлечение признаков
    try:
        result_df = extract_all_features()
        logger.info("Извлечение признаков завершено!")
    except KeyboardInterrupt:
        logger.info("Извлечение признаков прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1) 
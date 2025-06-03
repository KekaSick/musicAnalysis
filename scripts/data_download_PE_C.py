import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import subprocess
from tqdm import tqdm
import librosa
import antropy as ent
import scipy.signal as sig
import math
import subprocess
import ordpy
from scipy.fft import fft, ifft
from scipy.signal import hilbert, stft
from scipy.io import wavfile
import soundfile as sf
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────── 1. Audio Loader ──────────────────────
def _load_mono_audio(file_path):
    """
    Считываем WAV (scipy) или любой другой формат (soundfile/librosa backend)
    и нормализуем до float32 [-1, 1].
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        sr, data = wavfile.read(file_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0:          # int16 → float
            data /= np.abs(data).max() + 1e-9
        logger.info(f"WAV '{file_path}' loaded, sr={sr}")
    else:  # mp3, flac, …
        data, sr = sf.read(file_path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        logger.info(f"Audio '{file_path}' loaded via soundfile, sr={sr}")
    return data, sr


# ───────────────────────── helpers ────────────────────────────
def _frame_signal(x, frame_len, hop_len):
    """np.lib.stride_tricks.sliding_window_view wrapper → shape (n_frames, frame_len)"""
    if len(x) < frame_len:
        return np.empty((0, frame_len))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len]
    return frames


def _ordpy_pe_ce(ts, dim, tau):
    """Безопасный вызов ordpy.complexity_entropy с нормализацией."""
    if len(ts) <= dim:
        return np.nan, np.nan
    H, C = ordpy.complexity_entropy(ts, dx=dim, taux=tau)
    return H, C


# ═══════════════════ 2. Амплитудный хаос ═══════════════════════
def ordpy_amplitude(file_path, dim_size=6, hop_size=2,
                    win_len=2048, hop_len=1024):
    """Hilbert envelope → окно-среднее → PE-CE."""
    # 1. Загрузка сигнала
    x, sr = _load_mono_audio(file_path)

    # 2. Амплитудная огибающая
    env = np.abs(hilbert(x))

    # 3. Фрейминг и RMS
    frames = _frame_signal(env, win_len, hop_len)
    env_rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)

    # 4. Логарифм + нормализация
    env_log = np.log1p(env_rms)
    env_norm = (env_log - env_log.mean()) / (env_log.std() + 1e-12)

    return _ordpy_pe_ce(env_norm, dim_size, hop_size)


# ═══════════════════ 3. Тембровый флюкс ════════════════════════
def ordpy_flux(file_path, dim_size=6, hop_size=2,
               n_fft=2048, hop_len=512, smooth=7):
    """
    STFT через scipy.signal.stft → spectral flux →
    ordpy PE-CE.
    """
    x, sr = _load_mono_audio(file_path)

    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_len,
                   window="hann", padded=False, boundary=None)
    mag = np.abs(Z)                         # shape (n_bins, n_frames)
    if mag.shape[1] < 2:
        return np.nan, np.nan
      # ── FLUX ──────────────────────────────────────────────────────────
    # 1) лог-магнитуда смягчает пики
    mag_log = np.log1p(np.abs(Z))
    diff    = np.diff(mag_log, axis=1)
    flux    = np.sqrt((diff**2).sum(axis=0))   # 1-D

    # 2) сглаживание (rolling mean)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=float) / smooth
        flux = np.convolve(flux, kernel, mode="valid")
    diff = np.diff(mag, axis=1)

    return _ordpy_pe_ce(flux, dim_size, hop_size)


# ═══════════════ 4. Гармонический хаос (CQT-хрома) ════════════════
def ordpy_harmony(file_path,
                  dim_size: int = 6,
                  hop_size: int = 1,
                  bins_per_octave: int = 36,   # ≥ 12  (кратно 12!)
                  hop_len: int = 2048,
                  n_octaves: int = 7,          # совокупно 7 октав ≈ 32 Гц → 4 кГц
                  fmin: float | None = None,
                  use_auto_tempo = False):
    """
    Constant-Q chroma → угол на круге квинт → PE-CE.

    • CQT даёт полутоновую лог-сетку: одинаковое разрешение для баса и верха.  
    • Суммируем все октавы → 12-мерная хрома.  
    • Переводим хрому в комплексный «вектор тональности» и берём аргумент.  
    • Ряд углов скармливаем в ordpy.complexity_entropy.

    Параметры
    ---------
    bins_per_octave : напр. 36 (три бина на полутон) — хороший баланс
    hop_len         : 1024 сэмпл @ 44.1 кГц → 23 мс / кадр
    n_octaves       : сколько октав покрыть (C1–C8 ≈ 7 октав)
    fmin            : нижняя нота C1 (≈ 32.7 Гц) по умолчанию
    """

    if bins_per_octave is None:
        bins_per_octave = 36
    if n_octaves is None:
        n_octaves = 6
    if hop_len is None:
        hop_len = 1024

    # ── LOAD ──
    x, sr = _load_mono_audio(file_path)

    # ── TEMPO-АВТО ──
    if use_auto_tempo:
        tempo, _ = librosa.beat.beat_track(y=x, sr=sr)
        quarter_sec = 60.0 / tempo[0] if isinstance(tempo, np.ndarray) else 60.0 / tempo # Ensure tempo is scalar
        hop_len = 2 ** round(math.log2(float(sr * quarter_sec / 4)))
        print(hop_len)
    else:
        hop_len = hop_len or 1024

    # ── CQT + хрома (как в предыдущем коде) ──
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    n_bins = bins_per_octave * n_octaves
    C_mag = np.abs(librosa.cqt(x, sr=sr, hop_length=hop_len,
                               fmin=fmin, n_bins=n_bins,
                               bins_per_octave=bins_per_octave))**2
    chroma = np.zeros((12, C_mag.shape[1]), dtype=np.float32)
    for pc in range(12):
        chroma[pc] = C_mag[pc::bins_per_octave].sum(axis=0)

    angles = np.arange(12) * 2*np.pi/12
    vector  = (chroma * np.exp(1j*angles[:, None])).sum(axis=0)
    phi     = np.angle(vector)

    return _ordpy_pe_ce(phi, dim_size, hop_size)


# ═════════════════ 5. «Энтропия энтропии» (SE ➜ PE) ═══════════
def ordpy_specentropy(file_path,
                      dim_size: int = 6,
                      hop_size: int = 1,
                      N_FFT: int = 2048,
                      HOP_LENGTH_STFT: int = 512):
    """
    • Фреймируем аудио (), считаем spectral-entropy на каждом кадре
      через antropy.spectral_entropy(normalize=True, method='fft').
    • Передаём полученный 1-D ряд в ordpy.complexity_entropy.
    • Без fallback-ов: если antropy не отработает — будет исключение.

    Returns
    -------
    (H_norm, C)  or (np.nan, np.nan)  если ряд < dim_size.
    """
   # Параметры для фреймирования аудио и STFT (в случае фоллбэка)

    # ── загрузка ────────────────────────────────────────────────────
    ext = os.path.splitext(file_path)[1].lower()
    data, sr = _load_mono_audio(file_path)

    spectral_entropy_ts = None

    try:
        # ── СПЕКТРАЛЬНАЯ ЭНТРОПИЯ (antropy, основной метод) ───────────
        # Расчет на фреймах сырого аудиосигнала
        logger.debug(f"Attempting antropy.spectral_entropy (on raw audio frames) for {file_path}")
        
        # 1. Фреймирование сырого аудио
        # axis=0 для формата (длина_фрейма, количество_фреймов)
        audio_frames = librosa.util.frame(data, frame_length=N_FFT, hop_length=HOP_LENGTH_STFT, axis=0)
        
        # 2. Применение antropy.spectral_entropy к каждому аудио-фрейму
        spectral_entropy_ts = np.apply_along_axis(
            lambda audio_segment: ent.spectral_entropy(
                audio_segment, 
                sf=sr, 
                method="fft", 
                normalize=True  # Как в вашем коде, результат 0-1
            ) if np.any(audio_segment) else 0.0, # Обработка тихих фреймов
            axis=0, # Применить к каждому столбцу (аудио-фрейму)
            arr=audio_frames
        )
        logger.info(f"Successfully used antropy.spectral_entropy (on raw audio frames) for {file_path}")

    except (AttributeError, NameError, ImportError, ValueError) as e_antropy:
        logger.warning(
            f"antropy.spectral_entropy (on raw audio frames) failed for {file_path} ({type(e_antropy).__name__}: {e_antropy}). "
            "Falling back to manual spectral entropy calculation using librosa.stft."
        )
        # ── СПЕКТРАЛЬНАЯ ЭНТРОПИЯ (ручной STFT-based, фоллбэк) ──────
        # 1. STFT и магнитудный спектр
        S_complex = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LENGTH_STFT)
        S_mag = np.abs(S_complex)**2 # Магнитудный спектр
        
        # 2. Нормализация и расчет энтропии Шеннона для каждого фрейма S_mag
        epsilon = np.finfo(S_mag.dtype).eps if S_mag.dtype in [np.float32, np.float64] else np.finfo(float).eps
        prob_dist = S_mag / (np.sum(S_mag, axis=0, keepdims=True) + epsilon)
        prob_dist = np.nan_to_num(prob_dist, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Энтропия Шеннона (в битах)
        entropy_values_bits = -np.sum(prob_dist * np.log2(prob_dist + epsilon), axis=0)
        
        # Нормализация (0-1), если хотим соответствовать antropy c normalize=True
        # np.log2(S_mag.shape[0]) это log2(количество_частотных_бинов)
        # Добавляем epsilon в знаменатель для стабильности, если S_mag.shape[0] <= 1
        num_freq_bins = S_mag.shape[0]
        if num_freq_bins > 1:
            spectral_entropy_ts = entropy_values_bits / (np.log2(num_freq_bins) + epsilon)
        else: # Если всего 1 бин, энтропия должна быть 0 (после нормализации)
            spectral_entropy_ts = np.zeros_like(entropy_values_bits)
        
        logger.info(f"Used manual STFT-based calculation for spectral entropy for {file_path}")

    # Обработка NaN, если что-то пошло не так
    if spectral_entropy_ts is None:
        logger.error(f"Spectral entropy calculation failed completely for {file_path}. Defaulting to zeros.")
        num_expected_frames = int(np.floor((len(data) - N_FFT) / HOP_LENGTH_STFT)) + 1 if len(data) >= N_FFT else 1
        spectral_entropy_ts = np.zeros(num_expected_frames)
    
    spectral_entropy_ts = np.nan_to_num(spectral_entropy_ts) # Финальная очистка

    # ── проверки перед ordpy ───────────────────────────────────────
    logger.info(f"Length of spectral_entropy_ts for {file_path} before ordpy: {len(spectral_entropy_ts)}") # DEBUG PRINT
    if len(spectral_entropy_ts) <= dim_size:
        logger.warning(f"{file_path}: series (len={len(spectral_entropy_ts)}) too short for ordpy → NaN")
        return np.nan, np.nan

    logger.info(f"Calling ordpy.complexity_entropy for {file_path} with dim_size={dim_size}, hop_size(taux)={hop_size}") # DEBUG PRINT
 
    return _ordpy_pe_ce(spectral_entropy_ts, dim_size, hop_size)



def compute_log_mel_spectrogram(x, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Строит мел-спектрограмму и возвращает её логарифм (log1p).
    
    Параметры:
      - x: 1D numpy array, аудио-сигнал
      - sr: частота дискретизации
      - n_fft: размер окна FFT
      - hop_length: шаг (hop length) для STFT
      - n_mels: число мел-банок
    
    Возвращает:
      S_log: 2D numpy array формы (n_mels, n_frames)
    """
    S = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0
    )
    S_log = np.log1p(S)
    return S_log

def binarize_spectrogram(S_log, method="mean"):
    """
    Преобразует лог-мел-спектрограмму в бинарную матрицу 0/1 по порогу.
    
    Параметры:
      - S_log: 2D numpy array (логарифм мел-спектра)
      - method: способ выбора порога. Поддерживается:
          * "median"   — медиана всех значений S_log
          * "mean"     — среднее всех значений S_log
          * float      — прямое число, используем как порог
          * callable   — функция, принимающая S_log и возвращающая порог
    
    Возвращает:
      B: 2D numpy array тех же размеров, dtype=int, с элементами {0,1}
    """
    if isinstance(method, str):
        if method == "median":
            tau = np.median(S_log)
        elif method == "mean":
            tau = np.mean(S_log)
        else:
            raise ValueError("Недопустимый метод порога: choose 'median', 'mean', or provide a float/callable.")
    elif isinstance(method, (int, float)):
        tau = float(method)
    elif callable(method):
        tau = method(S_log)
    else:
        raise ValueError("method должно быть 'median', 'mean', число или callable.")
    
    B = (S_log > tau).astype(int)
    return B

def box_counting_fd_2d(B, epsilons):
    """
    Вычисляет количество занятых ячеек N(ε) для каждой ε методом box-counting по бинарной матрице B.
    
    Параметры:
      - B: 2D numpy array (0/1), shape = (H, W)
      - epsilons: список целых размеров ячейки (EPS), например [1,2,4,8,16,...]
    
    Возвращает:
      N_e: 1D numpy array длины len(epsilons), где N_e[i] = число ячеек размера epsilons[i],
           содержащих хотя бы одну единицу в B.
    """
    H, W = B.shape
    N_e = []
    
    for e in epsilons:
        n_h = math.ceil(H / e)
        n_w = math.ceil(W / e)
        count = 0
        for i_block in range(n_h):
            i0 = i_block * e
            i1 = min(i0 + e, H)
            for j_block in range(n_w):
                j0 = j_block * e
                j1 = min(j0 + e, W)
                # Если в этом блоке есть хотя бы одна «1»
                if np.any(B[i0:i1, j0:j1] > 0):
                    count += 1
        N_e.append(count)
    
    return np.array(N_e)

def compute_box_counting_fd(file_path,
                            n_fft=2048,
                            hop_length=512,
                            n_mels=128,
                            threshold_method="median",
                            epsilons=None):
    """
    Полная функция: загружает WAV, строит лог-мел-спектр, бинаризует, считает FD методом box-counting.
    
    Параметры:
      - file_path: str, путь к WAV-файлу
      - n_fft: int, размер окна FFT для мел-спектрограммы
      - hop_length: int, hop length для мел-спектрограммы
      - n_mels: int, число мел-банок
      - threshold_method: str или float или callable; метод бинаризации ("median", "mean", число или функция)
      - epsilons: list of int; размеры ячеек для box-counting.
                   Если None, по умолчанию берём [1,2,4,8,16,32,64]
    
    Возвращает:
      D_box: float, оценка фрактальной размерности (наклон в регрессии ln N(ε) vs ln(1/ε))
    """
    # 1) Загрузка аудио
    x, sr = _load_mono_audio(file_path)
    if x.ndim != 1:
        x = librosa.to_mono(x)
    
    # 2) Строим лог-мел-спектрограмму
    S_log = compute_log_mel_spectrogram(x, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # 3) Бинаризация
    B = binarize_spectrogram(S_log, method=threshold_method)
    
    # 4) Подготовка epsilons
    if epsilons is None:
        # Максимально возможный размер ячейки не должен превышать min(n_mels, n_frames)
        H, W = B.shape
        max_power = int(math.floor(math.log2(min(H, W))))
        # Генерируем [2^0, 2^1, ..., 2^max_power]
        epsilons = [2 ** i for i in range(max_power + 1)]
    
    # 5) Считаем N(ε) для каждого ε
    N_e = box_counting_fd_2d(B, epsilons)
    
    # 6) Убираем нулевые N(ε) (не валидные точки)
    eps_arr = np.array(epsilons, dtype=float)
    mask = N_e > 0
    if np.sum(mask) < 2:
        return np.nan  # недостаточно точек для регрессии
    
    ln_eps = np.log(1.0 / eps_arr[mask])   # ln(1/ε)
    ln_Ne  = np.log(N_e[mask])             # ln N(ε)
    
    # 7) Линейная регрессия ln N(ε) = D * ln(1/ε) + const
    #    => D = slope
    slope, intercept = np.polyfit(ln_eps, ln_Ne, 1)
    D_box = slope
    
    return D_box

def higuchi_fd(x, k_max):
    """
    Оценивает фрактальную размерность сигнала x методом Хигучи.
    x: 1D numpy array, длина N
    k_max: максимальный шаг (обычно N//10 или N//20)

    Возвращает: fd (скаляр) и (лог k, лог L(k)) для отладки/построения.
    """
    N = x.shape[0]
    Lk = np.zeros(k_max)
    ln_k = np.zeros(k_max)
    ln_Lk = np.zeros(k_max)

    for k in range(1, k_max+1):
        Lm = np.zeros(k)
        for m in range(k):
            # сколько точек в подпоследовательности при этом k
            num = (N - m - 1) // k
            if num < 1:
                continue
            # вычисляем абсолютные разности вдоль этой подпоследовательности
            idxs = m + np.arange(num + 1) * k
            # разности |x[idxs[i+1]] - x[idxs[i]]|
            dist_sum = np.sum(np.abs(x[idxs[1:] ] - x[idxs[:-1]]))
            # нормировка
            Lm[m] = (dist_sum * (N - 1)) / (num * k)
        # если все Lm == 0 (слишком коротко), оставим 0
        Lk[k-1] = np.mean(Lm[np.nonzero(Lm)] ) if np.any(Lm) else 0
        ln_k[k-1] = np.log(1.0 / k)
        ln_Lk[k-1] = np.log(Lk[k-1]) if Lk[k-1] > 0 else 0

    # Берём лишь те k, где Lk>0
    mask = Lk > 0
    # Линейная регрессия: ln_Lk = D * ln_k + b => D = slope
    D, b = np.polyfit(ln_k[mask], ln_Lk[mask], 1)
    # Фрактальная размерность ≈ D
    return D, (ln_k[mask], ln_Lk[mask])


# Default paths
AUDIO_OUTPUT_DIR = os.path.join("data", "music", "audio")
RESULTS_PATH = os.path.join("data", "csv_spotify", "csv", "fractal_dimensions.csv")
MISSING_TRACKS_JSON = os.path.join("data", "music", "tracks_to_download.json")

def sanitize_filename(filename):
    """Remove invalid characters from filename, and truncate if too long."""
    sanitized = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in filename).rstrip()
    sanitized = sanitized.replace(" ", "_")
    max_len = 100  # Max length for a filename component
    return sanitized[:max_len]

def download_track(track_info, output_dir):
    """Download a track using youtube-dl."""
    try:
        track_name = track_info['track_name']
        artists = track_info['artists']
        genre = track_info.get('track_genre', 'unknown')
        
        # Create a sanitized filename
        filename = f"{genre}_{sanitize_filename(track_name)}_{sanitize_filename(artists)}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return output_path
            
        search_query = f"{track_name} {artists}"
        logger.info(f"Downloading: {track_name} {artists} (Genre: {genre})")
        
        # Use yt-dlp instead of youtube-dl for better reliability
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
        logger.error(f"Exception while downloading '{track_name}': {str(e)}")
        return None

def process_audio_file(wav_file, track_info):
    """Process a single audio file to extract features."""
    try:
        # Calculate entropy and complexity measures
        h_amp, c_amp = ordpy_amplitude(wav_file)
        h_flux, c_flux = ordpy_flux(wav_file)
        h_harm, c_harm = ordpy_harmony(wav_file)
        h_spec, c_spec = ordpy_specentropy(wav_file)
        
        # Return all features as a dictionary
        return {
            'entropy_amplitude': h_amp,
            'complexity_amplitude': c_amp,
            'entropy_flux': h_flux,
            'complexity_flux': c_flux,
            'entropy_harmony': h_harm,
            'complexity_harmony': c_harm,
            'entropy_spectral': h_spec,
            'complexity_spectral': c_spec
        }
    except Exception as e:
        logger.error(f"Error processing {track_info['track_name']}: {str(e)}")
        return None

def get_track_filename(track_info):
    """Generate expected filename for a track."""
    genre = sanitize_filename(track_info['track_genre'])
    track_name = sanitize_filename(track_info['track_name'])
    artist_name = sanitize_filename(track_info['artists'])
    return f"{genre}_{track_name}_{artist_name}.wav"

def find_existing_track(track_info, output_dir):
    """Find if track already exists in the output directory."""
    expected_filename = get_track_filename(track_info)
    expected_filename_no_ext = expected_filename[:-4]  # remove .wav
    
    for file in os.listdir(output_dir):
        if file.startswith(expected_filename_no_ext) and file.endswith('.wav'):
            return os.path.join(output_dir, file)
    return None

def clean_and_sample_dataset():
    """Clean dataset and calculate fractal dimensions for each track."""
    # Create output directories
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    # Read the dataset
    try:
        df = pd.read_csv('data/csv_spotify/csv/sampled_dataset_with_fd.csv')
        logger.info(f"Loaded {len(df)} tracks from dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Convert DataFrame to list of dicts
    tracks_to_process = df.to_dict('records')
    
    # Step 1: Check existing files
    existing_files = {}  # track_name -> wav_file_path
    tracks_to_download = []
    
    logger.info("Checking for existing files...")
    for track in tracks_to_process:
        existing_file = find_existing_track(track, AUDIO_OUTPUT_DIR)
        if existing_file:
            logger.info(f"Found existing file for: {track['track_name']}")
            existing_files[track['track_name']] = existing_file
        else:
            tracks_to_download.append(track)
    
    logger.info(f"Found {len(existing_files)} existing tracks")
    logger.info(f"Missing {len(tracks_to_download)} tracks (skipping download)")
    
    # Save tracks to download info to JSON for future reference
    tracks_to_download_info = {
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tracks_in_dataset": len(tracks_to_process),
        "tracks_to_download": [
            {
                "track_name": track["track_name"],
                "artists": track["artists"],
                "track_genre": track["track_genre"],
                "search_query": f"{track['track_name']} {track['artists']}"
            }
            for track in tracks_to_download
        ],
        "statistics": {
            "total_tracks": len(tracks_to_process),
            "already_downloaded": len(existing_files),
            "to_download": len(tracks_to_download),
            "by_genre": {}
        }
    }
    
    # Add genre statistics
    genre_stats = {}
    for track in tracks_to_download:
        genre = track["track_genre"]
        if genre not in genre_stats:
            genre_stats[genre] = 0
        genre_stats[genre] += 1
    tracks_to_download_info["statistics"]["by_genre"] = genre_stats
    
    # Save to JSON with pretty formatting
    with open(MISSING_TRACKS_JSON, 'w', encoding='utf-8') as f:
        json.dump(tracks_to_download_info, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved tracks to download info to {MISSING_TRACKS_JSON}")
    
    # Process existing files
    results = []
    processing_failed = []
    
    logger.info("\nProcessing existing audio files...")
    with tqdm(total=len(existing_files), desc="Processing tracks") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 3) as executor:
            future_to_track = {
                executor.submit(process_audio_file, wav_file, track): (wav_file, track)
                for track_name, wav_file in existing_files.items()
                for track in tracks_to_process if track['track_name'] == track_name
            }
            
            for future in as_completed(future_to_track):
                wav_file, track = future_to_track[future]
                try:
                    result, error = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Processed: {track['track_name']}")
                    else:
                        processing_failed.append((track, error))
                        logger.warning(f"Failed to process: {track['track_name']} - {error}")
                except Exception as e:
                    processing_failed.append((track, str(e)))
                    logger.error(f"Error processing {track['track_name']}: {e}")
                
                pbar.update(1)

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        # Sort by genre to maintain the order
        results_df = results_df.sort_values('track_genre').reset_index(drop=True)
        results_df.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Saved results to {RESULTS_PATH}")

    # Print summary
    logger.info("\n=== Processing Summary ===")
    logger.info(f"Total tracks in dataset: {len(tracks_to_process)}")
    logger.info(f"Available tracks: {len(existing_files)}")
    logger.info(f"Successfully processed: {len(results)} tracks")
    logger.info(f"Processing failed: {len(processing_failed)} tracks")
    logger.info(f"Missing tracks: {len(tracks_to_download)} tracks")
    
    if processing_failed:
        logger.info("\nProcessing failed tracks:")
        for track, error in processing_failed:
            logger.info(f"  - {track['track_name']} by {track['artists']} ({track['track_genre']}): {error}")

def process_single_track_fd(args):
    """Process a single track to calculate its fractal dimensions."""
    idx, track = args
    
    track_info = {
        'track_id': track['track_id'],
        'artists': track['artists'],
        'track_name': track['track_name'],
        'track_genre': track['track_genre']
    }
    
    try:
        # Download the track
        output_dir = os.path.join('data', 'audio')
        wav_file = download_track(track_info, output_dir)
        if wav_file is None or not os.path.exists(wav_file):
            logger.warning(f"Failed to download track {track['track_id']}")
            return idx, None
        
        # Calculate fractal dimensions
        # 1. Box-counting FD from spectrogram
        box_fd = compute_box_counting_fd(wav_file)
        
        # 2. Higuchi FD from raw signal
        x, sr = _load_mono_audio(wav_file)
        higuchi_fd_value, _ = higuchi_fd(x, 200)  # Fixed k_max value
        
        results = {
            'higuchi_fd': higuchi_fd_value,
            'box_counting_fd': box_fd
        }
        
        logger.info(f"Calculated FDs for {track['track_name']}: Higuchi={higuchi_fd_value:.3f}, Box-counting={box_fd:.3f}")
        return idx, results
        
    except Exception as e:
        logger.error(f"Error processing track {track['track_id']}: {str(e)}")
        return idx, None

def process_missing_tracks():
    """
    Find tracks with missing fractal dimensions (Higuchi FD and Box-counting FD),
    download them, calculate the missing values, and update the CSV file.
    """
    # Load the dataset
    csv_path = 'data/csv_spotify/csv/sampled_dataset_with_fd.csv'
    df = pd.read_csv(csv_path)
    
    # Find tracks with missing fractal dimensions
    fd_cols = ['higuchi_fd', 'box_counting_fd']
    missing_tracks = df[df[fd_cols].isna().any(axis=1)].copy()
    logger.info(f"Found {len(missing_tracks)} tracks with missing fractal dimensions")
    
    if len(missing_tracks) == 0:
        logger.info("No tracks with missing fractal dimensions found")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('data', 'audio')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process tracks in parallel with progress bar
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    n_cores = multiprocessing.cpu_count()
    logger.info(f"Using {n_cores} CPU cores for parallel processing")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, row) for idx, row in missing_tracks.iterrows()]
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_single_track_fd, args) for args in args_list]
        
        # Process results as they complete with progress bar
        from tqdm import tqdm
        for future in tqdm(futures, total=len(missing_tracks), 
                          desc="Processing tracks", unit="track"):
            try:
                idx, results = future.result()
                if results is not None:
                    # Update the dataframe with new values
                    for key, value in results.items():
                        df.loc[idx, key] = value
                    # Save after each successful track processing
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Updated fractal dimensions for track at index {idx}")
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}")
                continue
    
    logger.info("Dataset update completed")

if __name__ == "__main__":
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    process_missing_tracks()

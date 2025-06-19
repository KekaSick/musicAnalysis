import os, logging
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, stft
import ordpy
import soundfile as sf
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import antropy as ent
import math
import librosa
import antropy as ent


logger = logging.getLogger(__name__)


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
    # FLUX 
    # 1) лог-магнитуда смягчает пики
    mag_log = np.log1p(np.abs(Z))
    diff    = np.diff(mag_log, axis=1)
    flux    = np.sqrt((diff**2).sum(axis=0))   # 1-D

    # 2) сглаживание (rolling mean)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=float) / smooth
        flux = np.convolve(flux, kernel, mode="valid")

    return _ordpy_pe_ce(flux, dim_size, hop_size)



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

    x, sr = _load_mono_audio(file_path)

    # TEMPO-АUТО
    if use_auto_tempo:
        tempo, _ = librosa.beat.beat_track(y=x, sr=sr)
        quarter_sec = 60.0 / tempo[0] if isinstance(tempo, np.ndarray) else 60.0 / tempo # Ensure tempo is scalar
        hop_len = 2 ** round(math.log2(float(sr * quarter_sec / 4)))
        print(hop_len)
    else:
        hop_len = hop_len or 1024

    # CQT + хрома
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


# «Энтропия энтропии» (SE ➜ PE)   
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

    # загрузка 
    ext = os.path.splitext(file_path)[1].lower()
    data, sr = _load_mono_audio(file_path)

    spectral_entropy_ts = None

    # СПЕКТРАЛЬНАЯ ЭНТРОПИЯ (antropy, основной метод) 
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


def ordpy_process_folder(folder_path, method_name, dim_size=6, hop_size=1):
    """
    Processes all audio files in a folder using the specified ordpy method.

    Args:
        folder_path (str): Path to the folder containing audio files.
        method_name (str): Name of the method to use.
                           Valid options: "amplitude", "flux", "harmony", "specentropy".
        dim_size (int): Dimension size for ordpy calculations (passed to the method).
        hop_size (int): Hop size for ordpy calculations (passed to the method).
    """
    METHODS = {
        "amplitude": ordpy_amplitude,
        "flux": ordpy_flux,
        "harmony": ordpy_harmony,
        "specentropy": ordpy_specentropy,
    }

    if method_name not in METHODS:
        logger.error(f"Invalid method_name: {method_name}. Available methods: {list(METHODS.keys())}")
        return [], [], []

    selected_method = METHODS[method_name]

    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    mp3_files = glob.glob(os.path.join(folder_path, "*.mp3"))
    all_files = wav_files + mp3_files

    if not all_files:
        logger.info(f"No audio files in {folder_path}")
        return [], [], []

    Hs, Cs, labels = [], [], []
    for file_path in tqdm(all_files,
                          desc=f"{folder_path} method={method_name} dim={dim_size} hop={hop_size}"):
        try:
            H, C = selected_method(file_path, dim_size=dim_size, hop_size=hop_size)
            Hs.append(H); Cs.append(C); labels.append(os.path.basename(file_path))
        except Exception as e:
            logger.exception(f"Error processing {file_path} with method {method_name}: {e}")
    return Hs, Cs, labels


def plot_graph_ordpy(folder_path, dim, hop, method, folder="plots"):
    logger.info(f"Plotting {folder_path} (dim={dim}, hop={hop})")
    Hs, Cs, labels = ordpy_process_folder(folder_path, method, dim, hop)
    if not labels:
        logger.info("Nothing to plot.")
        return

    df = pd.DataFrame({
        "Entropy": Hs,
        "Complexity": Cs,
        "File": labels
    }).dropna()

    # ordpy границы 
    # Для границ всегда используем tau=1, так как hop в ordpy работает по-другому
    max_HC = ordpy.maximum_complexity_entropy(dim, 1)
    min_HC = ordpy.minimum_complexity_entropy(dim, 1)

    # Matplotlib 
    plt.figure(figsize=(8, 6))
    if not df.empty:
        plt.scatter(df["Entropy"], df["Complexity"], s=70,
                    c="blue", edgecolors="black")
    plt.plot(max_HC[:, 0], max_HC[:, 1], "r--")
    plt.plot(min_HC[:, 0], min_HC[:, 1], "g--")
    plt.xlabel("Normalized Permutation Entropy")
    plt.ylabel("Normalized Complexity")
    plt.title(f"{folder_path}\n(dim={dim}, hop={hop}, method={method})")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.grid(alpha=0.3)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()

    parent_dir = os.path.basename(os.path.dirname(folder_path))
    folder = f"plots/{method}/{parent_dir}/{os.path.basename(folder_path)}"

    os.makedirs(folder, exist_ok=True)
    out_png = os.path.join(folder, f"PE_C_{os.path.basename(folder_path)}_d{dim}_h{hop}.png")
    plt.savefig(out_png); plt.close()
    logger.info(f"Saved plot → {out_png}")

    # Plotly 
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df["Entropy"], y=df["Complexity"],
            mode="markers",
            marker=dict(size=10, color="blue", line=dict(width=1, color="black")),
            hovertext=df["File"],
            name="Files"
        ))
    fig.add_trace(go.Scatter(x=max_HC[:, 0], y=max_HC[:, 1],
                             mode="lines", line=dict(color="red", dash="dash"),
                             name="Max C"))
    fig.add_trace(go.Scatter(x=min_HC[:, 0], y=min_HC[:, 1],
                             mode="lines", line=dict(color="green", dash="dash"),
                             name="Min C"))
    fig.update_layout(title=f"{folder_path} (Plotly)\n(dim={dim}, hop={hop}, method={method})",
                      xaxis_title="Normalized Permutation Entropy",
                      yaxis_title="Normalized Complexity",
                      xaxis=dict(range=[0, 1]),
                      yaxis=dict(range=[0, 1]),
                      template="plotly_white",
                      width=800, height=600)
    fig.show()


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

def compute_box_counting_fd_folder(folder_path,
                                n_fft=2048,
                                hop_length=512,
                                n_mels=128,
                                threshold_method="median",
                                epsilons=None):
    """
    Вычисляет box-counting фрактальную размерность для всех WAV-файлов в папке.
    
    Параметры:
      - folder_path: путь к папке с WAV-файлами
      - n_fft: размер окна FFT для мел-спектрограммы
      - hop_length: шаг для мел-спектрограммы
      - n_mels: число мел-банок
      - threshold_method: метод бинаризации ("median", "mean", число или функция)
      - epsilons: список размеров ячеек. Если None, вычисляется автоматически
    
    Возвращает:
      min_fd  – минимальное значение FD среди всех файлов
      max_fd  – максимальное значение FD среди всех файлов
      std_fd  – стандартное отклонение по FD
      mean_fd – среднее значение FD
      fd_dict – словарь {имя_файла: FD}
    """
    # Проверяем существование папки
    if not os.path.exists(folder_path):
        logger.error(f"Directory not found: {folder_path}")
        return np.nan, np.nan, np.nan, np.nan, {}

    # Получаем список всех wav файлов
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not wav_files:
        logger.warning(f"No .wav files found in {folder_path}")
        return np.nan, np.nan, np.nan, np.nan, {}

    fd_dict = {}
    fds = []

    for file_path in tqdm(wav_files, desc=f"Processing {os.path.basename(folder_path)}"):
        fname = os.path.basename(file_path)
        try:
            fd = compute_box_counting_fd(
                file_path=file_path,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                threshold_method=threshold_method,
                epsilons=epsilons
            )
            
            fd_dict[fname] = fd
            if not np.isnan(fd):
                fds.append(fd)

        except Exception as e:
            logger.exception(f"Error processing {fname}: {e}")
            fd_dict[fname] = np.nan

    fds = np.array(fds)
    if fds.size == 0:
        logger.warning("No valid FD values computed")
        min_fd = max_fd = std_fd = mean_fd = np.nan
    else:
        min_fd = np.min(fds)
        max_fd = np.max(fds)
        mean_fd = np.mean(fds)
        std_fd = np.std(fds)
        
        # Выводим статистику
        logger.info(f"\nFolder statistics for {os.path.basename(folder_path)}:")
        logger.info(f"Mean FD: {mean_fd:.3f} ± {std_fd:.3f}")
        logger.info(f"Range: [{min_fd:.3f}, {max_fd:.3f}]")
        logger.info(f"Processed {len(fds)} files successfully")

    return min_fd, max_fd, std_fd, mean_fd, fd_dict


# ═════════════════ 7. Фрактальная размерность ════════════════════════
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

def compute_folder_higuchi_fd(folder_path, k_max=200):
    """
    Проходит по всем .wav-файлам в указанной папке, вычисляет для каждого 
    фрактальную размерность по методу Хигучи и возвращает статистику.

    Возвращает:
      min_fd  – минимальное значение FD среди всех файлов
      max_fd  – максимальное значение FD среди всех файлов
      std_fd  – стандартное отклонение по FD
      mean_fd – среднее значение FD
      fd_dict – словарь {имя_файла: FD}

    Параметры:
      - folder_path: путь к папке с .wav-файлами
      - k_max: максимальный шаг k для алгоритма Хигучи (по умолчанию 200)
    """
    # Проверяем существование папки
    if not os.path.exists(folder_path):
        logger.error(f"Directory not found: {folder_path}")
        return np.nan, np.nan, np.nan, np.nan, {}

    fd_dict = {}
    fds = []

    # Получаем список всех wav файлов
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not wav_files:
        logger.warning(f"No .wav files found in {folder_path}")
        return np.nan, np.nan, np.nan, np.nan, {}

    for file_path in tqdm(wav_files, desc=f"Processing {os.path.basename(folder_path)}"):
        fname = os.path.basename(file_path)
        try:
            # Загрузка моно-аудио и удаление DC-смещения
            x, sr = librosa.load(file_path, sr=None, mono=True)
            x = x - np.mean(x)

            # Если сигнал слишком короткий, пропускаем
            if x.shape[0] < 2:
                fd = np.nan
            else:
                # higuchi_fd возвращает (D, (ln_k, ln_Lk)), нам нужно только D
                fd = ent.higuchi_fd(x, k_max)

            fd_dict[fname] = fd
            if not np.isnan(fd):
                fds.append(fd)

        except Exception as e:
            logger.exception(f"Error processing {fname}: {e}")
            fd_dict[fname] = np.nan

    fds = np.array(fds)
    if fds.size == 0:
        logger.warning("No valid FD values computed")
        min_fd = max_fd = std_fd = mean_fd = np.nan
    else:
        min_fd = np.min(fds)
        max_fd = np.max(fds)
        mean_fd = np.mean(fds)
        std_fd = np.std(fds)
        
        # Выводим статистику
        logger.info(f"\nFolder statistics for {os.path.basename(folder_path)}:")
        logger.info(f"Mean FD: {mean_fd:.3f} ± {std_fd:.3f}")
        logger.info(f"Range: [{min_fd:.3f}, {max_fd:.3f}]")
        logger.info(f"Processed {len(fds)} files successfully")

    return min_fd, max_fd, std_fd, mean_fd, fd_dict

# # hop 1
# folder_path = "data/genres_30sec/rock"
# dim = 6
# hop = 2
# plot_graph_ordpy(folder_path, dim=dim, hop=hop, method="amplitude")

# # hop 2

# folder_path = "data/genres_30sec/rock"
# dim = 6
# hop = 2
# plot_graph_ordpy(folder_path, dim=dim, hop=hop, method="flux")


# # #hop 1

# folder_path = "data/genres_30sec/country"
# dim = 6
# hop = 1
# plot_graph_ordpy(folder_path, dim=dim, hop=hop, method="harmony")


# # hop 1

# folder_path = "data/genres_30sec/rock"
# dim = 6
# hop = 1
# plot_graph_ordpy(folder_path, dim=dim, hop=hop, method="specentropy")


# fd1 = compute_folder_higuchi_fd("data/genres_30sec/blues", k_max=200)
# print(*fd1)


# # Пример использования новой функции
# folder_path = "data/scales/aeolian/A-1-aeolian-0.wav"
# fd2 = compute_box_counting_fd_folder(
#     folder_path=folder_path,
#     n_fft=2048,
#     hop_length=512,
#     n_mels=128,
#     threshold_method="median"
# )
d = compute_box_counting_fd("data/chords/aug/A-1-aug-chord-0.wav")
print(d)
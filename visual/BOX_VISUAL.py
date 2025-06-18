import numpy as np
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _load_mono_audio(file_path):
    x, sr = librosa.load(file_path, sr=None)
    if x.ndim != 1:
        x = librosa.to_mono(x)
    return x, sr

def compute_log_mel_spectrogram(x, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Builds a mel spectrogram and returns its logarithm (log1p).
    
    Parameters:
      - x: 1D numpy array, audio signal
      - sr: sampling rate
      - n_fft: FFT window size
      - hop_length: hop length for STFT
      - n_mels: number of mel bands
    
    Returns:
      S_log: 2D numpy array of shape (n_mels, n_frames)
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

def binarize_spectrogram(S_log, method="median"):
    """
    Converts log-mel-spectrogram to binary matrix 0/1 using threshold.
    
    Parameters:
      - S_log: 2D numpy array (log mel spectrum)
      - method: threshold selection method. Supported:
          * "median"   — median of all S_log values
          * "mean"     — mean of all S_log values
          * float      — direct number used as threshold
          * callable   — function that takes S_log and returns threshold
    
    Returns:
      B: 2D numpy array of same dimensions, dtype=int, with elements {0,1}
    """
    if isinstance(method, str):
        if method == "median":
            tau = np.median(S_log)
        elif method == "mean":
            tau = np.mean(S_log)
        else:
            raise ValueError("Invalid threshold method: choose 'median', 'mean', or provide a float/callable.")
    elif isinstance(method, (int, float)):
        tau = float(method)
    elif callable(method):
        tau = method(S_log)
    else:
        raise ValueError("method must be 'median', 'mean', number or callable.")
    
    B = (S_log > tau).astype(int)
    return B

def box_counting_fd_2d(B, epsilons):
    """
    Calculates the number of occupied cells N(ε) for each ε using box-counting method on binary matrix B.
    
    Parameters:
      - B: 2D numpy array (0/1), shape = (H, W)
      - epsilons: list of integer cell sizes (EPS), e.g. [1,2,4,8,16,...]
    
    Returns:
      N_e: 1D numpy array of length len(epsilons), where N_e[i] = number of cells of size epsilons[i]
           containing at least one '1' in B.
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
                # If there is at least one '1' in this block
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
    Complete function: loads WAV, builds log-mel-spectrogram, binarizes, calculates FD using box-counting.
    
    Parameters:
      - file_path: str, path to WAV file
      - n_fft: int, FFT window size for mel spectrogram
      - hop_length: int, hop length for mel spectrogram
      - n_mels: int, number of mel bands
      - threshold_method: str or float or callable; binarization method ("median", "mean", number or function)
      - epsilons: list of int; cell sizes for box-counting.
                   If None, defaults to [1,2,4,8,16,32,64]
    
    Returns:
      D_box: float, fractal dimension estimate (slope in regression ln N(ε) vs ln(1/ε))
    """
    # 1) Load audio
    x, sr = _load_mono_audio(file_path)
    if x.ndim != 1:
        x = librosa.to_mono(x)
    
    # 2) Build log-mel-spectrogram
    S_log = compute_log_mel_spectrogram(x, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # 3) Binarization
    B = binarize_spectrogram(S_log, method=threshold_method)
    
    # 4) Prepare epsilons
    if epsilons is None:
        # Maximum possible cell size should not exceed min(n_mels, n_frames)
        H, W = B.shape
        max_power = int(math.floor(math.log2(min(H, W))))
        # Generate [2^0, 2^1, ..., 2^max_power]
        epsilons = [2 ** i for i in range(max_power + 1)]
    
    # 5) Calculate N(ε) for each ε
    N_e = box_counting_fd_2d(B, epsilons)
    
    # 6) Remove zero N(ε) (invalid points)
    eps_arr = np.array(epsilons, dtype=float)
    mask = N_e > 0
    if np.sum(mask) < 2:
        return np.nan  # not enough points for regression
    
    ln_eps = np.log(1.0 / eps_arr[mask])   # ln(1/ε)
    ln_Ne  = np.log(N_e[mask])             # ln N(ε)
    
    # 7) Linear regression ln N(ε) = D * ln(1/ε) + const
    #    => D = slope
    slope, intercept = np.polyfit(ln_eps, ln_Ne, 1)
    D_box = slope
    
    return D_box

def visualize_spectrograms(audio_path, n_fft=2048, hop_length=512, n_mels=128, threshold_method="mean"):
    """
    Visualizes mel spectrogram and its binarized version.
    
    Parameters:
    - audio_path: path to audio file
    - n_fft: FFT window size
    - hop_length: hop length for STFT
    - n_mels: number of mel bands
    - threshold_method: binarization method
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Get spectrograms
    S_log = compute_log_mel_spectrogram(y, sr, n_fft, hop_length, n_mels)
    B = binarize_spectrogram(S_log, method=threshold_method)
    
    # Create figure with two plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Display mel spectrogram
    img1 = librosa.display.specshow(S_log, y_axis='mel', x_axis='time', 
                                  hop_length=hop_length, ax=ax1)
    ax1.set_title('Logarithmic Mel Spectrogram')
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
    
    # Display binarized spectrogram
    img2 = ax2.imshow(B, aspect='auto', origin='lower', cmap='binary')
    ax2.set_title('Binarized Spectrogram')
    fig.colorbar(img2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def visualize_box_counting(B, epsilons=[1, 2, 4, 8, 16, 32]):
    """
    Visualizes box counting process for different cell sizes.
    
    Parameters:
    - B: binary matrix
    - epsilons: list of cell sizes for visualization
    """
    n_plots = len(epsilons)
    n_cols = min(3, n_plots)  # maximum 3 plots per row
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    H, W = B.shape
    
    for idx, (ax, e) in enumerate(zip(axes, epsilons)):
        # Show binary matrix
        ax.imshow(B, cmap='binary', aspect='auto', interpolation='nearest')
        
        # Draw grid
        n_h = math.ceil(H / e)
        n_w = math.ceil(W / e)
        
        # Vertical grid lines
        for j in range(n_w + 1):
            x = j * e - 0.5
            ax.axvline(x=x, color='blue', alpha=0, linestyle='-')
        
        # Horizontal grid lines
        for i in range(n_h + 1):
            y = i * e - 0.5
            ax.axhline(y=y, color='blue', alpha=0, linestyle='-')
        
        # Count and mark occupied cells
        occupied_boxes = 0
        
        for i_block in range(n_h):
            i0 = i_block * e
            i1 = min(i0 + e, H)
            for j_block in range(n_w):
                j0 = j_block * e
                j1 = min(j0 + e, W)
                
                # If there is at least one '1' in the block
                if np.any(B[i0:i1, j0:j1] > 0):
                    occupied_boxes += 1
                    # Draw semi-transparent rectangle over occupied cell
                    rect = patches.Rectangle((j0-0.5, i0-0.5), e, e, 
                                          linewidth=1.5, 
                                          edgecolor='red',
                                          facecolor='red',
                                          alpha=0.3)
                    ax.add_patch(rect)
        
        ax.set_title(f'Box size = {e}, Occupied boxes = {occupied_boxes}')
    
    # Remove empty subplots
    for idx in range(len(epsilons), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def visualize_box_counting_progression(B, n_steps=10):
    """
    Визуализирует прогрессию box counting от мелкой до крупной сетки.
    Показывает как постепенно укрупняются ячейки и какие из них остаются занятыми.
    
    Параметры:
    - B: бинарная матрица
    - n_steps: количество шагов визуализации
    """
    H, W = B.shape
    
    # Определяем размеры ячеек для визуализации, начиная с 3
    max_size = min(H, W) // 4  # максимальный размер не больше четверти изображения
    epsilons = np.geomspace(3, max_size, n_steps).astype(int)
    epsilons = np.unique(epsilons)  # убираем повторяющиеся значения
    
    # Создаем сетку графиков
    n_plots = len(epsilons)
    n_cols = 5  # 5 графиков в ряд
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 4*n_rows))
    plt.suptitle('Прогрессия Box Counting: увеличение размера ячеек', fontsize=16, y=1.02)
    
    for idx, e in enumerate(epsilons, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        
        # Показываем бинарную матрицу
        ax.imshow(B, cmap='binary', aspect='auto', interpolation='nearest')
        
        # Рисуем сетку
        n_h = math.ceil(H / e)
        n_w = math.ceil(W / e)
        
        # Вертикальные линии сетки
        for j in range(n_w + 1):
            x = j * e - 0.5
            ax.axvline(x=x, color='blue', alpha=0, linestyle='-')
        
        # Горизонтальные линии сетки
        for i in range(n_h + 1):
            y = i * e - 0.5
            ax.axhline(y=y, color='blue', alpha=0, linestyle='-')
        
        # Подсчитываем и отмечаем занятые ячейки
        occupied_boxes = 0
        
        for i_block in range(n_h):
            i0 = i_block * e
            i1 = min(i0 + e, H)
            for j_block in range(n_w):
                j0 = j_block * e
                j1 = min(j0 + e, W)
                
                # Если в блоке есть хотя бы одна единица
                if np.any(B[i0:i1, j0:j1] > 0):
                    occupied_boxes += 1
                    # Рисуем полупрозрачный прямоугольник поверх занятой ячейки
                    rect = patches.Rectangle((j0-0.5, i0-0.5), e, e, 
                                          linewidth=1.5, 
                                          edgecolor='red',
                                          facecolor='red',
                                          alpha=0.2)
                    ax.add_patch(rect)
        
        ax.set_title(f'ε = {e}\nN(ε) = {occupied_boxes}')
        
        # Устанавливаем пределы осей
        ax.set_xlim(-1, W)
        ax.set_ylim(-1, H)
        
        # Убираем метки осей
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Добавляем график ln N(ε) vs ln(1/ε)
    if n_plots < n_rows * n_cols:
        ax_plot = plt.subplot(n_rows, n_cols, n_plots + 1)
        
        # Вычисляем точки для графика
        N_values = []
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
                    if np.any(B[i0:i1, j0:j1] > 0):
                        count += 1
            N_values.append(count)
        
        N_values = np.array(N_values)
        ln_eps = np.log(1.0 / epsilons)
        ln_N = np.log(N_values)
        
        # Линейная регрессия
        slope, intercept = np.polyfit(ln_eps, ln_N, 1)
        
        # Построение графика
        ax_plot.plot(ln_eps, ln_N, 'bo-', label='Данные')
        ax_plot.plot(ln_eps, slope * ln_eps + intercept, 'r--', 
                    label=f'D ≈ {slope:.3f}')
        
        ax_plot.set_xlabel('ln(1/ε)')
        ax_plot.set_ylabel('ln N(ε)')
        ax_plot.set_title('График ln N(ε) vs ln(1/ε)')
        ax_plot.legend()
        ax_plot.grid(True)
    
    plt.show()

def visualize_full_analysis(audio_path, n_fft=2048, hop_length=512, n_mels=128, threshold_method="median"):
    """
    Показывает полный анализ: спектрограммы и box counting.
    """
    # Загрузка и обработка аудио
    y, sr = _load_mono_audio(audio_path)
    
    # Получение спектрограмм
    S_log = compute_log_mel_spectrogram(y, sr, n_fft, hop_length, n_mels)
    B = binarize_spectrogram(S_log, method=threshold_method)
    
    # Визуализация спектрограмм
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    img1 = librosa.display.specshow(S_log, y_axis='mel', x_axis='time', 
                                  hop_length=hop_length)
    plt.title('Логарифмическая мел-спектрограмма')
    plt.colorbar(img1, format='%+2.0f dB')
    
    plt.subplot(2, 1, 2)
    img2 = plt.imshow(B, aspect='auto', origin='lower', cmap='binary', interpolation='nearest')
    plt.title('Бинаризованная спектрограмма')
    plt.colorbar(img2)
    
    plt.tight_layout()
    plt.show()
    
    # Визуализация прогрессии box counting
    visualize_box_counting_progression(B)
    
    # Вычисление и вывод фрактальной размерности
    D = compute_box_counting_fd(audio_path)
    print(f"Фрактальная размерность: {D:.3f}")

def generate_binary_spectrogram(audio_path, n_fft=2048, hop_length=512, n_mels=128, threshold_method="median"):
    """
    Генерирует только бинаризованную спектрограмму без дополнительных визуализаций.
    
    Параметры:
    - audio_path: путь к аудио файлу
    - n_fft: размер окна FFT
    - hop_length: шаг для STFT
    - n_mels: число мел-банок
    - threshold_method: метод бинаризации
    
    Возвращает:
    - B: бинарная матрица спектрограммы
    """
    # Загрузка и обработка аудио
    y, sr = _load_mono_audio(audio_path)
    
    # Получение спектрограмм
    S_log = compute_log_mel_spectrogram(y, sr, n_fft, hop_length, n_mels)
    B = binarize_spectrogram(S_log, method=threshold_method)
    
    # Создаем чистое изображение
    plt.figure(figsize=(10, 5))
    plt.imshow(B, aspect='auto', cmap='binary', interpolation='nearest')
    plt.axis('off')  # Убираем оси
    plt.tight_layout(pad=0)  # Убираем отступы
    plt.show()
    
    return B

def visualize_pure_box_counting(B, n_steps=10):
    """
    Показывает только одну сетку box counting с ε = 4.
    
    Параметры:
    - B: бинарная матрица
    - n_steps: количество шагов визуализации (не используется)
    """
    H, W = B.shape
    
    # Используем только один маленький масштаб
    epsilon = 4
    
    # Создаем одно большое изображение
    fig = plt.figure(figsize=(20, 8))
    plt.axis('off')
    
    # Создаем сетку 1x2 (1 для box counting + 1 для регрессии)
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1])
    
    # Рисуем box counting
    ax = fig.add_subplot(gs[0, 0])
    
    # Показываем бинарную матрицу
    ax.imshow(B, cmap='binary', aspect='auto', interpolation='nearest')
    
    # Рисуем сетку
    n_h = math.ceil(H / epsilon)
    n_w = math.ceil(W / epsilon)
    
    # Подсчитываем занятые ячейки и рисуем их
    occupied_boxes = 0
    for i_block in range(n_h):
        i0 = i_block * epsilon
        i1 = min(i0 + epsilon, H)
        for j_block in range(n_w):
            j0 = j_block * epsilon
            j1 = min(j0 + epsilon, W)
            if np.any(B[i0:i1, j0:j1] > 0):
                occupied_boxes += 1
                # Рисуем красный прямоугольник только для занятых ячеек
                rect = patches.Rectangle((j0-0.5, i0-0.5), epsilon, epsilon, 
                                      linewidth=0.5, 
                                      edgecolor='red',
                                      facecolor='red',
                                      alpha=0.1)
                ax.add_patch(rect)
    
    # Добавляем заголовок с размером ячейки
    ax.set_title(f'Box Counting с ε = {epsilon}\nOccipied boxes N(ε) = {occupied_boxes}', 
                fontsize=14, pad=10)
    
    # Убираем все оси и метки
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Добавляем график регрессии в правую колонку
    ax_plot = fig.add_subplot(gs[0, 1])
    
    # Для регрессии используем больше точек, включая маленькие epsilon
    all_epsilons = [2, 4, 8, 16, 32, 48, 64, 96]
    
    # Вычисляем точки для графика
    N_values = []
    for e in all_epsilons:
        n_h = math.ceil(H / e)
        n_w = math.ceil(W / e)
        count = 0
        for i_block in range(n_h):
            i0 = i_block * e
            i1 = min(i0 + e, H)
            for j_block in range(n_w):
                j0 = j_block * e
                j1 = min(j0 + e, W)
                if np.any(B[i0:i1, j0:j1] > 0):
                    count += 1
        N_values.append(count)
    
    N_values = np.array(N_values)
    ln_eps = np.log(1.0 / np.array(all_epsilons))
    ln_N = np.log(N_values)
    
    # Линейная регрессия
    slope, intercept = np.polyfit(ln_eps, ln_N, 1)
    
    # Построение графика
    ax_plot.plot(ln_eps, ln_N, 'bo-', label='Calculations', markersize=8, alpha=0.5)
    ax_plot.plot(ln_eps, slope * ln_eps + intercept, 'r--', 
                label=f'D ≈ {slope:.3f}', linewidth=2)
    
    # Выделяем точку ε = 4
    idx = all_epsilons.index(epsilon)
    ax_plot.plot(ln_eps[idx], ln_N[idx], 'ro', markersize=12, 
                label=f'ε = {epsilon}')
    
    ax_plot.set_xlabel('ln(1/ε)', fontsize=12)
    ax_plot.set_ylabel('ln N(ε)', fontsize=12)
    ax_plot.set_title('Fractal dimension\n estimation', 
                     fontsize=14, pad=20)
    ax_plot.legend(fontsize=10)
    ax_plot.grid(True)
    
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.show()

def process_audio_for_box_counting(audio_path, n_fft=2048, hop_length=512, n_mels=128, threshold_method="median"):
    """
    Подготавливает аудио файл, показывает спектрограммы и box counting.
    """
    # Загрузка и обработка аудио
    y, sr = _load_mono_audio(audio_path)
    
    # Получение спектрограмм
    S_log = compute_log_mel_spectrogram(y, sr, n_fft, hop_length, n_mels)
    B = binarize_spectrogram(S_log, method=threshold_method)
    
    # 1. Сначала показываем спектрограммы
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    img1 = librosa.display.specshow(S_log, y_axis='mel', x_axis='time', 
                                  hop_length=hop_length)
    plt.title('Логарифмическая мел-спектрограмма')
    plt.colorbar(img1, format='%+2.0f dB')
    
    plt.subplot(2, 1, 2)
    img2 = plt.imshow(B, aspect='auto', origin='lower', cmap='binary', interpolation='nearest')
    plt.title('Бинаризованная спектрограмма')
    plt.colorbar(img2)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Затем показываем box counting
    visualize_pure_box_counting(B)
    return B

# Показываем спектрограммы и box counting
process_audio_for_box_counting("data/test/ALBLAK 52-Nice To Meet You-kissvk.com.mp3")

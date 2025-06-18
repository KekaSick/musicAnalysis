import os
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def _load_mono_audio(file_path):
    """
    Read WAV (scipy) or any other format (soundfile/librosa backend)
    and normalize to float32 [-1, 1].
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


def higuchi_fd(x, k_max=200):
    """
    Estimates the fractal dimension of signal x using Higuchi's method.
    x: 1D numpy array, length N
    k_max: maximum step (usually N//10 or N//20)

    Returns: fd (scalar) and (log k, log L(k)) for debugging/plotting.
    """
    N = x.shape[0]
    Lk = np.zeros(k_max)
    ln_k = np.zeros(k_max)
    ln_Lk = np.zeros(k_max)

    for k in range(1, k_max+1):
        Lm = np.zeros(k)
        for m in range(k):
            # number of points in subsequence for this k
            num = (N - m - 1) // k
            if num < 1:
                continue
            # calculate absolute differences along this subsequence
            idxs = m + np.arange(num + 1) * k
            # differences |x[idxs[i+1]] - x[idxs[i]]|
            dist_sum = np.sum(np.abs(x[idxs[1:] ] - x[idxs[:-1]]))
            # normalization
            Lm[m] = (dist_sum * (N - 1)) / (num * k)
        # if all Lm == 0 (too short), leave as 0
        Lk[k-1] = np.mean(Lm[np.nonzero(Lm)] ) if np.any(Lm) else 0
        ln_k[k-1] = np.log(1.0 / k)
        ln_Lk[k-1] = np.log(Lk[k-1]) if Lk[k-1] > 0 else 0

    # Take only those k where Lk>0
    mask = Lk > 0
    # Linear regression: ln_Lk = D * ln_k + b => D = slope
    D, b = np.polyfit(ln_k[mask], ln_Lk[mask], 1)
    # Fractal dimension ≈ D
    return D, (ln_k[mask], ln_Lk[mask])

def visualize_higuchi_steps(file_path, k_max=200):
    """
    Visualizes the original signal, decomposition for k=1 and k=200, and full regression.
    
    Parameters:
    -----------
    file_path : str
        Path to audio file
    k_max : int
        Maximum k value for the algorithm
    """
    # Create plot directory if it doesn't exist
    plot_dir = 'plot/temp'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load audio
    x, sr = _load_mono_audio(file_path)
    
    # Save full signal
    x_full = x.copy()
    
    # Take 20ms fragment for k=1 and 1 second for k=200
    fragment_samples_short = int(sr * 0.02)  # 20ms for k=1
    fragment_samples_long = sr  # 1 second for k=200
    start_sample = sr // 4  # Take from middle of first second
    
    # Fragment for k=1
    x_short = x_full[start_sample:start_sample + fragment_samples_short]
    # Fragment for k=200
    x_long = x_full[start_sample:start_sample + fragment_samples_long]
    
    N_short = len(x_short)
    N_long = len(x_long)
    
    # Colors for subsequences
    colors = ['#E74C3C', '#2ECC71', '#9B59B6', '#F1C40F', '#E67E22']
    
    # 0. Full original signal
    plt.figure(figsize=(15, 5))
    t_full = np.arange(len(x_full)) / sr
    plt.plot(t_full, x_full, color='#2E86C1', linewidth=1, label='Original Signal')
    plt.title('Original Audio Signal', pad=10, fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_dir, '1_original_signal.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    # Highlight areas that will be shown in detailed analysis
    highlight_start = start_sample / sr
    highlight_end_short = (start_sample + fragment_samples_short) / sr
    highlight_end_long = (start_sample + fragment_samples_long) / sr
    
    # Highlight short fragment with different color
    plt.axvspan(highlight_start, highlight_end_short, color='#F7DC6F', alpha=0.3, 
                label='Analysis area k=1 (20ms)')
    # Highlight long fragment
    plt.axvspan(highlight_start, highlight_end_long, color='#82E0AA', alpha=0.15, 
                label='Analysis area k=200 (1s)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, '1_highlight_areas.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    # 1. Visualization for k=1
    plt.figure(figsize=(15, 5))
    t_short = np.arange(N_short) / sr + highlight_start
    plt.plot(t_short, x_short, color='#2E86C1', linestyle='-', 
             linewidth=2, label='Original Signal')
    
    k = 1
    for m in range(k):
        indices = np.arange(m, N_short, k)
        plt.plot(t_short[indices], x_short[indices],
                color=colors[m],
                marker='o',
                markersize=8,
                linestyle='-',
                linewidth=2,
                label='Subsequence')
        
        # Add vertical lines
        for idx in indices[:-1]:
            if idx + k < N_short:
                plt.vlines(t_short[idx], x_short[idx], x_short[idx+k], 
                          colors[m], alpha=0.3, linestyles='--')
    
    plt.title(f'Signal Decomposition for k=1\n(no thinning, 20ms)', 
              pad=10, fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_dir, '2_k1_decomposition.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    # 2. Visualization for k=200
    plt.figure(figsize=(15, 5))
    t_long = np.arange(N_long) / sr + highlight_start
    plt.plot(t_long, x_long, color='#2E86C1', linestyle='-', 
             linewidth=2, label='Original Signal')
    
    # Show only one subsequence
    k = 200
    indices = np.arange(0, N_long, k)  # m=0
    if len(indices) > 0:
        plt.plot(t_long[indices], x_long[indices],
                color=colors[0],
                marker='o',
                markersize=8,
                linestyle='none',
                label='Thinned Sequence')
    
    plt.title(f'Signal Decomposition for k=200\n(strong thinning, every 200th point, 1s)', 
              pad=10, fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_dir, '3_k200_decomposition.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    # 3. Full regression
    plt.figure(figsize=(15, 5))
    D, (ln_k, ln_Lk) = higuchi_fd(x_full, k_max)
    
    # Regression line
    reg_line = np.polyval([D, np.mean(ln_Lk - D * ln_k)], ln_k)
    
    # Show all points in gray
    plt.scatter(ln_k, ln_Lk, 
                color='#95A5A6', s=50, alpha=0.5,
                label='All Points', zorder=4)
    
    # Highlight points k=1 and k=200
    special_k_indices = [0, -1]  # First and last indices for k=1 and k=200
    plt.scatter(ln_k[special_k_indices], ln_Lk[special_k_indices], 
                color='#2ECC71', s=150, 
                label='k=1 and k=200', zorder=5)
    
    # Add labels to these points
    plt.annotate('k=1', (ln_k[0], ln_Lk[0]), 
                xytext=(10, 10), textcoords='offset points',
                color='#27AE60', fontweight='bold')
    plt.annotate('k=200', (ln_k[-1], ln_Lk[-1]), 
                xytext=(10, -10), textcoords='offset points',
                color='#27AE60', fontweight='bold')
    
    # Regression line
    plt.plot(ln_k, reg_line, color='#E74C3C', 
             linewidth=2, label=f'Regression (D ≈ {D:.3f})')
    
    plt.title('Fractal Dimension Estimation using Higuchi Method', 
              pad=10, fontsize=12, fontweight='bold')
    plt.xlabel('ln(1/k)')
    plt.ylabel('ln L(k)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, '4_higuchi_regression.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    return D


print(visualize_higuchi_steps('data/test/ALBLAK 52-Nice To Meet You-kissvk.com.mp3'))
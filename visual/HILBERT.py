import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import hilbert

def load_audio(file_path):
    """Load audio file (supports WAV and MP3 via librosa)"""
    x, sr = librosa.load(file_path, sr=44100, mono=True)
    return x, sr

def frame_signal(x, frame_len, hop_len):
    """Split signal into frames"""
    if len(x) < frame_len:
        return np.empty((0, frame_len))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len]
    return frames

# Function for creating plots
def plot_signal_processing(file_path, win_len=2048, hop_len=1024):
    # Load signal
    x, sr = load_audio(file_path)
    t = np.arange(len(x)) / sr  # Time in seconds

    # 1. Hilbert transform for amplitude envelope
    env = np.abs(hilbert(x))

    # 2. Frame splitting and RMS envelope calculation
    frames = frame_signal(env, win_len, hop_len)
    env_rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)

    # 3. Logarithm and normalization
    env_log = np.log1p(env_rms)
    env_norm = (env_log - env_log.mean()) / (env_log.std() + 1e-12)

    # Create plots
    plt.figure(figsize=(10, 8))

    # 1. Original signal
    plt.subplot(4, 1, 1)
    plt.plot(t[:len(x)], x)
    plt.title('Original Sine Wave')

    # 2. Amplitude envelope
    plt.subplot(4, 1, 2)
    plt.plot(t[:len(env)], env)
    plt.title('Amplitude Envelope (Hilbert)')

    # 3. RMS envelope
    plt.subplot(4, 1, 3)
    plt.plot(np.linspace(0, len(x) / sr, len(env_rms)), env_rms)
    plt.title('RMS Envelope')

    # 4. Logarithmic and normalized RMS envelope
    plt.subplot(4, 1, 4)
    plt.plot(np.linspace(0, len(x) / sr, len(env_log)), env_log)
    plt.title('Logarithmic and Normalized RMS Envelope')

    plt.tight_layout()
    plt.show()

# Example usage: pass the path to your audio file (MP3 or WAV)
file_path = 'data/test/ALBLAK 52-Nice To Meet You-kissvk.com.mp3'  # Replace with your file path
plot_signal_processing(file_path)

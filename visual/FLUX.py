import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import librosa

# Function to load WAV/MP3 file
def load_audio(file_path):
    x, sr = librosa.load(file_path, sr=44100, mono=True)
    return x, sr

# Parameters
n_fft = 2048       # Window size for STFT
hop_len = 1024     # Window overlap
smooth = 7         # Smoothing parameter

# Specify path to your WAV/MP3 file
file_path = 'data/test/ALBLAK 52-Nice To Meet You-kissvk.com.mp3'  # Replace with your file path

# Load audio
x, sr = load_audio(file_path)
t_len = len(x) / sr  # Audio duration in seconds
t = np.linspace(0, t_len, len(x))

# Calculate STFT
f, t_stft, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_len,
                    window="hann", padded=False, boundary=None)

# Spectrum magnitude
mag = np.abs(Z)

# Logarithmic amplitude for better visibility
mag_log = np.log1p(mag)

# Differentiation for flux
diff = np.diff(mag_log, axis=1)
flux = np.sqrt((diff**2).sum(axis=0))   # 1-D spectral flux

# Smoothing flux (rolling mean)
kernel = np.ones(smooth, dtype=float) / smooth
flux_smooth = np.convolve(flux, kernel, mode="valid")

# Visualize STFT with logarithmic amplitude
plt.figure(figsize=(12, 6))

# Limit frequencies to 5000 Hz
freq_mask = f <= 5000
f_limited = f[freq_mask]
mag_log_limited = mag_log[freq_mask, :]

plt.imshow(mag_log_limited, aspect='auto', origin='lower', cmap='viridis',
           extent=[t_stft.min(), t_stft.max(), f_limited.min(), f_limited.max()])
plt.title('STFT (Magnitude Spectrum)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Log(Magnitude)')
plt.tight_layout()
plt.show()

# Visualize other stages (original signal, fluxes and smoothing)
fig, axs = plt.subplots(4, 1, figsize=(12, 16))

# 1. Original signal (all points)
axs[0].plot(t, x)  # All signal points
axs[0].set_title('Original Signal')
axs[0].set_ylabel('Amplitude')
axs[0].set_xticklabels([])

# 2. Logarithmic amplitude
axs[1].plot(t_stft[:-1], np.log1p(mag[:, :-1]).mean(axis=0))
axs[1].set_title('Logarithmic Amplitude')
axs[1].set_ylabel('Log(Amplitude)')
axs[1].set_xticklabels([])

# 3. Spectral flux
axs[2].plot(t_stft[:-1], flux)
axs[2].set_title('Spectral Flux')
axs[2].set_ylabel('Flux')
axs[2].set_xticklabels([])

# 4. Smoothed spectral flux
axs[3].plot(t_stft[7:], flux_smooth)
axs[3].set_title('Smoothed Spectral Flux')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Flux')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import stft

# Set global font size
plt.rcParams.update({'font.size': 14})

# Load audio
def load_audio(file_path):
    x, sr = librosa.load(file_path, sr=44100, mono=True)
    return x, sr

# Parameters
file_path = 'data/test/ALBLAK 52-Nice To Meet You-kissvk.com.mp3'  # Path to your file

# Load audio
x, sr = load_audio(file_path)
t_len = len(x) / sr  # Audio duration in seconds
t = np.linspace(0, t_len, len(x))

# 1. Original signal (time domain)
plt.figure(figsize=(12, 6))
plt.plot(t, x)  # Show first 500 points
plt.title('Original Signal', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.tight_layout()
plt.show()

# 2. CQT (Constant-Q Transform)
n_octaves = 7
bins_per_octave = 36
fmin = librosa.note_to_hz('C1')  # Lower frequency for CQT
C = librosa.cqt(x, sr=sr, hop_length=512, fmin=fmin, n_bins=bins_per_octave * n_octaves, bins_per_octave=bins_per_octave)

# Visualize CQT
plt.figure(figsize=(12, 6))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(C), ref=np.max), y_axis='log', x_axis='time', sr=sr)
plt.title('CQT (Constant-Q Transform)', fontsize=16)
plt.colorbar(label='Amplitude (dB)')
plt.tight_layout()
plt.show()

# 3. Chroma
chroma = librosa.feature.chroma_cqt(C=C, sr=sr, bins_per_octave=bins_per_octave)

# Visualize chroma
plt.figure(figsize=(12, 6))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
plt.title('Chroma', fontsize=16)
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()

# 4. Tonality vector (angles on circle of fifths)
angles = np.arange(12) * 2 * np.pi / 12  # Angles for 12 tones
vector = (chroma * np.exp(1j * angles[:, None])).sum(axis=0)  # Complex vector
phi = np.angle(vector)  # Angle on circle of fifths

# Visualize angles
plt.figure(figsize=(12, 6))
plt.plot(t[:len(phi)], phi)
plt.title('Tonality Vector (Angles)', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Angle (rad)', fontsize=14)
plt.tight_layout()
plt.show()

# 5. Analysis result (e.g., PE-CE)
# Here we can perform further analysis using ordpy.complexity_entropy (for example).

import numpy as np
import matplotlib.pyplot as plt
import librosa
import antropy as ent   # pip install antropy

# ─── parameters ────────────────────────────────────────────────
FILE = 'data/test/ALBLAK 52-Nice To Meet You-kissvk.com.mp3'
SR   = 44_100
N_FFT = 2048
HOP   = 512

# ─── loading ─────────────────────────────────────────────────
y, sr = librosa.load(FILE, sr=SR, mono=True)

# ─── framing: (n_frames, frame_len) ─────────────────────
frames = librosa.util.frame(y, frame_length=N_FFT,
                            hop_length=HOP).T          # ← transpose!

# ─── spectral entropy for each frame ────────────────
spec_ent = np.apply_along_axis(
    lambda seg: ent.spectral_entropy(seg, sf=sr,
                                     method="fft",
                                     normalize=True),
    axis=1,
    arr=frames
)

# time for each point
times = librosa.frames_to_time(np.arange(len(spec_ent)),
                               sr=sr, hop_length=HOP)

# ─── visualization ─────────────────────────────────────────────
plt.figure(figsize=(14, 5))
plt.plot(times, spec_ent)
plt.title('Spectral Entropy (Full Track)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Entropy')
plt.tight_layout()
plt.show()

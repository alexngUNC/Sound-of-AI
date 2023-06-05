import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import requests
from pathlib import Path


recording_path = Path("blues.00000.wav")
# if recording_path.is_dir():
#   print(f"{recording_path} already exists")
# else:
#   print(f"{recording_path} doesn't exist, creating one...")
#   recording_path.mkdir(parents=True, exist_ok=True)

with open(recording_path, "wb") as f:
  request = requests.get("https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/44a0e1880eee57a523780a1862cb8bf44963fbe8/11-%20Preprocessing%20audio%20data%20for%20deep%20learning/code/blues.00000.wav")
  print("Downloading recording file...")
  f.write(request.content)

# waveform
signal, sr = librosa.load(str(recording_path), sr=22050) # sr * T -> 22050 * 30

# visualize
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show();

# FFT -> specturm
fft = np.fft.fft(signal)

# abs value on complex values, contribution of each frequency on the sound
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
plt.plot(frequency, magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show();

# only use the first half
left_frequency = frequency[:int(len(frequency) / 2)]
left_magnitude = magnitude[:int(len(magnitude) / 2)]
plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show();

# spectrogram gives info about amplitude as function of freq and time
# stft -> spect
# num samples per FFT
n_fft = 2048

# amount fft is shifted to the right
hop_length = 512
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

# complex nums -> magnitude
spectrogram = np.abs(stft)

# plot spectrogram
librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
#plt.colorbar()
plt.show();

# Log spectrogram
log_spectrogram = librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.title("Log Spectrogram")
plt.ylabel("Frequency")
#plt.colorbar()
plt.show();


# MFCCs
# n_mfcc -> num coefficients to be extracted
MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
# plt.colorbar()
plt.show();
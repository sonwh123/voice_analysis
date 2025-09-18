from pydub import AudioSegment
import numpy as np, noisereduce as nr, soundfile as sf

# m4a → wav 변환 (메모리상)
audio = AudioSegment.from_file("test.m4a", format="m4a")
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
samples = samples / (1<<15)  # int16 → float(-1~1)

sr = audio.frame_rate

# 노이즈 제거
noise_clip = samples[:int(0.5*sr)]
y_denoised = nr.reduce_noise(y=samples, sr=sr, y_noise=noise_clip)

sf.write("voice_denoised.wav", y_denoised, sr)

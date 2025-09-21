import noisereduce as nr
import librosa
import soundfile as sf

# 오디오 로드
y, sr = librosa.load("test.m4a", sr=None)

# 노이즈 제거
reduced = nr.reduce_noise(y=y, sr=sr)

# 저장
sf.write("output.wav", reduced, sr)

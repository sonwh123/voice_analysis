from pydub import AudioSegment

audio = AudioSegment.from_file("voice.m4a")

start_ms = 4.48 * 1000   # 밀리초 변환
end_ms = 4.79 * 1000

# 슬라이싱
segment = audio[start_ms:end_ms]

# 저장
segment.export("audio_cut.wav", format="wav")

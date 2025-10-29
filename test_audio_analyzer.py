from audio_analyzer import analyze_segments

# 분석할 음성 파일 경로 (wav, mp3 모두 가능)
audio_path = "voice.m4a"

# 분석 실행
result = analyze_segments(audio_path, model_name="base", language="ko")

# 결과 출력
import json
print(json.dumps(result, indent=2, ensure_ascii=False))

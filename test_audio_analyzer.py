from audio_analyzer import analyze_segments
import json
import os

# 분석할 음성 파일 경로 (wav, mp3 모두 가능)
audio_path = "voice.m4a"

# 분석 실행
result = analyze_segments(audio_path, model_name="base", language="ko")

# # 결과 출력
# print(json.dumps(result, indent=2, ensure_ascii=False))

# JSON 파일 저장
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

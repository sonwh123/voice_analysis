from audio_analyzer import analyze_segments
from clova_stt import make_voice_to_stt
from make_feedback import make_feedback
import json
import os

# 분석할 음성 파일 경로 (wav, mp3 모두 가능)
audio_path = "voice1.m4a"

# STT 변환 실행(.json 파일 생성)
make_voice_to_stt(audio_path)

# 분석 실행
result = analyze_segments(audio_path)

# # 결과 출력
# print(json.dumps(result, indent=2, ensure_ascii=False))

# JSON 파일 저장
with open("result1.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 피드백 생성
feedbacks = make_feedback(result)

# 피드백 결과 저장
base_name = os.path.splitext(audio_path)[0]
json_output_file = base_name + '_feedback.json'
with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(feedbacks, f, ensure_ascii=False, indent=2)
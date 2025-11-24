# LLM 피드백 생성
import json
from clova_LLM import get_feedback_from_LLM
from analyzer_function import classify_energy_cv, classify_pitch_cv, classify_rate_wpm

with open("result.json", "r", encoding="utf-8") as f:
    result = json.load(f)

# 문장별 분석 결과 추출
for seg in result["segments"]:
    # 초기화
    analyzed = {
        "volume_stability": {},
        "ending_pattern": {},
        "pitch_stability": {},
        "pitch_ending": {},
        "rate_level": {}
    }

    # CV 값 기반 레이블링
    # 1. Volume Stability
    cv_energy = seg["energy"]["cv"]
    label, comment = classify_energy_cv(cv_energy)
    analyzed["volume_stability"] = {
        "label": label,
        "comment": comment
    }

    # 2. Pitch Stability
    cv_pitch = seg["pitch"]["cv"]
    label, comment = classify_pitch_cv(cv_pitch)
    analyzed["pitch_stability"] = {
        "label": label,
        "comment": comment
    }

    # 3. Rate Level
    rate_level = seg["wpm"]["rate_wpm"]
    label, comment = classify_rate_wpm(rate_level)
    analyzed["rate_level"] = {
        "label": label,
        "comment": comment
    }

    # print(analyzed)
    # LLM 피드백 생성
    feedback = get_feedback_from_LLM(analyzed)
    print("=== LLM Feedback ===")
    print(feedback)

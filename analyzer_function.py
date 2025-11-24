import numpy as np

# ===== 공통: 유성(발화) 마스크 생성 =====
def get_voiced_mask_from_words(rms: np.ndarray,
                               sr: int,
                               hop_length: int,
                               stt_json: dict,
                               margin: float = 0.03) -> np.ndarray:
    """
    STT JSON에 들어있는 word 단위 타임라인을 이용해서
    '유성(말한) 구간'만 True인 프레임 마스크를 만든다.

    - rms        : librosa.feature.rms(y=...) 결과 (shape: (1, T) 또는 (T,))
    - sr         : 샘플링 레이트
    - hop_length : rms 계산에 사용한 hop_length
    - stt_json   : 네가 준 형태의 STT 결과(dict)
                   stt_json["segments"][i]["words"][j]["start"|"end"]
    - margin     : 각 단어 앞뒤로 포함할 여유 시간(초)

    반환값:
    - mask: shape (T,) 의 bool 배열
    """
    # 정규화 (확인용)
    r = np.asarray(rms).astype(float).squeeze()
    num_frames = r.shape[0]

    # 각 프레임 중심 시간 (초 단위)
    frame_idx = np.arange(num_frames)
    frame_times = (frame_idx * hop_length + hop_length / 2.0) / sr

    mask = np.zeros(num_frames, dtype=bool)

    # 모든 segment/word의 (start, end) 구간을 모아서 mask에 반영
    for seg in stt_json:
        for w in seg["words"]:
            start = w[0]/1000  # ms -> s
            end = w[1]/1000 # ms -> s

            s = max(0.0, start - margin)
            e = end + margin

            mask |= (frame_times >= s) & (frame_times <= e)

    return mask

# ===== 1. Energy CV (유성 구간만 사용, RMS linear) =====
def compute_energy_stats_segment(rms: np.ndarray,
                                 silence_thresh: float = 1e-6):
    """
    rms        : 전체 음성에 대해 계산된 RMS (shape: (T,))
    seg_mask   : 이 segment(문장)에 해당하는 프레임만 True인 bool mask
                 (예: in_seg & voiced_mask_global 같은 것)
    silence_thresh : 실질적 무음으로 간주할 RMS threshold

    반환:
      mean_r, std_r, cv_energy
      (유효 프레임 < 2이면 모두 np.nan)
    """
    # 실질적인 무음 제거 + NaN 제거
    valid = np.isfinite(rms) & (rms > silence_thresh)
    r = rms[valid]

    if r.size < 2:
        return np.nan, np.nan, np.nan

    mean_r = float(np.mean(r))
    std_r = float(np.std(r))

    if mean_r <= 0:
        return mean_r, std_r, np.nan

    cv_energy = std_r / mean_r
    return mean_r, std_r, float(cv_energy)

# ===== 2. Pitch CV (유성 구간만 사용) =====
def compute_pitch_cv_segment(f0_hz: np.ndarray,
                             f0_min: float = 1e-3):
    """
    f0_hz   : 전체 음성에 대해 pyin 등으로 얻은 F0(Hz), NaN 포함 가능 (shape: (T,))
    seg_mask: 이 segment(문장)에 해당하는 프레임만 True인 bool mask
              (예: in_seg & voiced_mask_global 같이, 이미 유성/무성까지 반영된 mask)
    f0_min  : 0 또는 비정상적으로 작은 F0 필터링용 threshold

    반환:
      mean_st, std_st, cv_pitch  (semitone 스케일 기준)
      (유효 프레임 < 2이면 모두 np.nan)
    """
    # NaN 제거 + 0 / 비정상적으로 작은 값 제거
    valid = np.isfinite(f0_hz) & (f0_hz > f0_min)
    f0 = f0_hz[valid]

    if f0.size < 2:
        return np.nan, np.nan, np.nan

    # 55 Hz 기준 semitone 변환
    f0_semitone = 12.0 * np.log2(f0 / 55.0)
    mean_st = float(np.mean(f0_semitone))
    std_st = float(np.std(f0_semitone))

    if mean_st <= 0:
        return mean_st, std_st, np.nan

    cv_pitch = std_st / mean_st
    return mean_st, std_st, float(cv_pitch)

# ===== 3. CV 기반 레이블링 함수들 =====
## ===== 3-1. Pitch CV =====
def classify_pitch_cv(cv_pitch: float):
    if not np.isfinite(cv_pitch):
        return "UNKNOWN", "유성 구간이 너무 짧거나 피치 측정이 불안정해서 음높이 변동을 평가하기 어렵습니다."

    if cv_pitch < 0.15:
        label = "VERY_LOW"
        comment = (
            "음높이 변동이 매우 적은 편입니다. 전체적으로 단조롭게 들릴 수 있어, "
            "강조가 필요한 부분에서 피치를 조금 더 올리거나 내려 주면 좋습니다."
        )
    elif cv_pitch < 0.23:
        label = "LOW_NORMAL"
        comment = (
            "음높이 변동이 비교적 작은 편입니다. 차분하고 안정적인 인상을 주지만, "
            "설명이나 발표에서는 핵심 키워드에서 피치를 조금 더 변화시키면 전달력이 좋아질 수 있습니다."
        )
    elif cv_pitch < 0.33:
        label = "TYPICAL"
        comment = (
            "일반적인 말하기에서 자주 관찰되는 수준의 음높이 변동입니다. "
            "억양이 자연스러워서 듣기에 무난한 편입니다."
        )
    elif cv_pitch < 0.45:
        label = "HIGH"
        comment = (
            "음높이 변동이 비교적 큰 편입니다. 감정 표현이나 강조가 풍부하게 느껴질 수 있지만, "
            "일부 구간에서는 다소 요동치는 느낌을 줄 수도 있습니다."
        )
    else:
        label = "VERY_HIGH"
        comment = (
            "음높이 변동이 매우 큰 편입니다. 감정이 많이 실린 말하기이거나, "
            "긴장을 많이 한 상태일 수 있습니다. 중요한 부분 위주로만 피치 변화를 크게 사용하면 "
            "전체 흐름이 더 안정적으로 느껴질 수 있습니다."
        )

    return label, comment

## ==== 3-2. Energy CV =====
def classify_energy_cv(cv_energy: float):
    if not np.isfinite(cv_energy):
        return "UNKNOWN", "유성 구간이 너무 짧거나 RMS 측정이 불안정해서 크기 변동을 평가하기 어렵습니다."

    if cv_energy < 0.25:
        label = "VERY_LOW"
        comment = (
            "음성 크기 변동이 매우 적습니다. 전체적으로 볼륨이 거의 일정해서 안정적으로 들리지만, "
            "강조해야 할 단어나 문장에서 크기 변화를 조금 더 주면 전달력이 좋아질 수 있습니다."
        )
    elif cv_energy < 0.45:
        label = "LOW_NORMAL"
        comment = (
            "음성 크기 변동이 비교적 작은 편입니다. 차분한 인상을 주지만, "
            "핵심 부분에서 크기를 살짝 더 키우면 메시지가 더 뚜렷하게 전달될 수 있습니다."
        )
    elif cv_energy < 0.70:
        label = "TYPICAL"
        comment = (
            "일반적인 말하기에서 흔히 보이는 수준의 크기 변동입니다. "
            "문장과 단어에 따라 적당한 강약이 들어가 있어 자연스럽게 들립니다."
        )
    elif cv_energy < 1:
        label = "HIGH"
        comment = (
            "음성 크기 변동이 꽤 큰 편입니다. 강조가 많아서 에너지가 느껴질 수 있지만, "
            "일부 구간에서는 소리가 갑자기 커지거나 작아지는 느낌을 줄 수도 있습니다."
        )
    else:
        label = "VERY_HIGH"
        comment = (
            "음성 크기 변동이 매우 큽니다. 마이크 거리 변화, 몸 움직임, 혹은 볼륨 조절이 "
            "불안정했을 수 있습니다. 중요한 부분을 제외하고는 크기 변화를 조금 줄이면 "
            "청자가 더 편하게 들을 수 있습니다."
        )

    return label, comment

## === 3-3. Rate WPM =====
def classify_rate_wpm(rate_wpm: float):
    if not np.isfinite(rate_wpm):
        return "UNKNOWN", "말하기 속도를 평가하기 어렵습니다."
    
    if rate_wpm < 110:
        label = "SLOW"
        comment = (
            "차분하고 또렷하지만 전반적으로 느린 편. 이해하기 쉬운 속도이지만, 에너지가 낮거나 "
            "흐름이 단조롭게 느껴질 수 있음. 중요한 발표에서는 강조 포인트에서 속도 조절이 필요함."
        )
    elif rate_wpm < 160:
        label = "TYPICAL"
        comment = (
            "발표에서 가장 흔하고 듣기 편한 정상 속도. 내용 이해·집중·전달력의 균형이 좋은 구간."
            "대부분의 프레젠테이션이 이 범위에 들어감."
        )
    else:
        label = "FAST"
        comment = (
            "빠른 말하기. 에너지가 높고 생동감 있지만, 정보량이 많을 경우 청중의 이해가 떨어질 수 있음."
            "긴장으로 인해 빨라졌을 가능성도 있으며, 문장 끝 처리나 뉘앙스가 희생될 수 있음."
        )

    return label, comment





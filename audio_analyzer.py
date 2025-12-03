# app/utils/audio_analyzer.py
import os
import json
import librosa
import numpy as np
from analyzer_function import get_voiced_mask_from_words, compute_energy_stats_segment, compute_pitch_cv_segment, compute_final_boundary_features_for_segment

def analyze_segments(audio_path: str):
    # clova stt로 부터 json 결과 받기
    base_name = os.path.splitext(audio_path)[0]
    json_output_file = base_name + '.json'

    with open(json_output_file, "r", encoding="utf-8") as f:
        result = json.load(f)

    # 음성 파일 로드 및 RMS 계산
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = 2048
    hop_length = 256

    # RMS (음성 크기)
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]  # (1,T) -> (T,)

    # F0 (음성 높낮이)
    f0, _, _ = librosa.pyin(
        y,
        fmin=80,
        fmax=300,
        frame_length=frame_length,
        hop_length=hop_length,
        sr=sr
    )

    # 프레임 시간 (초)
    frame_idx = np.arange(rms.shape[0])
    frame_times = (frame_idx * hop_length + hop_length / 2.0) / sr

    # STT 결과 기반 유성 마스크 생성
    full_voice_masked = get_voiced_mask_from_words(rms, sr, hop_length, result)

    # 초기화
    result_text = ""
    analyzed = []
    id=0

    for seg in result:
        # 문장 단위 정보 (시간, 텍스트)
        seg_start, seg_end = seg["start"]/1000, seg["end"]/1000
        seg_text = seg["text"].strip()
        result_text += " " + seg_text

        # 이 문장에 속하는 프레임 인덱스
        y_seg = (frame_times >= seg_start) & (frame_times <= seg_end)

        if len(y_seg) > 0:

            # 이 문장 구간 + 유성 마스크 둘 다 만족하는 프레임
            seg_voice_masked = y_seg & full_voice_masked

            rms_seg = rms[seg_voice_masked]
            f0_seg  = f0[seg_voice_masked]

            # dB 계산
            mean_r, std_r, cv_energy = compute_energy_stats_segment(
                rms=rms_seg,
                silence_thresh=1e-6
            )

            # pitch 계산
            mean_st, std_st, cv_pitch = compute_pitch_cv_segment(
                f0_hz=f0_seg,
                f0_min=1e-3
            )
            
            # 문장 끝 경계 특징 계산
            final_db_drop, final_db_slope, final_pitch_drop, final_pitch_slope = compute_final_boundary_features_for_segment(
                rms=rms,
                f0_hz=f0,
                frame_times=frame_times,
                seg_start=seg_start,
                seg_end=seg_end
            )

            # 말하기 속도 계산 (wpm)
            words_count = len(seg_text.split())
            duration_min = (seg_end - seg_start) / 60
            rate_wpm = words_count / duration_min if duration_min > 0 else 0

        # 문장 단위 정보 구성
        segment_info ={
            "id": id,
            "text": seg_text,
            "start": seg_start,
            "end": seg_end,
            "energy": {
                "mean_rms": round(mean_r, 2),
                "std_rms": round(std_r, 2),
                "cv": round(cv_energy, 4)
            },
            "pitch": {
                "mean_st": round(mean_st, 2),
                "std_st": round(std_st, 2),
                "cv": round(cv_pitch, 4)
            },
            "wpm":{
                "word_count": words_count,
                "rate_wpm": round(rate_wpm, 1),
                "duration_sec": round(seg_end - seg_start, 3)
            },
            "final_boundary": {
                "final_db_drop": round(final_db_drop, 2),
                "final_db_slope": round(final_db_slope, 4),
                "final_pitch_drop_semitone": round(final_pitch_drop, 2),
                "final_pitch_slope": round(final_pitch_slope, 4)
            },
            "words" : []
        }
        id+=1

        # 단어 단위 분석
        if "words" in seg:
            for w in seg["words"]:
                w_text = w[2].strip()
                w_start, w_end = w[0]/1000, w[1]/1000
                w_start_samp, w_end_samp = int(w_start*sr), int(w_end*sr)
                y_word = y[w_start_samp:w_end_samp]

                if len(y_word) == 0:
                    continue

                # --- dB
                w_rms = librosa.feature.rms(y=y_word)
                db = float(np.mean(librosa.amplitude_to_db(w_rms, ref=1.0)))

                # --- pitch
                w_f0, _, _ = librosa.pyin(
                    y_word,
                    fmin=80,
                    fmax=300,
                    sr=sr
                )
                pitch_vals = w_f0[~np.isnan(w_f0)]
                pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
                pitch_std = float(np.std(pitch_vals)) if pitch_vals.size else 0.0

                duration = w_end - w_start

                segment_info["words"].append({
                    "text": w_text,
                    "start": w_start,
                    "end": w_end,
                    "metrics": {
                        "dB": round(db, 2),
                        "pitch_mean_hz": round(pitch_mean, 2),
                        "pitch_std_hz": round(pitch_std, 2),
                        "duration_sec": round(duration, 3)
                    }
                })

        analyzed.append(segment_info)

    return {"text": result_text, "segments": analyzed, "duration": float(len(y) / sr)}

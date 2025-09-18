# pip install numpy scipy librosa soundfile
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import wiener

# -------------------------
# 유틸 함수들
# -------------------------
def load_audio(path, sr=16000, mono=True):
    # librosa는 m4a도 로딩 가능(시스템에 ffmpeg 필요). 안정성을 위해 wav 권장.
    y, _sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr

def save_audio(path, y, sr):
    sf.write(path, y, sr)

def rms(x, eps=1e-12):
    return np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps)

def apply_gain(x, gain, peak_guard=0.999):
    y = x * gain
    peak = np.max(np.abs(y)) + 1e-12
    if peak > peak_guard:
        y = y * (peak_guard / peak)  # 간단한 피크 리미트
    return y

def fade_io(x, sr, fade_ms=10):
    n = int(sr * fade_ms / 1000.0)
    if n <= 1: 
        return x
    w = np.linspace(0, 1, n)
    y = x.copy()
    y[:n] *= w           # fade-in
    y[-n:] *= w[::-1]    # fade-out
    return y

def wiener_denoise(x, mysize=29, noise_power=None):
    """
    scipy.signal.wiener는 1D 위너 필터(로컬 평균 기반)로 간단한 노이즈 저감 효과.
    더 강력한 spectral Wiener를 원하면 noisereduce 등으로 대체 가능.
    """
    # mysize는 홀수 권장. (프레임 길이 느낌)
    if mysize % 2 == 0:
        mysize += 1
    y = wiener(x, mysize=mysize, noise=noise_power)
    return y.astype(np.float32)

def time_to_sample(t, sr):
    return max(0, int(round(t * sr)))

# -------------------------
# 핵심 파이프라인
# -------------------------
def normalize_redo_segment(
    orig_path: str,
    redo_path: str,
    ref_start: float,
    ref_end: float,
    out_seg_path: str = "redo_normalized.wav",
    out_mix_path: str | None = None,
    target_sr: int = 16000,
    wiener_size: int = 29,
    gain_clip: tuple[float, float] = (0.25, 4.0),  # 과도한 게인 방지
    fade_ms: int = 10
):
    """
    orig_path : 원본 전체 파일(참조 구간 포함)
    redo_path : 재녹음 파일(해당 문장만 따로 녹음)
    ref_start/ref_end : 원본에서 '그 문장의' 기준 구간 (초)
    out_seg_path : 보정된 재녹음 구간 저장 경로
    out_mix_path : (선택) 원본 타임라인에 보정된 구간을 삽입한 전체 믹스 저장 경로
    """
    # 1) 로드
    orig, sr = load_audio(orig_path, sr=target_sr)
    redo, _ = load_audio(redo_path, sr=target_sr)

    # 2) 기준 구간 추출(RMS)
    a = time_to_sample(ref_start, sr)
    b = time_to_sample(ref_end, sr)
    ref_seg = orig[a:b]
    if len(ref_seg) < sr * 0.2:
        raise ValueError("참조 구간이 너무 짧습니다(≥0.2 s 권장).")

    ref_rms = rms(ref_seg)
    redo_rms = rms(redo)

    # 3) 게인(에너지 매칭)
    if redo_rms < 1e-9:
        raise ValueError("재녹음 구간의 RMS가 0에 가깝습니다. 녹음 파일을 확인하세요.")
    raw_gain = ref_rms / redo_rms
    # 과도한 게인 방지
    gain = np.clip(raw_gain, gain_clip[0], gain_clip[1])

    redo_gain = apply_gain(redo, gain)

    # 4) Wiener 필터(노이즈 저감)
    redo_denoised = wiener_denoise(redo_gain, mysize=wiener_size)

    # 5) 페이드 인/아웃(경계 클릭 방지)
    redo_final = fade_io(redo_denoised, sr, fade_ms=fade_ms)

    # 6) 결과 저장(보정된 재녹음 구간)
    save_audio(out_seg_path, redo_final, sr)

    # 7) (선택) 원본 타임라인에 삽입하여 완성본 출력
    if out_mix_path is not None:
        mix = orig.copy()
        # 길이 차이를 고려: ref 구간 길이와 재녹음 길이가 다르면 두 가지 전략:
        # (A) 원본 길이에 맞춰 재샘플링/타임스트레치 (보통 비권장)
        # (B) ref 구간을 그대로 덮고, 길이 차이는 허용(타이밍이 중요하면 타임스트레치 별도 수행)
        repl_len = min(len(mix[a:b]), len(redo_final))
        mix[a:a+repl_len] = redo_final[:repl_len]
        save_audio(out_mix_path, mix, sr)

    return {
        "sr": sr,
        "ref_rms": float(ref_rms),
        "redo_rms": float(redo_rms),
        "raw_gain": float(raw_gain),
        "applied_gain": float(gain),
        "out_seg_path": out_seg_path,
        "out_mix_path": out_mix_path,
    }

# -------------------------
# 사용 예시
# -------------------------
if __name__ == "__main__":
    """
    예) 원본(orig.wav)에서 10.2~14.8초가 해당 문장이고,
        문장만 다시 녹음한 파일이 redo.wav 라고 할 때:
    """
    info = normalize_redo_segment(
        orig_path="test.m4a",
        redo_path="redo.m4a",
        ref_start=10.2,
        ref_end=14.8,
        out_seg_path="redo_normalized.m4a",
        out_mix_path="mixed_out.m4a",   # 원본에 덮어쓴 결과까지 만들고 싶으면 지정
        target_sr=16000,
        wiener_size=29,      # 21~41 정도에서 튜닝
        gain_clip=(0.25, 4.0),
        fade_ms=12
    )
    print(info)

# dfn_full_pipeline.py
# 사용법(예):
#   python dfn_full_pipeline.py --in "C:/audio/voice.m4a" --outdir "C:/audio/out" --alpha 0.7 --atten-lim -12 --sr 16000
#
# 동작:
#   1) (필요시) m4a -> wav 변환 (모노, 지정 SR)
#   2) DeepFilterNet(DFN)으로 노이즈 제거 (CLI 호출)
#   3) 원본 wav와 DFN 결과를 alpha 비율로 블렌딩 -> <stem>_blend.wav 저장
#
# 메모:
#   - DFN의 -o/--output-dir는 "폴더"입니다(파일명 아님). 결과는 <stem>_df.wav 형식으로 저장됩니다.
#   - ffmpeg와 deepFilter가 PATH에 있어야 합니다. (터미널에서 `ffmpeg -version`, `deepFilter --help`로 확인)

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def run(cmd: list[str], ok_codes=(0,)) -> subprocess.CompletedProcess:
    """subprocess 실행 헬퍼 (stdout/stderr를 모두 출력, 에러시 예외)"""
    print(">>", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        # deepFilter/ffmpeg는 정상 로그를 stderr로 출력하는 경우가 흔합니다.
        print(proc.stderr.strip())
    if proc.returncode not in ok_codes:
        raise RuntimeError(f"Command failed (code {proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def ensure_tool(name: str, version_args: list[str]):
    """외부 실행파일이 PATH에 있는지 확인"""
    try:
        run([name] + version_args)
    except Exception as e:
        raise EnvironmentError(f"'{name}' 실행 파일을 찾을 수 없습니다. PATH를 확인하세요. ({e})")


def m4a_to_wav(m4a: Path, wav: Path, sr: int = 16000):
    """ffmpeg로 m4a/mp3 등을 wav(모노, sr)로 변환"""
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(m4a), "-ac", "1", "-ar", str(sr), str(wav)]
    run(cmd)


def deepfilter(wav_in: Path, out_dir: Path, atten_lim: float = -12) -> Path:
    """DeepFilterNet CLI로 노이즈 제거 -> 반환: 생성된 *_df.wav 경로"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["deepFilter", str(wav_in), "-o", str(out_dir), "--atten-lim", str(int(atten_lim))]
    run(cmd)
    # 기본 출력 파일명 규칙: <stem>_df.wav
    expect = out_dir / f"{wav_in.stem}_df.wav"
    if expect.exists():
        return expect
    # 혹시 규칙이 다른 경우를 대비해 유사 파일 탐색
    cands = sorted(out_dir.glob(f"{wav_in.stem}*_DeepFilterNet3.wav"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"DFN 출력 파일을 찾지 못했습니다: {expect}")


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def peak_normalize(x: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    """간단 피크 노멀라이즈 (True peak 근사)"""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-9:
        return x
    target = 10 ** (peak_dbfs / 20.0)
    gain = min(1.0, target / peak)
    return x * gain


def blend(orig_wav: Path, dfn_wav: Path, out_wav: Path, alpha: float = 0.7):
    """alpha*DFN + (1-alpha)*Original 블렌딩 후 저장"""
    y0, sr0 = sf.read(orig_wav)
    y1, sr1 = sf.read(dfn_wav)
    y0 = to_mono(y0)
    y1 = to_mono(y1)
    if sr0 != sr1:
        raise ValueError(f"SR mismatch: original={sr0}, dfn={sr1}")

    # 길이 맞추기
    L = max(len(y0), len(y1))
    if len(y0) < L: y0 = np.pad(y0, (0, L - len(y0)))
    if len(y1) < L: y1 = np.pad(y1, (0, L - len(y1)))

    y = float(alpha) * y1 + (1.0 - float(alpha)) * y0
    y = peak_normalize(y, -1.0)  # 피크 -1 dBFS
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, y.astype(np.float32), sr0)
    print(f"[OK] blended -> {out_wav}")


def main():
    ap = argparse.ArgumentParser(description="m4a->wav -> DFN denoise -> blending 파이프라인")
    ap.add_argument("--in", dest="in_path", required=True, help="입력 오디오 경로 (m4a/wav)")
    ap.add_argument("--outdir", required=True, help="출력 폴더")
    ap.add_argument("--alpha", type=float, default=0.7, help="블렌딩 DFN 가중치(0~1)")
    ap.add_argument("--atten-lim", type=float, default=-12, help="DFN atten-lim (음수 dB)")
    ap.add_argument("--sr", type=int, default=16000, help="WAV 변환 샘플레이트(모노)")
    ap.add_argument("--keep-tmp", action="store_true", help="임시 wav 파일 보존")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.outdir)
    if not in_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {in_path}")

    # 외부 도구 확인
    ensure_tool("ffmpeg", ["-version"])
    ensure_tool("deepFilter", ["--help"])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    # 1) 입력이 m4a/mp3면 wav로 변환, wav면 복사/정규화 경로로 사용
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_wav = tmp_dir / f"{stem}.wav"

    if in_path.suffix.lower() == ".wav":
        # 그대로 작업용 위치로 복사 (SR/채널이 다를 수 있으니 ffmpeg로 강제 통일을 권장)
        run(["ffmpeg", "-y", "-i", str(in_path), "-ac", "1", "-ar", str(args.sr), str(work_wav)])
    else:
        # m4a/mp3 등 -> wav(모노, sr)
        m4a_to_wav(in_path, work_wav, sr=args.sr)

    # 2) DeepFilterNet 노이즈 제거
    dfn_dir = out_dir / "denoised"
    dfn_wav = deepfilter(work_wav, dfn_dir, atten_lim=args["atten_lim"] if isinstance(args, dict) and "atten_lim" in args else args.atten_lim)

    # 3) 블렌딩
    out_blend = out_dir / f"{stem}_blend.wav"
    blend(work_wav, dfn_wav, out_blend, alpha=args.alpha)

    # 4) 임시 파일 정리
    if not args.keep_tmp:
        try:
            if work_wav.exists():
                work_wav.unlink()
            # 비었으면 _tmp 삭제
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except Exception as e:
            print(f"[WARN] 임시 파일 정리 중 경고: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()

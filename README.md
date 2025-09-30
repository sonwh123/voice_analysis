# 1. 가상환경 만들기

DFN은 Python 3.10 권장합니다.

```powershell
conda create -n dfn python=3.10 -y
conda activate dfn
```

---

# 2. 필수 툴 설치

### (A) Python 패키지

```powershell
pip install numpy soundfile
```

### (B) DeepFilterNet

```powershell
pip install deepfilternet
```

### (C) FFmpeg

- 방법 1: Conda로 설치
    
    ```powershell
    conda install -c conda-forge ffmpeg -y
    ```
    
- 방법 2: [ffmpeg.org](https://ffmpeg.org/download.html)에서 Windows build를 받아 설치하고, `bin` 폴더를 PATH에 추가.

설치 확인:

```powershell
ffmpeg -version
```

### (D) DeepFilterNet 모델 다운로드

```powershell
deepFilter --download-model dfn3
```

---

# 3. 경로/환경 확인

1. `deepFilter --help` 실행 → CLI가 정상 출력되면 OK.
2. `%LOCALAPPDATA%\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3` 안에 모델이 다운로드되어야 합니다.

---

# 4. 스크립트 실행

```powershell
conda activate dfn

python dfn_full_pipeline.py --in "C:\audio\voice.m4a" --outdir "C:\audio\out" --alpha 0.5 --atten-lim -15 --sr 48000
```

출력 결과:

- `C:\audio\out\denoised\voice_DeepFilterNet3.wav` → DFN 노이즈 제거 결과
- `C:\audio\out\voice_blend.wav` → 최종 블렌딩 파일

---

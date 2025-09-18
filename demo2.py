from faster_whisper import WhisperModel

def analyze_segments(audio_path, model_name="large-v3", language="ko"):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        vad_parameters={"threshold": 0.5},
        word_timestamps=True,   # 단어 단위 시간 확보
        beam_size=5,
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"text": w.word, "start": w.start, "end": w.end})
    return words


# 사용 예시
if __name__ == "__main__":
    data = analyze_segments("test.m4a", model_name="large-v3", language="ko")
    import json
    print(json.dumps(data, ensure_ascii=False, indent=2))

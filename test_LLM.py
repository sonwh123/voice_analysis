from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1️⃣ 모델 설정
model_name = "EleutherAI/polyglot-ko-1.3b"
device = "cpu"  # Mac MPS 대신 CPU로 강제 설정
print(f"현재 사용 디바이스: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 2️⃣ 프롬프트
pitch = 210
speed = 5.8
volume = 68

prompt = f"""
너는 한국어 스피치 코치야.
다음 데이터를 보고 한 문장으로 피드백을 작성해줘.
- 평균 피치: {pitch} Hz
- 속도: {speed} 음절/초
- 볼륨: {volume} dB
"""

# 3️⃣ 실행
inputs = tokenizer(prompt, return_tensors="pt").to(device)
if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)

# 4️⃣ 결과 정제
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
result = result.split("dB")[-1].strip()

print("\n[🔍 스피치 피드백 결과]\n")
print(result)

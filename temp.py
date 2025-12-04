from classify_text_service import classify_text_into_sections
from temp2 import split_llm_sections_into_sentences_with_clova_timestamps
import json
import os


# clova stt로 부터 json 결과 받기
base_name = os.path.splitext("voice1.m4a")[0]
json_output_file = base_name + '.json'

with open(json_output_file, "r", encoding="utf-8") as f:
    result = json.load(f)

text = ""

for seg in result:
    seg_text = seg["text"].strip()
    text += " " + seg_text

# print(text)

sections = classify_text_into_sections(text)
# 결과 출력
# sections_result.json 파일 저장
json_output_file = "sections_result.json"
with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=2)

# print(word_timestamps)

final = split_llm_sections_into_sentences_with_clova_timestamps(sections, result)

with open("sections_final_result.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
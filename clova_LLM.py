# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import requests
import json
import os
import uuid

load_dotenv()

class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        collected_text = ""

        with requests.post(self._host + '/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8").strip()

                # ✅ 스트리밍 종료
                if decoded in ["data:[DONE]", "data: [DONE]"]:
                    break

                # ✅ event:result인 경우에만 처리
                if decoded.startswith("event:result"):
                    continue  # event 이름은 건너뜀

                if decoded.startswith("data:"):
                    try:
                        data_json = json.loads(decoded.replace("data:", "").strip())

                        # ✅ event:result 데이터만 잡기
                        if data_json.get("message") and data_json["message"]["role"] == "assistant":
                            # 최종 결과 저장
                            result_content = data_json["message"].get("content", None)
                    except json.JSONDecodeError:
                        pass

        return result_content

def get_feedback_from_LLM(sentence_info: dict):
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id=str(uuid.uuid4())
    )

    preset_text = {
        "role":"system",
        "content":"""
        당신은 전문 스피치 코치입니다.

        아래 JSON은 한 문장에 대한 음성 분석 결과입니다.
        각 항목에는 (1) 라벨(label), (2) 해당 라벨에 대한 간단한 설명(comment)이 포함되어 있습니다.
        라벨 이름 자체를 해석하려고 하지 말고, comment에 적힌 내용을 바탕으로 자연스럽고 구체적인 피드백을 작성해주세요.

        [입력 JSON]
        {sentence_json}

        [작성 지침]

        1. 먼저 이 문장의 말하기 방식에서 좋은 점을 1문장 정도로 짚어주세요.
        2. 그 다음, volume_stability / ending_pattern / pitch_stability / pitch_ending / rate_level에 해당하는 comment들을 종합해서,
        이 문장에서 드러나는 말하기 습관과 아쉬운 점을 1~2문장으로 설명해주세요.
        3. 마지막으로, 연습할 때 바로 적용할 수 있는 구체적인 개선 방향을 1문장 정도로 제안해주세요.
        4. JSON 안의 label 이름이나 키 이름은 그대로 반복하지 말고, comment 내용을 자연스럽게 풀어서 말해주세요.
        5. 문체는 따뜻하고 격려하는 톤으로, 비난보다는 “이렇게 하면 더 좋아질 것 같다”는 방향으로 작성해주세요.
        6. 전체 길이는 3~4문장 정도의 하나의 단락으로 작성해주세요.
        7. 답변은 반드시 한국어로 작성해주세요.

        """
    }

    user_message = {
        "role": "user",
        "content": f"{sentence_info}"
    }

    request_data = {
        'messages': [preset_text,user_message],
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1500,
        'temperature': 0.5,
        'repeatPenalty': 1.05,
        'stopBefore': [],
        'includeAiFilters': True
    }

    # print(preset_text)
    # completion_executor.execute(request_data)
    answer = completion_executor.execute(request_data)
    # print(answer)

    # 1️⃣ 문자열에 있는 이스케이프 제거 → 실제 JSON 객체로 변환
    cleaned_answer = answer

    # 2️⃣ 보기 좋게 출력
    # print(json.dumps(cleaned, indent=2, ensure_ascii=False))
    with open("LLM_result.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_answer, f, ensure_ascii=False, indent=2)

    return cleaned_answer
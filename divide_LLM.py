# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import requests
import json
import os

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

        with requests.post(self._host + '/v3/chat-completions/HCX-005',
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

if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id='7cb606aa44574fc0b35cc2c620f5e7d3'
    )

    preset_text = [{
        "role":"system",
        "content":"""
        당신은 전문적인 한국어 스피치 코치입니다.발표나 면접 등 다양한 상황에서 화자의 대본을 내용별로 문단을 나누는 것이 당신의 역할입니다.
        
        앞으로 사용자의 대본을 줄 것입니다.
        출력 형식에 맞춰서 데이터를 주면 내용별로 문단을 나눠서 출력합니다.
                
        [요청사항]
        - 대본의 순서나 내용을 바꾸지 말고 그대로 출력하세요.
        - 답변을 json 형식으로 변환해서 답변합니다.
        - section은 7개 미만으로 나눕니다.
        - part의 내용은 서론, 본론, 결론으로만 나타냅니다.
        - content는 반드시 대본의 순서대로 대본의 전체를 포함해야 합니다.
        - 처음과 마지막이 인사거나 마무리 멘트이더라도 생략하지 말고 포함해야 합니다.
        - JSON 외에 설명, 인사말, 코드블록 표시(````json`, ``` 등)은 절대 포함하지 마세요.
                
        [출력 형식]
        [
            {
            "title": "대본 주제",
            "sections": [
                {
                "id": 1,
                "part": "서론",
                "content": "대본1"
                },
                {
                "id": 2,
                "part": "본론1",
                "content": "대본2"
                },
                {
                "id": 2,
                "part": "본론2",
                "content": "대본2"
                },
            ...
                {
                "id": 3,
                "part": "결론",
                "content": "대본3"
                }
            }
        ]
        """
    }]

    # 대본 가져오기
    file_path = "sample.txt"  # 불러올 파일 경로

    with open(file_path, "r", encoding="utf-8") as f:
        script = f.read() 

    user_message = [{
        "role": "user",
        "content": """
        {script}
        """
    }]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1024,
        'temperature': 0.5,
        'repeatPenalty': 1.1,
        'stopBefore': [],
        'includeAiFilters': True
    }

    # print(preset_text)
    # completion_executor.execute(request_data)
    answer = completion_executor.execute(request_data)
    print(answer)

    # 1️⃣ 문자열에 있는 이스케이프 제거 → 실제 JSON 객체로 변환
    cleaned_answer = json.loads(answer)

    # 2️⃣ 보기 좋게 출력
    # print(json.dumps(cleaned, indent=2, ensure_ascii=False))
    with open("LLM_divide_result.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_answer, f, ensure_ascii=False, indent=2)
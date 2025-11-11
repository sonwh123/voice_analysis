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
        request_id=str(uuid.uuid4())
    )

    preset_text = {
        "role":"system",
        "content":"""
        당신은 전문적인 한국어 스피치 코치입니다.  
        발표나 면접 등 다양한 상황에서 화자의 대본을 주제 흐름에 따라 문단 단위로 나누는 것이 당신의 역할입니다.

        [목표]
        - 사용자가 제공한 대본을 **요약하거나 수정하지 말고**, 
        **원문 그대로** 각 문단의 내용을 "content" 필드에 포함시켜야 합니다.
        - 문단의 구분만 수행하세요. 문장의 삭제, 순서 변경, 문체 변경, 요약은 절대 하지 마세요.

        [요청사항]
        - 대본의 순서나 내용을 바꾸지 마세요.
        - 답변은 반드시 순수 JSON 형식으로만 출력하세요.
        - JSON 외의 설명, 인사말, 코드블록(````json`, ``` 등)은 절대 포함하지 마세요.
        - section은 7개 미만으로 나누세요.
        - part의 내용은 "서론", "본론", "결론" 중 하나로만 지정하세요.
        - "content"에는 반드시 원문 전체를 포함하세요.
        - 인사나 마무리 멘트도 생략하지 말고 포함하세요.
        - 문단 구분 외에는 어떠한 요약, 재구성, 문장 수정도 하지 마세요.
                        
        [출력 형식]
        [
            {
                "title": "대본 주제",
                "sections": [
                    {
                        "id": 1,
                        "part": "서론",
                        "content": "..."
                    },
                    {
                        "id": 2,
                        "part": "본론1",
                        "content": "..."
                    },
                    {
                        "id": 3,
                        "part": "본론2",
                        "content": "..."
                    },
                    ...
                    {
                        "id": 4,
                        "part": "결론",
                        "content": "..."
                    }
                ]
            }
        ]
        """
    }

    # 대본 가져오기
    file_path = "sample.txt"  # 불러올 파일 경로

    with open(file_path, "r", encoding="utf-8") as f:
        script = f.read()

    user_message = {
        "role": "user",
        "content": f"""
        아래는 발표 대본입니다. 
        대본을 원문 그대로 문단만 나누세요.

        [대본 시작]
        {script}
        """
    }

    request_data = {
        'messages': [preset_text,user_message],
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
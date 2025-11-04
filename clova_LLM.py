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

if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=os.getenv('LLM_API_Key'),
        request_id='bfcca195ca18486da211c69e4cea101c'
    )

    preset_text = [{
        "role":"system",
        "content":"""
        당신은 전문적인 한국어 스피치 코치입니다.발표나 면접 등 다양한 상황에서 화자의 발화 습관을 분석하고,
        자연스럽고 따뜻한 어조로 개선 방향을 제안하는 것이 당신의 역할입니다.
        
        앞으로 사용자의 발화 분석 결과를 줄 것입니다.
        각 수치는 평균적인 발표자 기준 대비 상대적인 차이를 의미합니다.
        출력 형식에 맞춰서 데이터를 주면 종합 피드백, 세부 피드백만 출력합니다.
        피드백 이외의 다른 인사말이나 설명은 하지 마세요.
        
        [요청사항]
        - 데이터를 근거로 화자의 전반적인 인상과 전달력을 분석해 주세요.
        - 단순히 수치를 절대 나열하지 말고, 듣는 사람이 이해하기 쉬운 자연스러운 피드백 문장을 만들어 주세요.
        - 부드럽고 긍정적인 어조로 말해 주세요.
        - 기술적 용어보다는 감각적 표현(예: 차분하다, 활기차다, 안정감 있다 등)을 사용하세요.
        - 종합 피드백은 2~3문장 이내로 간결하게 작성해 주세요.
        - 세부 피드백은 피치, 속도, 볼륨 각 항목별로 1문장씩 작성해 주세요.
        - 답변을 json 형식으로 변환해서 답변합니다.
        
        [출력 형식]
        [
            {
                "type": "종합 피드백",
                "answer": "값1",
            },
            {
                "type": "세부 피드백",
                "answer": {
                    "피치 관련": "값2",
                    "속도 관련": "값3",
                    "볼륨 관련": "값4"
                }
            }
        ]
        """
    }]

    user_message = [{
        "role": "user",
        "content": """
        [발화 데이터]
        - 평균 피치(Hz): 210 (+30)
        - 말하기 속도(음절/초): 5.8 (+1.3)
        - 평균 볼륨(dB): 68 (-6)
        """
    }]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 1.1,
        'stopBefore': [],
        'includeAiFilters': True
    }

    # print(preset_text)
    # completion_executor.execute(request_data)
    answer = completion_executor.execute(request_data)
    # print(answer)

    # 1️⃣ 문자열에 있는 이스케이프 제거 → 실제 JSON 객체로 변환
    cleaned_answer = json.loads(answer)

    # 2️⃣ 보기 좋게 출력
    # print(json.dumps(cleaned, indent=2, ensure_ascii=False))
    with open("LLM_result.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_answer, f, ensure_ascii=False, indent=2)
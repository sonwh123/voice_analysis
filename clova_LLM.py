# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import requests
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

        with requests.post(self._host + '/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    print(line.decode("utf-8"))


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
        - 단순히 수치를 나열하지 말고, 듣는 사람이 이해하기 쉬운 자연스러운 피드백 문장을 만들어 주세요.
        - 부드럽고 긍정적인 어조로 말해 주세요.
        - 기술적 용어보다는 감각적 표현(예: 차분하다, 활기차다, 안정감 있다 등)을 사용하세요.
        
        [출력 형식]
        1. 종합 피드백 (2~3문장)
        2. 세부 피드백 (각각 1문장)
            - 피치 관련:
            - 속도 관련:
            - 볼륨 관련:
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

    print(preset_text)
    completion_executor.execute(request_data)

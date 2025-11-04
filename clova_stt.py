from dotenv import load_dotenv
import requests
import json
import os

load_dotenv()

class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = os.getenv('Clova_Speech_Invoke_URL')
    # Clova Speech secret key
    secret = os.getenv('Clova_Speech_Secret_Key')

    def req_url(self, url, completion, callback=None, userdata=None,
    	forbiddens=None, boostings=None, wordAlignment=True,
        	fullText=True, diarization=None, sed=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None,
    	userdata=None, forbiddens=None, boostings=None,wordAlignment=True,
        	fullText=True, diarization=None, sed=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None,
    	forbiddens=None, boostings=None, wordAlignment=True, 
        	fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body,
            			ensure_ascii=False).encode('UTF-8'),
                        		'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url
        			+ '/recognizer/upload', files=files)
        return response

if __name__ == '__main__':
    res = ClovaSpeechClient().req_upload(file='./voice.m4a',
    		completion='sync')
    
    # --- ⭐ 응답 객체(res)만을 사용한 결과 처리 및 출력 ⭐ ---
    if res.status_code == 200:
        try:
            result = res.json()
        except json.JSONDecodeError:
            print(f"\n❌ JSON 디코딩 오류: 응답 텍스트를 JSON으로 변환할 수 없습니다.")
            print(f"응답 텍스트:\n{res.text}")
            exit()
            
        # 1. 'segments' 필드만 추출하여 저장 준비
        segments_data = result.get("segments", [])
        
        # 2. segments 배열을 순회하며 'diarization' 및 'speaker' 필드 제거
        final_segments = []
        for seg in segments_data:
            # 딕셔너리 복사 (원본 데이터 보호)
            clean_seg = seg.copy() 
            
            # 'speaker' 정보가 포함된 필드 삭제
            if 'speaker' in clean_seg:
                del clean_seg['speaker']
            
            # 'diarization' 정보가 포함된 필드 삭제 (API 응답 구조에 따라 존재 가능)
            if 'diarization' in clean_seg:
                del clean_seg['diarization']
                
            final_segments.append(clean_seg)
            
        
        # 3. JSON 파일 저장
        base_name = os.path.splitext("./voice")[0]
        json_output_file = base_name + '_segments_pure.json'
        
        # 순수 segments 배열만 파일에 저장
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_segments, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 'segments' 순수 JSON 파일 저장 완료: {json_output_file}")

    else:
        print(f"\n--- ❌ API 요청 실패 ---")
        print(f"상태 코드: {res.status_code}")
        print(f"응답 내용:\n{res.text}")
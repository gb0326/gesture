from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators import gzip
from tensorflow.keras.models import load_model
from django.conf import settings

import openai
import cv2
import mediapipe as mp
import numpy as np
import os
import atexit

actions = ['meet', 'nice', 'hello']
seq_length = 30

model = load_model('models/model_b.keras')

def hand_gesture(request):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 웹캠 스트리밍을 위한 generator 함수 정의
    def gen_frames():
        cap = cv2.VideoCapture(0)
        seq = []
        prev_action = None
        
        @atexit.register
        def cleanup():
            # 카메라가 꺼질 때 example.txt 파일 초기화
            user_id = request.session.session_key
            file_path = get_user_file_path(user_id)
            if os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("")

        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                    angle = np.degrees(angle)
                    d = np.concatenate([joint.flatten(), angle])
                    seq.append(d)

                    # 좌표 시퀀스가 시퀀스 길이에 도달하면 모델에 입력하고 결과 예측
                    if len(seq) >= seq_length:
                        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                        y_pred = model.predict(input_data).squeeze()

                        user_id = request.session.session_key
                        if not user_id:
                        # 세션이 없다면 새로 생성
                            request.session.create()
                            user_id = request.session.session_key

                        # 이전 동작과 현재 동작이 다르거나, 현재 동작의 확률이 일정 이상인 경우에만 자막 출력
                        if prev_action != actions[np.argmax(y_pred)] or y_pred.max() >= 0.9:
                            i_pred = int(np.argmax(y_pred))
                            action = actions[i_pred]
                            
                            if prev_action != action:
                                cv2.putText(img, f'{action.upper()}', org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                file_path = get_user_file_path(user_id)
                                append_to_file(file_path, action)
                                prev_action = action
                                          

            # 카메라 프레임 반환
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



def get_user_file_path(user_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_dir = os.path.join(base_dir, 'data', user_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, 'example.txt')
    return file_path

def append_to_file(file_path, text):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{text}\n")
        
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def get_completion(prompt):
    try:
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message['content']
        print(message)
        return message
    except openai.error.OpenAIError as e:
        if e.code == 'quota_exceeded':
            return "You have exceeded your quota. Please check your OpenAI plan and billing details."
        else:
            print(f"Error occurred: {e}")
            return "There was an error processing your request."

def query_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        prompt = str(prompt)
        
        user_id = request.session.session_key
        if not user_id:
            request.session.create()
            user_id = request.session.session_key

        file_path = get_user_file_path(user_id)
        print(f"파일 경로: {file_path}")
        
        if not os.path.exists(file_path):
            print("파일이 존재하지 않습니다.")
            return JsonResponse({'response': "파일을 찾을 수 없습니다."})

        content = read_text_file(file_path)
        if content is None:
            return JsonResponse({'response': "파일을 읽을 수 없습니다."})

        content += "\n지금까지 나온 단어들을 중복된 단어가 있다면 한 단어만 사용해서 한글로 번역 후 자연스러운 문장으로 만들어줘"
        response = get_completion(content)
        return JsonResponse({'response': response})
    return render(request, 'index.html')
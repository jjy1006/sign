import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

# 인식하는 최대 손 개수
max_num_hands = 1
# 수어 알파벳을 딕셔너리에 저장 
gesture = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i',
           9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
           18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'spacing', 27: 'clear'}

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils  
# 손가락 detection 모듈을 초기화
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

f = open('test.txt', 'w')
# f 라는 변수에 데이터셋을 추가하고 데이터셋 안의 데이터 수치는 ,로 구분을 하며 배열로 저장 
# 각 제스처들의 라벨과 각도가 저장되어 있음
file = np.genfromtxt('dataSet.txt', delimiter=',')
# 마지막열을 제외한 모든 행과 열을 각도 배열에 저장하고, 마지막 열을 라벨 배열에 저장
angleFile = file[:, :-1]
labelFile = file[:, -1]
# 학습데이터:angle, 응답 행렬: label 
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

# knn 알고리즘
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 학습된 KNN 모델 저장
knn.save('knn_model.xml')  # 모델을 'knn_model.xml'로 저장

# 카메라 
cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1
while True:
    # 웹캠에서 이미지 입력 받고 RGB에서 BGR로 변환 
    ret, img = cap.read()
    if not ret:
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    # 손이 감지되면  
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))  # joint = 랜드마크에서 빨간 점, joint는 21개가 있고 x,y,z 좌표이므로 21,3
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]  # 각 joint마다 x,y,z 좌표 저장
            
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1
            
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] 
            
            comparev1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            comparev2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.arccos(np.einsum('nt,nt->n', comparev1, comparev2))

            angle = np.degrees(angle)  # 라디안을 각도로 변환
            if keyboard.is_pressed('a'):
                for num in angle:
                    num = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                f.write('27.000000')
                f.write('\n')
                print('next')
            
            data = np.array([angle], dtype=np.float32)  # 구한 각도 값을 32bit로 변환하여 저장
            
            ret, result, neighbours, dist = knn.findNearest(data, 3) 
            
            index = int(result[0][0])  # 인덱스를 저장
            
            # 만약 인덱스 중 위의 딕셔너리에 같은 것이 있다면 출력 
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += '_'
                        elif index == 27:
                            sentence = sentence[:-1]
                        else:
                            sentence += gesture[index]
                        startTime = time.time()
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)  # 손에 랜드마크 표시
        cv2.putText(img, sentence, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv2.imshow('HandTracking', img)
        cv2.waitKey(1)

        if keyboard.is_pressed('b'):
            break

f.close()

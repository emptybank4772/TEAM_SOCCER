import cv2
import numpy as np
from ultralytics import YOLO
import os

# ----------------- 1. 팀 색상 임계값 정의 (HSV 공간) -----------------
# 주의: 이 값은 사용자의 실제 유니폼 색상에 맞게 조정해야 합니다.
# [Hue, Saturation, Value]

# --- 🅰️ 우리 팀: 붉은색 계열 (Red) ---
# 붉은색 1 (H: 0~10)
LOWER_RED_1 = np.array([0, 100, 50])
UPPER_RED_1 = np.array([10, 255, 255])
# 붉은색 2 (H: 170~180)
LOWER_RED_2 = np.array([170, 100, 50])
UPPER_RED_2 = np.array([180, 255, 255])

# --- 🅱️ 상대 팀: 흰색~하늘색 계열 (Light Blue/Sky Blue) ---
# 하늘색/밝은 파란색 범위 (H: 90~130)
LOWER_OPPONENT = np.array([90, 50, 100])
UPPER_OPPONENT = np.array([130, 255, 255])


# ----------------- 2. 팀 분류 함수 정의 -----------------
def classify_team(image, box_coords):
    """바운딩 박스 내부의 색상을 분석하여 팀 (OURS 또는 OPPONENT)을 반환"""

    x1, y1, x2, y2 = map(int, box_coords)

    # 유니폼 상의만 분석하기 위해 박스의 상단 1/3 ~ 2/3 지점만 사용
    roi = image[y1 + (y2 - y1) // 3: y1 + 2 * (y2 - y1) // 3, x1:x2]

    if roi.size == 0:
        return "OTHER"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # --- A. 붉은색 (우리 팀) 마스크 생성 ---
    mask_red_1 = cv2.inRange(hsv_roi, LOWER_RED_1, UPPER_RED_1)
    mask_red_2 = cv2.inRange(hsv_roi, LOWER_RED_2, UPPER_RED_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    red_pixels = np.sum(mask_red > 0)

    # --- B. 하늘색/흰색 (상대 팀) 마스크 생성 ---
    # 흰색 유니폼도 이 범위에 어느 정도 포함될 수 있도록 채도(S) 최소값을 낮췄습니다 (50)
    mask_opponent = cv2.inRange(hsv_roi, LOWER_OPPONENT, UPPER_OPPONENT)
    opponent_pixels = np.sum(mask_opponent > 0)

    # --- C. 분류 로직 ---
    MIN_PIXEL_THRESHOLD = 100  # 노이즈를 걸러내기 위한 최소 픽셀 수

    if red_pixels > opponent_pixels and red_pixels > MIN_PIXEL_THRESHOLD:
        return "TEAM_OURS"  # 붉은색이 우세
    elif opponent_pixels > red_pixels and opponent_pixels > MIN_PIXEL_THRESHOLD:
        return "TEAM_OPPONENT"  # 하늘색/흰색이 우세
    else:
        # 두 색상 모두 명확하지 않은 경우 (예: 흰색 유니폼이 너무 밝아 마스크에 안 걸리거나, 다른 색상인 경우)
        return "OTHER"

    # ----------------- 3. 메인 실행 코드 -----------------


# 경로는 사용자 환경에 맞게 조정하세요.
model = YOLO('best.pt')
TEST_IMAGE_PATH = 'test_photo.jpg'

img = cv2.imread(TEST_IMAGE_PATH)
results = model.predict(source=TEST_IMAGE_PATH, conf=0.5, save=False)

identified_players = []

if results and results[0].boxes:
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box, cls in zip(boxes, results[0].boxes.cls.cpu().numpy()):
        if int(cls) == 0:  # 'players' 클래스에 대해서만
            team_id = classify_team(img, box)  # 팀 분류 함수 호출

            # (이름은 위치 기반으로 임시 부여)
            player_name = "Player_" + str(len(identified_players) + 1)

            identified_players.append({
                'name': player_name,
                'team': team_id,
                'box': box
            })

# ----------------- 4. 팀별 시각화 (확인용) -----------------
for player in identified_players:
    x1, y1, x2, y2 = map(int, player['box'])
    name = player['name']
    team = player['team']

    color = (0, 0, 0)  # 기본: 검정
    if team == "TEAM_OURS":
        color = (0, 0, 255)  # 붉은색
    elif team == "TEAM_OPPONENT":
        color = (255, 255, 0)  # 노란색 (하늘색과 대비를 위해 사용)
    else:
        color = (100, 100, 100)  # 기타: 회색

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{name} ({team})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite('team_classified_photo_final.jpg', img)
print("✅ 팀 분류 및 시각화 완료. team_classified_photo_final.jpg 확인.")
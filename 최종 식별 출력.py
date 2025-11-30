import cv2
from ultralytics import YOLO
import numpy as np

# ----------------- 1. 수동 이름 매핑 정의 -----------------
# 주의: 이 딕셔너리는 '각 사진'의 특정 '선수 ID'에 따라 수동으로 정의해야 합니다.
# YOLO는 탐지된 순서대로 객체에 ID를 부여할 수 있지만, 안정성을 위해 위치 정보를 사용합니다.
# 여기서는 편의상 탐지된 모든 'player'에게 임시 이름 리스트를 순서대로 부여한다고 가정합니다.

# ★ 실제 선수 이름 리스트 (탐지된 선수 수에 맞게 준비) ★
PLAYER_NAMES = ["Son Heung-min", "Lee Kang-in", "Hwang Hee-chan", "Kim Min-jae", "Gue-sung Cho", "Unidentified Player"]

# ----------------- 2. 모델 및 이미지 로드 -----------------
model = YOLO('best.pt')
TEST_IMAGE_PATH = 'test_photo.jpg'

# 3. 이미지 로드 및 예측 실행
img = cv2.imread(TEST_IMAGE_PATH)
if img is None:
    print("이미지 파일을 로드할 수 없습니다.")
    exit()

# 예측 (predict는 결과를 객체 리스트로 반환)
results = model.predict(source=TEST_IMAGE_PATH, conf=0.5, save=False)

# ----------------- 4. 결과 분석 및 시각화 -----------------
if results and results[0].boxes:
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표 (xyxy)

    # 선수 객체만 필터링 (클래스 ID '0'이 'player'라고 가정)
    player_boxes = [box for box, cls in zip(boxes, results[0].boxes.cls.cpu().numpy()) if int(cls) == 0]

    # 바운딩 박스를 왼쪽에서 오른쪽 순으로 정렬 (위치 기반 식별을 돕기 위함)
    # x_min 좌표를 기준으로 정렬
    player_boxes.sort(key=lambda x: x[0])

    # 이미지에 박스와 이름 그리기
    for i, box in enumerate(player_boxes):
        x1, y1, x2, y2 = map(int, box)

        # 이름 가져오기 (정렬된 순서대로 이름 리스트에서 가져옴)
        name = PLAYER_NAMES[i] if i < len(PLAYER_NAMES) else PLAYER_NAMES[-1]

        # 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 이름 배경 및 텍스트 추가
        (text_w, text_h), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(img, name, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# ----------------- 5. 최종 이미지 저장 -----------------
OUTPUT_PATH = 'identified_photo.jpg'
cv2.imwrite(OUTPUT_PATH, img)
print(f"✅ 선수 식별이 완료된 이미지가 {OUTPUT_PATH}에 저장되었습니다.")
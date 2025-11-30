from ultralytics import YOLO
import os

# 1. 학습된 최적 모델 로드
# 현재 best.pt 파일이 프로젝트 루트 폴더에 있다고 가정합니다.
model = YOLO('best.pt')

# 2. 테스트 이미지 파일 경로 설정 (준비한 파일 이름으로 변경하세요)
TEST_IMAGE_PATH = 'test_photo.jpg'

# 3. 탐지(Predict) 실행
# source: 탐지할 이미지 파일
# save=True: 탐지 결과가 그려진 새 이미지를 저장
# conf=0.5: 신뢰도(Confidence) 50% 이상의 객체만 박스를 표시 (필요시 조절 가능)
results = model.predict(source=TEST_IMAGE_PATH,
                        save=True,
                        conf=0.5)

print(f"✅ 탐지 시각화 완료. 결과는 runs/detect/predict 폴더에 저장됩니다.")
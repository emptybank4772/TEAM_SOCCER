from ultralytics import YOLO

# 모델 로드: yolov8n.pt (nano, 가장 빠르고 작은 모델) 사용 권장
model = YOLO('../디지털 영상처리 팀 프로젝트/yolov8n.pt')

# 학습 시작
# data: 3단계에서 만든 .yaml 파일 경로
# epochs: 학습 횟수 (예: 100)
# imgsz: 이미지 크기 (640 또는 1280 권장)
# 100회까지 돌리되, mAP가 10회 동안 개선되지 않으면 학습을 멈춤
results = model.train(data='football_data.yaml', epochs=120, patience=10, imgsz=640, batch=32)
print("✅ YOLOv8 학습이 시작되었습니다. GPU 환경에 따라 수 시간이 소요될 수 있습니다.")
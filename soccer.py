import json
import os
from glob import glob

# **************** 경로 설정 ****************
# 현재 JSON과 JPG 파일이 모두 들어있는 디렉토리
DATA_ROOT = 'C:/Users/k0104/PycharmProjects/PythonProject/fittogether'
# YOLO 라벨을 저장할 디렉토리 (새로 생성할 폴더)
YOLO_LABELS_ROOT = os.path.join(DATA_ROOT, 'yolo_labels')
os.makedirs(YOLO_LABELS_ROOT, exist_ok=True)
# *******************************************

# 클래스 이름과 ID 매핑 정의
CLASS_MAPPING = {
    'players': 0,
    'ball': 1,
    'others': 2
}


def convert_to_yolo_format(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. JSON 파일에서 이미지 크기 가져오기
    W = data['imageWidth']
    H = data['imageHeight']

    yolo_content = []

    for shape in data['shapes']:
        label = shape['label']
        if label in CLASS_MAPPING:
            class_id = CLASS_MAPPING[label]

            # 2. 바운딩 박스 좌표 추출: [[x1, y1], [x2, y2]]
            points = shape['points']
            x_min, y_min = points[0]
            x_max, y_max = points[1]

            # 3. YOLO 형식으로 변환 (정규화 공식 적용)
            x_center = ((x_max + x_min) / 2) / W
            y_center = ((y_max + y_min) / 2) / H
            width = (x_max - x_min) / W
            height = (y_max - y_min) / H

            # 4. TXT 파일에 저장할 내용 포맷 (소수점 6자리까지)
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_content.append(line)

    # 5. TXT 파일로 저장
    # 'video_01_000036.json' -> 'video_01_000036.txt'
    base_name = os.path.basename(json_file_path).replace('.json', '.txt')
    txt_file_path = os.path.join(YOLO_LABELS_ROOT, base_name)

    with open(txt_file_path, 'w') as f:
        f.write('\n'.join(yolo_content))


# DATA_ROOT 내의 모든 JSON 파일을 찾아 변환 스크립트 실행
json_files = glob(os.path.join(DATA_ROOT, '*.json'))
for file in json_files:
    convert_to_yolo_format(file)

print(f"✅ 총 {len(json_files)}개의 JSON 파일을 YOLO 형식으로 변환 완료. 결과는 {YOLO_LABELS_ROOT}에 저장됨.")
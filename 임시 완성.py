import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- 1. 설정 및 도구 정의 -----------------
model = YOLO('best.pt')
TEST_IMAGE_PATH = 'test_photo.jpg'
K_CLUSTERS = 3
BALL_CLASS_ID = 1

OTHER = ["OTHER"]
TEAM_OPPONENT = ["Son Heung-min", "Lee Kang-in", "Hwang Hee-chan", "Kim Min-jae", "Gue-sung Cho", "Unidentified Player"]
TEAM_OURS = ["Son Heung-min", "Lee Kang-in", "Hwang Hee-chan", "Kim Min-jae", "Gue-sung Cho", "Unidentified Player"]

def get_representative_color(img, box_coords):
    x1, y1, x2, y2 = map(int, box_coords)
    roi = img[y1 + (y2 - y1) // 3: y1 + 2 * (y2 - y1) // 3, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = np.float32(roi_rgb.reshape((-1, 3)))
    return pixels


def kmeans_clustering(all_pixels, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, labels, centers = cv2.kmeans(all_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, np.uint8(centers)

# ----------------- 2. 메인 실행 로직 (YOLO 및 K-means) -----------------
img = cv2.imread(TEST_IMAGE_PATH)
if img is None:
    print(f"❌ 오류: '{TEST_IMAGE_PATH}' 파일을 읽을 수 없습니다. 경로를 확인해주세요.")
    exit()

results = model.predict(source=TEST_IMAGE_PATH, conf=0.005,
                        iou=0.1, save=False, verbose=False)

all_player_pixels = []
player_boxes_list = []

if results and results[0].boxes:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:
            pixels = get_representative_color(img, box)
            all_player_pixels.append(pixels)
            player_boxes_list.append(box)

if not all_player_pixels:
    print("탐지된 선수 객체가 없어 K-means를 실행할 수 없습니다.")
    exit()

all_pixels_combined = np.vstack(all_player_pixels)

# 3. K-means 클러스터링 실행 (labels와 cluster_centers 획득)
labels, cluster_centers = kmeans_clustering(all_pixels_combined, K_CLUSTERS)

# ----------------- 4. 클러스터 인덱스 수동 지정 -----------------

GRASS_CLUSTER_INDEX = 0
TEAM_OURS_INDEX = 1
TEAM_OPPONENT_INDEX = 2

team_clusters = [TEAM_OURS_INDEX, TEAM_OPPONENT_INDEX]

# ----------------- 5. 각 선수 객체에 팀 레이블 할당 -----------------
current_pixel_index = 0
final_player_data = []

for i, box in enumerate(player_boxes_list):
    x1, y1, x2, y2 = map(int, box)

    roi_height = ((2 * (y2 - y1) // 3) - ((y2 - y1) // 3))
    pixel_count = (x2 - x1) * roi_height

    current_labels = labels[current_pixel_index: current_pixel_index + pixel_count]
    (unique, counts) = np.unique(current_labels, return_counts=True)

    valid_counts = {}
    for idx, cluster_label in enumerate(unique):
        if cluster_label != GRASS_CLUSTER_INDEX:
            valid_counts[cluster_label] = counts[idx]

    player_cluster = -1
    if valid_counts:
        player_cluster = max(valid_counts, key=valid_counts.get)

    team_id = "OTHER"
    if player_cluster == TEAM_OURS_INDEX:
        team_id = "TEAM_OURS"
    elif player_cluster == TEAM_OPPONENT_INDEX:
        team_id = "TEAM_OPPONENT"

    final_player_data.append({
        'team': team_id,
        'box': box
    })

    current_pixel_index += pixel_count

# ----------------- 6. 팀별 및 공 시각화 통합 -----------------
img_classified = img.copy()

COLOR_OURS = (0, 255, 0)
COLOR_OPPONENT = (0, 0, 255)
COLOR_OTHER = (100, 100, 100)
COLOR_BALL = (0, 255, 255)

# 6-1. 선수 시각화
for player in final_player_data:
    x1, y1, x2, y2 = map(int, player['box'])
    team = player['team']

    color = COLOR_OTHER
    if team == "TEAM_OURS":
        color = COLOR_OURS
    elif team == "TEAM_OPPONENT":
        color = COLOR_OPPONENT

    cv2.rectangle(img_classified, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_classified, team, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 6-2. 공 시각화
if results and results[0].boxes:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    ball_count = 0

    for box, cls in zip(boxes, classes):
        if int(cls) == BALL_CLASS_ID:
            x1, y1, x2, y2 = map(int, box)
            ball_count += 1
            text_label = f"BALL {ball_count}"

            cv2.rectangle(img_classified, (x1, y1), (x2, y2), COLOR_BALL, 3)
            (text_w, text_h), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_classified, (x1, y1 - text_h - baseline), (x1 + text_w, y1), COLOR_BALL, -1)
            cv2.putText(img_classified, text_label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 7. 결과 저장 및 출력
output_filename = '임시.jpg'
cv2.imwrite(output_filename, img_classified)
print(f"✅ K-means 팀 분류 최종 수정 완료. '{output_filename}' 파일을 확인해주세요.")
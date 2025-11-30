import os
import shutil
import random

# 설정
DATA_DIR = 'C:/Users/k0104/PycharmProjects/PythonProject/fittogether'
TARGET_DIR = os.path.join(DATA_DIR, 'football_dataset')
LABELS_DIR = os.path.join(DATA_DIR, 'yolo_labels')  # 1단계에서 생성한 TXT 폴더
SPLIT_RATIO = {'train': 0.8, 'val': 0.2}  # 80:20 분할

# 폴더 생성
for split in SPLIT_RATIO.keys():
    os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels', split), exist_ok=True)

# 모든 이미지 파일 목록 가져오기
all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
random.shuffle(all_images)

# 분할 지점 계산
train_end = int(len(all_images) * SPLIT_RATIO['train'])

# 파일 분할 및 이동
file_splits = {
    'train': all_images[:train_end],
    'val': all_images[train_end:]
}

for split, file_list in file_splits.items():
    print(f"--- {split} 세트: {len(file_list)}개 파일 이동 시작 ---")
    for img_file in file_list:
        # 이미지 이동
        src_img = os.path.join(DATA_DIR, img_file)
        dst_img = os.path.join(TARGET_DIR, 'images', split, img_file)
        shutil.move(src_img, dst_img)

        # 라벨 이동 (확장자 변경)
        label_file = img_file.replace('.jpg', '.txt')
        src_label = os.path.join(LABELS_DIR, label_file)
        dst_label = os.path.join(TARGET_DIR, 'labels', split, label_file)

        # 라벨 파일이 있는지 확인하고 이동 (없으면 에러 방지)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"경고: {label_file} 라벨 파일이 없습니다. 건너뜁니다.")

print("✅ 데이터셋 분할 및 이동 완료!")
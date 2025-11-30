import os
import random
import shutil

# **************** 경로 설정 ****************
TRAIN_IMG_DIR = 'C:/Users/k0104/PycharmProjects/PythonProject/fittogether/football_dataset/images/train'
TRAIN_LBL_DIR = 'C:/Users/k0104/PycharmProjects/PythonProject/fittogether/football_dataset/labels/train'
NEW_TRAIN_IMG_DIR = 'C:/Users/k0104/PycharmProjects/PythonProject/fittogether/football_dataset/images/train_10percent'
NEW_TRAIN_LBL_DIR = 'C:/Users/k0104/PycharmProjects/PythonProject/fittogether/football_dataset/labels/train_10percent'
os.makedirs(NEW_TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(NEW_TRAIN_LBL_DIR, exist_ok=True)
# *******************************************

SAMPLE_RATIO = 0.10  # 10% 샘플링

# 1. 학습용 이미지 목록 가져오기
all_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.jpg')]
num_to_sample = int(len(all_images) * SAMPLE_RATIO)

# 2. 무작위 샘플링
sampled_images = random.sample(all_images, num_to_sample)

# 3. 샘플링된 이미지와 라벨 이동 (복사)
for img_file in sampled_images:
    # 이미지 복사
    shutil.copy(os.path.join(TRAIN_IMG_DIR, img_file), os.path.join(NEW_TRAIN_IMG_DIR, img_file))

    # 라벨 파일명 생성 및 복사
    label_file = img_file.replace('.jpg', '.txt')
    shutil.copy(os.path.join(TRAIN_LBL_DIR, label_file), os.path.join(NEW_TRAIN_LBL_DIR, label_file))

print(f"✅ 총 {len(all_images)}개 중 {num_to_sample}개의 파일이 성공적으로 샘플링되었습니다.")
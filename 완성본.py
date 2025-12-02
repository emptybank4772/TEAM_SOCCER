import customtkinter as ctk
#import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
#import math

# =========================================================
# I. 상수 및 헬퍼 함수 정의
# =========================================================

MODEL_PATH = 'best.pt'
TEST_IMAGE_PATH = 'test_photo.jpg'
K_CLUSTERS = 3
BALL_CLASS_ID = 1

# K-means 클러스터 인덱스
GRASS_CLUSTER_INDEX = 0
TEAM_OURS_INDEX = 1
TEAM_OPPONENT_INDEX = 2

# 시각화 색상 (BGR 포맷)
COLOR_OURS = (0, 255, 0)  # Green
COLOR_OPPONENT = (0, 0, 255)  # Red
COLOR_OTHER = (100, 100, 100)  # Gray
COLOR_BALL = (0, 255, 255)  # Yellow

# (수정) 1번 방식: 수동 목록 매칭을 위한 선수 이름
# 정렬된 순서대로 이름이 할당되므로, 번호 순서를 임의로 지정하여 사용합니다.
OURS_NAMES = ["goal", "def", "def", "def", "def", "Ours Player 6",
              "Ours Player 7", "Ours Player 8", "Ours Player 9", "Ours Player 10", "Ours Player 11"]
OPPONENT_NAMES = ["Opponent 1", "Opponent 2", "Opponent 3", "Opponent 4", "Opponent 5", "Opponent 6", "Opponent 7",
                  "Opponent 8", "Opponent 9", "Opponent 10", "Opponent 11"]


def get_representative_color(img, box_coords):
    """선수의 바운딩 박스 중앙 1/3 영역의 픽셀을 추출합니다."""
    x1, y1, x2, y2 = map(int, box_coords)
    y_start = y1 + (y2 - y1) // 3
    y_end = y1 + 2 * (y2 - y1) // 3
    roi = img[y_start:y_end, x1:x2]
    if roi.size == 0:
        return None
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = np.float32(roi_rgb.reshape((-1, 3)))
    return pixels


def kmeans_clustering(all_pixels, k):
    """K-means 클러스터링을 실행합니다."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if all_pixels.size == 0 or len(all_pixels) < k:
        return np.array([]), np.array([])
    ret, labels, centers = cv2.kmeans(all_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, np.uint8(centers)


def get_center_coords(box):
    """바운딩 박스의 중심 좌표를 계산합니다."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


# =========================================================
# II. 핵심 처리 함수: 탐지 및 팀 분류 (X-좌표 정렬 로직 적용)
# =========================================================

def process_detection_and_classification(model, image_path):
    """
    YOLO를 실행하여 선수와 공을 탐지하고, K-means를 사용하여 선수에게 팀을 분류한 후,
    X-좌표 순으로 이름을 할당합니다.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, [], []

    try:
        results = model.predict(source=image_path, conf=0.01, iou=0.1, save=False, verbose=False)
    except Exception as e:
        print(f"YOLO 모델 예측 중 오류 발생: {e}")
        return img, [], []

    if not results or not results[0].boxes:
        return img, [], []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    all_player_pixels = []
    player_boxes_list = []
    ball_boxes = []

    # 1. 선수 픽셀 수집 및 공 박스 저장
    for box, cls in zip(boxes, classes):
        box_list = box.tolist()
        if int(cls) == 0:  # Player
            pixels = get_representative_color(img, box_list)
            if pixels is not None:
                all_player_pixels.append(pixels)
                player_boxes_list.append(box_list)
        elif int(cls) == BALL_CLASS_ID:  # Ball
            ball_boxes.append(box_list)

    if not all_player_pixels:
        print("탐지된 선수 객체가 없어 K-means를 실행할 수 없습니다.")
        return img, [], ball_boxes

    all_pixels_combined = np.vstack(all_player_pixels)

    # 2. K-means 클러스터링
    labels, cluster_centers = kmeans_clustering(all_pixels_combined, K_CLUSTERS)

    if labels.size == 0:
        print("K-means 클러스터링에 문제가 발생했습니다.")
        return img, [], ball_boxes

    # 3. 각 선수에 팀 레이블만 임시 할당
    current_pixel_index = 0
    temp_player_data = []

    for box in player_boxes_list:
        x1, y1, x2, y2 = map(int, box)

        # ROI 픽셀 수 계산
        roi_height = ((2 * (y2 - y1) // 3) - ((y2 - y1) // 3))
        pixel_count = (x2 - x1) * roi_height

        current_labels = labels[current_pixel_index: current_pixel_index + pixel_count]
        (unique, counts) = np.unique(current_labels, return_counts=True)

        valid_counts = {}
        for idx, cluster_label in enumerate(unique):
            if cluster_label != GRASS_CLUSTER_INDEX:
                valid_counts[cluster_label] = counts[idx]

        team_id = "OTHER"

        if valid_counts:
            player_cluster = max(valid_counts, key=valid_counts.get)

            if player_cluster == TEAM_OURS_INDEX:
                team_id = "TEAM_OURS"
            elif player_cluster == TEAM_OPPONENT_INDEX:
                team_id = "TEAM_OPPONENT"

        temp_player_data.append({
            'team': team_id,
            'box': box,
            'center': get_center_coords(box)
        })

        current_pixel_index += pixel_count

    # 🚨 4. 팀별로 분리 및 X-좌표(가로 위치) 기준으로 정렬
    ours_list = [p for p in temp_player_data if p['team'] == 'TEAM_OURS']
    opponent_list = [p for p in temp_player_data if p['team'] == 'TEAM_OPPONENT']
    other_list = [p for p in temp_player_data if p['team'] == 'OTHER']

    # X-좌표(center[0])가 작은 순서대로 정렬 (왼쪽 -> 오른쪽)
    ours_list.sort(key=lambda p: p['center'][0])
    opponent_list.sort(key=lambda p: p['center'][0])

    # 5. 정렬된 순서대로 이름 할당 및 최종 리스트 병합
    final_player_data = []

    # Our Team Naming
    for i, player in enumerate(ours_list):
        name = OURS_NAMES[i] if i < len(OURS_NAMES) else f"Ours Player {i + 1} (Sorted X)"
        player['name'] = name
        final_player_data.append(player)

    # Opponent Team Naming
    for i, player in enumerate(opponent_list):
        name = OPPONENT_NAMES[i] if i < len(OPPONENT_NAMES) else f"Opponent Player {i + 1} (Sorted X)"
        player['name'] = name
        final_player_data.append(player)

    # Other Team Naming
    for i, player in enumerate(other_list):
        player['name'] = f"Other {i + 1}"
        final_player_data.append(player)

    return img, final_player_data, ball_boxes


# =========================================================
# III. 시각화 함수: 필터링 및 그리기 (변경 없음)
# =========================================================

def visualize_results(original_img, player_data, ball_boxes, filter_type):
    """
    필터 타입에 따라 선수와 공을 시각화한 이미지를 반환합니다.
    filter_type: 'NONE', 'ALL', 'OURS', 'OPPONENT', 'NEAREST_BALL'

    반환 값: 튜플 (시각화된 이미지, 표시할 선수 목록)
    """
    if original_img is None:
        return None, []

    # 'NONE' 필터일 경우 원본 이미지를 그대로 반환하고 선수 목록은 비워둡니다.
    if filter_type == 'NONE':
        return original_img.copy(), []

    img_classified = original_img.copy()
    players_to_draw = []
    nearest_player = None

    # 1. 필터링 로직
    if filter_type == 'OURS':
        players_to_draw = [p for p in player_data if p['team'] == 'TEAM_OURS']
    elif filter_type == 'OPPONENT':
        players_to_draw = [p for p in player_data if p['team'] == 'TEAM_OPPONENT']
    elif filter_type == 'ALL':
        players_to_draw = player_data
    elif filter_type == 'NEAREST_BALL':
        min_distance_sq = float('inf')

        if ball_boxes:
            ball_centers = [get_center_coords(box) for box in ball_boxes]

            for player in player_data:
                p_center_x, p_center_y = player['center']

                # 모든 공과의 최소 거리 찾기
                for b_center_x, b_center_y in ball_centers:
                    dist_sq = (p_center_x - b_center_x) ** 2 + (p_center_y - b_center_y) ** 2

                    if dist_sq < min_distance_sq:
                        min_distance_sq = dist_sq
                        nearest_player = player

            if nearest_player:
                players_to_draw = [nearest_player]

    # 2. 선수 시각화
    for player in players_to_draw:
        x1, y1, x2, y2 = map(int, player['box'])
        team = player['team']
        name = player['name']

        color = COLOR_OTHER
        if team == "TEAM_OURS":
            color = COLOR_OURS
        elif team == "TEAM_OPPONENT":
            color = COLOR_OPPONENT

        # 일반 바운딩 박스
        cv2.rectangle(img_classified, (x1, y1), (x2, y2), color, 2)

        text_label = name

        # 'NEAREST_BALL'인 경우, 특별히 표시
        if filter_type == 'NEAREST_BALL' and player is nearest_player:
            # 가장 가까운 선수에게 노란색 테두리 추가
            cv2.rectangle(img_classified, (x1, y1), (x2, y2), (0, 255, 255), 4)

            # 이름 텍스트 출력
        cv2.putText(img_classified, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. 공 시각화
    for box in ball_boxes:
        x1, y1, x2, y2 = map(int, box)
        text_label = f"BALL"

        cv2.rectangle(img_classified, (x1, y1), (x2, y2), COLOR_BALL, 3)
        (text_w, text_h), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_classified, (x1, y1 - text_h - baseline), (x1 + text_w, y1), COLOR_BALL, -1)
        cv2.putText(img_classified, text_label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_classified, players_to_draw


# =========================================================
# IV. CustomTkinter GUI 클래스 (변경 없음)
# =========================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 창 설정
        self.title("축구선수 팀 분류 및 시각화 앱 (Zoom 지원)")
        self.geometry("1280x850")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.model = YOLO(MODEL_PATH)
        self.player_data = []
        self.ball_boxes = []
        self.original_img = None
        self.current_tk_image = None
        self.current_filter_type = 'NONE'

        # 이미지 크기 설정
        self.FRAME_WIDTH = 900
        self.zoom_factor = 1.0  # 줌 배율

        # 2. 메인 콘텐츠 (왼쪽) - 이미지 프레임
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)

        # 3. 이미지 표시를 위한 캔버스 설정 (Zoom 기능 구현)
        self.image_canvas = ctk.CTkCanvas(self.main_content_frame,
                                          bg=self._apply_appearance_mode(
                                              ctk.ThemeManager.theme["CTkFrame"]["fg_color"]),
                                          highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        self.image_item = self.image_canvas.create_image(0, 0, anchor="nw", image=None)

        # 줌 이벤트 바인딩
        self.image_canvas.bind("<MouseWheel>", self.zoom_handler)  # Windows/macOS
        self.image_canvas.bind("<Button-4>", lambda event: self.zoom_handler(event, 1))  # Linux 휠 업
        self.image_canvas.bind("<Button-5>", lambda event: self.zoom_handler(event, -1))  # Linux 휠 다운
        self.image_canvas.bind("<ButtonPress-1>", self.move_start)  # 이동 시작
        self.image_canvas.bind("<B1-Motion>", self.move_move)  # 이동 중

        # 캔버스 크기 변경 이벤트 바인딩
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        # 초기 로딩 텍스트
        self.loading_text = self.image_canvas.create_text(
            self.FRAME_WIDTH // 2,
            self.FRAME_WIDTH // 2,
            text="이미지 로딩 및 분석 중...",
            fill="white",
            font=("Arial", 20)
        )

        # 🚨 1. 데이터 전처리 및 분류 실행 (딱 한 번)
        print("시작: YOLO 탐지 및 K-means 팀 분류...")
        self.original_img, self.player_data, self.ball_boxes = process_detection_and_classification(self.model,
                                                                                                    TEST_IMAGE_PATH)
        print("완료: 데이터 처리 완료.")

        if self.original_img is None:
            self.display_error_ui(f"❌ 오류: 이미지 로드 또는 모델 처리 실패.\n경로 확인: '{TEST_IMAGE_PATH}' 또는 '{MODEL_PATH}'")
            return

        # 4. 목록 패널 설정
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=10, fg_color="#444444")
        self.sidebar_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
        for i in range(4): self.sidebar_frame.grid_rowconfigure(i, weight=0)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        self.sidebar_title = ctk.CTkLabel(self.sidebar_frame,
                                          text="선수 보기 옵션 및 현재 목록",
                                          font=ctk.CTkFont(size=18, weight="bold"))
        self.sidebar_title.grid(row=0, column=0, padx=10, pady=(15, 5), sticky="n")

        # 5. 필터 체크박스 섹션
        self.checkbox_scroll_frame = ctk.CTkFrame(self.sidebar_frame, height=120)
        self.checkbox_scroll_frame.grid(row=1, column=0, padx=10, pady=(1, 1), sticky="nsew")

        checkbox_items = [
            ("공과 가장 가까운 선수", "NEAREST_BALL"),
            ("우리 팀만 보기 (OURS)", "OURS"),
            ("상대 팀만 보기 (OPPONENT)", "OPPONENT"),
            ("전체 선수 보기 (ALL)", "ALL")
        ]

        self.checkboxes = {}
        for i, (text, filter_key) in enumerate(checkbox_items):
            checkbox = ctk.CTkCheckBox(self.checkbox_scroll_frame,
                                       text=text,
                                       font=("Arial", 15),
                                       height=18,
                                       checkbox_height=18,
                                       checkbox_width=18,
                                       command=lambda fk=filter_key: self.toggle_checkboxes(fk))

            self.checkboxes[filter_key] = checkbox
            checkbox.pack(pady=(5, 3), padx=10, anchor="w")

        # 6. 초기화 버튼
        self.reset_button = ctk.CTkButton(self.sidebar_frame,
                                          text="필터 초기화 (원본 이미지)",
                                          command=self.reset_filters,
                                          fg_color="#a03030",
                                          hover_color="#c04040")

        self.reset_button.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")

        # 7. 선수 목록 프레임
        self.player_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame,
                                                        label_text="현재 표시 선수 목록 (0명)",
                                                        fg_color="transparent")

        self.player_list_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")

        self.player_labels = []  # 동적으로 생성된 라벨을 저장할 리스트

        # 8. 초기 상태 설정 (필터 미선택, 원본 이미지 표시)
        self.reset_filters()

    def display_error_ui(self, message):
        """오류 발생 시 오류 메시지만 표시하는 UI로 대체"""
        error_label = ctk.CTkLabel(self, text=message, text_color="red", font=ctk.CTkFont(size=20, weight="bold"))
        error_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        self.image_canvas.delete(self.loading_text)

    # =========================================================
    # V. Zoom 및 Canvas 관련 함수
    # =========================================================

    def move_start(self, event):
        """이미지 이동 시작 시점 (마우스 버튼 클릭)"""
        self.image_canvas.scan_mark(event.x, event.y)

    def move_move(self, event):
        """이미지 이동 중 (마우스 드래그)"""
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom_handler(self, event, direction=None):
        """마우스 휠 이벤트 발생 시 줌 인/아웃 처리"""
        scale_factor = 1.1

        # 휠 방향 확인 (Windows: event.delta, Linux: direction)
        if direction is not None:
            if direction == 1:  # 휠 업
                delta = 1
            else:  # 휠 다운
                delta = -1
        else:  # Windows/macOS
            delta = event.delta / 120

        if delta > 0:  # Zoom In
            self.zoom_factor *= scale_factor
        elif delta < 0:  # Zoom Out
            self.zoom_factor /= scale_factor

        # 최소/최대 줌 배율 제한 (원본 대비 0.5배 ~ 5배)
        self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))

        # 이미지 크기 및 캔버스 스케일 업데이트
        self.update_image_display(self.current_filter_type, zoom_update=True)

    def on_canvas_resize(self, event):
        """캔버스 크기가 변경될 때 이미지를 중앙에 다시 배치합니다."""
        # 캔버스 크기 변경 시 zoom_factor를 초기화하지 않고 현재 줌 레벨을 유지하며 중앙 재배치
        self.update_image_display(self.current_filter_type, zoom_update=True)

    def update_image_display(self, filter_type, zoom_update=False):
        """
        필터 타입에 따라 이미지를 업데이트하고 GUI에 표시합니다.
        """
        if self.original_img is None:
            return

        self.current_filter_type = filter_type

        # 1. 시각화 및 선수 목록 필터링
        classified_img_bgr, players_to_draw = visualize_results(self.original_img, self.player_data, self.ball_boxes,
                                                                filter_type)

        # 2. BGR -> RGB 변환 및 PIL Image로 변환
        img_rgb = cv2.cvtColor(classified_img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # 3. GUI 크기 및 줌 배율에 맞게 이미지 리사이즈
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            # 캔버스 크기가 유효하지 않은 경우 (초기 로딩 단계 등), 기본값을 사용
            canvas_width = self.FRAME_WIDTH
            canvas_height = self.FRAME_WIDTH * pil_image.height // pil_image.width

        # 캔버스 크기에 맞게 초기 이미지 크기 조정 (aspect ratio 유지)
        aspect_ratio = pil_image.width / pil_image.height

        # 캔버스에 꽉 차게 이미지 크기 조정
        if canvas_width / aspect_ratio <= canvas_height:
            display_width = canvas_width
            display_height = int(display_width / aspect_ratio)
        else:
            display_height = canvas_height
            display_width = int(display_height * aspect_ratio)

        # 줌 팩터 적용
        display_width = int(display_width * self.zoom_factor)
        display_height = int(display_height * self.zoom_factor)

        resized_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # 4. CustomTkinter 이미지 객체 생성 및 캔버스 업데이트
        self.current_tk_image = ImageTk.PhotoImage(image=resized_image)

        # 이미지 중앙 배치 좌표 계산
        center_x = canvas_width // 2
        center_y = canvas_height // 2

        # 로딩 텍스트 삭제
        self.image_canvas.delete(self.loading_text)

        self.image_canvas.itemconfig(self.image_item, image=self.current_tk_image)

        # 캔버스에서 이미지 위치 업데이트 (중앙 정렬)
        self.image_canvas.coords(self.image_item, center_x - display_width // 2, center_y - display_height // 2)

        # 5. 선수 목록 업데이트 호출
        self.update_player_list(players_to_draw)

    # =========================================================
    # VI. 기능 함수 정의
    # =========================================================

    def update_player_list(self, players_to_draw):
        """
        현재 화면에 표시된 선수들만 목록에 업데이트합니다.
        """
        # 기존 라벨 모두 삭제
        for label in self.player_labels:
            label.destroy()
        self.player_labels.clear()

        # 새 라벨 생성
        for player in players_to_draw:
            player_info = f"[{player['team'].replace('TEAM_', '')}] {player['name']}"

            player_label = ctk.CTkLabel(self.player_list_frame,
                                        text=player_info,
                                        anchor="w",
                                        padx=5,
                                        pady=2)
            self.player_labels.append(player_label)
            player_label.pack(fill="x", padx=5, pady=2)

        # 라벨 업데이트
        self.player_list_frame.configure(label_text=f"현재 표시 선수 목록 ({len(players_to_draw)}명)")

    def reset_filters(self):
        """
        모든 체크박스를 해제하고 'NONE' (원본 이미지) 필터를 기본으로 설정합니다.
        """
        for checkbox in self.checkboxes.values():
            checkbox.deselect()

        self.zoom_factor = 1.0  # 줌 배율 초기화
        # 캔버스 스크롤 위치 초기화
        self.image_canvas.xview_moveto(0)
        self.image_canvas.yview_moveto(0)

        self.update_image_display('NONE')  # 원본 이미지를 표시
        print("모든 필터가 초기화되었습니다. (원본 이미지 보기)")

    def toggle_checkboxes(self, selected_filter_key):
        """
        단일 선택을 구현하고, 선택된 필터에 따라 이미지를 업데이트합니다.
        """
        selected_checkbox = self.checkboxes[selected_filter_key]

        # 1. 체크박스가 선택된 경우: 다른 모든 체크박스 해제 및 이미지 업데이트
        if selected_checkbox.get() == 1:
            for filter_key, checkbox in self.checkboxes.items():
                if filter_key != selected_filter_key:
                    checkbox.deselect()

            # 선택된 필터로 이미지 업데이트
            self.update_image_display(selected_filter_key)

        # 2. 체크박스가 해제된 경우: 모든 체크박스가 해제되었다면 원본 이미지 상태로 전환
        else:
            is_any_checked = any(cb.get() == 1 for cb in self.checkboxes.values())

            if not is_any_checked:
                self.update_image_display('NONE')  # 원본 이미지 상태로 전환


# =========================================================
# VII. 앱 실행
# =========================================================

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"애플리케이션 실행 중 치명적인 오류 발생: {e}")
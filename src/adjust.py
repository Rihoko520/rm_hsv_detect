import cv2
import numpy as np
from skimage import exposure

# 全局变量
gamma = 70.0  # 将 gamma 默认值设置为 70.00
threshold_value = 128
h_min, s_min, v_min = 0, 100, 100
h_max, s_max, v_max = 10, 255, 255

url = 'http://192.168.3.195:4747/video/'
current_frame = None

# 更新阈值
def update_threshold(val):
    global threshold_value
    threshold_value = val
    apply_changes_without_hsv(current_frame)  # 重新应用变化

# 更新 gamma 值
def update_gamma(val):
    global gamma
    gamma = val / 100.0  # 将滑动条值转换为实际 gamma 值
    apply_changes_without_hsv(current_frame)  # 重新应用变化

# 更新 HSV 范围
def update_hue_min(val):
    global h_min
    h_min = val
    apply_changes_with_hsv(current_frame)  # 重新应用变化

def update_hue_max(val):
    global h_max
    h_max = val
    apply_changes_with_hsv(current_frame)  # 重新应用变化

def update_saturation_min(val):
    global s_min
    s_min = val
    apply_changes_with_hsv(current_frame)  # 重新应用变化

def update_saturation_max(val):
    global s_max
    s_max = val
    apply_changes_with_hsv(current_frame)  # 重新应用变化

def update_value_min(val):
    global v_min
    v_min = val
    apply_changes_with_hsv(current_frame)  # 重新应用变化

def update_value_max(val):
    global v_max
    v_max = val
    apply_changes_with_hsv(current_frame)  # 重新应用变化

# 应用变化（调试 gamma、threshold 和 HSV）
def apply_changes_with_hsv(frame):
    if frame is not None:
        # Gamma 校正
        gamma_corrected = exposure.adjust_gamma(frame, gamma)
        gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        
        # HSV 处理
        hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)  # 使用 gamma 校正的图像
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        result = cv2.bitwise_and(gamma_corrected, gamma_corrected, mask=mask)

        # 显示处理后的帧
        cv2.imshow('Gamma and Threshold', binary_image)
        cv2.imshow('HSV Adjustment', result)

# 应用变化（仅调试 gamma 和 threshold）
def apply_changes_without_hsv(frame):
    if frame is not None:
        # Gamma 校正
        gamma_corrected = exposure.adjust_gamma(frame, gamma)
        gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 显示处理后的帧
        cv2.imshow('Gamma and Threshold', binary_image)

# 设置窗口和滑动条
def setup_windows(use_hsv=True):
    # Gamma 和阈值窗口
    cv2.namedWindow('Gamma and Threshold')
    cv2.createTrackbar('Threshold', 'Gamma and Threshold', threshold_value, 255, update_threshold)
    cv2.createTrackbar('Gamma', 'Gamma and Threshold', int(gamma * 100), 9999, update_gamma)  # 将上限设为 9999

    # 如果需要 HSV 调节，则创建 HSV 窗口
    if use_hsv:
        cv2.namedWindow('HSV Adjustment')
        cv2.createTrackbar('H_min', 'HSV Adjustment', h_min, 179, update_hue_min)
        cv2.createTrackbar('H_max', 'HSV Adjustment', h_max, 179, update_hue_max)
        cv2.createTrackbar('S_min', 'HSV Adjustment', s_min, 255, update_saturation_min)
        cv2.createTrackbar('S_max', 'HSV Adjustment', s_max, 255, update_saturation_max)
        cv2.createTrackbar('V_min', 'HSV Adjustment', v_min, 255, update_value_min)
        cv2.createTrackbar('V_max', 'HSV Adjustment', v_max, 255, update_value_max)

# Resize 图像
def resize_frame(frame):
    return cv2.resize(frame, (640, 480))

# 运行调试界面
def run_adjustment(use_hsv=True):
    setup_windows(use_hsv)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("错误: 无法打开视频。")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取帧。")
            break

        global current_frame
        current_frame = resize_frame(frame)  # 调整大小

        # Gamma 校正
        gamma_corrected = exposure.adjust_gamma(current_frame, gamma)

        # 根据是否需要 HSV 调试调用不同的函数
        if use_hsv:
            apply_changes_with_hsv(gamma_corrected)  # 调用 HSV 变化应用
        else:
            apply_changes_without_hsv(gamma_corrected)  # 调用仅 gamma 和阈值变化应用

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 处理单张图片
def process_image(image_path, use_hsv=True):
    global current_frame
    current_frame = cv2.imread(image_path)
    if current_frame is None:
        print("错误: 无法读取图像。请检查路径:", image_path)
        return

    current_frame = resize_frame(current_frame)  # 调整大小
    setup_windows(use_hsv)  # 根据需要设置窗口

    # 进行 gamma 校正
    gamma_corrected = exposure.adjust_gamma(current_frame, gamma)

    if use_hsv:
        apply_changes_with_hsv(gamma_corrected)  # 调用变化应用，包括 HSV
    else:
        apply_changes_without_hsv(gamma_corrected)  # 调用变化应用，仅 gamma 和阈值

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
import cv2
from skimage import exposure

# 全局变量
gamma = 0.7
threshold_value = 128
mode = 0  # 0 表示处理静态图像，1 表示处理视频流
path = './photo/1.png'
url = 'http://192.168.3.195:4747/video/'

def update_threshold(val):
    global threshold_value
    threshold_value = val
    apply_changes(current_frame)

def update_gamma(val):
    global gamma
    gamma = val / 100.0
    apply_changes(current_frame)

def apply_changes(frame):
    gamma_corrected = exposure.adjust_gamma(frame, gamma)
    gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', binary_image)

def setup_window():
    cv2.namedWindow('Binary Image')
    cv2.createTrackbar('Threshold', 'Binary Image', threshold_value, 255, update_threshold)
    cv2.createTrackbar('Gamma', 'Binary Image', int(gamma * 100), 5000, update_gamma)

def process_image(image):
    if image is not None:
        image = cv2.resize(image, (640, 480))
        global current_frame
        current_frame = image
        apply_changes(current_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("错误: 无法读取图像。请检查路径:", path)

def process_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取帧。")
            break
        frame = cv2.resize(frame, (640, 480))
        global current_frame
        current_frame = frame
        apply_changes(current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 主程序
current_frame = None
setup_window()

if mode == 0:
    image = cv2.imread(path)
    process_image(image)
elif mode == 1:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("错误: 无法打开视频。")
        exit()
    process_video(cap)
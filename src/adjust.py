import cv2
from skimage import exposure

# 全局变量
gamma = 0.7
threshold_value = 128
mode = 1  # 0 for image, 1 for video stream
path = './photo/2.jpg'
url='http://192.168.3.195:4747/video/'
def update_threshold(val):
    global threshold_value
    threshold_value = val
    apply_changes()

def update_gamma(val):
    global gamma
    gamma = val / 100.0
    apply_changes()

def apply_changes(frame):
    gamma_corrected = exposure.adjust_gamma(frame, gamma)
    gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', binary_image)

# 选择运行模式
if mode == 0:
    # 处理静态图像
    image = cv2.imread(path)
    if image is not None:
        image = cv2.resize(image, (640, 480))
        cv2.namedWindow('Binary Image')
        cv2.createTrackbar('Threshold', 'Binary Image', threshold_value, 255, update_threshold)
        cv2.createTrackbar('Gamma', 'Binary Image', int(gamma * 100), 5000, update_gamma)
        apply_changes(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not read the image.")

elif mode == 1:
    # 处理视频流
    cap = cv2.VideoCapture(url)  # 使用默认摄像头
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    cv2.namedWindow('Binary Image')
    cv2.createTrackbar('Threshold', 'Binary Image', threshold_value, 255, update_threshold)
    cv2.createTrackbar('Gamma', 'Binary Image', int(gamma * 100), 5000, update_gamma)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame = cv2.resize(frame, (640, 480))
        apply_changes(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
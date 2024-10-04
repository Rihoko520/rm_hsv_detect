import cv2
import numpy as np

# 选择运行模式：0 表示识别图片，1 表示视频流
mode = 1  # 修改为 0 以识别图片，1 以使用视频流
url='http://192.168.3.195:4747/video/'

# 读取图像并调整大小（仅在模式为 0 时使用）
if mode == 0:
    image = cv2.imread('./photo/2.jpg')
    image = cv2.resize(image, (640, 480))
else:
    # 使用视频流
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("无法打开视频源")
        exit()

# 初始化 HSV 范围
h_min, s_min, v_min = 0, 100, 100
h_max, s_max, v_max = 10, 255, 255

# 创建窗口和滑动条
cv2.namedWindow('Color Thresholding')
for i, name in enumerate(['H_min', 'S_min', 'V_min', 'H_max', 'S_max', 'V_max']):
    cv2.createTrackbar(name, 'Color Thresholding', [h_min, s_min, v_min, h_max, s_max, v_max][i], 255 if i % 3 else 179, lambda x: None)

while True:
    if mode == 0:
        # 仅在模式为 0 时处理图像
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        # 从视频流捕获帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频流")
            break
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取滑动条的值
    h_min, s_min, v_min = [cv2.getTrackbarPos(name, 'Color Thresholding') for name in ['H_min', 'S_min', 'V_min']]
    h_max, s_max, v_max = [cv2.getTrackbarPos(name, 'Color Thresholding') for name in ['H_max', 'S_max', 'V_max']]

    # 创建掩模并显示结果
    mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
    result = cv2.bitwise_and(frame if mode == 1 else image, frame if mode == 1 else image, mask=mask)
    cv2.imshow('Color Thresholding', result)

    # 等待按键，如果按下 'q' 则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象（仅在模式为 1 时）
if mode == 1:
    cap.release()

# 销毁所有窗口
cv2.destroyAllWindows()
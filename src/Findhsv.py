import cv2  
import numpy as np  

# 读取图像并调整大小  
image = cv2.imread('./photo/2.jpg')  
image = cv2.resize(image, (640, 480))  

# 初始化HSV范围  
h_min, s_min, v_min = 0, 100, 100  
h_max, s_max, v_max = 10, 255, 255  

# 转换到HSV颜色空间  
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  

# 创建窗口和滑动条  
cv2.namedWindow('Color Thresholding')  
for i, name in enumerate(['H_min', 'S_min', 'V_min', 'H_max', 'S_max', 'V_max']):
    cv2.createTrackbar(name, 'Color Thresholding', [h_min, s_min, v_min, h_max, s_max, v_max][i], 255 if i % 3 else 179, lambda x: None)

while True:  
    # 获取滑动条的值  
    h_min, s_min, v_min = [cv2.getTrackbarPos(name, 'Color Thresholding') for name in ['H_min', 'S_min', 'V_min']]  
    h_max, s_max, v_max = [cv2.getTrackbarPos(name, 'Color Thresholding') for name in ['H_max', 'S_max', 'V_max']]  

    # 创建掩模并显示结果  
    mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))  
    result = cv2.bitwise_and(image, image, mask=mask)  
    cv2.imshow('Color Thresholding', result)  

    # 等待按键，如果按下'q'则退出循环  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# 销毁所有窗口  
cv2.destroyAllWindows()
import cv2
from skimage import exposure

# 全局变量
gamma = 0.7
threshold_value = 128

def update_threshold(val):
    global threshold_value
    threshold_value = val
    apply_changes()

def update_gamma(val):
    global gamma
    gamma = val / 100.0
    apply_changes()

def apply_changes():
    gamma_corrected = exposure.adjust_gamma(image, gamma)
    gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', binary_image)

image = cv2.imread('./photo/1.jpg')

if image is not None:
    image = cv2.resize(image, (640, 480))
    cv2.namedWindow('Binary Image')
    cv2.createTrackbar('Threshold', 'Binary Image', threshold_value, 255, update_threshold)
    cv2.createTrackbar('Gamma', 'Binary Image', int(gamma * 100), 5000, update_gamma)
    apply_changes()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not read the image.")
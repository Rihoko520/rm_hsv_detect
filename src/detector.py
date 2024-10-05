import cv2
import numpy as np
from skimage import exposure, img_as_float

def adjust_rotated_rect(rect):
    """调整旋转矩形的宽高和角度，使宽始终小于高。"""
    c, (w, h), angle = rect
    if w > h:
        w, h = h, w
        angle = (angle + 90) % 360
        angle = angle - 360 if angle > 180 else angle - 180 if angle > 90 else angle
    return c, (w, h), angle

def is_close(rect1, rect2, angle_tol, height_tol, width_tol, cy_tol):
    """检查两个旋转矩形是否足够接近。"""
    (cx1, cy1), (w1, h1), angle1 = rect1
    (cx2, cy2), (w2, h2), angle2 = rect2
    
    # 计算中心点之间的距离
    distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    if distance > 60:
        return (min(abs(angle1 - angle2), 360 - abs(angle1 - angle2)) <= angle_tol and
                abs(h1 - h2) <= height_tol and
                abs(w1 - w2) <= width_tol and
                abs(cy1 - cy2) <= cy_tol)

def put_text(img, rect):
    """在图像上绘制中心坐标文本。"""
    center_x, center_y = map(int, rect[0])
    cv2.putText(img, f"({center_x}, {center_y})", (center_x, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def do_polygons_intersect(a, b):
    """使用分离轴定理检查两个多边形是否相交。"""
    for polygon in (a, b):
        for i in range(len(polygon)):
            p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
            normal = (p2[1] - p1[1], p1[0] - p2[0])
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)
            if max_a < min_b or max_b < min_a:
                return False

def project_polygon(polygon, axis):
    """将多边形投影到给定的轴上。"""
    projections = np.dot(polygon, axis)
    return projections.min(), projections.max()

def img_processed(img, val, gamma):
    """处理图像，返回二值图像、调整大小的图像和浮点图像。"""    
    resized_img = cv2.resize(img, (640, 480))  # 调整图像大小
    img_float = (exposure.adjust_gamma(img_as_float(resized_img), gamma) * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(img_gray, val, 255, cv2.THRESH_BINARY)
    return binary_image, resized_img, img_float

def find_light(color, img_binary, img_float):
    """查找图像中的光源并返回旋转矩形。"""
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_rotated_rects = [
        adjust_rotated_rect(cv2.minAreaRect(contour))
        for contour in contours if cv2.contourArea(contour) > 30 ###100-30
    ]

    filtered_rotated_rects = [
        rect for rect in filtered_rotated_rects if -45 <= rect[2] <= 45 and not any(
            do_polygons_intersect(cv2.boxPoints(rect).astype(int), cv2.boxPoints(other_rect).astype(int))
            for other_rect in filtered_rotated_rects
        )
    ]

    for rect in filtered_rotated_rects:
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(img_float, [box], 0, color, 3)  # 修改为在 img_float 上绘制
        cv2.circle(img_float, tuple(map(int, rect[0])), 3, color, -1)

    return filtered_rotated_rects  # 返回找到的旋转矩形列表

def track_armor(img, img_float, rotated_rects, angle_tol=5, height_tol=50, width_tol=40, cy_tol=40):##angle_tol=5-10, height_tol=20-50, width_tol=20-40, cy_tol=40
    """跟踪装甲并返回装甲字典。"""
    all_groups = []  # 存储所有装甲组

    while rotated_rects:  # 当还有待处理的矩形时
        rect1 = rotated_rects.pop(0)  # 取出第一个矩形
        # 查找与rect1接近的矩形
        close_indices = [i for i in range(len(rotated_rects)) if is_close(rect1, rotated_rects[i], angle_tol, height_tol, width_tol, cy_tol)]
        
        if close_indices:  # 如果找到接近的矩形
            group = [rect1] + [rotated_rects.pop(i) for i in sorted(close_indices, reverse=True)]
            all_groups.append(group)  # 将当前组添加到所有组中

    armor_rects = []  # 存储合并后的装甲矩形
    for rect_group in all_groups:  # 遍历所有装甲组
        if rect_group:  # 如果组不为空
            points = np.concatenate([cv2.boxPoints(rect) for rect in rect_group])  # 获取所有矩形的四个顶点
            merged_rect = cv2.minAreaRect(points)  # 计算最小外接矩形

            # 检查合并后的矩形是否符合条件
            if 2000 <= merged_rect[1][0] * merged_rect[1][1] <= 10000 and 0 <= merged_rect[1][0] / merged_rect[1][1] <= 4:
                armor_rects.append(adjust_rotated_rect(merged_rect))  # 调整并添加到装甲矩形列表

    armors_dict = {}  # 存储装甲信息的字典
    color_map = {1: (255, 0, 0), 0: (0, 0, 255)}  # 蓝色和红色的颜色映射
    class_map = {1: 1, 0: 7}  # 蓝色和红色的class_id映射

    for armor_rect in armor_rects:  # 遍历所有装甲矩形
        center, (width, height), angle = armor_rect  # 获取装甲矩形的中心、宽高和角度
        max_size = max(width, height)  # 计算最大尺寸
        box = cv2.boxPoints(((center[0], center[1]), (max_size, max_size), angle)).astype(int)  # 获取装甲的四个顶点
        class_id = armortype(img_float, armor_rect)  # 识别装甲类型

        if class_id in color_map:  # 如果class_id在颜色映射中
            armors_dict[f"{int(center[0])}"] = {
                "class_id": class_map[class_id],  # 添加装甲信息到字典
                "height": int(max_size),
                "center": [int(center[0]), int(center[1])]
            }
            draw_armor(img_float, box, center, color_map[class_id], armor_rect)  # 修改为在 img_float 上绘制

    return armors_dict  # 返回装甲字典

def armortype(img_float, rotated_rect):
    """判断装甲类型并返回相应的类 ID。"""
    try:
        points = np.int0(cv2.boxPoints(rotated_rect))
        mask = np.zeros_like(img_float[:, :, 0], dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        roi = cv2.bitwise_and(img_float, img_float, mask=mask)
        x, y, w, h = cv2.boundingRect(points)
        roi = roi[y:y + h, x:x + w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_ranges = {
            'red': (np.array([0, 43, 46]), np.array([10, 255, 255])),
            'red_upper': (np.array([160, 100, 100]), np.array([180, 255, 255])),
            'blue': (np.array([100, 100, 100]), np.array([140, 255, 255])),
            'black': (np.array([0, 0, 0]), np.array([180, 255, 50]))
        }

        masks = {color: cv2.inRange(hsv, *ranges) for color, ranges in color_ranges.items()}
        total_pixels = roi.size // 3
        
        if total_pixels == 0:
            return -1

        red_pixels = sum(np.count_nonzero(masks[color]) for color in ['red', 'red_upper'])
        blue_pixels = sum(np.count_nonzero(masks[color]) for color in ['blue'])
        black_pixels = np.count_nonzero(masks['black'])
        total_non_black = total_pixels - black_pixels

        if total_non_black <= 0:
            return -1

        red_ratio = red_pixels / total_non_black
        blue_ratio = blue_pixels / total_non_black

        if red_ratio > blue_ratio and red_ratio > 0.5:
            return 0  # 红色
        elif blue_ratio > red_ratio and blue_ratio > 0.5:
            return 1  # 蓝色
        return -1  # 无主导颜色

    except Exception:
        return -1  # 出现异常返回 -1

def draw_armor(img_float, box, center, color, armor_rect):
    """在图像上绘制装甲。"""
    cv2.drawContours(img_float, [box], 0, color, 3)  # 绘制装甲的轮廓
    cv2.circle(img_float, (int(center[0]), int(center[1])), 5, color, -1)  # 绘制装甲中心点
    put_text(img_float, armor_rect)  # 在图像上添加文本信息

def detect_armor(img, val, gamma, color=(0, 0, 0)):
    """检测装甲并返回结果字典。"""
    img_binary, resized_img, img_float = img_processed(img, val, gamma)  # 处理图像，获取二值图、缩放图和浮点图
    rotated_rects = find_light(color, img_binary, img_float)  # 查找光照下的矩形
    armors_dict = track_armor(resized_img, img_float, rotated_rects)  # 跟踪装甲并获取装甲字典
    cv2.imshow("Detecting", img_float)  # 显示检测结果
    return armors_dict  # 返回装甲字典

def destroy():
    """销毁所有窗口。"""
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread('./photo/3.jpg')
    val = 89
    gamma = 4.31
    armors_dict = detect_armor(img, val, gamma)
    print(armors_dict)
    destroy()
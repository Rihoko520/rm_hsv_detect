# RM_DETECTOR

---

## 使用传统视觉识别装甲板

---

## 运行代码的准备工作

### 1. 克隆代码库
使用 Git 克隆代码仓库：
```bash
git clone https://github.com/Rihoko520/rm_detector.git
```

### 2. 环境设置
- **安装 Python**: 确保已安装 Python 3.x。

### 3. 安装依赖库
运行以下命令安装所需库：
```bash
pip install opencv-python numpy scikit-image pillow
```

### 4. 图像准备
- **准备待检测图像**: 确保图像格式为 JPG、PNG 等。
- **设置图像路径**: 修改代码中图像文件的路径：
```python
img = cv2.imread('path/to/your/image.jpg')
```

---

## 功能模块介绍

### 1. `adjust.py` 功能介绍
`adjust.py` 通过调整 **伽马值** 和 **阈值** 来实现图像的 **二值化** 处理，用于调整处理图像的动态参数。

#### 使用说明
- 修改 `mode` 变量以选择处理模式（图像或视频流）。
- 确保在图像模式下文件路径正确，或在视频流模式下确保视频源可用。
- 运行脚本后，通过滑动条调整伽马值和阈值，以查看不同处理效果的实时结果。

### 2. `cam.py` 功能介绍
`cam.py` 负责处理相机输入，支持实时目标检测和静态图像处理。它结合了 `adjust.py` 的参数调整功能，允许用户根据需要进行实时调节。

#### 使用说明
- 设置 `mode` 变量以选择不同的处理模式。
- 运行脚本后，根据选择的模式进行图像或视频流的处理。

### 3. `detector.py` 函数功能描述
`detector.py` 是主要的检测代码，负责装甲板的识别和处理。

#### 主要函数：
1. `adjust_rotated_rect(rect)`：
   - **功能**：调整旋转矩形的宽高和角度，使宽始终小于高。

2. `is_close(rect1, rect2, angle_tol, height_tol, width_tol, cy_tol)`：
   - **功能**：检查两个旋转矩形是否足够接近。

3. `put_text(img, color, rect)`：
   - **功能**：在图像上绘制旋转矩形的中心坐标文本。

4. `do_polygons_intersect(a, b)`：
   - **功能**：检查两个多边形是否相交。

5. `project_polygon(polygon, axis)`：
   - **功能**：将多边形投影到给定的轴上。

6. `img_processed(img, val, gamma)`：
   - **功能**：处理输入图像，返回二值图像、调整大小的图像和经过伽马校正的浮点图像。

7. `find_light(color, img_binary, img)`：
   - **功能**：查找图像中的光源并返回旋转矩形。

8. `armortype(img_blur, rotated_rect)`：
   - **功能**：判断装甲的类型并返回相应的类 ID。

9. `track_armor(img, img_blur, rotated_rects, angle_tol=15, height_tol=100, width_tol=100, cy_tol=100)`：
   - **功能**：跟踪装甲并返回装甲字典。

10. `destroy()`：
    - **功能**：销毁所有 OpenCV 窗口，释放资源。

#### 主要变量：
- `angle_tol`: 允许的角度误差，默认值为 `15`。
- `height_tol`: 允许的高度误差，默认值为 `100`。
- `width_tol`: 允许的宽度误差，默认值为 `100`。
- `cy_tol`: 中心 y 坐标之间的误差，默认值为 `100`。
- `gamma`: 伽马校正值，默认值为 `70.0`。
- `val`: 二值化阈值，默认值为 `128`。
- `color`: 绘制文本的颜色，跟随识别到的装甲板灯条颜色改变。
- `hsv`: 用于颜色检测的 HSV 色彩空间值，帮助识别不同颜色的装甲板。

### 4. `square.py` 功能介绍
`square.py` 用于创建一个白色背景上的正方形图像。虽然功能不重要，但可以作为简单的图像生成示例。

#### 使用说明
- 运行 `square.py` 将生成一个 640x480 像素的白色背景图像，并在中心绘制一个边长为 84 的正方形。图像将保存为 `output_image.png` 并自动显示。

---

## 装甲板识别流程

### 1. 图像处理
- **`img_processed(img, val, gamma)`**:
  - **调整图像大小**: 将图像调整为 `(640, 480)`。
  - **伽马校正**: 应用伽马校正以增强图像亮度。
  - **灰度转换**: 将图像转换为灰度图进行二值化处理。
  - **二值化**: 使用阈值 `val` 生成二值图像，便于后续的轮廓检测。

### 2. 查找光源
- **`find_light(color, img_binary, img)`**:
  - **轮廓检测**: 使用 `cv2.findContours` 查找图像中的轮廓，并返回旋转矩形。
  - **过滤小面积轮廓**: 过滤小面积的轮廓，确保只处理较大的轮廓。
  - **轮廓相交检测**: 检查轮廓之间是否相交，避免重复检测。

### 3. 跟踪装甲
- **`track_armor(img, img_blur, rotated_rects, angle_tol=15, height_tol=100, width_tol=100, cy_tol=100)`**:
  - **分组旋转矩形**: 将相近的旋转矩形分组，以便合并处理。
  - **合并相邻矩形**: 合并相邻的旋转矩形，过滤掉面积过小的矩形。
  - **判断装甲类型**: 调用 `armortype(img_blur, rotated_rect)` 函数判断装甲类型，返回相应的类 ID。
  - **生成装甲字典**: 包含每个装甲的类 ID、高度和中心坐标，并在图像上绘制检测结果。
![detect_armor](/photo/detect.jpg) 
### 4. 返回结果返回包含检测到的装甲信息的字典：```{'526': {'class_id': 7, 'height': 78, 'center': [526, 288]}}```
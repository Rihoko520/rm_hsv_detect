import cv2
from detector import detect_armor
import adjust  # 导入调试代码

# 设置模式：0 - 调整参数（包括 HSV），1 - 目标检测（仅 gamma 和阈值），2 - 处理静态图像，3 - 静态图像调节并实时检测
global mode 
mode = 3 # 将此值改为 0 以进行参数调整，1 进行目标检测，2 进行静态图像处理，3 进行静态图像调节并实时检测
global image_path   
image_path = './photo/3.jpg' # 替换为你的图像路径

def main():
    if mode == 0:
        # 调整参数，包括 HSV
        adjust.run_adjustment(use_hsv=True)
    elif mode == 1:
        # 目标检测
        url = 'http://192.168.3.195:4747/video/'
        video_stream = cv2.VideoCapture(url)

        if not video_stream.isOpened():
            print("错误: 无法打开视频流。")
            return
        
        # 不需要 HSV 调试，直接设置 gamma 和阈值的窗口
        adjust.setup_windows(use_hsv=False)  # 设置滑动条窗口

        while True:
            ret, frame = video_stream.read()
            if not ret:
                print("错误: 无法读取帧")
                break
            
            # 通过 adjust 处理帧
            frame = adjust.resize_frame(frame)  # 调整大小
            
            # 仅调试 gamma 和阈值
            adjust.apply_changes_without_hsv(frame)  # 仅调试 gamma 和阈值
            
            # 获取当前调节的参数
            threshold_value = adjust.threshold_value  # 获取当前阈值
            gamma = adjust.gamma  # 获取当前 gamma 值
            
            # 目标检测，传入阈值和 gamma
            armors_dict = detect_armor(frame, threshold_value, gamma)
            if armors_dict :
                print(armors_dict)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_stream.release()
        cv2.destroyAllWindows()
    elif mode == 2:
        # 处理静态图像
        current_frame = cv2.imread(image_path)
        adjust.setup_windows(use_hsv=True)  # 设置滑动条窗口        
        if current_frame is None:
            print("错误: 无法读取图像。请检查路径:", image_path)
            return

        current_frame = adjust.resize_frame(current_frame)  # 调整大小

        while True:
            # 显示初始图像
            adjust.apply_changes_with_hsv(current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    elif mode == 3:
        # 静态图像调节并实时进行目标检测

        adjust.setup_windows(use_hsv=False)  # 设置滑动条窗口
        current_frame = cv2.imread(image_path)

        if current_frame is None:
            print("错误: 无法读取图像。请检查路径:", image_path)
            return

        current_frame = adjust.resize_frame(current_frame)  # 调整大小

        while True:
            # 显示初始图像
            adjust.apply_changes_without_hsv(current_frame)

            # 获取当前调节的参数
            threshold_value = adjust.threshold_value  # 获取当前阈值
            gamma = adjust.gamma  # 获取当前 gamma 值

            # 进行目标检测
            armors_dict = detect_armor(current_frame, threshold_value, gamma)
            if armors_dict :
                print(armors_dict)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    else:
        print("无效的模式，程序结束。")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体支持中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def detect_lane_lines(image_path):
    # 1. 读取图像并预处理
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img_rgb.copy()
    height, width = img.shape[:2]

    # 2. 颜色过滤：增强黄色检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 调整黄色阈值，扩大检测范围
    yellow_low = np.array([15, 80, 80])    # 降低饱和度和明度下限
    yellow_high = np.array([35, 255, 255]) # 扩大色调范围
    yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
    # 白色车道线掩码
    white_low = np.array([0, 0, 200])
    white_high = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv, white_low, white_high)
    color_mask = cv2.bitwise_or(yellow_mask, white_mask)
    color_filtered = cv2.bitwise_and(img, img, mask=color_mask)

    # 3. 边缘检测
    gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 稍大核减少噪点
    edges = cv2.Canny(blur, 50, 150)          # 降低阈值提高检测灵敏度

    # 4. 优化感兴趣区域：更贴合实际车道形状
    mask = np.zeros_like(edges)
    # 调整多边形顶点，使区域更贴近中间黄色车道线
    vertices = np.array([[
        (width*0.4, height),         # 左下：向右移动，靠近中间车道
        (width*0.55, height*0.3),    # 左上：调整倾斜角度
        (width*0.42, height*0.3),    # 右上：调整倾斜角度
        (width*0.65, height)          # 右下：向左移动，靠近中间车道
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 5. 霍夫变换：优化参数检测连续线段
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=15,           # 降低阈值检测更多线段
        minLineLength=50,       # 接受较短线段
        maxLineGap=60           # 允许更大间隙连接断续线
    )

    # 6. 线段拟合优化：针对中间车道线调整斜率过滤
    if lines is not None:
        # 增加中间车道线检测逻辑（针对单条黄色车道线场景）
        middle_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # 放宽斜率限制，允许接近垂直的线段（适应中间车道线）
            if abs(slope) < 0.8 and abs(slope) > 0.1:  # 更平缓的斜率范围
                middle_lines.append(line)

        # 拟合中间车道线（针对单条黄色车道线优化）
        if middle_lines:
            mid_points = np.array([[x1, y1, x2, y2] for line in middle_lines]).reshape(-1, 2)
            mid_vx, mid_vy, mid_x, mid_y = cv2.fitLine(
                mid_points, cv2.DIST_L2, 0, 0.01, 0.01
            )
            # 调整绘制范围，使线条更贴合实际车道长度
            y1_mid = height
            y2_mid = int(height * 0.2)  # 延长检测线长度
            x1_mid = int((y1_mid - mid_y[0]) * mid_vx[0] / mid_vy[0] + mid_x[0])
            x2_mid = int((y2_mid - mid_y[0]) * mid_vx[0] / mid_vy[0] + mid_x[0])
            # 绘制中间车道线
            cv2.line(result, (x1_mid, y1_mid), (x2_mid, y2_mid), (0, 255, 0), 100)

    # 7. 显示结果
    plt.figure(figsize=(12, 10))
    plt.subplot(221), plt.imshow(img_rgb), plt.title("原始图像")
    plt.subplot(222), plt.imshow(color_filtered), plt.title("颜色过滤后")
    plt.subplot(223), plt.imshow(masked_edges, cmap='gray'), plt.title("掩码边缘")
    plt.subplot(224), plt.imshow(result), plt.title("车道线检测结果")
    plt.tight_layout()
    plt.show()

    # 保存结果
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("lane_detection_result.jpg", result_bgr)
    print("检测结果已保存为 lane_detection_result.jpg")

if __name__ == "__main__":
    image_path = "campus_road.jpg"  # 替换为你的图像路径
    detect_lane_lines(image_path)
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 加载YOLOv5模型
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model.conf = 0.25  # 降低阈值（关键修改）
    model.iou = 0.45
    model.classes = [1]  # 仍保留自行车类别
    return model

# 检测共享单车
def detect_bikes(model, image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 进行目标检测
    results = model(img_rgb)

    # 筛选出自行车类别 (YOLOv5中自行车的类别ID是1)
    bikes = []
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 1:  # 1对应自行车类别
            x1, y1, x2, y2 = map(int, box)
            bikes.append((x1, y1, x2, y2, float(conf)))

    return img_rgb, bikes


# 绘制检测结果
def draw_results(img, bikes):
    img_copy = img.copy()
    for (x1, y1, x2, y2, conf) in bikes:
        # 绘制 bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制置信度
        text = f'Bike: {conf:.2f}'
        cv2.putText(img_copy, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img_copy


# 主函数
def main(image_path):
    # 加载模型
    model = load_model()
    print("模型加载完成")

    # 检测共享单车
    img, bikes = detect_bikes(model, image_path)
    print(f"检测到 {len(bikes)} 辆共享单车")

    # 绘制结果
    result_img = draw_results(img, bikes)

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()

    # 保存结果
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('bike_detection_result.jpg', result_img_bgr)
    print("检测结果已保存为 bike_detection_result.jpg")


if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = "bike.png"  # 请将此处替换为图一的实际路径
    main(image_path)
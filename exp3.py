import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# ----------------------------
# 1. 环境配置
# ----------------------------
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ----------------------------
# 2. 图像预处理函数（针对学号照片）
# ----------------------------
def preprocess_image(image_path):
    """
    处理学号照片：灰度化、二值化、分割单个数字、归一化
    返回：分割后的单个数字图像列表（28×28，与MNIST格式一致）
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 1. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 二值化（反相：数字为白色，背景为黑色，与MNIST一致）
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. 去除噪声（形态学操作）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭合操作填补空洞

    # 4. 轮廓检测（获取数字区域）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 筛选并排序轮廓（按x坐标排序，确保数字顺序正确）
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤过小的噪声轮廓
        if w > 10 and h > 10:
            digit_contours.append((x, y, w, h))

    # 按x坐标排序（从左到右）
    digit_contours.sort(key=lambda c: c[0])

    # 6. 分割并处理单个数字
    digits = []
    for x, y, w, h in digit_contours:
        # 提取数字区域
        digit = binary[y:y + h, x:x + w]

        # 调整尺寸为28×28（MNIST格式）
        # 先在数字周围填充边框，保持比例
        pad_size = max(w, h)
        pad_w = (pad_size - w) // 2
        pad_h = (pad_size - h) // 2
        digit_padded = cv2.copyMakeBorder(
            digit, pad_h, pad_size - h - pad_h, pad_w, pad_size - w - pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        # 缩放至28×28
        digit_resized = cv2.resize(digit_padded, (28, 28), interpolation=cv2.INTER_AREA)

        digits.append(digit_resized)

    return digits


# ----------------------------
# 3. 定义CNN模型
# ----------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # 卷积层：输入1通道（灰度图），输出32通道，卷积核3×3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 卷积层：32通道→64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层：2×2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层：64*7*7（池化后尺寸）→128→10（数字0-9）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # 激活函数和 dropout（防止过拟合）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积→激活→池化：(1,28,28)→(32,28,28)→(32,14,14)
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积→激活→池化：(32,14,14)→(64,14,14)→(64,7,7)
        x = self.pool(self.relu(self.conv2(x)))
        # 展平：(64,7,7)→(64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        # 全连接→激活→dropout
        x = self.dropout(self.relu(self.fc1(x)))
        # 输出层（无激活，配合交叉熵损失）
        x = self.fc2(x)
        return x


# ----------------------------
# 4. 训练模型（使用MNIST数据集）
# ----------------------------
def train_model():
    # 数据预处理（与MNIST一致：归一化到0-1，转为Tensor）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    # 数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # 计算训练集平均损失
        train_loss /= len(train_loader.dataset)

        # 在测试集上验证
        model.eval()  # 评估模式
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)

                # 统计准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / total

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'digit_cnn_model.pth')
    print("模型已保存为 digit_cnn_model.pth")
    return model


# ----------------------------
# 5. 预测学号数字
# ----------------------------
def predict_student_id(image_path, model):
    # 预处理图像，获取单个数字
    digits = preprocess_image(image_path)
    if not digits:
        print("未检测到数字区域")
        return []

    # 转换为模型输入格式
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 预测每个数字
    model.eval()
    result = []
    with torch.no_grad():
        for digit in digits:
            # 转换为Tensor并添加批次维度
            digit_tensor = transform(digit).unsqueeze(0).to(device)
            # 预测
            output = model(digit_tensor)
            _, predicted = torch.max(output.data, 1)
            result.append(str(predicted.item()))

    return result


# ----------------------------
# 主函数：执行训练和预测
# ----------------------------
if __name__ == "__main__":
    # 训练模型（首次运行时执行，后续可注释掉直接加载模型）
    #  model = train_model()

    # 加载已训练的模型
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load('digit_cnn_model.pth', map_location=device))

    # 预测学号（替换为你的学号照片路径）
    image_path = "student_id.jpg"  # 请将你的学号照片命名为此路径
    student_id = predict_student_id(image_path, model)

    print(f"识别结果：{''.join(student_id)}")
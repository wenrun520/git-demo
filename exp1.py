import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from numba import jit  # 加速循环

#实验一
# ===================== 通用工具函数 =====================
def read_image(path):
    """读取图像（支持PNG/JPG格式），转为RGB浮点型数组（0-255）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件不存在：{path}")
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def save_image(img, save_path):
    """保存图像（自动归一化到0-255并转为uint8）"""
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 处理单通道图像（LBP是单通道，需扩展为3通道才能正常保存）
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)  # 单通道→三通道（灰度图）
    Image.fromarray(img).save(save_path)
    print(f"图像已保存到：{save_path}")


# ===================== 1. 卷积操作 =====================
@jit(nopython=True)  # Numba加速卷积循环
def convolution_jit(padded_img, kernel, h, w, c, k_size):
    """卷积核心逻辑（Numba加速）"""
    output = np.zeros((h, w, c), dtype=np.float32)
    for channel in range(c):
        for i in range(h):
            for j in range(w):
                neighbor = padded_img[i:i + k_size, j:j + k_size, channel]
                output[i, j, channel] = np.sum(neighbor * kernel)
    return output


def convolution(img, kernel, padding_mode="constant"):
    """通用卷积函数（支持多通道，优化大图像性能）"""
    h, w, c = img.shape
    k_size = kernel.shape[0]
    if k_size % 2 == 0:
        raise ValueError("卷积核尺寸必须为奇数")
    pad = k_size // 2

    # 补边
    padded_img = np.pad(
        img,
        pad_width=((pad, pad), (pad, pad), (0, 0)),
        mode=padding_mode
    )

    # 调用加速后的卷积逻辑
    output = convolution_jit(padded_img, kernel, h, w, c, k_size)
    return output


def sobel_edge_detection(img):
    """Sobel边缘检测（x+y方向合并）"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    edge_x = convolution(img, sobel_x)
    edge_y = convolution(img, sobel_y)
    sobel_combined = np.sqrt(np.square(edge_x) + np.square(edge_y))
    return sobel_combined, edge_x, edge_y


# ===================== 2. 颜色直方图 =====================
def extract_color_histogram(img, save_path="color_histogram.png"):
    """提取RGB三通道颜色直方图"""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img_uint8.shape[:2]

    # 优化：用numpy内置函数统计（比双重循环快100倍+，大图像必备）
    hist_r, _ = np.histogram(img_uint8[..., 0].flatten(), bins=256, range=(0, 255))
    hist_g, _ = np.histogram(img_uint8[..., 1].flatten(), bins=256, range=(0, 255))
    hist_b, _ = np.histogram(img_uint8[..., 2].flatten(), bins=256, range=(0, 255))

    # 绘制柱状图
    plt.figure(figsize=(12, 5))
    x = np.arange(256)
    width = 1.0  # 柱宽（填满bin）
    plt.bar(x, hist_r, width=width, color="red", alpha=0.7, label="Red Channel")
    plt.bar(x, hist_g, width=width, color="green", alpha=0.7, label="Green Channel")
    plt.bar(x, hist_b, width=width, color="blue", alpha=0.7, label="Blue Channel")

    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Pixel Count")
    plt.title("RGB Color Histogram (Bar Chart)")
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.xlim(0, 255)  # 限制x轴范围，避免空白
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"颜色直方图已保存到：{save_path}")
    return [hist_r, hist_g, hist_b]


# ===================== 3. LBP纹理特征 =====================
@jit(nopython=True)  # 加速LBP循环
def lbp_jit(gray, h, w):
    """LBP核心逻辑（Numba加速）"""
    lbp_feature = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            lbp_code = 0
            # 8邻域顺时针编码
            neighbors = [
                gray[i - 1, j - 1], gray[i - 1, j], gray[i - 1, j + 1],
                gray[i, j + 1], gray[i + 1, j + 1], gray[i + 1, j],
                gray[i + 1, j - 1], gray[i, j - 1]
            ]
            for k in range(8):
                if neighbors[k] > center:
                    lbp_code |= (1 << (7 - k))
            lbp_feature[i, j] = lbp_code
    return lbp_feature


def extract_lbp_texture(img, save_path="lbp_texture.npy"):
    """提取LBP纹理特征"""
    # 转为灰度图
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    h, w = gray.shape

    # 调用加速后的LBP逻辑
    lbp_feature = lbp_jit(gray, h, w)

    # 保存npy文件
    np.save(save_path, lbp_feature)
    print(f"LBP纹理特征已保存到：{save_path}")

    # 保存可视化图（单通道→三通道，避免保存错误）
    save_image(lbp_feature, "experiment_results/lbp_visualization.png")
    return lbp_feature


# ===================== 主函数（整合所有任务） =====================
def main(image_path, output_dir="experiment_results"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    print("正在读取图像...")
    img = read_image(image_path)
    print(f"图像尺寸：{img.shape}")

    # Sobel边缘检测
    print("正在执行Sobel边缘检测...")
    sobel_combined, sobel_x, sobel_y = sobel_edge_detection(img)
    save_image(sobel_combined, os.path.join(output_dir, "sobel_combined.png"))
    save_image(sobel_x, os.path.join(output_dir, "sobel_x.png"))
    save_image(sobel_y, os.path.join(output_dir, "sobel_y.png"))

    # 给定核滤波
    print("正在执行给定核滤波...")
    given_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    given_kernel_result = convolution(img, given_kernel)
    save_image(given_kernel_result, os.path.join(output_dir, "given_kernel_filter.png"))

    # 颜色直方图（柱状图）
    print("正在提取颜色直方图...")
    extract_color_histogram(img, os.path.join(output_dir, "color_histogram.png"))

    # LBP纹理特征
    print("正在提取LBP纹理特征...")
    extract_lbp_texture(img, os.path.join(output_dir, "lbp_texture.npy"))

    print("\n所有任务执行完成！结果保存在：", output_dir)


# ===================== 运行实验 =====================
if __name__ == "__main__":
    INPUT_IMAGE_PATH = "1764329365855.jpg"
    try:
        main(INPUT_IMAGE_PATH)
    except Exception as e:
        print(f"实验执行出错：{e}")
        # 打印详细错误信息
        import traceback

        traceback.print_exc()
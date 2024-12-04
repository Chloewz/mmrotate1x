import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


# TODO: 还没有尝试过，无需求，有需要再调整这个
# Step 1: 生成热力图函数
def generate_heatmap(feature_map):
    """
    通过对特征图进行通道均值操作，生成热力图
    :param feature_map: Tensor, shape (batch_size, channels, height, width)
    :return: 归一化后的热力图，numpy数组
    """
    # 取通道均值，得到每个空间位置的响应
    heatmap = feature_map.mean(dim=1, keepdim=True)  # (batch_size, 1, height, width)

    # 去掉 batch 维度，转为二维数据
    heatmap = heatmap.squeeze().cpu().numpy()  # 转为 numpy 数组

    # 归一化到 [0, 1]
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()

    return heatmap


# Step 2: 可视化热力图
def plot_heatmap(heatmap, output_path):
    """
    将热力图绘制并保存为图像
    :param heatmap: 生成的热力图 (二维 numpy 数组)
    :param output_path: 保存路径
    """
    plt.imshow(heatmap, cmap="jet")  # 使用 jet 配色
    plt.colorbar()  # 显示颜色条
    plt.savefig(output_path)  # 保存图像
    plt.close()  # 关闭 plt


# Step 3: 将热力图叠加到原图
def overlay_heatmap_on_image(image_path, heatmap, alpha=0.5):
    """
    将热力图叠加到原始图像上
    :param image_path: 原始图像的路径
    :param heatmap: 生成的热力图 (二维 numpy 数组)
    :param alpha: 热力图叠加的透明度
    :return: 叠加后的图像
    """
    # 读取原始图像
    image = cv2.imread(image_path)

    # 对热力图进行调整，确保大小与原图一致
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # 将热力图转换为伪彩色图像
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # 将热力图和原图进行叠加
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


# Step 4: 主程序
def process_feature_map_and_generate_heatmap(
    feature_map, image_path, save_dir, output_filename="heatmap.png"
):
    """
    主程序，执行从特征图到热力图的生成与保存
    :param feature_map: 输入的特征图 (Tensor, shape: (batch_size, channels, height, width))
    :param image_path: 原始图像路径，用于叠加热力图
    :param save_dir: 热力图保存目录
    :param output_filename: 输出的热力图文件名
    """
    # 生成热力图
    heatmap = generate_heatmap(feature_map)

    # 可视化并保存热力图
    heatmap_output_path = f"{save_dir}/{output_filename}"
    plot_heatmap(heatmap, heatmap_output_path)
    print(f"Heatmap saved to {heatmap_output_path}")

    # 将热力图叠加到原始图像
    overlay_image = overlay_heatmap_on_image(image_path, heatmap)
    overlay_output_path = f"{save_dir}/overlay_{output_filename}"
    cv2.imwrite(overlay_output_path, overlay_image)
    print(f"Overlay image saved to {overlay_output_path}")


# 例如，假设你已经有了一个 feature_map 和一个原始图像路径
# feature_map 是模型输出的特征图 (Tensor, shape: (batch_size, channels, height, width))
# image_path 是你想叠加热力图的原始图像文件路径

# 示例：生成热力图并保存
# 假设 feature_map 是形状为 (1, 256, 64, 64) 的特征图
feature_map = torch.randn(1, 256, 64, 64)  # 模拟特征图
image_path = "path_to_your_image.jpg"  # 你想叠加热力图的原始图像路径
save_dir = "path_to_save_directory"  # 保存热力图的目录

# 运行主程序
process_feature_map_and_generate_heatmap(feature_map, image_path, save_dir)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap


# TODO: 只是简单的打印出来Heat Map，没有对可视化进行美观方面的调整
# Step 1: 生成热力图函数
def generate_heatmap(feature_map):
    """
    通过对特征图进行通道均值操作，生成热力图
    :param feature_map: Tensor, shape (batch_size, channels, height, width)
    :return: 归一化后的热力图，numpy数组
    """
    # 取通道均值，得到每个空间位置的响应
    heatmap = feature_map.mean(axis=1, keepdims=True)  # (batch_size, 1, height, width)

    # 去掉 batch 维度，转为二维数据
    heatmap = heatmap.squeeze()  # 转为 numpy 数组

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


# Step 3: 主程序
def visualize_heatmap(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                heatmap_filename = file.split(".npy")[0] + "_heatmap.png"
                png_path = os.path.join(root, heatmap_filename)

                feature_map = np.load(npy_path)
                heatmap = generate_heatmap(feature_map)

                plot_heatmap(heatmap, png_path)
                print(f"Heatmap saved to {png_path}")


if __name__ == "__main__":
    feature_map = ""

    # 运行主程序
    visualize_heatmap(feature_map)

import numpy as np
import matplotlib.pyplot as plt
import os


# TODO: 只是简单的打印出来Feature Map，没有对可视化进行美观方面的调整
def visualize_feature_map(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                heatmap_filename = file.split(".npy")[0] + "_featmap.png"
                png_path = os.path.join(root, heatmap_filename)
                # 加载特征图
                feature_map = np.load(npy_path)
                feature_map = feature_map[0]

                # 可视化第一个通道
                plt.imshow(feature_map[0], cmap="viridis")
                plt.colorbar()
                plt.savefig(png_path)
                plt.show()
                plt.close()


if __name__ == "__main__":
    folder_path = ""
    visualize_feature_map(folder_path)

"""
从预测数据集中快速复制出具有相似类别的图片
"""

import shutil
import os


def copy_images(source_folder, target_folder, image_files):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 复制文件
    for image in image_files:
        src_path = os.path.join(source_folder, image)
        dest_path = os.path.join(target_folder, image)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"已复制: {image}")
        else:
            print(f"文件未找到: {image}")


if __name__ == "__main__":
    # 示例使用
    source_folder = "/mnt/d/exp/sodaa_sob/datasets-similar/images-all/"  # 源文件夹路径
    target_folder = "/mnt/d/try/"  # 目标文件夹路径
    image_files = [
        "00010__800__1950___650.jpg",
        "00010__800__3250___650.jpg",
        "00010__800__3900___650.jpg",
        "00010__800__4000___650.jpg",
        "00022__800__1300___1300.jpg",
        "00022__800__1300___1944.jpg",
        "00022__800__1950___1300.jpg",
        "00022__800__3900___1944.jpg",
        "00022__800__4000___1944.jpg",
        "00105__800__1300___650.jpg",
        "00105__800__1300___1950.jpg",
        "00105__800__3200___0.jpg",
        "00105__800__3250___650.jpg",
        "01071__800__2600___0.jpg",
        "02434__800__650___650.jpg",
        "02434__800__3955___1300.jpg",
        "02434__800__3955___1985.jpg",
        "02461__800__3900___650.jpg",
        "02461__800__3900___1300.jpg",
        "02461__800__3900___1950.jpg",
    ]  # 要复制的图片文件列表

    copy_images(source_folder, target_folder, image_files)

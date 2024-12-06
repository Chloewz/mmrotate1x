import os
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmrotate.structures import rbox2qbox


def crop_image(image, crop_size, overlap):
    """裁剪大图成小块"""
    h, w, _ = image.shape
    crops = []
    coords = []
    step = crop_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + crop_size, w)
            y_end = min(y + crop_size, h)
            crop = image[y:y_end, x:x_end]
            crops.append(crop)
            coords.append((x, y))  # 保存裁剪块的起始坐标
    return crops, coords


def detect_and_merge(crops, coords, model, crop_size, overlap):
    """对每个小块进行检测并合并结果"""
    results = []

    for i, (crop, coord) in enumerate(zip(crops, coords)):
        x_offset, y_offset = coord
        # 保存当前裁剪块到本地（可选，用于调试）
        crop_path = os.path.join(output_dir, f"crop_{i}.jpg")
        cv2.imwrite(crop_path, crop)

        # 使用模型检测
        detections = inference_detector(model, crop)

        # 提取 DOTA 格式结果
        bboxes = detections.pred_instances.bboxes
        confs = detections.pred_instances.scores

        for i in range(13):  # 针对一个类别的框
            class_name = "ship"
            # cx, cy, w, h, angle = bboxes[i]
            score = confs[i].cpu().numpy()
            poly = rbox2qbox(bboxes[i])
            poly_global = [
                poly[0] + x_offset,
                poly[1] + y_offset,
                poly[2] + x_offset,
                poly[3] + y_offset,
                poly[4] + x_offset,
                poly[5] + y_offset,
                poly[6] + x_offset,
                poly[7] + y_offset,
            ]
            results.append(poly_global + [class_name, score])

    return results


def save_results(results, save_path):
    """保存合并后的结果到DOTA格式的txt文件"""
    with open(save_path, "w") as f:
        for res in results:
            x1, y1, x2, y2, x3, y3, x4, y4, class_name, score = res
            line = f"DOTA {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} {x4:.2f} {y4:.2f}\n"
            f.write(line)


if __name__ == "__main__":
    # 配置路径
    config_file = "/mnt/d/exp/"  # mmrotate模型配置文件路径
    checkpoint_file = "/mnt/d/exp/"  # 训练好的模型权重路径
    img_path = "/mnt/d/exp/"  # 输入大图路径
    output_dir = "/mnt/d/exp/"  # 输出目录路径
    crop_size = 1024  # 裁剪块的尺寸
    overlap = 200  # 裁剪块之间的重叠区域

    # 初始化模型
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载大图
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法加载图像 {img_path}")

    # 裁剪大图
    crops, coords = crop_image(image, crop_size, overlap)

    # 检测并合并结果
    results = detect_and_merge(crops, coords, model, crop_size, overlap)

    # 保存结果到大图的DOTA格式文件
    result_path = os.path.join(output_dir, "detection_results.txt")
    save_results(results, result_path)

    print(f"检测完成！结果保存在 {result_path}")

"""
统计一张预测图中的预测结果中某个类别的得分的均值有多少
即对一张图像进行推理
得到的是模型对这张图像进行预测的结果，这些结果的类别数量总计和平均得分
"""

import mmcv
from mmdet.apis import inference_detector, init_detector

# config_file = "mmrotate1x/configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py"  # 调试时的文件夹目录
config_file = "configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py" # 运行时的文件夹目录
checkpoint_file = "/mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth"
img = "/mnt/d/exp/sodaa_sob/datasets-similiar/images/00002__800__0___0.jpg"
img = mmcv.imread(img)

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 推理图像
result = inference_detector(model, img)
print(result)

class_stats = {}

# 打印类别得分
if isinstance(result, list):
    for data_sample in result:
        # 提取预测的边界框、类别ID和得分
        pred_instances = data_sample.pred_instances
        for i, instance in enumerate(pred_instances):
            cls_score = instance.score
            cls_id = instance.clses_id
            bbox = instance.bbox
            print(f"Class {cls_id}, Score: {cls_score:.4f}, BBox: {bbox}")
else:
    # result不是一个列表，直接处理单个DetDataSample对象
    for i, instance in enumerate(result.pred_instances):
        # print(instance)
        cls_score = instance.scores.cpu().item()
        cls_id = instance.labels.cpu().item()
        bbox = instance.bboxes.cpu().numpy()

        # 格式化bbox，使得每个坐标值都保留四位小数
        formatted_bbox = [f"{x:.4f}" for x in bbox[0]]

        # 更新类别的预测次数和得分总和
        if cls_id not in class_stats:
            class_stats[cls_id] = {'count':0, 'score_sum': 0.0}
        class_stats[cls_id]['count']+=1
        class_stats[cls_id]['score_sum']+=cls_score

        print(f"Class {cls_id}, Score: {cls_score:.4f}, BBox: {formatted_bbox}")

# 计算每个类别的得分均值
for cls_id, stats in class_stats.items():
    mean_score = stats['score_sum']/stats['count']
    print(f"Class {cls_id}:")
    print(f" - Predicted {stats['count']} times")
    print(f" - Average score: {mean_score:.4f}\n")

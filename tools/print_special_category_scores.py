import mmcv
import mmengine
import torch
from mmdet.apis import init_detector
from mmengine.dataset import Compose
import torch.nn.functional as F

# 配置文件和模型权重路径
config_file = "configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py"
checkpoint_file = "/mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth"
img_path = "/mnt/d/exp/sodaa_sob/datasets-similiar/images/00022__800__1300___1944.jpg"

# 读取配置文件
cfg = mmengine.Config.fromfile(config_file)

# 初始化模型
model = init_detector(config_file, checkpoint_file, device="cuda:0")
model.eval()  # 切换模型到评估模式，确保不会应用 softmax

# 加载图像数据
img = mmcv.imread(img_path)
data = dict(img=img, img_path=img_path, img_id=0)  # 添加 img_id 字段

# 构建预处理流水线
pipeline = Compose(cfg.test_pipeline)

# 通过预处理流水线
data = pipeline(data)

# 增加 batch 维度并移动到 GPU
data['inputs'] = data['inputs'].unsqueeze(0).to('cuda:0')
# 将输入转换为浮点格式并归一化
data['inputs'] = data['inputs'].float() / 255.0

# 将图像送入模型的 backbone 和 bbox_head，获得 logits
with torch.no_grad():
    features = model.extract_feat(data['inputs'])  # 提取特征
    cls_scores, _ = model.bbox_head(features)  # 获取分类分数（logits）

# 获取类别数量
num_classes = cfg.model.bbox_head.num_classes

# 初始化列表存储每个实例的 softmax 后的概率向量
softmax_vectors = []

# 对每个实例的 logits 应用 softmax
for cls_score in cls_scores:
    # 应用 softmax 转换为概率
    softmax_probs = F.softmax(cls_score, dim=-1)  # softmax 需要在类别维度应用
    softmax_probs = softmax_probs.squeeze().cpu().numpy()  # 将其转换为 NumPy 数组并移除 batch 维度
    softmax_vectors.append(softmax_probs)

# 打印每个实例的 softmax 后的概率向量
for i, softmax_vector in enumerate(softmax_vectors):
    print(f"Instance {i} softmax vector: {softmax_vector[:num_classes]}")


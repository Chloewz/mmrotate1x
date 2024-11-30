import numpy as np
import matplotlib.pyplot as plt

# 加载特征图
feature_map = np.load('/mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/featuremap/batch_0_layer_4.npy')

feature_map = feature_map[0]

# 可视化第一个通道
plt.imshow(feature_map[0], cmap='viridis')
plt.colorbar()
plt.savefig('/mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/featuremap/layer_4.png')
plt.show()

"""
对相似类别的预测过程进行分析
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns

scores_path = (
    "/home/odysseus/pyFiles/mmrotate1x/tools/special_category/retinanet_105_1300_650_scores_scale_2.csv"
)
idx_path = "/home/odysseus/pyFiles/mmrotate1x/tools/special_category/retinanet_105_1300_650_idx_scale_2.csv"

scores_pd = pd.read_csv(scores_path)
idx_pd = pd.read_csv(idx_path)

scores_pd = scores_pd.iloc[:, 1:]
idx_values = idx_pd.iloc[:, 1].drop_duplicates().values
result = scores_pd.loc[idx_values]

x = result.iloc[:, 3]
y = result.iloc[:, 5]

"""
散点图的形式观察
"""
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue')
#
# plt.title('Similar Category Analysis')
# plt.xlabel('large-vehicle')
# plt.ylabel('container')
# plt.show()

"""
相关系数的形式观察
"""
# corr, _ = spearmanr(x, y)
# print(corr)

"""
SeaBorn的形式观察
"""
p = sns.jointplot(x=result.columns[3], y=result.columns[5], data=result, kind="scatter",color='blue')
# p = sns.jointplot(x=scores_pd.columns[3], y=scores_pd.columns[5], data=scores_pd, kind="scatter",color='blue')

# filter_result = result[(result[result.columns[3]] > 0.05) | (result[result.columns[5]] > 0.05)]
# sns.jointplot(x=filter_result.columns[3], y=filter_result.columns[5], data=filter_result, kind="scatter")
#
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(0.015))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.015))
plt.xlim(0,0.1)
plt.ylim(0,0.1)

plt.suptitle("Similar Category RetinaNet Scale-2")
plt.xlabel("large-vehicle")
plt.ylabel("container")

plt.savefig("retinanet_105_1300_650_scale_2_.png")
plt.show()


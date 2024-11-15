"""
对相似类别的预测过程进行分析
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

scores_path = (
    "/home/odysseus/pyFiles/mmrotate1x/tools/special_category/s2anet_refine_scores_scale_2.csv"
)
idx_path = "/home/odysseus/pyFiles/mmrotate1x/tools/special_category/s2anet_refine_idx_scale_2.csv"

scores_pd = pd.read_csv(scores_path)
idx_pd = pd.read_csv(idx_path)

scores_pd = scores_pd.iloc[:, 1:]
idx_values = idx_pd.iloc[:, 1].drop_duplicates().values
result = scores_pd.loc[idx_values]

x = result.iloc[:, 3]
y = result.iloc[:, 5]

"""
相关系数的形式观察
"""
corr, _ = spearmanr(x, y)
print(corr)

"""
SeaBorn的形式观察
"""
p = sns.jointplot(x=result.columns[2], y=result.columns[7], data=result, kind="scatter",color='blue')
# p = sns.jointplot(x=scores_pd.columns[3], y=scores_pd.columns[5], data=scores_pd, kind="scatter",color='blue')

# filter_result = result[(result[result.columns[3]] > 0.05) | (result[result.columns[5]] > 0.05)]
# sns.jointplot(x=filter_result.columns[3], y=filter_result.columns[5], data=filter_result, kind="scatter")

ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(0.08))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.08))
plt.xlim(0,0.8)
plt.ylim(0,0.8)

plt.suptitle("Different Category S2ANet Scale-2")
# plt.xlabel("large-vehicle")
plt.xlabel("small-vehicle")
# plt.ylabel("container")
plt.ylabel("swimming-pool")

plt.savefig("tools/special_category/s2anet_scale_2.png")
plt.show()


"""
比例设置：
RetinaNet：
    Scale-1: MultipleLocater-0.06, xlim-0.6
    Scale-2: MultipleLocater-0.01, xlim-0.1

R3Det:
    Scale-1: MultipleLocater-0.09, xlim-0.9
    Scale-2: MultipleLocater-0.05, xlim-0.5
    Scale-3: MultipleLocater-0.01875, xlim-0.1875
    
S2ANet:
    Scale-1: MultipleLocater-0.1, xlim-1
    Scale-2: MultipleLocater-0.08, xlim-0.8
"""

"""
不同类别的比例设置
S2ANet:
    Scale-1: MultipleLocater-0.1, xlim-1
    Scale-2: MultipleLocater-0.08, xlim-0.8
"""


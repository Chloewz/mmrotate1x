"""
对相似类别的预测过程进行分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1, rc={"figure.figsize": (12,8)})

scores_path = (
    "/home/odysseus/pyFiles/mmrotate1x/tools/special_category/s2anet_refine_scores_scale_1.csv"
)
idx_path = "/home/odysseus/pyFiles/mmrotate1x/tools/special_category/s2anet_refine_idx_scale_1.csv"

scores_pd = pd.read_csv(scores_path)
idx_pd = pd.read_csv(idx_path)

scores_pd = scores_pd.iloc[:, 1:]
idx_values = idx_pd.iloc[:, 1].drop_duplicates().values
result = scores_pd.loc[idx_values]

"""
相关系数的形式观察
"""
# corr, _ = spearmanr(x, y)
# print(corr)

"""
SeaBorn的形式观察

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

# p = sns.jointplot(x=result.columns[2], y=result.columns[7], data=result, kind="scatter",color='blue')
# # p = sns.jointplot(x=scores_pd.columns[3], y=scores_pd.columns[5], data=scores_pd, kind="scatter",color='blue')
#
# ax = plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.08))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.08))
# plt.xlim(0,0.8)
# plt.ylim(0,0.8)
#
# plt.suptitle("Different Category S2ANet Scale-2")
# # plt.xlabel("large-vehicle")
# plt.xlabel("small-vehicle")
# # plt.ylabel("container")
# plt.ylabel("swimming-pool")
#
# plt.savefig("tools/special_category/s2anet_scale_2.png")
# plt.show()

"""
以45°为区分线，回归两个类别的图
"""
x = result.iloc[:, 3]
y = result.iloc[:, 5]

similar = result.iloc[:,[3,5]].copy()
similar['label']=similar.apply(lambda row: 'large-vehicle' if row.iloc[0]>row.iloc[1] else 'container', axis=1)

similar_x = similar[similar['label']=='large-vehicle']
similar_y = similar[similar['label']=='container']

slope_x, intercept_x,_,_,_= linregress(similar_x.iloc[:,0], similar_x.iloc[:,1])
slope_y, intercept_y,_,_,_= linregress(similar_y.iloc[:,0], similar_y.iloc[:,1])

angle = np.arctan(np.abs((slope_y-slope_x)/(1+slope_y*slope_x)))
angle_degrees = np.degrees(angle)

p = sns.lmplot(x=similar.columns[0], y=similar.columns[1], hue="label",data=similar, truncate=True,scatter_kws={'s':15},height=5)
p.set_axis_labels("large_vehicle", "container")

# x_vals = np.linspace(0,1,500)
# plt.figure(figsize=(10,7))
#
# plt.scatter(similar_x.iloc[:,0],similar_x.iloc[:,1],color='blue',label='large-vehicle')
# plt.scatter(similar_y.iloc[:,0],similar_y.iloc[:,1],color='orange',label='container')
#
# plt.plot(x_vals, slope_x*x_vals+intercept_x, color='blue', linestyle='-', linewidth=2)
# plt.plot(x_vals, slope_y*x_vals+intercept_y, color='orange', linestyle='-', linewidth=2)
#
plt.text(
    0.95, 0.95, f'Angle: {angle_degrees:.2f}°',
    transform=plt.gca().transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
)

ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.xlim(0,1)
plt.ylim(0,1)

plt.title(f"Similar Category S2ANet Scale-1")
# plt.savefig("tools/special_category/s2anet_scale_1.png",bbox_inches='tight')
plt.savefig("s2anet_scale_1.png",bbox_inches='tight')
plt.show()


# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

"""
===============================================================================
脚本名称: clustering_logbinning_strict.py
功能说明:
    1. 读取网络 GEXF 文件，计算节点度 k 与聚类系数 C。
    2. 使用指数增长的对数分箱方法，对不同度值区间内的聚类系数进行统计。
       - 分箱边界按固定比例 BIN_RATIO 递增。
       - 分箱横坐标使用几何中心。
       - 每个分箱统计聚类系数均值 Mean C 和中位数 Median C。
    3. 绘制双对数散点图。
       - 背景: 原始节点散点。
       - 前景: 分箱均值与分箱中位数。
    4. 图框、坐标轴、字体和颜色与 power_law2.py 保持一致。
===============================================================================
"""

# ==========================================
# 1. 全局样式设置
# ==========================================
FONT_FAMILY = 'Arial'
BASE_COLOR = '#444444'

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = BASE_COLOR
plt.rcParams['axes.labelcolor'] = BASE_COLOR
plt.rcParams['xtick.color'] = BASE_COLOR
plt.rcParams['ytick.color'] = BASE_COLOR
plt.rcParams['axes.edgecolor'] = BASE_COLOR
plt.rcParams['font.size'] = 11

# ==========================================
# 2. 配置与数据加载
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
feature_dir = os.path.dirname(script_dir)
data_dir = os.path.join(feature_dir, "data")
output_dir = os.path.join(feature_dir, "figures")

file_path = os.path.join(data_dir, "net_2006-2025_newman_S2.gexf")
filename_base = 'clustering_statistics_logbin_strict'

MIN_DEGREE = 5
BIN_RATIO = 1.2589

print(f"Reading file: {file_path} ...")
try:
    graph = nx.read_gexf(file_path)

    degree_dict = dict(graph.degree())
    clustering_dict = nx.clustering(graph)

    data_rows = []
    for node, degree in degree_dict.items():
        if degree >= MIN_DEGREE:
            clustering = clustering_dict.get(node, 0)
            if clustering > 0:
                data_rows.append({'k': degree, 'C': clustering})

    df = pd.DataFrame(data_rows)

    if df.empty:
        print("Error: no valid data for k >= MIN_DEGREE and C > 0.")
        exit()

    print(f"Valid nodes: {len(df)}")
    print(f"Degree range: {df['k'].min()} - {df['k'].max()}")

except Exception as exc:
    print(f"Read or calculation error: {exc}")
    exit()


# ==========================================
# 3. 对数分箱统计
# ==========================================
def log_binning_clustering_stats(k_list, c_list, bin_ratio=1.2589):
    """
    对节点度进行对数分箱，并计算每个分箱内聚类系数的均值和中位数。

    Args:
        k_list: 节点度列表，作为分箱横坐标。
        c_list: 聚类系数列表，作为统计对象。
        bin_ratio: 分箱边界递增比例。

    Returns:
        centers: 每个分箱的几何中心。
        means: 每个分箱内聚类系数的均值。
        medians: 每个分箱内聚类系数的中位数。
    """
    data = np.array(k_list)
    c_data = np.array(c_list)

    min_x = min(data)
    max_x = max(data)

    bins = [min_x]
    while bins[-1] * bin_ratio <= max_x:
        bins.append(bins[-1] * bin_ratio)
    if bins[-1] < max_x:
        bins.append(bins[-1] * bin_ratio)

    bins_array = np.array(bins)
    if bins_array[0] == 0:
        bins_array[0] = 1e-9

    bin_centers = 10 ** ((np.log10(bins_array[:-1]) + np.log10(bins_array[1:])) / 2)

    counts, _ = np.histogram(data, bins=bins_array)
    c_sums, _ = np.histogram(data, bins=bins_array, weights=c_data)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_c = c_sums / counts

    indices = np.digitize(data, bins=bins_array)
    median_c = []
    for i in range(1, len(bins_array)):
        mask = (indices == i)
        if np.any(mask):
            median_c.append(np.median(c_data[mask]))
        else:
            median_c.append(np.nan)

    median_c = np.array(median_c)
    mask = counts > 0
    return bin_centers[mask], mean_c[mask], median_c[mask]


print("Running log-binning statistics...")
bin_centers, bin_means, bin_medians = log_binning_clustering_stats(
    df['k'], df['C'], bin_ratio=BIN_RATIO
)
print(f"Log-binning complete: {len(bin_centers)} statistic points.")

# ==========================================
# 4. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(4.55, 3.34), dpi=600)
ax.set_position([0.189231, 0.220848, 0.571477, 0.707526])

# 原始节点散点
ax.scatter(df['k'], df['C'],
           color='#808080',
           alpha=0.08,
           s=12,
           label='Original data',
           edgecolor='none',
           zorder=1)

# 分箱均值
ax.scatter(bin_centers, bin_means,
           marker='o',
           color='#0172be',
           s=32,
           label='Mean data',
           edgecolor='white',
           linewidth=0.5,
           alpha=1,
           zorder=5)

# 分箱中位数
ax.scatter(bin_centers, bin_medians,
           marker='D',
           color='#ed7d2f',
           s=24,
           label='Median data',
           edgecolor='white',
           linewidth=0.5,
           alpha=1,
           zorder=6)

# ==========================================
# 5. 坐标轴与图框
# ==========================================
ax.set_xscale('log')
ax.set_yscale('log')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(BASE_COLOR)

ax.xaxis.tick_bottom()
ax.xaxis.set_label_position('bottom')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

ax.tick_params(axis='x', which='both', direction='in',
               top=False, bottom=True, labelsize=11, rotation=0)
for label in ax.get_xticklabels():
    label.set_horizontalalignment('center')
    label.set_rotation_mode(None)
ax.tick_params(axis='y', which='both', direction='in',
               left=False, right=True, labelleft=False, labelright=True, labelsize=11)

ax.set_xlabel('Degree', fontsize=13, labelpad=5)
ax.set_ylabel('Clustering coefficient', fontsize=13,
              labelpad=24, rotation=270)

# ==========================================
# 6. 图例
# ==========================================
legend_handles = [
    Line2D([0], [0],
           marker='o',
           linestyle='none',
           markersize=6,
           markerfacecolor='#808080',
           markeredgecolor='none',
           alpha=0.08,
           label='Original data'),
    Line2D([0], [0],
           marker='o',
           linestyle='none',
           markersize=7.5,
           markerfacecolor='#0172be',
           markeredgecolor='white',
           markeredgewidth=0.8,
           label='Mean data'),
    Line2D([0], [0],
           marker='D',
           linestyle='none',
           markersize=7,
           markerfacecolor='#ed7d2f',
           markeredgecolor='white',
           markeredgewidth=0.8,
           label='Median data')
]

legend = ax.legend(
    handles=legend_handles,
    frameon=False,
    loc='lower left',
    fontsize=11,
    handletextpad=0.4,
    handlelength=1.5,
    borderpad=0.1,
    labelspacing=0.3
)
for text in legend.get_texts():
    text.set_color(BASE_COLOR)

# ==========================================
# 7. 保存输出
# ==========================================
ax.set_position([0.189231, 0.220848, 0.571477, 0.707526])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ext in ['png', 'svg', 'tiff', 'jpg']:
    save_path = os.path.join(output_dir, f"{filename_base}.{ext}")
    fig.savefig(save_path, format=ext, dpi=600)
    print(f"Saved: {save_path}")

plt.show()

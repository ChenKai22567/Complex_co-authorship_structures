import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import Counter
import os

"""
================================================================================
脚本名称: degree_distribution_styled_fixed.py
功能描述:
    1. 读取 GEXF 网络文件并计算节点度。
    2. 使用对数分箱 (Log-Binning) 处理数据。
    3. 执行线性回归拟合幂律分布。
    4. 绘制符合学术风格的双对数散点图，并修正颜色格式与视觉深度。

主要修正点:
    - 修复无效颜色代码 '#h172be' -> '#8172be' (紫蓝色)。
    - 加深全局字体与边框颜色 (LIGHT_COLOR: #888888 -> #444444)。
    - 加深原始数据点颜色 (#C2C2C2 -> #808080)。
================================================================================
"""

# ————— 全局字体与颜色设置 —————
FONT_FAMILY = 'Arial'  # 设置全局字体
BASE_COLOR = '#444444'

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = BASE_COLOR
plt.rcParams['axes.labelcolor'] = BASE_COLOR
plt.rcParams['xtick.color'] = BASE_COLOR
plt.rcParams['ytick.color'] = BASE_COLOR
plt.rcParams['axes.edgecolor'] = BASE_COLOR
plt.rcParams['font.size'] = 11

# ————— 1. 用户配置参数 —————

# GEXF 文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
feature_dir = os.path.dirname(script_dir)
DATA_DIR = os.path.join(feature_dir, "data")
FILE_PATH = os.path.join(DATA_DIR, "net_2006-2025_newman_S2.gexf")

# 图片保存目录
OUTPUT_DIR = os.path.join(feature_dir, "figures")
IMG_FILENAME = 'degree_distribution_top_right_v2'  # 更新文件名以免覆盖

# [关键参数] 纳入统计的最小度阈值
MIN_DEGREE_THRESHOLD = 5

# ————— 2. 数据加载与处理 —————
print(f"正在读取文件: {FILE_PATH} ...")
try:
    G = nx.read_gexf(FILE_PATH)
    # 计算节点度 (Degree)
    degrees_dict = dict(G.degree())
    all_degrees = list(degrees_dict.values())

    # [数据过滤] 仅保留度数 >= 5 的节点
    records_filtered = [d for d in all_degrees if d >= MIN_DEGREE_THRESHOLD]

    if not records_filtered:
        print(f"错误: 过滤条件 (Degree >= {MIN_DEGREE_THRESHOLD}) 下无数据。")
        exit()

    print(f"原始节点数: {len(all_degrees)}")
    print(f"有效节点数 (>= {MIN_DEGREE_THRESHOLD}): {len(records_filtered)}")

    # 统计频率
    counter_records = Counter(records_filtered)

except Exception as e:
    print(f"读取或处理文件出错: {e}")
    exit()


# ————— 3. 对数分箱算法 (Log-Binning) —————
def log_binning_corrected(data, bin_ratio=1.2589):
    """
    对数分箱：解决长尾分布尾部稀疏问题
    """
    min_x = min(data)
    max_x = max(data)

    # 生成指数增长的箱子边界
    bins = [min_x]
    while bins[-1] * bin_ratio <= max_x:
        bins.append(bins[-1] * bin_ratio)
    if bins[-1] < max_x:
        bins.append(bins[-1] * bin_ratio)

    bins_array = np.array(bins)
    if bins_array[0] == 0: bins_array[0] = 1e-9  # 避免 log(0)

    # 计算几何中心
    bin_centers = 10 ** ((np.log10(bins_array[:-1]) + np.log10(bins_array[1:])) / 2)

    # 统计
    counts, _ = np.histogram(data, bins=bins_array)
    freqs, _ = np.histogram(data, bins=bins_array, weights=[counter_records[x] for x in data])

    # 计算平均频率
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_frequencies = freqs / counts

    # 过滤空箱子
    mask = mean_frequencies > 0
    return bin_centers[mask], mean_frequencies[mask]


# 执行分箱
bin_centers, mean_freqs = log_binning_corrected(records_filtered)

# ————— 4. 线性回归拟合 —————
X_log = np.log10(bin_centers).reshape(-1, 1)
y_log = np.log10(mean_freqs)

regr = LinearRegression()
regr.fit(X_log, y_log)

# 获取指标
alpha = -regr.coef_[0]
intercept = regr.intercept_
y_pred_log = regr.predict(X_log)
r2 = r2_score(y_log, y_pred_log)

print("-" * 30)
print(f"拟合结果 (Min Degree >= {MIN_DEGREE_THRESHOLD}):")
print(f"Alpha: {alpha:.4f}")
print(f"R^2:   {r2:.4f}")
print("-" * 30)

# 生成拟合线数据
fit_x = np.linspace(min(bin_centers), max(bin_centers), 100)
fit_y = 10 ** (intercept + regr.coef_[0] * np.log10(fit_x))

# ————— 5. 绘图 (样式已修正) —————

# 创建画布
fig, ax = plt.subplots(figsize=(4.55, 3.34), dpi=600)
ax.set_position([0.189231, 0.220848, 0.571477, 0.707526])

# (1) 绘制原始数据点 (背景)
# [修正] 加深颜色，从 #C2C2C2 改为 #808080，使其在白色背景下更明显
orig_x = list(counter_records.keys())
orig_y = list(counter_records.values())
ax.scatter(orig_x, orig_y,
           color='#808080',  # 加深的灰色
           alpha=0.4,
           s=12,
           label='Original',
           edgecolor='none')

# (2) 绘制分箱数据点 - 三角形
ax.scatter(bin_centers, mean_freqs,
           marker='^',  # 三角形
           color='#0172be',  # 蓝色
           s=48,
           label='Binned data',
           edgecolor='white',
           linewidth=0.5,
           zorder=5)

# (3) 绘制拟合线 - 橙色虚线
ax.plot(fit_x, fit_y,
        color='#ed7d2f',  # 橙色
        linestyle='--',
        linewidth=1.5,
        label='Fit',
        zorder=10)

# ————— 6. 坐标轴风格调整 —————

ax.set_xscale('log')
ax.set_yscale('log')

# 边框与刻度设置，与 average_degree_time.py 主图保持一致
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(BASE_COLOR)

ax.tick_params(axis='x', which='both', direction='in',
               top=False, bottom=True, labelsize=11)
ax.tick_params(axis='y', which='both', direction='in',
               left=False, right=True, labelleft=False, labelright=True, labelsize=11)
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

# 标签
ax.set_xlabel('Degree', fontsize=13, labelpad=5)
ax.set_ylabel('Frequency', fontsize=13, labelpad=24, rotation=270)
ax.text(0.74, 0.94,
        f"$\\alpha$={alpha:.2f}\n$R^2$={r2:.2f}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        linespacing=1.35,
        color=BASE_COLOR)

# ————— 7. 图例设置 —————
legend_handles = [
    Line2D([0], [0],
           marker='o',
           linestyle='none',
           markersize=6,
           markerfacecolor='#808080',
           markeredgecolor='none',
           alpha=0.4,
           label='Original'),
    Line2D([0], [0],
           marker='^',
           linestyle='none',
           markersize=7.5,
           markerfacecolor='#0172be',
           markeredgecolor='white',
           markeredgewidth=0.8,
           label='Binned data'),
    Line2D([0], [0],
           color='#ed7d2f',
           linestyle='--',
           linewidth=1.8,
           label='Fit')
]
leg = ax.legend(
    handles=legend_handles,
    frameon=False,
    loc='lower left',
    fontsize=11,
    handletextpad=0.4,
    handlelength=1.5
)
for text in leg.get_texts():
    text.set_color(BASE_COLOR)

# ————— 8. 保存与显示 —————
ax.set_position([0.189231, 0.220848, 0.571477, 0.707526])

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for ext in ['png', 'svg', 'tiff', 'jpg']:
    save_path = os.path.join(OUTPUT_DIR, f"{IMG_FILENAME}.{ext}")
    fig.savefig(save_path, format=ext, dpi=600)
    print(f"已保存: {save_path}")

plt.show()

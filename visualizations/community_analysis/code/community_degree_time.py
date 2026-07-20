# -*- coding: utf-8 -*-
"""
Plot: Community Degree Density Heatmap & Community Count Evolution
(社区度密度热力图 & 社区数量演化图)

功能描述:
    1. [主图] 绘制社区“平均度”的频率密度热力图 (Heatmap)。
       - X轴: 年份 (2006-2025)。
       - Y轴: 社区平均度 (Avg Degree, Log Scale)。
       - 颜色: 橙色系，代表该年份、该度数区间内的社区数量。
    2. [嵌入图] 绘制“每年社区总数量”随时间的演化折线图。
       - 计算公式: Count(CommunityID) group by Year。
       - 数据源: stats_evolution_community_details.csv

修改记录:
    - [2026-01-16] 嵌入图数据修改为“每年的社区数量”。
"""

import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import matplotlib.colors as mcolors  # 颜色映射库
import numpy as np  # 数值计算库
from matplotlib.ticker import MaxNLocator, FuncFormatter  # 刻度格式化工具
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 嵌入子图工具
import os  # 文件路径操作

# ==========================================
# 1. 全局风格设置 (Global Style)
# ==========================================
FONT_FAMILY = 'Arial'  # 全局字体
BASE_COLOR = '#444444'  # 基础文字颜色

# 更新 matplotlib 配置参数
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = BASE_COLOR
plt.rcParams['axes.labelcolor'] = BASE_COLOR
plt.rcParams['xtick.color'] = BASE_COLOR
plt.rcParams['ytick.color'] = BASE_COLOR
plt.rcParams['axes.edgecolor'] = BASE_COLOR
plt.rcParams['font.size'] = 11  # 基础字号

# ==========================================
# 2. 配置与数据加载 (Configuration)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
community_dir = os.path.dirname(script_dir)
input_dir = os.path.join(community_dir, "data")
output_dir = os.path.join(community_dir, "figures")
# 输出文件名
filename_base = "plot_comm_degree_density_count_orange"

# 定义文件路径 (使用详细指标表)
path_details = os.path.join(input_dir, "stats_evolution_community_details.csv")

# 尝试读取数据
try:
    if os.path.exists(path_details):
        # 读取 CSV 数据
        df_details = pd.read_csv(path_details)
        print(f"Loaded details data: {len(df_details)} records.")
    else:
        raise FileNotFoundError(f"{path_details} not found.")

except FileNotFoundError:
    print("Error: Files not found. Generating dummy data for demonstration.")
    # 生成模拟数据 (如果文件不存在，用于代码测试)
    years = np.arange(2006, 2026)
    data = []
    for y in years:
        # 模拟社区数量随年份增加
        n_comms = int(y - 2000) * 8
        # 模拟度分布 (对数正态分布)
        degrees = np.random.lognormal(mean=1.5, sigma=0.8, size=n_comms)
        degrees = np.maximum(degrees, 0.5)
        for d in degrees:
            data.append({"Year": y, "Avg_Degree": d})
    df_details = pd.DataFrame(data)

# ==========================================
# 3. 数据预处理 (Preprocessing)
# ==========================================
# 过滤年份范围 (2006-2024, 假设2025不完整或作为最终态单独展示)
df_details = df_details[(df_details['Year'] >= 2006) & (df_details['Year'] < 2025)]

# --- [主图数据准备] ---
x = df_details['Year']  # X轴: 年份
y = df_details['Avg_Degree']  # Y轴: 社区平均度

# --- [嵌入图数据准备: 修改为社区数量] ---
# 逻辑: 按年份分组，统计每年的行数 (每一行代表一个社区)
# reset_index(name='Count') 将统计结果列命名为 'Count'
df_trend = df_details.groupby('Year').size().reset_index(name='Count')

# ==========================================
# 4. 大图绘制: 社区度频率热力图 (Main Heatmap)
# ==========================================
# --- 分箱 (Binning) ---
# X轴分箱: 每年一个格子
x_bins = np.arange(2006, 2026, 1)

# Y轴对数分箱 (针对 Degree)
# 确定数据范围
min_val = max(0.1, y.min())
max_val = y.max()

# 生成对数刻度的箱子边界 (Logarithmic bins)
# 30个区间能较好地展示分布细节
raw_log_bins = np.logspace(np.log10(min_val), np.log10(max_val + 0.1), 30)
y_bins = np.unique(raw_log_bins)

# --- 计算二维直方图 (2D Histogram) ---
# H 是频次矩阵
H, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
H_plot = H.T  # 转置矩阵以适配 pcolormesh 的坐标系
# 掩盖 0 值 (Mask zeros)，使背景显示为白色而不是颜色条最低端的颜色
H_masked = np.ma.masked_where(H_plot == 0, H_plot)

# --- 自定义颜色映射 (Colormap) ---
# 定义橙色渐变: 白色(背景) -> 浅橙 -> 深红橙
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "CustomOranges", ["white", "#ffecb3", "#e65100"]
)

# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(4.55, 3.34), dpi=600)  # 创建画布
ax.set_position([0.189231, 0.220848, 0.542903, 0.707526])

# 绘制热力图 (pcolormesh)
# norm=LogNorm: 颜色映射使用对数标度，能更好地展示频次差异巨大的数据
mesh = ax.pcolormesh(x_edges, y_edges, H_masked,
                     norm=mcolors.LogNorm(vmin=1, vmax=H.max()),
                     cmap=custom_cmap, shading='flat')

# --- 大图样式设置 ---
ax.set_xlabel("Year", fontsize=13, labelpad=5)
ax.set_ylabel("Community average degree", fontsize=13, labelpad=8)

ax.set_yscale('log')  # Y轴设为对数坐标
ax.set_ylim(bottom=min_val, top=max_val * 1.5)  # 顶部留白给嵌入图


# 自定义 Y轴 对数刻度显示格式
def log_formatter(x, pos):
    """仅显示特定的主要刻度"""
    if x < 0.1: return ""
    if x in [0.1, 1, 10, 100, 1000]:
        return f"{x:g}"  # :g 自动去掉无用的 .0
    return ""


ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
ax.set_xticks(range(2008, 2025, 4))
ax.set_xlim(2006, 2024)  # X轴范围
ax.tick_params(axis='x', rotation=30)  # X轴标签旋转
# 设置刻度线朝内
ax.tick_params(axis='both', which='both', direction='in',
               top=False, right=False, bottom=True, left=True, labelsize=11)
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

# 添加颜色条 (Colorbar)
cax = fig.add_axes([0.761726, 0.220848, 0.026000, 0.707526])
cbar = fig.colorbar(mesh, cax=cax)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Frequency (count)', fontsize=11, rotation=270, labelpad=15)

# ==========================================
# 5. 嵌入图绘制: 社区数量演化 (Inset: Community Count)
# ==========================================
if not df_trend.empty:
    # 创建嵌入轴 [x, y, width, height] (相对于父坐标轴的比例)
    # 位置设置在左上/中上方的空白区域
    ax_inset = ax.inset_axes([0.19, 0.60, 0.36, 0.36])

    # 设置嵌入图背景
    ax_inset.patch.set_facecolor('white')
    ax_inset.patch.set_alpha(0.85)

    # 定义嵌入图颜色 (深橙色，与主图呼应)
    COLOR_INSET = "#e65100"

    # 绘制折线图
    ax_inset.plot(df_trend['Year'], df_trend['Count'],
                  color=COLOR_INSET,
                  ls="-", marker="o", markersize=2.5, linewidth=0.8,
                  markerfacecolor="white", markeredgecolor=COLOR_INSET,
                  label="Community count")

    # --- 嵌入图样式 ---
    # 修改 Y轴 标签为 "Num Comms"
    ax_inset.set_ylabel("Community count", fontsize=9, labelpad=1, color=COLOR_INSET)

    # 设置刻度样式
    ax_inset.tick_params(axis='both', which='both', direction='in',
                         top=False, right=False, bottom=True, left=True,
                         width=0.4, length=2, labelsize=8)
    ax_inset.tick_params(axis='y', labelcolor=COLOR_INSET)
    ax_inset.set_ylim(0, 160)
    ax_inset.set_yticks([0, 50, 100, 150])
    ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))  # X轴刻度稀疏显示
    ax_inset.grid(False)  # 移除网格

    # 设置边框颜色
    ax_inset.spines['left'].set_color(BASE_COLOR)
    ax_inset.spines['bottom'].set_color(BASE_COLOR)
    # ax_inset.spines['top'].set_visible(False) # 可选：隐藏顶部边框
    # ax_inset.spines['right'].set_visible(False) # 可选：隐藏右侧边框

else:
    print("Skipping inset plot due to empty trend data.")

# ==========================================
# 6. 保存与显示 (Output)
# ==========================================
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ax.set_position([0.189231, 0.220848, 0.542903, 0.707526])
cbar.ax.set_position([0.761726, 0.220848, 0.026000, 0.707526])

# 保存为多种格式
for ext in ['png', 'svg', 'tiff', 'jpg']:
    save_path = os.path.join(output_dir, f"{filename_base}.{ext}")
    try:
        fig.savefig(save_path, format=ext, dpi=600)  # 高分辨率保存
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error saving {ext}: {e}")

plt.show()  # 显示图像

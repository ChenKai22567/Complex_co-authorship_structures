# -*- coding: utf-8 -*-
"""
Plot: Community Size Density Heatmap & Average Size Evolution
功能描述:
    1. [主图] 绘制社区大小的频率密度热力图 (Heatmap)。
       - X轴: 年份 (2006-2025)。
       - Y轴: 社区大小 (Log Scale)。
       - 颜色: 代表该年份、该大小区间内的社区数量 (Frequency Count)。
    2. [嵌入图] 绘制“社区平均大小”随时间的演化折线图。
       - 计算公式: Avg_Comm_Size = 当年总节点数 / 当年社区总数。
       - 数据源: 使用 stats_evolution_summary_rich.csv 中的 Nodes 和 Communities 列。

修改记录:
    - [2026-01-16] 嵌入图由“平均度”改为“社区平均大小”。
"""

import pandas as pd  # 用于数据处理
import matplotlib.pyplot as plt  # 用于绘图
import matplotlib.colors as mcolors  # 用于自定义颜色映射
import numpy as np  # 用于数值计算
from matplotlib.ticker import MaxNLocator, FuncFormatter  # 用于自定义刻度格式
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 用于创建嵌入子图
import os  # 用于文件路径操作

# ==========================================
# 1. 全局风格设置 (Global Style)
# ==========================================
FONT_FAMILY = 'Arial'  # 设置全局字体为 Arial
BASE_COLOR = '#444444'  # 设置基础颜色为深灰色

# 更新 matplotlib 配置参数
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = BASE_COLOR
plt.rcParams['axes.labelcolor'] = BASE_COLOR
plt.rcParams['xtick.color'] = BASE_COLOR
plt.rcParams['ytick.color'] = BASE_COLOR
plt.rcParams['axes.edgecolor'] = BASE_COLOR
plt.rcParams['font.size'] = 11  # 基础字号设置为 11

# ==========================================
# 2. 配置与数据加载 (Configuration)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
community_dir = os.path.dirname(script_dir)
input_dir = os.path.join(community_dir, "data")
output_dir = os.path.join(community_dir, "figures")
filename_base = "plot_comm_size_density_avg_size"  # 输出文件名前缀

# 定义文件路径
path_comm_sizes = os.path.join(input_dir, "stats_evolution_comm_sizes.csv")  # 社区大小明细数据
path_summary = os.path.join(input_dir, "stats_evolution_summary_rich.csv")  # 年度汇总数据 (用于计算平均大小)

# 尝试读取数据
try:
    # 读取社区大小明细数据 (Year, Size)
    df_sizes = pd.read_csv(path_comm_sizes)
    print(f"Loaded sizes data: {len(df_sizes)} records.")

    # 读取年度汇总数据 (Year, Nodes, Communities, ...)
    if os.path.exists(path_summary):
        df_summary = pd.read_csv(path_summary)
        print(f"Loaded summary data: {len(df_summary)} records.")
    else:
        # 如果找不到 rich 版，尝试普通版
        alt_path = os.path.join(input_dir, "stats_evolution_summary.csv")
        if os.path.exists(alt_path):
            df_summary = pd.read_csv(alt_path)
            print(f"Loaded summary data (basic): {len(df_summary)} records.")
        else:
            df_summary = pd.DataFrame()
            print("Warning: Summary file not found. Inset plot will be empty.")

except FileNotFoundError:
    print("Error: Files not found. Generating dummy data.")
    # 生成模拟数据用于演示
    years = np.arange(2006, 2026)
    size_data = []
    summary_data = []
    for y in years:
        n_comms = int(y - 2000) * 5
        # 模拟社区大小分布
        sizes = np.random.lognormal(mean=2.0, sigma=1.0, size=n_comms).astype(int)
        sizes = np.maximum(sizes, 2)
        for s in sizes:
            size_data.append({"Year": y, "Size": s})

        # 模拟节点和社区数
        nodes = np.sum(sizes)
        summary_data.append({"Year": y, "Nodes": nodes, "Communities": n_comms})

    df_sizes = pd.DataFrame(size_data)
    df_summary = pd.DataFrame(summary_data)

# ==========================================
# 3. 数据预处理 (Preprocessing)
# ==========================================
# 过滤年份范围 (2006-2024)
df_sizes = df_sizes[(df_sizes['Year'] >= 2006) & (df_sizes['Year'] < 2025)]

if not df_summary.empty:
    df_summary = df_summary[(df_summary['Year'] >= 2006) & (df_summary['Year'] < 2025)]
    # [核心计算] 计算平均社区大小 = 总节点数 / 总社区数
    # 注意处理除零异常
    df_summary['Avg_Comm_Size'] = df_summary.apply(
        lambda row: row['Nodes'] / row['Communities'] if row['Communities'] > 0 else 0, axis=1
    )

# 提取大图绘图数据
x = df_sizes['Year']
y = df_sizes['Size']

# ==========================================
# 4. 大图绘制: 社区大小频次热力图 (Main Heatmap)
# ==========================================
# --- 分箱 (Binning) ---
x_bins = np.arange(2006, 2026, 1)  # X轴分箱：每年一个

# Y轴对数分箱 (Log Scale Binning for Size)
min_size = max(1, y.min())
max_size = y.max()
# 生成对数刻度的箱子边界，约35个区间
raw_log_bins = np.logspace(np.log10(min_size), np.log10(max_size + 1), 35)
y_bins = np.unique(np.floor(raw_log_bins))
if len(y_bins) < 2: y_bins = np.array([1, 10, 100])  # 防止数据过少报错

# --- 计算频次矩阵 (Frequency Count) ---
H, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
H_plot = H.T  # 转置以适应 pcolormesh
H_masked = np.ma.masked_where(H_plot == 0, H_plot)  # 掩盖 0 值以便显示白色背景

# 自定义渐变色板 (白色 -> 浅蓝 -> 深蓝)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("CustomBlues", ["white", "#d0e1f0", "#0172be"])

# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(4.55, 3.34), dpi=600)  # 设置画布大小
ax.set_position([0.189231, 0.220848, 0.542903, 0.707526])

# 绘制热力图
# vmin=1: 频次为1时开始着色; vmax=max: 最大频次
mesh = ax.pcolormesh(x_edges, y_edges, H_masked,
                     norm=mcolors.LogNorm(vmin=1, vmax=H.max()),
                     cmap=custom_cmap, shading='flat')

# --- 大图样式设置 ---
ax.set_xlabel("Year", fontsize=13, labelpad=5)  # X轴标签
ax.set_ylabel("Community size", fontsize=13, labelpad=8)  # Y轴标签

ax.set_yscale('log')  # Y轴设为对数坐标
ax.set_ylim(bottom=min_size, top=max_size * 2)  # 设置Y轴范围，顶部留白给嵌入图


# 自定义Y轴刻度显示格式
def log_formatter(x, pos):
    if x < 1: return ""
    # 只显示主要整数刻度
    return f"$10^{{{int(np.log10(x))}}}$"


ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
ax.set_xticks(range(2008, 2025, 4))
ax.set_xlim(2006, 2024)  # X轴范围
ax.tick_params(axis='x', rotation=30)  # X轴标签旋转30度
# 刻度线朝内，只显示左侧和底部
ax.tick_params(axis='x', rotation=30)
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
# 5. 嵌入图绘制: 平均社区大小 (Inset: Avg Size)
# ==========================================
if not df_summary.empty:
    # 设置嵌入图位置 [x, y, w, h] (相对于主坐标轴)
    # 放在右上角空白处
    ax_inset = ax.inset_axes([0.19, 0.60, 0.36, 0.36])
    ax_inset.patch.set_facecolor('white')  # 背景白色
    ax_inset.patch.set_alpha(0.85)  # 半透明

    # 定义颜色 (与热力图主色调一致)
    COLOR_INSET = "#0172be"

    # 绘制折线图
    ax_inset.plot(df_summary['Year'], df_summary['Avg_Comm_Size'],
                  color=COLOR_INSET,
                  ls="-", marker="o", markersize=2.5, linewidth=0.8,
                  markerfacecolor="white", markeredgecolor=COLOR_INSET,
                  label="Average size")

    # --- 嵌入图样式设置 ---
    ax_inset.set_ylabel("Average size", fontsize=9, labelpad=1, color=COLOR_INSET)  # Y轴标签
    # 设置刻度样式
    ax_inset.tick_params(axis='both', which='both', direction='in',
                         top=False, right=False, bottom=True, left=True,
                         width=0.4, length=2, labelsize=8)
    ax_inset.tick_params(axis='y', labelcolor=COLOR_INSET)
    ax_inset.set_ylim(0, 180)
    ax_inset.set_yticks([0, 50, 100, 150])
    ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax_inset.grid(False)  # 无网格



    # 隐藏顶部和右侧边框，保持极简风格
    # ax_inset.spines['top'].set_visible(False)
    # ax_inset.spines['right'].set_visible(False)
    # 左侧和底部边框颜色
    ax_inset.spines['left'].set_color(BASE_COLOR)
    ax_inset.spines['bottom'].set_color(BASE_COLOR)

else:
    print("Skipping inset plot due to missing summary data.")

# ==========================================
# 6. 保存与显示 (Save & Show)
# ==========================================

# 确保输出目录存在
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

plt.show()  # 显示图片

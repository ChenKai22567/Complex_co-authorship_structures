# -*- coding: utf-8 -*-
"""
step2_plot_rg_flow.py (V6 - Final Publication Style)

【程序功能】
1. 读取 .npz 数据。
2. 使用 Log-Binning 优化内嵌图。
3. 绘制最终图表，满足以下定制要求：
   - 拟合优度 R2 合并入 Lambda 图例。
   - 状态标签置于图例上方。
   - 输出 tif, png, svg 多种格式。
   - 包含详细代码注释。

"""

import os  # 用于文件路径操作
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
# 输入数据路径 (请根据实际情况修改)
script_dir = os.path.dirname(os.path.abspath(__file__))
fractal_dir = os.path.dirname(script_dir)
input_dir = os.path.join(fractal_dir, "data")
output_dir = os.path.join(fractal_dir, "figures")
DATA_FILENAME = "rg_flow_data.npz"
DATA_PATH = os.path.join(input_dir, DATA_FILENAME)

# 字体设置
FONT_FAMILY = 'Arial'  # 论文常用字体
BASE_COLOR = '#444444'

# --- 颜色配置 ---
# 主图颜色
COLOR_DATA_EDGE = '#1f77b4'  # 数据点边缘颜色 (蓝色)
COLOR_DATA_FACE = 'none'  # 数据点填充颜色 (空心)
COLOR_FIT = '#ff7f0e'  # 拟合线颜色 (橙色)

# 内嵌图颜色
COLOR_G0_RAW = '#808080'  # 原始网络 G0 点颜色 (灰色)
COLOR_G3_BIN = '#ed7d2f'  # 重整化网络 G3 点颜色 (橙色)
BIN_RATIO = 1.2589  # 分箱增长比率

# 文本颜色
TEXT_COLOR = BASE_COLOR  # 深灰色文本

# 图片尺寸与分辨率
FIG_SIZE = (4.55, 3.34)  # 图片尺寸 (英寸)
DPI = 600  # 输出分辨率
AX_POSITION = [0.189231, 0.220848, 0.571477, 0.707526]
INSET_POS = [0.60, 0.60, 0.35, 0.35]

# 全局绘图参数设置
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['axes.edgecolor'] = TEXT_COLOR
plt.rcParams['font.size'] = 11


# ==========================================
# 2. 数据处理工具 (Data Utilities)
# ==========================================

def get_raw_dist(degree_seq):
    """
    计算原始度分布 (不分箱)。
    用于绘制内嵌图的背景 G0。
    """
    degree_seq = degree_seq[degree_seq > 0]  # 过滤度为0的节点
    if len(degree_seq) == 0: return [], []  # 空数据保护

    from collections import Counter
    counts = Counter(degree_seq)  # 统计每个度的频次
    total = len(degree_seq)  # 总节点数

    x = sorted(counts.keys())  # x轴: 度 k
    y = [counts[k] / total for k in x]  # y轴: 概率 P(k)

    return np.array(x), np.array(y)


def log_binning_probability(degree_seq, bin_ratio=1.2589):
    """
    对数分箱计算概率密度 P(k)。
    用于绘制内嵌图的前景 G3 (平滑曲线)。
    """
    data = np.array(degree_seq)
    data = data[data > 0]  # 过滤0值
    if len(data) == 0: return [], []

    min_x = np.min(data)
    max_x = np.max(data)

    # 生成对数增长的箱子边界
    bins = [min_x]
    while bins[-1] * bin_ratio <= max_x:
        bins.append(bins[-1] * bin_ratio)
    if bins[-1] < max_x:
        bins.append(bins[-1] * bin_ratio)

    bins_array = np.array(bins)

    if len(bins_array) < 2: return [min_x], [1.0]  # 数据点过少处理

    # 计算箱子几何中心 (Geometric Mean)
    bin_centers = 10 ** ((np.log10(bins_array[:-1]) + np.log10(bins_array[1:])) / 2)

    # 统计落入每个箱子的数量
    counts, _ = np.histogram(data, bins=bins_array)
    # 计算每个箱子的宽度
    bin_widths = bins_array[1:] - bins_array[:-1]

    # 计算概率密度: Count / (Total * Width)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdf = counts / (len(data) * bin_widths)

    mask = counts > 0  # 过滤空箱子
    return bin_centers[mask], pdf[mask]


# ==========================================
# 3. 绘图逻辑 (Plotting Logic)
# ==========================================
def plot_rg_flow_optimized():
    print(f">>> Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("Error: File not found! Run step1 first.")
        return

    # 加载 .npz 数据
    with np.load(DATA_PATH, allow_pickle=True) as data:
        # 读取 RG Flow 主数据
        mode = str(data['mode'])
        x = data['rg_x']
        y = data['rg_y']

        # 读取拟合参数 (处理标量读取问题)
        lambda_val = data['rg_lambda'].item() if data['rg_lambda'].ndim == 0 else data['rg_lambda'][0]
        r2_val = data['rg_R2'].item() if data['rg_R2'].ndim == 0 else data['rg_R2'][0]
        fit_line = data['rg_fit_line']

        # 读取原始度序列 (用于内嵌图)
        k0_raw = data['k0_raw']
        k3_raw = data['k3_raw']

    print(f"    Mode: {mode}, Lambda: {lambda_val:.2f}, R2: {r2_val:.2f}")

    # 创建画布
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_position(AX_POSITION)

    # --- 1. 绘制主图 RG Flow ---

    # 根据相位模式确定坐标轴含义和标签
    if mode == 'stable':
        label_txt = r'Data ($z_b$)'
        ylabel_txt = r"Renormalized degree $z_b$"
        status_txt = "Stable phase (fractal)"
    else:
        label_txt = r'Data ($z_b - z_0$)'
        ylabel_txt = r"$z_b - z_0$"
        status_txt = "Unstable phase (small-world)"

    # 绘制数据点: 空心蓝色圆圈
    ax.loglog(x, y, 'o',
              markerfacecolor=COLOR_DATA_FACE,  # 内部透明
              markeredgecolor=COLOR_DATA_EDGE,  # 边缘蓝色
              markeredgewidth=1.5,  # 边缘线宽
              ms=6,  # 点大小
              label=label_txt,  # 图例标签
              zorder=10)  # 层级置顶

    # 绘制拟合线: 橙色虚线
    # 【修改1】将 R2 合并到拟合线的图例中
    if not np.isnan(lambda_val):
        fit_label = rf"$\lambda={lambda_val:.2f}$ ($R^2={r2_val:.2f}$)"
        ax.loglog(x, fit_line, '--',
                  color=COLOR_FIT,
                  lw=1.5,
                  alpha=0.8,
                  label=fit_label)

    # 设置坐标轴标签
    ax.set_xlabel(r"$x_b \equiv N_0 / N_b$", fontsize=13, labelpad=5)
    ax.set_ylabel(ylabel_txt, fontsize=13, labelpad=8)

    # 设置刻度样式 (朝内)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, bottom=True, left=True,
                   labelsize=11)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], ylims[1] * 1.2)  # 上限扩大

    # 显示状态标签
    # 位置设置在左下角 (0.02, 0.22)，刚好位于图例上方
    ax.text(0.04, 0.20, status_txt, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            color=TEXT_COLOR)

    # 显示图例 (左下角)
    ax.legend(loc='lower left', frameon=False, fontsize=8)

    # --- 2. 绘制内嵌图 (Inset: self-similarity) ---

    ax_ins = ax.inset_axes(INSET_POS)
    ax_ins.patch.set_facecolor('white')
    ax_ins.patch.set_alpha(0.85)

    # (A) 绘制原始网络 G0 (背景)
    raw_x, raw_y = get_raw_dist(k0_raw)
    ax_ins.loglog(raw_x, raw_y, '.', color=COLOR_G0_RAW, alpha=0.2,
                  markersize=3, label='$G_0$', zorder=1)

    # (B) 绘制重整化网络 G3 (前景)
    bin_x, bin_y = log_binning_probability(k3_raw, bin_ratio=BIN_RATIO)
    ax_ins.loglog(bin_x, bin_y, '^',
                  markerfacecolor=COLOR_G3_BIN,  # 橙色填充
                  markeredgecolor='w',  # 白色描边
                  ms=4.5,  # 大小
                  markeredgewidth=0.8,
                  label='$G_{l_B=3}$', zorder=2)

    # 内嵌图标签
    ax_ins.set_xlabel(r"$k$", fontsize=9, labelpad=1)
    ax_ins.set_ylabel(r"$P(k)$", fontsize=9, labelpad=1)

    # 内嵌图刻度调整
    ax_ins.tick_params(axis='both', which='both', direction='in',
                       top=False, right=False, bottom=True, left=True,
                       labelsize=8, width=0.4, length=2)
    ax_ins.tick_params(which='minor', length=0)  # 隐藏次刻度
    ax_ins.grid(False)
    ax_ins.legend(fontsize=6, frameon=False, loc='upper right',
                  handlelength=1.0, labelspacing=0.2, borderpad=0.1)


    # --- 3. 保存与展示 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ax.set_position(AX_POSITION)

    # 【修改3】保存为多种格式 (tif, png, svg)
    formats = ['png', 'svg', 'tiff', 'jpg']
    base_name = os.path.join(output_dir, "figure_e_rg_flow_final_styled")

    for fmt in formats:
        save_path = f"{base_name}.{fmt}"
        plt.savefig(save_path, dpi=DPI)
        print(f">>> Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_rg_flow_optimized()

# -*- coding: utf-8 -*-
"""
Plot: Degree Correlation (Ratio Profile) - Styled Version
功能描述:
    绘制度-度相关性热力图 (R matrix)，即 Song et al. (Nature 2005) 图 2c 的复刻与美化版。
    - X轴: 节点度数 k1 (Log Scale)
    - Y轴: 节点度数 k2 (Log Scale)
    - 颜色: 连接比率 R(k1, k2) = P(k1,k2) / P_rand(k1,k2)
      - R > 1 (深蓝): 连接比随机情况更频繁 (Rich-club / Assortative)
      - R < 1 (浅色): 连接被抑制 (Hub Repulsion / Disassortative)

    [本次更新]:
    1. 无数据/低值区域背景改为"极浅蓝色" (#f2f8fd)，消除突兀的纯白。
    2. 对角线辅助线改为白色虚线，增强对比度。
    3. 增加了详细注释，方便微调。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatterMathtext, LogLocator

# ==========================================
# 1. 全局风格设置 (Global Style)
# ==========================================
# 设置全局字体为 Arial (符合学术出版标准)
FONT_FAMILY = 'Arial'
# 基础颜色 (深灰色)，比纯黑更柔和
BASE_COLOR = '#444444'

# 定义背景色 (极浅的蓝色，替代纯白)
# 作用：让无数据的区域显示为这种颜色，而不是刺眼的纯白，使图表更整体。
# 您可以微调这个 HEX 值: #f0f6fa (偏冷), #eef7ff (偏亮), #f2f8fd (推荐)
PALE_BLUE_BG = "#f4f9fd"

# 更新 matplotlib 全局参数
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = BASE_COLOR       # 文本颜色
plt.rcParams['axes.labelcolor'] = BASE_COLOR  # 坐标轴标签颜色
plt.rcParams['xtick.color'] = BASE_COLOR      # X轴刻度颜色
plt.rcParams['ytick.color'] = BASE_COLOR      # Y轴刻度颜色
plt.rcParams['axes.edgecolor'] = BASE_COLOR   # 边框颜色
plt.rcParams['font.size'] = 11                 # 基础字号

# ==========================================
# 2. 配置 (Configuration)
# ==========================================
# 输入数据目录 (应包含 step1 生成的 .npz 文件)
script_dir = os.path.dirname(os.path.abspath(__file__))
fractal_dir = os.path.dirname(script_dir)
DATA_DIR = os.path.join(fractal_dir, "data")
# 数据文件名
DATA_FILENAME = "figure_c_heatmap_data.npz"

# 输出设置
OUTPUT_DIR = os.path.join(fractal_dir, "figures")
OUTPUT_FILENAME_BASE = "figure_c_ratio_profile_final_style" # 输出文件名前缀

# ==========================================
# 3. 绘图函数 (Plot Function)
# ==========================================
def plot_ratio_heatmap(R, xedges, yedges):
    """
    绘制热力图的核心函数。
    参数:
        R: 2D numpy array, 比率矩阵 R(k1, k2)
        xedges: 1D array, X轴的分箱边界
        yedges: 1D array, Y轴的分箱边界
    """
    print(">>> Plotting Styled Heatmap...")

    # 设置画布大小 (宽4.2英寸, 高3.5英寸)，分辨率600dpi
    fig, ax = plt.subplots(figsize=(4.55, 3.34), dpi=600)
    ax.set_position([0.189231, 0.220848, 0.515885, 0.707526])

    # --- 数据预处理 ---
    # 使用 NumPy 掩码数组 (Masked Array) 隐藏无效数据
    # R <= 0 的区域 (即无数据或分母为0) 将被掩盖，显示为背景色
    R_masked = np.ma.masked_where(R <= 0, R)

    # --- 配色方案优化 (Colormap) ---
    # 1. 创建自定义色板：起点为极浅蓝(PALE_BLUE_BG)，过渡到浅蓝(#d0e1f0)，再到深蓝(#0172be)
    # 这种渐变模拟了从无相关性到强相关性的过渡
    colors = [PALE_BLUE_BG, "#d0e1f0", "#0172be"]
    # 创建线性分段色图
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("CustomPaleBlues", colors)

    # 2. 设置"坏值"(masked values)的颜色
    # 这确保了被掩盖的区域显示为我们设定的背景色，而不是默认的白色或灰色
    custom_cmap.set_bad(color=PALE_BLUE_BG)

    # --- 绘制热力图 (pcolormesh) ---
    # xedges, yedges: 网格边界
    # cmap: 使用自定义色板
    # norm=LogNorm: 对数颜色映射。因为 R 值跨度可能很大 (从 0.1 到 100)，对数映射能更好地展示细节。
    # vmin=0.1: 设置显示的下限。R < 0.1 的值都显示为最浅色。
    # vmax=R_masked.max(): 设置上限为数据的最大值。
    # shading='flat': 每个网格填充一种颜色。
    # edgecolors='face': 重要！让每个网格的边框颜色与其填充色一致。这消除了矢量图渲染时可能出现的网格间微小白线缝隙。
    mesh = ax.pcolormesh(xedges, yedges, R_masked,
                         cmap=custom_cmap,
                         norm=mcolors.LogNorm(vmin=0.1, vmax=R_masked.max()),
                         shading='flat',
                         edgecolors='face')

    # 设置 Axes 背景色，确保边缘没有任何白色缝隙漏出
    ax.set_facecolor(PALE_BLUE_BG)

    # --- 坐标轴格式化 (Axis Formatting) ---
    # 设置为双对数坐标
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 强制显示整数 (ScalarFormatter)
    # 默认 Log 轴会显示 10^1, 10^2。这里我们强制显示为 1, 10, 100，更直观。
    formatter = LogFormatterMathtext(base=10)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # 强制刻度位置
    # LogLocator(base=10.0) 确保主刻度只出现在 1, 10, 100 等位置
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

    # 刻度样式微调
    # direction='in': 刻度线朝内 (符合 Nature 风格)
    # top=False, right=False: 去掉上方和右侧的刻度线，保持简洁
    # bottom=True, left=True: 保留下方和左侧刻度
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, bottom=True, left=True,
                   labelsize=11)

    # 轴标签
    ax.set_xlabel(r"Degree ($k_1$)", fontsize=13, labelpad=5)
    ax.set_ylabel(r"Degree ($k_2$)", fontsize=13, labelpad=8)

    # --- 颜色条 (Colorbar) ---
    # fraction: 颜色条占用空间的比例
    # pad: 颜色条与主图的间距
    cax = fig.add_axes([0.734708, 0.220848, 0.026000, 0.707526])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(labelsize=10) # 颜色条刻度字体
    # 颜色条标签 (旋转270度垂直显示)
    cbar.set_label(r"Ratio $R(k_1, k_2)$", fontsize=11, rotation=270, labelpad=15)

    # --- 辅助线 (对角线) ---
    # 绘制 y=x 对角线。对于同配网络，热点通常集中在对角线附近。
    min_lim = min(xedges[0], yedges[0])
    max_lim = max(xedges[-1], yedges[-1])

    # color='white': 在深蓝色背景上，白色虚线对比度最高。
    # alpha=0.9: 设置透明度，使其既清晰又不至于太突兀。
    # lw=1.2: 线宽。
    ax.plot([min_lim, max_lim], [min_lim, max_lim],
            color='white', ls='--', lw=1.5, alpha=0.9)

    # 锁定坐标范围，防止辅助线撑大视图
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    # ==========================================
    # 4. 保存逻辑 (Save)
    # ==========================================
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ax.set_position([0.189231, 0.220848, 0.515885, 0.707526])
    cbar.ax.set_position([0.734708, 0.220848, 0.026000, 0.707526])

    print(f">>> Saving to {OUTPUT_DIR}...")
    # 循环保存为多种格式，满足不同需求
    # png: 预览/PPT用; svg/tiff: 论文排版用
    for ext in ['png', 'svg', 'tiff', 'jpg']:
        save_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_BASE}.{ext}")
        try:
            # facecolor=... 确保保存图片的边缘背景也是浅蓝色(如果需要纯白边框可去掉此参数)
            # dpi=600: 保证位图足够清晰
            fig.savefig(save_path, format=ext, dpi=600)
            print(f"    Saved: {save_path}")
        except Exception as e:
            print(f"    Error saving {ext}: {e}")

    plt.show() # 显示窗口


# =========================
# 5. 主程序入口
# =========================
def main():
    data_path = os.path.join(DATA_DIR, DATA_FILENAME)
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        print("Tip: Please run 'step1_calc_figure_c.py' first to generate data.")
        return

    # 加载 .npz 数据
    print(f">>> Loading data: {data_path}")
    data = np.load(data_path)

    # 调用绘图函数
    plot_ratio_heatmap(data['R'], data['xedges'], data['yedges'])


if __name__ == "__main__":
    main()

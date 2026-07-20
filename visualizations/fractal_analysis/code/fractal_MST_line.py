# -*- coding: utf-8 -*-
"""
【程序功能】
科研合作网络分形特征可视化：图 b (骨架分支率) 及其嵌入图 c (短路长度分布)。

【核心指标与计算方法回顾】
1. 骨架分支率 <b>:
   - 识别骨架及最大枢纽节点为根节点。
   - 计算各节点到根节点距离 d 下的平均分支数。
   - 围绕单位值 y=1 的平台区域表明符合临界分支树特征（分形结构）。
2. 短路长度分布 Ps(ds):
   - 定义剩余链接连接的两点沿骨架的距离为短路长度 ds。
   - 分形特征表现为以 ds=2 为峰值并随 ds 增大单调递减。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

FONT_FAMILY = 'Arial'
BASE_COLOR = '#444444'
DPI = 600
FIG_SIZE = (4.55, 3.34)
AX_POSITION = [0.189231, 0.220848, 0.571477, 0.707526]
INSET_POSITION = [0.60, 0.60, 0.35, 0.35]

script_dir = os.path.dirname(os.path.abspath(__file__))
fractal_dir = os.path.dirname(script_dir)
input_dir = os.path.join(fractal_dir, "data")
output_dir = os.path.join(fractal_dir, "figures")

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = BASE_COLOR
plt.rcParams['axes.labelcolor'] = BASE_COLOR
plt.rcParams['xtick.color'] = BASE_COLOR
plt.rcParams['ytick.color'] = BASE_COLOR
plt.rcParams['axes.edgecolor'] = BASE_COLOR
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


def plot_branching_with_shortcut_inset(csv_b, csv_c, base_name="fractal_analysis"):
    """
    绘制带嵌入图的分形特征图，支持多种保存格式并合并图例
    :param csv_b: 分支率数据 CSV
    :param csv_c: 短路分布数据 CSV
    :param base_name: 输出文件的基础名称
    """
    # --- 1. 数据安全性检查 ---
    if not os.path.exists(csv_b) or not os.path.exists(csv_c):
        print("Error: 未找到指定的 CSV 数据文件。")
        return

    df_b = pd.read_csv(csv_b)
    df_c = pd.read_csv(csv_c)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_position(AX_POSITION)

    # 颜色配置
    blue_color = '#1f77b4'  # 主图色调
    orange_color = '#ff7f0e'  # 嵌入图色调

    # --- 3. 绘制主图 (图 b: 骨架分支率) ---
    line_b, = ax.plot(df_b['d'], df_b['mean_b'], 'o-', color=blue_color,
                      linewidth=1.5, markersize=6, markerfacecolor='white',
                      markeredgewidth=1.2, label='Mean branching number ($b$)')

    ax.set_yscale('log')  # y 轴设置为对数坐标以展现分形特征

    # 绘制 y=1 临界参考线
    ax.axhline(y=1, color='#888888', linestyle='--', linewidth=1.0, alpha=0.7)

    # 主图坐标轴标签与刻度设置
    ax.set_xlabel('Distance from root ($d$)', fontsize=13, labelpad=5)
    ax.set_ylabel('Mean branching number ($b$)', fontsize=13, labelpad=8)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, bottom=True, left=True,
                   labelsize=11)

    # --- 4. 绘制嵌入图 (图 c: 短路跨度分布) ---
    ax_inset = ax.inset_axes(INSET_POSITION)
    ax_inset.patch.set_facecolor('white')
    ax_inset.patch.set_alpha(0.85)

    # 绘制嵌入图曲线
    line_c, = ax_inset.plot(df_c['ds'], df_c['Ps'], 'o-', color=orange_color,
                            linewidth=1, markersize=2, markerfacecolor='white',
                            label='Shortcut \ndistribution\n$P_s(d_s)$')

    # 嵌入图刻度与标签微调
    ax_inset.set_xlabel('$d_s$', fontsize=9, labelpad=1)
    ax_inset.set_ylabel('$P_s(d_s)$', fontsize=9, labelpad=1)

    # 动态设置刻度步长
    x_ticks = np.arange(4, int(max(df_c['ds'])) + 1, 6)
    ax_inset.set_xticks(x_ticks)
    ax_inset.tick_params(axis='both', which='both', direction='in',
                         top=False, right=False, bottom=True, left=True,
                         labelsize=8, width=0.4, length=2)
    ax_inset.grid(False)
    ax_inset.legend(fontsize=5.5, frameon=False, loc='upper right',
                    handlelength=1.0, labelspacing=0.2, borderpad=0.1)

    # --- 5. 主图图例 ---
    ax.legend(handles=[line_b], loc='lower left', fontsize=8, frameon=False)

    # --- 6. 多格式输出保存 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ax.set_position(AX_POSITION)

    # 定义保存路径及格式
    formats = ['png', 'svg', 'tiff', 'jpg']
    print("\n>>> 正在保存多种格式的图像文件...")

    if not os.path.isabs(base_name):
        base_name = os.path.join(output_dir, base_name)

    for fmt in formats:
        out_file = f"{base_name}.{fmt}"
        plt.savefig(out_file, dpi=DPI)
        print(f"    - 已保存: {os.path.abspath(out_file)}")

    plt.show()


if __name__ == "__main__":
    # 请确保统计代码生成的 CSV 结果文件在当前目录下
    CSV_BRANCHING = os.path.join(input_dir, "results_fig_b.csv")
    CSV_SHORTCUT = os.path.join(input_dir, "results_fig_c.csv")

    # 执行分析与保存
    plot_branching_with_shortcut_inset(CSV_BRANCHING, CSV_SHORTCUT)

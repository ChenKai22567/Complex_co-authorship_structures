# -*- coding: utf-8 -*-
"""
Plot-only script: Role Cartography (Z-P Plot) - Custom V6 (Multi-Export)
定制版 V6 更新内容：
1. 支持多种格式导出 (TIF, PNG, SVG)。
2. 输出分辨率提升至 600 DPI。
3. 保持 V5 的所有视觉样式（描边、背景透明度、颜色等）。

Inputs:
- CSV file containing 'P_weighted' and 'z_internal_strength'.
"""

# =======================
# 0) 参数配置区域 (PARAMETERS)
# =======================
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPLEMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SUPPLEMENT_DIR, "data")
FIGURE_DIR = os.path.join(SUPPLEMENT_DIR, "figures")
INPUT_CSV = os.path.join(DATA_DIR, "RWCP_3layer_node_roles_with_metrics.csv")
OUT_DIR = FIGURE_DIR

COL_P = "P_weighted"
COL_Z = "z_internal_strength"
COL_LAYER = "cp_layer"

# --- 导出设置 ---
EXPORT_DPI = 600  # [修改] 设置为 600 DPI
EXPORT_FORMATS = ['png', 'pdf', 'svg', 'tiff', 'jpg']  # [修改] 需要导出的格式列表

# --- 视觉平衡尺寸设置 ---
NODE_SIZES = {
    'periphery': 35,  # 圆形
    'semi_core': 27,  # 三角形
    'core': 25  # 方形
}

POINT_ALPHA = 0.5
SHOW_PLOT_WINDOW = False

# --- 颜色与样式配置 ---
FONT_FAMILY = 'Arial'
TEXT_COLOR = "#444444"
FIG_SIZE = (4.25, 3.12)
AX_POSITION = [0.17, 0.25, 0.508, 0.692]
AXIS_LINE_WIDTH = 0.8
MAIN_TICK_WIDTH = 0.8
MAIN_LABEL_SIZE = 12
MAIN_TICK_LABEL_SIZE = 10
MAIN_LEGEND_SIZE = 6.5
MAIN_X_LABEL_PAD = 7
MAIN_Y_LABEL_PAD = 10
MAIN_TICK_PAD = 3
MAIN_X_TICK_PAD = 5
MAIN_Y_TICK_PAD = 6
COLOR_AXIS_TEXT = TEXT_COLOR
COLOR_REGION_LABEL = TEXT_COLOR
COLOR_LEGEND_TEXT = TEXT_COLOR
LEGEND_MARKER_COLOR = '#d67eb7'

REGION_COLORS = {
    'R1': '#707070', 'R2': '#DE5A5A', 'R3': '#79B96C', 'R4': '#4B54A4',
    'R5': '#FDDB50', 'R6': '#d67eb7', 'R7': '#BBBBBB'
}

# =======================
# 1) 导入与全局设置
# =======================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# 设置全局字体为 Arial
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams["axes.edgecolor"] = TEXT_COLOR
plt.rcParams["axes.labelsize"] = MAIN_LABEL_SIZE
plt.rcParams["xtick.labelsize"] = MAIN_TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = MAIN_TICK_LABEL_SIZE
plt.rcParams["legend.fontsize"] = MAIN_LEGEND_SIZE
plt.rcParams["axes.linewidth"] = AXIS_LINE_WIDTH
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["figure.dpi"] = EXPORT_DPI


# =======================
# 2) 核心逻辑函数
# =======================

def get_region(p, z):
    if z >= 2.5:
        if p <= 0.30:
            return "R5"
        elif p <= 0.75:
            return "R6"
        else:
            return "R7"
    else:
        if p <= 0.05:
            return "R1"
        elif p <= 0.62:
            return "R2"
        elif p <= 0.80:
            return "R3"
        else:
            return "R4"


def print_statistics(df):
    stats_df = df.copy()
    stats_df['Region'] = stats_df.apply(lambda row: get_region(row[COL_P], row[COL_Z]), axis=1)
    counts = stats_df['Region'].value_counts()
    total = len(stats_df)

    print("\n" + "=" * 60)
    print(f"{'REGION':<6} | {'COUNT':<6} | {'PERCENT':<8}")
    print("-" * 60)
    for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
        c = counts.get(r, 0)
        pct = (c / total) * 100 if total > 0 else 0
        print(f"{r:<6} | {c:<6} | {pct:>6.2f}%")
    print("=" * 60 + "\n")


def plot_role_cartography(df, out_base_path):
    """
    out_base_path: 基础路径对象 (Path)，不包含具体后缀，或后缀将被忽略并替换
    """
    # --- 1. 数据清洗 ---
    df = df.copy()
    df[COL_P] = pd.to_numeric(df[COL_P], errors='coerce')
    df[COL_Z] = pd.to_numeric(df[COL_Z], errors='coerce')
    plot_data = df.dropna(subset=[COL_P, COL_Z])

    # --- 2. 初始化画布 ---
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=EXPORT_DPI)
    ax.set_position(AX_POSITION)

    # --- 3. 绘制背景 ---
    Y_MIN, Y_MAX = -2.0, 8.0
    HUB_Z = 2.5
    patches_coords = [
        (0.00, Y_MIN, 0.05, HUB_Z - Y_MIN, 'R1'),
        (0.05, Y_MIN, 0.57, HUB_Z - Y_MIN, 'R2'),
        (0.62, Y_MIN, 0.18, HUB_Z - Y_MIN, 'R3'),
        (0.80, Y_MIN, 0.20, HUB_Z - Y_MIN, 'R4'),
        (0.00, HUB_Z, 0.30, Y_MAX - HUB_Z, 'R5'),
        (0.30, HUB_Z, 0.45, Y_MAX - HUB_Z, 'R6'),
        (0.75, HUB_Z, 0.25, Y_MAX - HUB_Z, 'R7')
    ]

    for (x, y, w, h, tag) in patches_coords:
        # 背景条纹
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, hatch='//////',
            edgecolor=REGION_COLORS[tag], alpha=0.25, linewidth=0, zorder=0
        )
        ax.add_patch(rect)

        # R标签 + 描边
        cx = x + w / 2
        cy = y + h / 2
        txt = ax.text(cx, cy, tag, fontsize=MAIN_LABEL_SIZE, fontweight='normal',
                      ha='center', va='center',
                      color=COLOR_REGION_LABEL, alpha=0.9, zorder=120)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    # --- 4. 绘制散点 ---
    markers = {'periphery': '^', 'semi_core': 'o', 'core': 's'}
    order_lower = ['core', 'semi_core', 'periphery']
    order_upper = ['periphery', 'semi_core', 'core']

    data_lower = plot_data[plot_data[COL_Z] < 2.5]
    data_upper = plot_data[plot_data[COL_Z] >= 2.5]

    # Lower Z
    if not data_lower.empty and COL_LAYER in data_lower.columns:
        for i, role in enumerate(order_lower):
            subset = data_lower[data_lower[COL_LAYER] == role]
            if subset.empty: continue
            p_vals = subset[COL_P]
            z_vals = subset[COL_Z]
            colors = [REGION_COLORS[get_region(p, z)] for p, z in zip(p_vals, z_vals)]
            ax.scatter(p_vals, z_vals, s=NODE_SIZES.get(role, 30),
                       c=colors, marker=markers.get(role, 'o'),
                       alpha=POINT_ALPHA, edgecolors='w', linewidth=0.6,
                       zorder=10 + i)

    # Upper Z
    if not data_upper.empty and COL_LAYER in data_upper.columns:
        for i, role in enumerate(order_upper):
            subset = data_upper[data_upper[COL_LAYER] == role]
            if subset.empty: continue
            p_vals = subset[COL_P]
            z_vals = subset[COL_Z]
            colors = [REGION_COLORS[get_region(p, z)] for p, z in zip(p_vals, z_vals)]
            ax.scatter(p_vals, z_vals, s=NODE_SIZES.get(role, 30),
                       c=colors, marker=markers.get(role, 'o'),
                       alpha=POINT_ALPHA, edgecolors='w', linewidth=0.6,
                       zorder=20 + i)

    # --- 5. 轴与布局 ---
    ax.set_xlim(0, 1.0)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"Participation coefficient, $P$", fontsize=MAIN_LABEL_SIZE,
                  color=COLOR_AXIS_TEXT, labelpad=MAIN_X_LABEL_PAD)
    ax.set_ylabel(r"Within-module degree, $z$", fontsize=MAIN_LABEL_SIZE,
                  color=COLOR_AXIS_TEXT, labelpad=MAIN_Y_LABEL_PAD)
    ax.tick_params(direction='in', top=False, right=False,
                   colors=COLOR_AXIS_TEXT, labelcolor=COLOR_AXIS_TEXT,
                   width=MAIN_TICK_WIDTH, length=3, pad=MAIN_TICK_PAD,
                   labelsize=MAIN_TICK_LABEL_SIZE)
    ax.tick_params(axis='x', pad=MAIN_X_TICK_PAD)
    ax.tick_params(axis='y', pad=MAIN_Y_TICK_PAD)
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    if xticklabels:
        xticklabels[0].set_ha('left')
    if yticklabels:
        yticklabels[0].set_va('bottom')
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_AXIS_TEXT)
        spine.set_linewidth(AXIS_LINE_WIDTH)

    # --- 6. 图例 ---
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Periphery',
               markerfacecolor=LEGEND_MARKER_COLOR, markersize=7.5, markeredgecolor='w'),
        Line2D([0], [0], marker='o', color='w', label='Semi-periphery',
               markerfacecolor=LEGEND_MARKER_COLOR, markersize=6.5, markeredgecolor='w'),
        Line2D([0], [0], marker='s', color='w', label='Core',
               markerfacecolor=LEGEND_MARKER_COLOR, markersize=6.5, markeredgecolor='w')
    ]
    leg = ax.legend(handles=legend_elements,
                    loc='upper center', bbox_to_anchor=(0.5, -0.22),
                    ncol=3, frameon=False, fontsize=MAIN_LEGEND_SIZE,
                    handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)
    for text in leg.get_texts():
        text.set_color(COLOR_LEGEND_TEXT)

    # --- 7. 多格式保存 (tif, png, svg) ---
    # 确保传入的是Path对象
    base_path = Path(out_base_path)

    print("-" * 30)
    for fmt in EXPORT_FORMATS:
        # 替换或添加后缀
        save_path = base_path.with_suffix(f'.{fmt}')

        # 保存 (SVG是矢量图，dpi参数影响不明显但保留无妨；TIF和PNG会应用600dpi)
        ax.set_position(AX_POSITION)
        if fmt == "tiff":
            plt.savefig(save_path, dpi=EXPORT_DPI, format=fmt,
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(save_path, dpi=EXPORT_DPI, format=fmt)
        print(f"Saved [{fmt}] (DPI={EXPORT_DPI}): {save_path.name}")
    print("-" * 30)

    if SHOW_PLOT_WINDOW:
        plt.show()
    else:
        plt.close(fig)


# =======================
# 3) 主程序
# =======================
def main():
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        print(f"Error: File not found at {in_path}")
        return

    print(f"Reading: {in_path.name}")
    df = pd.read_csv(in_path)

    if COL_P not in df.columns or COL_Z not in df.columns:
        print("Error: Missing required P/Z columns.")
        return

    print_statistics(df)

    out_dir = Path(OUT_DIR) if OUT_DIR else in_path.parent
    if not out_dir.exists(): out_dir.mkdir(parents=True)

    # 定义基础文件名 (不带后缀，后缀由函数自动添加)
    out_base = out_dir / "Role_Cartography_MultiFormat"

    plot_role_cartography(df, out_base)
    print("All exports completed.")


if __name__ == "__main__":
    main()

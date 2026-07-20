# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

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
input_dir = os.path.join(feature_dir, "data")
output_dir = os.path.join(feature_dir, "figures")

csv_path = os.path.join(input_dir, "network_evolution_stats_fixed2.csv")
filename_base = "1plot_growth_edges_v28_1"

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Input file not found. Generating demo data...")
    data = []
    years = range(2006, 2026)
    for year in years:
        x = year - 2005
        new_overall = int(x ** 2.1 * 150 + 8000)
        cum_overall = int(x ** 2.5 * 300)
        new_active = int(x ** 1.5 * 80 + 1000)
        cum_active = int(x ** 1.8 * 200)
        new_backbone = int(x * 15 + 80)
        cum_backbone = int(x * 250)

        data.append({
            "year_end": year,
            "pipeline": "S2",
            "method": "newman",
            "new_edges_unique": new_overall,
            "cumulative_edges": cum_overall
        })
        data.append({
            "year_end": year,
            "pipeline": "S5-S2",
            "method": "newman",
            "new_edges_unique": new_active,
            "cumulative_edges": cum_active
        })
        data.append({
            "year_end": year,
            "pipeline": "S5-S3-S2",
            "method": "newman",
            "new_edges_unique": new_backbone,
            "cumulative_edges": cum_backbone
        })
    df = pd.DataFrame(data)

df_plot = df[df["method"] == "newman"]
df_plot = df_plot[df_plot["year_end"] < 2025]

styles = {
    "S2": {
        "marker": "s",
        "color": "#888888",
        "ls": "-",
        "label": "Overall"
    },
    "S5-S2": {
        "marker": "o",
        "color": "#0172be",
        "ls": "-",
        "mfc": "white",
        "label": "Active"
    },
    "S5-S3-S2": {
        "marker": "^",
        "color": "#ed7d2f",
        "ls": "--",
        "label": "Backbone"
    }
}

# 左轴显示 Overall/Active 的新增边，右轴显示 Backbone 的新增边。
max_val_left = df_plot[df_plot["pipeline"] == "S2"]["new_edges_unique"].max()
series_backbone = df_plot[df_plot["pipeline"] == "S5-S3-S2"]["new_edges_unique"]
max_val_right = series_backbone.max() if not series_backbone.empty else 100

LEFT_AXIS_TOP_LIMIT = max_val_left * 1.6
RIGHT_AXIS_TOP_LIMIT = max_val_right * 1.8

# ==========================================
# 3. 主图
# ==========================================
fig, ax_left = plt.subplots(figsize=(4.2, 3.34), dpi=600)
ax_left.set_position([0.205000, 0.220848, 0.619100, 0.707526])

ax_right = ax_left.twinx()
ax_right.set_position([0.205000, 0.220848, 0.619100, 0.707526])

for pipe in ["S2", "S5-S2", "S5-S3-S2"]:
    subset = df_plot[df_plot["pipeline"] == pipe].sort_values("year_end")
    if subset.empty:
        continue

    style = styles[pipe]
    target_ax = ax_right if pipe == "S5-S3-S2" else ax_left

    target_ax.plot(
        subset["year_end"],
        subset["new_edges_unique"],
        marker=style["marker"],
        color=style["color"],
        ls=style["ls"],
        mfc=style.get("mfc", style["color"]),
        label=style["label"],
        linewidth=1.2,
        markersize=3.6,
        markevery=1
    )

# X 轴两侧留出少量空白，避免端点贴边。
ax_left.set_xlim(2005.5, 2024.5)
ax_left.set_xticks(np.arange(2008, 2025, 4))
ax_left.set_xlabel("Year", fontsize=13, labelpad=5)
ax_left.tick_params(axis='x', which='both', direction='in',
                    top=False, bottom=True, labelsize=11, rotation=30)
for label in ax_left.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

# 左轴刻度间隔加大，减少刻度标签密度。
ax_left.set_ylabel("New nodes (overall & active)", fontsize=13, labelpad=5)
ax_left.set_ylim(0, LEFT_AXIS_TOP_LIMIT)
ax_left.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
ax_left.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax_left.tick_params(axis='y', which='both', direction='in',
                    right=False, left=True, labelsize=11)
ax_left.grid(False)

ax_right.set_ylabel("")
ax_right.set_ylim(0, RIGHT_AXIS_TOP_LIMIT)
ax_right.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax_right.tick_params(axis='y', which='both', direction='in',
                     left=False, right=True,
                     color=styles["S5-S3-S2"]["color"],
                     labelcolor=styles["S5-S3-S2"]["color"],
                     labelsize=11)
ax_right.grid(False)

for spine in ax_left.spines.values():
    spine.set_visible(True)
    spine.set_color(BASE_COLOR)
for side in ['left', 'top', 'bottom']:
    ax_right.spines[side].set_visible(False)
ax_right.spines['right'].set_visible(True)
ax_right.spines['right'].set_color(styles["S5-S3-S2"]["color"])
ax_right.spines['right'].set_linewidth(1.0)

# 图例符号尺寸与 node_dual_line2.py 对齐。
legend_handles = [
    Line2D([0], [0],
           color=styles[pipe]["color"],
           ls=styles[pipe]["ls"],
           marker=styles[pipe]["marker"],
           markersize=5.5,
           linewidth=1.4,
           markerfacecolor=styles[pipe].get("mfc", styles[pipe]["color"]),
           markeredgecolor=styles[pipe]["color"],
           markeredgewidth=1.1,
           label=styles[pipe]["label"])
    for pipe in ["S2", "S5-S2", "S5-S3-S2"]
]

legend = ax_left.legend(
    handles=legend_handles,
    loc='upper right',
    frameon=False,
    fontsize=11,
    handlelength=1.5,
    handletextpad=0.4,
    borderpad=0.3,
    bbox_to_anchor=(0.98, 0.98)
)
for text in legend.get_texts():
    text.set_color(BASE_COLOR)

# ==========================================
# 4. 嵌入图: 累计边数
# ==========================================
ax_inset = ax_left.inset_axes([0.18, 0.60, 0.34, 0.35])
ax_inset.patch.set_facecolor('white')
ax_inset.patch.set_alpha(0.9)

for pipe in ["S2", "S5-S2", "S5-S3-S2"]:
    subset = df_plot[df_plot["pipeline"] == pipe].sort_values("year_end")
    if subset.empty:
        continue

    style = styles[pipe]
    ax_inset.plot(
        subset["year_end"],
        subset["cumulative_edges"],
        marker=style["marker"],
        color=style["color"],
        ls=style["ls"],
        mfc=style.get("mfc", style["color"]),
        linewidth=0.8,
        markersize=2.1
    )

ax_inset.set_yscale('log')
ax_inset.set_ylabel("Cumulative edges", fontsize=9, labelpad=1)
ax_inset.grid(False)
ax_inset.tick_params(axis='both', which='both', direction='in',
                     top=False, right=False, bottom=True, left=True,
                     width=0.4, length=2, labelsize=8)
ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))

# ==========================================
# 5. 保存输出
# ==========================================
ax_left.set_position([0.205000, 0.220848, 0.619100, 0.707526])
ax_right.set_position([0.205000, 0.220848, 0.619100, 0.707526])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ext in ['png', 'svg', 'tiff', 'jpg']:
    save_path = os.path.join(output_dir, f"{filename_base}.{ext}")
    fig.savefig(save_path, format=ext, dpi=600)
    print(f"Saved: {save_path}")

plt.show()

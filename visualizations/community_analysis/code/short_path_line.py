import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# ==========================================
# 1. 全局风格设置
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
community_dir = os.path.dirname(script_dir)
input_dir = os.path.join(community_dir, "data")
output_dir = os.path.join(community_dir, "figures")

# 主数据路径 (平均最短路径)
csv_path_main = os.path.join(input_dir, "network_metrics_normalized_final2.csv")
# 嵌入图数据路径 (边相似度 - Jaccard)
csv_path_jaccard = os.path.join(input_dir, "network_metrics_v30_active_jaccard.csv")

filename_base = "00plot_avg_path_len_with_jaccard_inset"

# 配色定义
COLOR_FILL_GRAY = '#f2f2f2'
COLOR_FILL_YELLOW = '#fffbD5'
COLOR_FILL_GREEN = '#D9EEDF'
COLOR_VLINE = '#999999'

# --- 加载主数据 ---
try:
    df = pd.read_csv(csv_path_main)
    if "method" in df.columns:
        df = df[df["method"] == "newman"]
except FileNotFoundError:
    print("Main file not found, generating mock data...")
    data = []
    years = range(2006, 2026)
    for y in years:
        apl_a = 3.5 + (y - 2006) * 0.1
        apl_b = 3.2 + (y - 2006) * 0.08
        apl_c = 4.0 + (y - 2006) * 0.15 + (1 if y > 2020 else 0)
        data.append({"year": y, "pipeline": "S2", "avg_path_len": apl_a})
        data.append({"year": y, "pipeline": "S5-S2", "avg_path_len": apl_b})
        data.append({"year": y, "pipeline": "S5-S3-S2", "avg_path_len": apl_c})
    df = pd.DataFrame(data)

if 'year_end' in df.columns and 'year' not in df.columns:
    df.rename(columns={'year_end': 'year'}, inplace=True)
df_plot = df[df["year"] < 2025]

# --- 加载/生成嵌入图数据 (Jaccard) ---
try:
    df_jac = pd.read_csv(csv_path_jaccard)
except FileNotFoundError:
    print("Jaccard file not found, generating mock data...")
    data_j = []
    years = range(2006, 2026)
    for y in years:
        # 模拟数据: Overall 稳定, Active 波动低, Backbone 较高
        data_j.append({"year": y, "pipeline": "S2", "active_jaccard_edges": 0.8 + (y % 3) * 0.02})
        data_j.append({"year": y, "pipeline": "S5-S2", "active_jaccard_edges": 0.3 + (y % 4) * 0.03})
        data_j.append({"year": y, "pipeline": "S5-S3-S2", "active_jaccard_edges": 0.6 + (y % 2) * 0.05})
    df_jac = pd.DataFrame(data_j)
df_jac_plot = df_jac[df_jac["year"] < 2025]

# --- 样式定义 ---
styles = {
    "S2": {"marker": "s", "color": "#888888", "ls": "-", "label": "Overall"},
    "S5-S2": {"marker": "o", "color": "#0172be", "ls": "-", "mfc": "white", "label": "Active"},
    "S5-S3-S2": {"marker": "^", "color": "#ed7d2f", "ls": "--", "label": "Backbone"}
}

# ==========================================
# 3. 主图绘制
# ==========================================
# 稍微增加宽度以容纳右侧图例
fig, ax = plt.subplots(figsize=(4.2, 3.70), dpi=600)
ax.set_position([0.205000, 0.199360, 0.648370, 0.638686])

# --- A. 背景区域与辅助线 ---
ax.axvline(x=2012, color=COLOR_VLINE, linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
ax.axvline(x=2020, color=COLOR_VLINE, linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
ax.axvspan(2005, 2012, facecolor=COLOR_FILL_GRAY, alpha=0.5, zorder=0)
ax.axvspan(2012, 2020, facecolor=COLOR_FILL_YELLOW, alpha=0.3, zorder=0)
ax.axvspan(2020, 2025, facecolor=COLOR_FILL_GREEN, alpha=0.35, zorder=0)

# --- B. 主图数据绘制 ---
for pipe in ["S2", "S5-S2", "S5-S3-S2"]:
    subset = df_plot[df_plot["pipeline"] == pipe].sort_values("year")
    if subset.empty: continue
    st = styles[pipe]
    lw = 1.2 if pipe == "S2" else 1.0
    x_data = subset["year"] + (0.2 if pipe == "S5-S2" else 0)  # 偏移

    ax.plot(x_data, subset["avg_path_len"],
            marker=st["marker"], color=st["color"], ls=st["ls"],
            mfc=st.get("mfc", st["color"]), label=st["label"],
            linewidth=lw, markersize=4, markevery=1, zorder=2)

# --- C. 嵌入图绘制 (右上角) ---
# 位置: [x, y, width, height] (相对于主图坐标)
ax_inset = ax.inset_axes([0.19, 0.60, 0.36, 0.36])
ax_inset.patch.set_facecolor('white')
ax_inset.patch.set_alpha(1)


for pipe in ["S2", "S5-S2", "S5-S3-S2"]:
    subset = df_jac_plot[df_jac_plot["pipeline"] == pipe].sort_values("year")
    if subset.empty: continue
    st = styles[pipe]
    x_data = subset["year"] + (0.2 if pipe == "S5-S2" else 0)  # 保持一致的偏移

    # 绘制边相似度
    # 注意列名 active_jaccard_edges，如果文件列名不同请修改此处
    col_name = "active_jaccard_edges" if "active_jaccard_edges" in subset.columns else "jaccard_edges"

    ax_inset.plot(x_data, subset[col_name],
                  marker=st["marker"], color=st["color"], ls=st["ls"],
                  mfc=st.get("mfc", st["color"]),
                  linewidth=0.8, markersize=2.5)

# 嵌入图样式
ax_inset.set_ylabel("Edge similarity", fontsize=9, labelpad=1)
ax_inset.set_xlabel("", fontsize=9)
ax_inset.tick_params(axis='both', which='both', direction='in',
                     top=False, right=False, bottom=True, left=True,
                     width=0.4, length=2, labelsize=8)
ax_inset.set_xticks([2008, 2016, 2024])
ax_inset.set_ylim(0, 0.3)  # 相似度通常在0-1之间
ax_inset.set_yticks([0.0, 0.15, 0.3])
ax_inset.grid(False)

# --- D. 坐标轴与布局设置 ---
ax.set_xlabel("Year", fontsize=13, labelpad=5)
ax.set_xlim(2005, 2025)
ax.set_xticks(range(2008, 2025, 4))
ax.tick_params(axis='x', direction='in', top=False, bottom=True, labelsize=11, rotation=30)
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

# [核心修改] 强制 Y 轴只显示整数刻度
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

ax.set_ylabel("Average shortest path length", fontsize=13, labelpad=8)
ax.tick_params(axis='y', direction='in', right=False, left=True, labelsize=11)
ax.grid(False)

# [关键修改] 增加 Y 轴上限，为嵌入图留出空间
y_min = df_plot["avg_path_len"].min()
y_max = df_plot["avg_path_len"].max()
ax.set_ylim(1, y_max * 1.50)

# 图例移动到右侧内部
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.50, 0.935043),
           ncol=3,
           frameon=False, fontsize=11,
           handlelength=1.5, handletextpad=0.4,
           columnspacing=0.8, borderpad=0.2)

# ==========================================
# 4. 保存与显示
# ==========================================
# 调整布局以适应外部图例
# 自动调整布局，防止标签被裁剪
# Keep the exported canvas size consistent with edge_type_bar.py.

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 循环保存为 png, svg, tiff, jpg 四种格式
for ext in ['png', 'svg', 'tiff', 'jpg']:
    save_path = os.path.join(output_dir, f"{filename_base}.{ext}")
    fig.savefig(save_path, format=ext, dpi=600)
    print(f"已保存: {save_path}")

plt.show()

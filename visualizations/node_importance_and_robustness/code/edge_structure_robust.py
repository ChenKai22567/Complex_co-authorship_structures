# -*- coding: utf-8 -*-
"""
plot_robustness_structural_edge_final_v10.py

功能：
    绘制“边攻击”结构鲁棒性分析图（主图：LCC大小，嵌入图：连通分量数）。

完全沿用 plot_robustness_structural_final_v10.py 的风格（v10）：
    1. 【透明度高级控制】：
       - 线条和Marker彩色部分：半透明 (GLOBAL_ALPHA)。
       - 空心Marker白色填充 & 实心Marker白色描边：完全不透明，遮挡背景。
    2. 【嵌入图独立配置】：
       - 嵌入图线宽、Marker边缘线宽独立设置。
    3. 【基础保持】：
       - 去右上刻度、刻度向外、四边框线保留、深灰文本颜色、600dpi 输出多格式。

作者：AI 助手
"""

# 导入操作系统模块，用于处理文件路径和目录创建
import os
# 导入 NumPy 数值计算库，用于处理数组运算
import numpy as np
# 导入 Pandas 数据处理库，用于读取和操作 CSV 表格数据
import pandas as pd
# 导入 Matplotlib 绘图库的核心模块
import matplotlib.pyplot as plt
# 导入刻度定位器，用于控制坐标轴上刻度的数量
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, MaxNLocator
from matplotlib.patches import Rectangle
# 导入颜色转换工具，用于精确控制颜色的透明度通道 (RGBA)
from matplotlib.colors import to_rgba
# 导入警告管理模块
import warnings

# 忽略代码运行过程中的非致命警告，保持控制台输出整洁
warnings.filterwarnings("ignore")

# =========================
# 1) 全局路径与绘图风格设置
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORTANCE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(IMPORTANCE_DIR, "data")
OUTPUT_FIG_DIR = os.path.join(IMPORTANCE_DIR, "figures")
# 创建输出目录，如果目录已存在则不报错
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# 设置 Matplotlib 的全局字体为 Arial（学术常用无衬线字体）
plt.rcParams["font.family"] = "Arial"
TEXT_COLOR = "#444444"
FIG_SIZE = (4.25, 3.12)
DPI = 600
AX_POSITION = [0.17, 0.18, 0.602, 0.757]
INSET_POS = [0.61, 0.61, 0.35, 0.35]
AXIS_LINE_WIDTH = 0.8
MAIN_TICK_WIDTH = 0.8
INSET_TICK_WIDTH = 0.4
MAIN_LABEL_SIZE = 12
MAIN_TICK_LABEL_SIZE = 10
MAIN_LEGEND_SIZE = 6
INSET_LABEL_SIZE = 7.5
INSET_TICK_LABEL_SIZE = 7
MAIN_X_LABEL_PAD = 7
MAIN_Y_LABEL_PAD = 10
MAIN_TICK_PAD = 3
INSET_LABEL_PAD = 3
INSET_TICK_PAD = 3

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
plt.rcParams["figure.dpi"] = DPI

# =========================
# 2) 边攻击策略配置（Marker 个性化配置）
# =========================
EDGE_STRATEGIES = [
    # 策略1：随机删边（基准线）
    {"key": "random", "label": "Random", "color": "#999999", "style": "--",
     "file": "edge_random_mean_curve.csv", "marker": "o", "ms": 4.5, "hollow": False},

    # 策略2：弱权重优先移除（通常更“温和”）
    {"key": "weak_weight", "label": "Weak weight", "color": "#79B96C", "style": "-",
     "file": "edge_weak_weight_curve.csv", "marker": "s", "ms": 4, "hollow": False},

    # 策略3：强权重优先移除
    {"key": "strong_weight", "label": "Strong weight", "color": "#F8AD20", "style": "-",
     "file": "edge_strong_weight_curve.csv", "marker": "o", "ms": 4.5, "hollow": False},

    # 策略5：边介数（精确版本）
    {"key": "edge_betweenness", "label": "Edge betweenness", "color": "#4D97C6", "style": "-",
     "file": "edge_edge_betweenness_curve.csv", "marker": "D", "ms": 3.5, "hollow": True},

    # 策略6：边重叠度（低重叠优先移除，桥边/跨社群边）
    {"key": "overlap_low_first", "label": "Low overlap", "color": "#9467bd", "style": "-",
     "file": "edge_overlap_low_first_curve.csv", "marker": "^", "ms": 5.5, "hollow": False},
]

# =========================
# 3) 绘图参数配置（独立控制区）
# =========================

# --- A. 全局透明度 ---
GLOBAL_ALPHA = 0.85  # 线条和彩色Marker半透明

# --- B. 主图配置 ---
MAIN_TICK_LENGTH = 3
MAIN_LINE_WIDTH = 1.2
MAIN_HOLLOW_EDGE_WIDTH = 1.2
MAIN_SOLID_EDGE_WIDTH = 0.5

# --- C. 嵌入图独立配置 ---
INSET_TICK_LENGTH = 2
INSET_LINE_WIDTH = 1.0
INSET_MARKER_SCALE = 0.5
INSET_HOLLOW_EDGE_WIDTH = 0.8
INSET_SOLID_EDGE_WIDTH = 0.3

# =========================
# 4) 数据处理辅助函数
# =========================

def _get_removed_frac(df: pd.DataFrame) -> np.ndarray:
    """
    功能：从 DataFrame 中提取表示'移除比例'的列。
    兼容不同写法：removed_frac / removed_percent / removal_percentage 等。
    """
    # 优先使用标准列名 removed_frac
    if "removed_frac" in df.columns:
        x = pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).to_numpy()
        return x

    # 兼容其它列名
    for c in ["removed_percent", "removal_percentage", "removed_percentage"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            # 如果是 0-100 的百分比形式，则归一化到 0-1
            if np.nanmax(x) > 1.0:
                x = x / 100.0
            return x

    # 都找不到则抛出错误
    raise ValueError("❌ CSV 中找不到表示移除比例的列 (如 removed_frac)。")


def load_data(strategies, base_dir):
    """
    功能：根据策略配置批量读取 CSV 文件。
    返回：dict，key=策略名，value=DataFrame。
    """
    data_dict = {}
    print(f"[加载数据] 正在从 {base_dir} 读取...")

    count = 0
    for strat in strategies:
        path = os.path.join(base_dir, strat["file"])
        if os.path.exists(path):
            try:
                data_dict[strat["key"]] = pd.read_csv(path)
                count += 1
            except Exception:
                pass

    print(f"   成功加载 {count} 个文件。\n")
    return data_dict

# =========================
# 5) 轴美化工具（保留框线 + 去右上刻度）
# =========================

def beautify_axis_box(ax, tick_length=4, tick_width=0.8, tick_pad=3):
    """
    功能：美化坐标轴。
    - 保留四周边框线（矩形框）。
    - 只显示左侧/下方刻度线（去掉右侧/上侧刻度线）。
    """
    ax.tick_params(
        direction='in',
        top=False, right=False, left=True, bottom=True,
        length=tick_length, width=tick_width, pad=tick_pad,
        colors=TEXT_COLOR
    )

    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['right'].set_color(TEXT_COLOR)
    ax.spines['top'].set_color(TEXT_COLOR)
    ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['right'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['top'].set_linewidth(AXIS_LINE_WIDTH)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

# =========================
# 6) 核心绘图逻辑：边攻击结构鲁棒性
# =========================

def plot_edge_robustness_clean(strategies, data_dict, filename):
    """
    功能：绘制边攻击结构鲁棒性图：
    - 主图：lcc_ratio
    - 嵌入图：num_components
    """
    print(f"[绘制] {filename}")

    # 创建画布（与 v10 一致）
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_position(AX_POSITION)

    # 主图坐标轴美化
    beautify_axis_box(ax, tick_length=MAIN_TICK_LENGTH, tick_width=MAIN_TICK_WIDTH,
                      tick_pad=MAIN_TICK_PAD)

    # 坐标范围（与 v10 一致）
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.0)

    # --- 主图：绘制 LCC 曲线 ---
    for strat in strategies:
        key = strat["key"]
        if key not in data_dict:
            continue

        df = data_dict[key]

        # X：移除比例（降采样防拥挤）
        x = _get_removed_frac(df)[::2]
        # Y：LCC 相对规模（降采样）
        y = pd.to_numeric(df["lcc_ratio"], errors="coerce").fillna(np.nan).to_numpy()[::2]

        # 基础半透明颜色（用于线条和彩色Marker）
        transparent_base_color = to_rgba(strat["color"], alpha=GLOBAL_ALPHA)

        # Marker 样式：空心/实心
        if strat["hollow"]:
            face_col = "white"                       # 空心：白色填充（不透明）
            edge_col = transparent_base_color         # 边缘：半透明策略色
            edge_width = MAIN_HOLLOW_EDGE_WIDTH       # 空心边宽
        else:
            face_col = transparent_base_color         # 实心：半透明策略色填充
            edge_col = "white"                        # 实心：白色描边（不透明）
            edge_width = MAIN_SOLID_EDGE_WIDTH        # 实心边宽

        # 绘图
        ax.plot(
            x, y,
            label=strat["label"],
            color=transparent_base_color,
            linestyle=strat["style"],
            linewidth=MAIN_LINE_WIDTH if key != "random" else MAIN_LINE_WIDTH + 0.5,
            marker=strat["marker"],
            markersize=strat["ms"],
            markevery=1,
            markerfacecolor=face_col,
            markeredgecolor=edge_col,
            markeredgewidth=edge_width,
            zorder=10
        )

    # 主图标签（注意：节点->边）
    ax.set_xlabel("Fraction of edges removed", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_X_LABEL_PAD)
    ax.set_ylabel("Relative size of LCC", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_Y_LABEL_PAD)

    # 图例（延续 v10 布局）
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.035, 0.035),
        frameon=False,
        fontsize=MAIN_LEGEND_SIZE,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.35,
        borderaxespad=0.0,
        markerscale=0.85
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    legend.set_zorder(30)

    # --- 嵌入图：绘制连通分量数 ---
    inset_frame_bg = Rectangle(
        (INSET_POS[0] - 0.025, INSET_POS[1] - 0.025),
        INSET_POS[2] + 0.05,
        INSET_POS[3] + 0.05,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        zorder=20,
        clip_on=False
    )
    ax.add_patch(inset_frame_bg)
    ax_ins = ax.inset_axes(INSET_POS)
    ax_ins.set_zorder(25)
    beautify_axis_box(ax_ins, tick_length=INSET_TICK_LENGTH,
                      tick_width=INSET_TICK_WIDTH, tick_pad=INSET_TICK_PAD)

    ax_ins.set_facecolor("white")
    ax_ins.patch.set_alpha(1.0)
    ax_ins.patch.set_zorder(25)
    ax_ins.set_xlim(0.0, 1.0)

    for strat in strategies:
        key = strat["key"]
        if key not in data_dict:
            continue
        df = data_dict[key]

        x = _get_removed_frac(df)[::2]
        y2 = pd.to_numeric(df["num_components"], errors="coerce").fillna(np.nan).to_numpy()[::2]

        transparent_base_color = to_rgba(strat["color"], alpha=GLOBAL_ALPHA)

        if strat["hollow"]:
            face_col = "white"
            edge_col = transparent_base_color
            edge_width = INSET_HOLLOW_EDGE_WIDTH
        else:
            face_col = transparent_base_color
            edge_col = "white"
            edge_width = INSET_SOLID_EDGE_WIDTH

        ax_ins.plot(
            x, y2,
            color=transparent_base_color,
            linestyle=strat["style"],
            linewidth=INSET_LINE_WIDTH,
            marker=strat["marker"],
            markersize=strat["ms"] * INSET_MARKER_SCALE,
            markevery=1,
            markerfacecolor=face_col,
            markeredgecolor=edge_col,
            markeredgewidth=edge_width,
            zorder=10
        )

    # 嵌入图标题与标签（保持 v10 文案）
    inset_ymin, inset_ymax = ax_ins.get_ylim()
    ax_ins.set_ylim(inset_ymin, inset_ymax * 1.12)

    ax_ins.set_xlabel("Fraction removed", fontsize=INSET_LABEL_SIZE,
                      labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)
    ax_ins.set_ylabel("Components", fontsize=INSET_LABEL_SIZE,
                      labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)

    # 控制嵌入图刻度数量（与 v10 一致）
    ax_ins.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax_ins.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_ins.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_ins.tick_params(axis="both", which="major", direction="in",
                       top=False, right=False, bottom=True, left=True,
                       length=INSET_TICK_LENGTH, width=INSET_TICK_WIDTH,
                       pad=INSET_TICK_PAD,
                       labelsize=INSET_TICK_LABEL_SIZE, labelcolor=TEXT_COLOR)
    text_bg = {"facecolor": "white", "edgecolor": "none", "pad": 1}
    ax_ins.xaxis.label.set_bbox(text_bg)
    ax_ins.yaxis.label.set_bbox(text_bg)
    for tick_label in ax_ins.get_xticklabels() + ax_ins.get_yticklabels():
        tick_label.set_bbox(text_bg)

    # =========================
    # 7) 保存与显示（多格式）
    # =========================
    formats = ["png", "pdf", "svg", "tiff", "jpg"]
    base_filename = os.path.splitext(filename)[0]

    for fmt in formats:
        save_file = f"{base_filename}.{fmt}"
        save_path = os.path.join(OUTPUT_FIG_DIR, save_file)

        ax.set_position(AX_POSITION)
        if fmt == "tiff":
            plt.savefig(
                save_path, dpi=DPI, format=fmt,
                pil_kwargs={"compression": "tiff_lzw"}
            )
        else:
            plt.savefig(save_path, dpi=DPI, format=fmt)

        print(f"   已保存 ({fmt}): {save_path}")

    plt.show()
    plt.close()
    print("   绘图全部完成。\n")


def main():
    # 1) 读取边攻击数据
    edge_data = load_data(EDGE_STRATEGIES, INPUT_DIR)
    # 2) 绘制边攻击结构鲁棒性图
    plot_edge_robustness_clean(
        EDGE_STRATEGIES,
        edge_data,
        "Edge_Robustness_Clean_Style_v10.png"
    )


if __name__ == "__main__":
    main()

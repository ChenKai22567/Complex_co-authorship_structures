# -*- coding: utf-8 -*-
"""
plot_robustness_efficiency_capacity_edge_final_v8.py

功能：
    绘制“边攻击”功能鲁棒性分析图（主图：网络效率，嵌入图：引文承载量/保留率）。

修正点（保持 v8 样式）：
    1. 【独立设置】：嵌入图的线宽(Line Width)和Marker轮廓宽度(Edge Width)独立配置。
    2. 【透明度保持】：线条/彩色部分半透明，白色填充不透明（高级视觉效果）。
    3. 【配置区】：保留 Inset 的详细配置参数。

作者：AI 助手
"""

# 导入操作系统模块，用于处理文件路径、创建目录等
import os
# 导入 NumPy 数值计算库，用于数组操作和数学运算
import numpy as np
# 导入 Pandas 数据处理库，用于读取 CSV 文件和数据清洗
import pandas as pd
# 导入 Matplotlib 绘图库的核心模块
import matplotlib.pyplot as plt
# 导入刻度定位器，用于控制坐标轴上刻度的显示数量
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator
# 导入颜色转换工具，用于将颜色名转换为 (R, G, B, Alpha) 格式，以便精确控制透明度
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
# 创建输出目录，如果目录已存在则不报错（exist_ok=True）
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# --- Matplotlib 全局样式参数设置 ---
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
# 2) 攻击策略配置（替换为边攻击）
# =========================

# 定义“边攻击”策略列表（文件名与你 robustness_results_v1 的输出一致）
EDGE_STRATEGIES = [
    # 策略1：随机攻击（基准线，灰色虚线）
    {"key": "random", "label": "Random", "color": "#999999", "style": "--",
     "file": "edge_random_mean_curve.csv", "marker": "o", "ms": 4.5, "hollow": False},

    # 策略2：弱权重边优先移除（实心）
    {"key": "weak_weight", "label": "Weak wt.", "color": "#79B96C", "style": "-",
     "file": "edge_weak_weight_curve.csv", "marker": "s", "ms": 4, "hollow": False},

    # 策略3：强权重边优先移除（实心）
    {"key": "strong_weight", "label": "Strong wt.", "color": "#F8AD20", "style": "-",
     "file": "edge_strong_weight_curve.csv", "marker": "o", "ms": 4.5, "hollow": False},

    # 策略5：边介数（精确版本）（空心菱形 highlight）
    {"key": "edge_betweenness", "label": "Edge betw.", "color": "#4D97C6", "style": "-",
     "file": "edge_edge_betweenness_curve.csv", "marker": "D", "ms": 3.5, "hollow": True},

    # 策略6：边重叠度（低重叠优先移除）（实心叉号）
    {"key": "overlap_low_first", "label": "Low overlap", "color": "#9467bd", "style": "-",
     "file": "edge_overlap_low_first_curve.csv", "marker": "^", "ms": 5.5, "hollow": False},
]

# =========================
# 3) 绘图详细参数配置（保持 v8：Inset 独立控制）
# =========================

# --- A. 全局透明度 ---
GLOBAL_ALPHA = 0.85

# --- B. 主图 (Main Plot) 配置 ---
MAIN_TICK_LENGTH = 3
MAIN_LINE_WIDTH = 1.2
MAIN_HOLLOW_EDGE_WIDTH = 1.2
MAIN_SOLID_EDGE_WIDTH = 0.5

# --- C. 嵌入图 (Inset Plot) 独立配置 ---
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
    逻辑：兼容不同列名写法（removed_frac / removed_percent / removal_percentage 等）。
    """
    if "removed_frac" in df.columns:
        return pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).to_numpy()

    for c in ["removed_percent", "removal_percentage", "removed_percentage"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            if np.nanmax(x) > 1.0:
                x = x / 100.0
            return x

    raise ValueError("❌ CSV 中找不到表示移除比例的列。")


def load_data(strategies, base_dir):
    """
    功能：根据策略配置，批量读取 CSV 文件。
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
# 5) 轴美化工具（保持 v8：保留框线 + 去右上刻度）
# =========================

def beautify_axis_box(ax, tick_length=4, tick_width=0.8, tick_pad=3):
    """
    功能：美化坐标轴 (保留四周边框，隐藏右上刻度)。
    """
    ax.tick_params(direction='in', top=False, right=False, left=True, bottom=True,
                   length=tick_length, width=tick_width, pad=tick_pad,
                   colors=TEXT_COLOR)

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
# 6) 核心绘图逻辑（边攻击：Efficiency & Capacity）
# =========================

def plot_edge_robustness_efficiency(strategies, data_dict, filename):
    """
    功能：
        绘制边攻击功能鲁棒性图：
        - 主图：eff_ratio（或 efficiency）
        - 嵌入图：cap_in_lcc_ratio
    """
    print(f"[绘制] {filename}")

    # 创建图形画布
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_position(AX_POSITION)

    # 美化主图坐标轴
    beautify_axis_box(ax, tick_length=MAIN_TICK_LENGTH, tick_width=MAIN_TICK_WIDTH,
                      tick_pad=MAIN_TICK_PAD)

    # 设置主图范围
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.0)

    # ==========================
    # A) 主图：效率曲线（使用 MAIN 配置）
    # ==========================
    for strat in strategies:
        key = strat["key"]
        if key not in data_dict:
            continue

        df = data_dict[key]

        # 降采样（避免 marker 过密）
        x = _get_removed_frac(df)[::2]

        # 兼容列名：优先 eff_ratio，否则尝试 efficiency
        eff_col = "eff_ratio" if "eff_ratio" in df.columns else ("efficiency" if "efficiency" in df.columns else None)
        if eff_col is None:
            # 若没有效率列则跳过（避免报错）
            continue
        y = pd.to_numeric(df[eff_col], errors="coerce").fillna(np.nan).to_numpy()[::2]

        # 半透明策略色（线条 + 彩色部分）
        transparent_base_color = to_rgba(strat["color"], alpha=GLOBAL_ALPHA)

        # Marker 空心/实心逻辑（与 v8 完全一致）
        if strat["hollow"]:
            face_col = "white"                    # 空心：白色不透明填充
            edge_col = transparent_base_color      # 边缘：半透明策略色
            edge_width = MAIN_HOLLOW_EDGE_WIDTH
        else:
            face_col = transparent_base_color      # 实心：半透明策略色填充
            edge_col = "white"                     # 白色不透明描边
            edge_width = MAIN_SOLID_EDGE_WIDTH

        # 绘制主图曲线
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

    # 主图标签：节点->边（只改文字，不改风格）
    ax.set_xlabel("Fraction of edges removed", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_X_LABEL_PAD)
    ax.set_ylabel(r"Relative efficiency ($E/E_0$)", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_Y_LABEL_PAD)

    # 图例样式（保持 v8 的定位）
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

    # ==========================
    # B) 嵌入图：LCC 引文承载量/保留率（使用 INSET 独立配置）
    # ==========================
    ax_ins = ax.inset_axes(INSET_POS)
    beautify_axis_box(ax_ins, tick_length=INSET_TICK_LENGTH,
                      tick_width=INSET_TICK_WIDTH, tick_pad=INSET_TICK_PAD)

    ax_ins.set_facecolor("white")
    ax_ins.set_xlim(0.0, 1.0)

    for strat in strategies:
        key = strat["key"]
        if key not in data_dict:
            continue
        df = data_dict[key]

        x = _get_removed_frac(df)[::2]

        # 需要 cap_in_lcc_ratio 列
        if "cap_in_lcc_ratio" not in df.columns:
            continue
        y2 = pd.to_numeric(df["cap_in_lcc_ratio"], errors="coerce").fillna(np.nan).to_numpy()[::2]

        transparent_base_color = to_rgba(strat["color"], alpha=GLOBAL_ALPHA)

        # Inset Marker 空心/实心逻辑（独立线宽）
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

    # 嵌入图文案：保持你 v8 的语义，但适配“边攻击”
    ax_ins.set_ylim(-0.05, 1.0)

    ax_ins.set_xlabel("Fraction removed", fontsize=INSET_LABEL_SIZE,
                      labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)
    ax_ins.set_ylabel("Retention ratio", fontsize=INSET_LABEL_SIZE,
                      labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)

    # 限制嵌入图刻度数量，避免挤
    ax_ins.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax_ins.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_ins.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax_ins.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_ins.tick_params(axis="both", which="major", direction="in",
                       top=False, right=False, bottom=True, left=True,
                       length=INSET_TICK_LENGTH, width=INSET_TICK_WIDTH,
                       pad=INSET_TICK_PAD,
                       labelsize=INSET_TICK_LABEL_SIZE, labelcolor=TEXT_COLOR)

    # =========================
    # 7) 保存与显示（多格式输出）
    # =========================
    formats = ["png", "pdf", "svg", "tiff", "jpg"]
    base_filename = os.path.splitext(filename)[0]

    for fmt in formats:
        save_file = f"{base_filename}.{fmt}"
        save_path = os.path.join(OUTPUT_FIG_DIR, save_file)

        ax.set_position(AX_POSITION)
        if fmt == "tiff":
            plt.savefig(save_path, dpi=DPI, format=fmt,
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(save_path, dpi=DPI, format=fmt)

        print(f"   已保存 ({fmt}): {save_path}")

    plt.show()
    plt.close()
    print("   绘图全部完成。\n")


def main():
    # 1) 加载边攻击数据
    edge_data = load_data(EDGE_STRATEGIES, INPUT_DIR)
    # 2) 绘制边攻击功能鲁棒性图
    plot_edge_robustness_efficiency(
        EDGE_STRATEGIES,
        edge_data,
        "Edge_Functional_Robustness_V8.png"
    )


if __name__ == "__main__":
    main()

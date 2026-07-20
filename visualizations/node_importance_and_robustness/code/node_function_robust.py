# -*- coding: utf-8 -*-
"""
plot_robustness_efficiency_capacity_final_v8.py

功能：
    绘制节点功能鲁棒性分析图（主图：网络效率，嵌入图：引文承载量）。

修正点（本次）：
    1. 【独立设置】：嵌入图的 线宽(Line Width) 和 Marker轮廓宽度(Edge Width) 不再依赖主图，而是单独配置变量。
    2. 【透明度保持】：继续保持“线条/边缘半透明，白色填充不透明”的高级视觉效果。
    3. 【配置区】：增加了专门针对 Inset 的详细配置参数。

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
# 导入刻度定位器，用于控制坐标轴上刻度的显示数量（例如限制嵌入图刻度过密）
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator
# 导入颜色转换工具，用于将颜色名转换为 (R, G, B, Alpha) 格式，以便精确控制透明度
from matplotlib.colors import to_rgba
# 导入警告管理模块
import warnings

# 忽略代码运行过程中的非致命警告（如字体缺失警告等），保持控制台输出整洁
warnings.filterwarnings("ignore")

# =========================
# 1) 全局路径与绘图风格设置
# =========================

# 定义输入数据和图片输出目录，统一放在 importance 下的“数据”“图像”文件夹中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORTANCE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(IMPORTANCE_DIR, "data")
OUTPUT_FIG_DIR = os.path.join(IMPORTANCE_DIR, "figures")
# 创建输出目录，如果目录已存在则不报错（exist_ok=True），确保保存路径有效
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# --- Matplotlib 全局样式参数设置 ---
# 设置全局字体为 Arial，这是学术论文中通用的无衬线字体
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

# 设置全局文本的基础颜色，与 node_structure_robustness.py 保持一致
plt.rcParams["text.color"] = TEXT_COLOR
# 设置坐标轴标签（X轴名称、Y轴名称）的颜色
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
# 设置 X 轴刻度标签的颜色
plt.rcParams["xtick.color"] = TEXT_COLOR
# 设置 Y 轴刻度标签的颜色
plt.rcParams["ytick.color"] = TEXT_COLOR
# 设置坐标轴边框线（Spines）的颜色
plt.rcParams["axes.edgecolor"] = TEXT_COLOR
# 设置坐标轴标签的字体大小
plt.rcParams["axes.labelsize"] = MAIN_LABEL_SIZE
# 设置 X 轴刻度数字的字体大小
plt.rcParams["xtick.labelsize"] = MAIN_TICK_LABEL_SIZE
# 设置 Y 轴刻度数字的字体大小
plt.rcParams["ytick.labelsize"] = MAIN_TICK_LABEL_SIZE
# 设置图例中文本的字体大小
plt.rcParams["legend.fontsize"] = MAIN_LEGEND_SIZE
# 设置坐标轴边框线的粗细（1.0 磅）
plt.rcParams["axes.linewidth"] = AXIS_LINE_WIDTH
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"
# 设置绘图的分辨率为 600 DPI，确保输出图片高清，满足印刷出版要求
plt.rcParams["figure.dpi"] = DPI

# =========================
# 2) 攻击策略配置
# =========================

# 定义节点攻击策略列表，包含每个策略对应的文件、颜色、线型和标记样式
# key: 数据对应的键名
# label: 图例中显示的名称
# color: 线条和标记的颜色
# style: 线型（实线'-' 或 虚线'--'）
# file: CSV文件名
# marker: 散点标记的形状（'o'圆, 's'方, '^'上三角, 'v'下三角, 'D'菱形, 'X'叉）
# ms: 标记的大小 (Marker Size)
# hollow: 是否为空心标记（True=空心，False=实心）
NODE_STRATEGIES = [
    # 策略1：随机攻击（作为基准线，使用灰色虚线）
    {"key": "random", "label": "Random", "color": "#999999", "style": "--", "file": "node_random_mean_curve.csv",
     "marker": "o", "ms": 6, "hollow": False},
    # 策略2：度中心性攻击（实心）
    {"key": "degree", "label": "Degree", "color": "#79B96C", "style": "-", "file": "node_degree_curve.csv",
     "marker": "s", "ms": 5, "hollow": False},
    # 策略3：强度（加权度）攻击（实心）
    {"key": "strength", "label": "Strength", "color": "#F46D43", "style": "-", "file": "node_strength_curve.csv",
     "marker": "^", "ms": 7, "hollow": False},
    # 策略4：介数中心性攻击（空心圆，highlight）
    {"key": "betweenness", "label": "Betweenness", "color": "#F8AD20", "style": "-",
     "file": "node_betweenness_curve.csv", "marker": "o", "ms": 6, "hollow": True},
    # 策略5：紧密中心性攻击（空心菱形，highlight）
    {"key": "closeness", "label": "Closeness", "color": "#4D97C6", "style": "-", "file": "node_closeness_curve.csv",
     "marker": "D", "ms": 4.5, "hollow": True},
    # 策略6：结构洞有效规模攻击（实心叉号）
    {"key": "effective_size", "label": "Effective size", "color": "#9467bd", "style": "-",
     "file": "node_effective_size_curve.csv", "marker": "X", "ms": 6, "hollow": False},
]

# =========================
# 3) 绘图详细参数配置 (核心修改区域：独立控制)
# =========================

# --- A. 全局透明度 ---
# 控制线条和 Marker 彩色部分（边缘或填充）的透明度 (0~1)
# 注意：这不影响空心 Marker 中间的白色填充，白色始终不透明以遮挡背景
GLOBAL_ALPHA = 0.85

# --- B. 主图 (Main Plot) 配置 ---
MAIN_TICK_LENGTH = 3  # 主图坐标轴刻度线的长度
MAIN_LINE_WIDTH = 1.2  # 主图折线的宽度
MAIN_HOLLOW_EDGE_WIDTH = 1.4  # 主图空心 Marker 的边缘线宽 (稍粗，为了看清颜色)
MAIN_SOLID_EDGE_WIDTH = 0.6  # 主图实心 Marker 的白色描边线宽 (稍细，仅用于区分重叠)

# --- C. 嵌入图 (Inset Plot) 独立配置 ---
# 这里实现了“小图参数独立设置，不与大图成比例”的需求
INSET_TICK_LENGTH = 2  # 嵌入图刻度线长度 (比主图短)
INSET_LINE_WIDTH = 1.0  # 嵌入图折线宽度 (比主图细，显得更精致)
INSET_MARKER_SCALE = 0.5  # 嵌入图 Marker 大小的缩放比例 (相对于主图 ms)
INSET_HOLLOW_EDGE_WIDTH = 0.8  # 嵌入图空心 Marker 的边缘线宽 (独立变细)
INSET_SOLID_EDGE_WIDTH = 0.3  # 嵌入图实心 Marker 的白色描边线宽 (独立变细)


# =========================
# 4) 数据处理辅助函数
# =========================

def _get_removed_frac(df: pd.DataFrame) -> np.ndarray:
    """
    功能：从 DataFrame 中提取表示'移除比例'的列。
    逻辑：兼容不同的列名写法（如 removed_frac, removed_percent）。
    """
    # 检查是否存在标准的 'removed_frac' 列
    if "removed_frac" in df.columns:
        # 转换为数值类型，填充缺失值为0
        x = pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).to_numpy()
        return x

    # 如果没找到，尝试检查其他可能的列名变体
    for c in ["removed_percent", "removal_percentage", "removed_percentage"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            # 如果数值范围是 0-100（百分比），则除以 100 归一化为 0-1
            if np.nanmax(x) > 1.0: x = x / 100.0
            return x

    # 如果所有列名都匹配失败，抛出异常
    raise ValueError("CSV 中找不到表示移除比例的列。")


def load_data(strategies, base_dir):
    """
    功能：根据策略配置，批量读取 CSV 文件。
    返回：一个字典，key 为策略名，value 为对应的 DataFrame。
    """
    data_dict = {}
    print(f"[加载数据] 正在从 {base_dir} 读取...")
    count = 0
    for strat in strategies:
        # 拼接完整的文件路径
        path = os.path.join(base_dir, strat["file"])
        # 检查文件是否存在
        if os.path.exists(path):
            try:
                # 读取 CSV 文件
                data_dict[strat["key"]] = pd.read_csv(path)
                count += 1
            except Exception:
                pass  # 读取失败则跳过
    print(f"   成功加载 {count} 个文件。\n")
    return data_dict


# =========================
# 5) 轴美化工具
# =========================

def beautify_axis_box(ax, tick_length=4, tick_width=0.8, tick_pad=3):
    """
    功能：美化坐标轴 (保留四周边框，隐藏右上刻度)。
    参数：
        ax: Matplotlib 的轴对象
        tick_length: 刻度线的长度 (由配置决定)
    """
    # 设置刻度参数：
    # direction='in': 刻度线朝内
    # top=False, right=False: 【关键】关闭上方和右侧的刻度线（小短线）
    # left=True, bottom=True: 开启左侧和下方的刻度线
    ax.tick_params(direction='in', top=False, right=False, left=True, bottom=True,
                   length=tick_length, width=tick_width, colors=TEXT_COLOR,
                   pad=tick_pad)

    # 设置四周边框线（Spines）的颜色
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['right'].set_color(TEXT_COLOR)
    ax.spines['top'].set_color(TEXT_COLOR)
    ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['right'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['top'].set_linewidth(AXIS_LINE_WIDTH)

    # 确保四周边框都是可见的（形成一个完整的矩形框）
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)


# =========================
# 6) 核心绘图逻辑
# =========================

def plot_node_robustness_efficiency(strategies, data_dict, filename):
    """执行绘图操作：Efficiency & Capacity"""
    print(f"[绘制] {filename}")

    # 创建图形画布，设置尺寸为 5x4 英寸，启用约束布局
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_position(AX_POSITION)

    # 美化主图坐标轴，使用主图配置的刻度长度
    beautify_axis_box(ax, tick_length=MAIN_TICK_LENGTH,
                      tick_width=MAIN_TICK_WIDTH, tick_pad=MAIN_TICK_PAD)

    # 设置主图坐标轴范围
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.0)

    # ==========================
    # A. 主图绘图循环 (使用 MAIN 配置)
    # ==========================
    for strat in strategies:
        key = strat["key"]
        # 如果数据未加载，跳过
        if key not in data_dict: continue

        df = data_dict[key]

        # 数据降采样：[::2] 表示每隔一个点取一个，避免 Marker 过于密集
        x = _get_removed_frac(df)[::2]
        # 获取效率数据 (兼容 eff_ratio 或 efficiency 列名)
        col_name = "eff_ratio" if "eff_ratio" in df.columns else "efficiency"
        y = pd.to_numeric(df[col_name], errors="coerce").fillna(np.nan).to_numpy()[::2]

        # --- 颜色与透明度计算 ---
        # 使用 to_rgba 将 HEX 颜色转为 RGBA，赋予 GLOBAL_ALPHA 透明度
        # 这样线条和实心填充就会是半透明的
        transparent_base_color = to_rgba(strat['color'], alpha=GLOBAL_ALPHA)

        # --- 主图 Marker 样式逻辑 ---
        if strat['hollow']:
            # 空心 Marker:
            # 填充 = 不透明白色 (遮挡背景网格)
            # 边缘 = 带透明度的策略色
            # 边框宽 = MAIN 配置
            face_col = 'white'
            edge_col = transparent_base_color
            edge_width = MAIN_HOLLOW_EDGE_WIDTH
        else:
            # 实心 Marker:
            # 填充 = 带透明度的策略色
            # 边缘 = 不透明白色 (描边效果，增加对比度)
            # 边框宽 = MAIN 配置
            face_col = transparent_base_color
            edge_col = 'white'
            edge_width = MAIN_SOLID_EDGE_WIDTH

        # 绘制主图曲线
        ax.plot(
            x, y,
            label=strat["label"],
            color=transparent_base_color,  # 线条颜色(带透明度)
            linestyle=strat["style"],
            linewidth=MAIN_LINE_WIDTH if key != "random" else MAIN_LINE_WIDTH + 0.5,
            # 【注意】：这里不设置全局 alpha，而是依靠 RGBA 颜色控制透明度
            marker=strat["marker"],
            markersize=strat["ms"],
            markevery=1,
            markerfacecolor=face_col,  # 填充色 (根据空心/实心逻辑已定)
            markeredgecolor=edge_col,  # 边缘色
            markeredgewidth=edge_width,  # 边缘粗细 (使用主图配置)
            zorder=10  # 层级设为10，确保覆盖网格
        )

    # 主图标签设置
    ax.set_xlabel("Fraction of nodes removed", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_X_LABEL_PAD)
    ax.set_ylabel(r"Relative efficiency ($E/E_0$)", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_Y_LABEL_PAD)

    # 主图图例设置
    legend = ax.legend(
        loc='lower right',  # 位置：右上角
        bbox_to_anchor=(0.99, 0.17),
        frameon=False,  # 无边框
        fontsize=MAIN_LEGEND_SIZE,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.35,
        borderaxespad=0.0,
        markerscale=0.85
    )
    # 强制设置图例文本颜色
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # ==========================
    # B. 嵌入图绘图循环 (使用 INSET 配置)
    # ==========================

    # 创建嵌入坐标轴 (位置: 左下 [x, y, width, height] 相对坐标)
    ax_ins = ax.inset_axes(INSET_POS)

    # 美化嵌入图坐标轴，使用嵌入图配置的刻度长度
    beautify_axis_box(ax_ins, tick_length=INSET_TICK_LENGTH,
                      tick_width=INSET_TICK_WIDTH, tick_pad=INSET_TICK_PAD)

    # 设置背景白且不透明，防止下方线条透出
    ax_ins.set_facecolor("white")
    ax_ins.set_xlim(0.0, 1.0)
    ax_ins.set_ylim(-0.05, 1.05)

    for strat in strategies:
        key = strat["key"]
        if key not in data_dict: continue
        df = data_dict[key]

        x = _get_removed_frac(df)[::2]
        # 获取容量数据 (Capacity)
        col_name_ins = "cap_in_lcc_ratio"
        if col_name_ins not in df.columns: continue
        y2 = pd.to_numeric(df[col_name_ins], errors="coerce").fillna(np.nan).to_numpy()[::2]

        # --- 颜色与透明度计算 (同上) ---
        transparent_base_color = to_rgba(strat['color'], alpha=GLOBAL_ALPHA)

        # --- 嵌入图 Marker 样式逻辑 (使用 INSET 独立配置) ---
        if strat['hollow']:
            face_col = 'white'
            edge_col = transparent_base_color
            edge_width = INSET_HOLLOW_EDGE_WIDTH  # <--- 使用独立的嵌入图空心线宽
        else:
            face_col = transparent_base_color
            edge_col = 'white'
            edge_width = INSET_SOLID_EDGE_WIDTH  # <--- 使用独立的嵌入图实心线宽

        # 绘制嵌入图曲线
        ax_ins.plot(
            x, y2,
            color=transparent_base_color,  # 线条颜色(带透明度)
            linestyle=strat["style"],
            linewidth=INSET_LINE_WIDTH,  # <--- 使用独立的嵌入图线宽
            marker=strat["marker"],
            markersize=strat["ms"] * INSET_MARKER_SCALE,  # Marker 大小按比例缩放
            markevery=1,
            markerfacecolor=face_col,
            markeredgecolor=edge_col,
            markeredgewidth=edge_width,  # <--- 使用独立的嵌入图边缘粗细
            zorder=10
        )

    # 嵌入图标签设置
    inset_ymin, inset_ymax = ax_ins.get_ylim()
    ax_ins.set_ylim(inset_ymin, inset_ymax * 1.12)

    ax_ins.set_xlabel("Fraction removed", fontsize=INSET_LABEL_SIZE,
                      labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)
    ax_ins.set_ylabel("Retention ratio", fontsize=INSET_LABEL_SIZE,
                      labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)

    # 嵌入图刻度限制 (防止刻度过密)
    ax_ins.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax_ins.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_ins.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_ins.tick_params(axis="both", which="major", direction="in",
                       top=False, right=False, bottom=True, left=True,
                       length=INSET_TICK_LENGTH, width=INSET_TICK_WIDTH,
                       pad=INSET_TICK_PAD,
                       labelsize=INSET_TICK_LABEL_SIZE, labelcolor=TEXT_COLOR)

    # --- 保存与显示 ---

    # 定义输出格式
    formats = ['png', 'pdf', 'svg', 'tiff', 'jpg']
    # 去除扩展名获取基础文件名
    base_filename = os.path.splitext(filename)[0]

    for fmt in formats:
        save_file = f"{base_filename}.{fmt}"
        save_path = os.path.join(OUTPUT_FIG_DIR, save_file)
        ax.set_position(AX_POSITION)

        # 保存图片，tiff 格式增加压缩参数
        if fmt == 'tiff':
            plt.savefig(save_path, dpi=DPI, format=fmt,
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(save_path, dpi=DPI, format=fmt)

        print(f"   已保存 ({fmt}): {save_path}")

    # 屏幕显示
    plt.show()
    plt.close()
    print("   绘图全部完成。\n")


def main():
    # 1. 加载所有数据
    node_data = load_data(NODE_STRATEGIES, INPUT_DIR)
    # 2. 执行绘图函数
    plot_node_robustness_efficiency(NODE_STRATEGIES, node_data, "Node_Functional_Robustness_V8.png")


# 程序入口检查
if __name__ == "__main__":
    main()

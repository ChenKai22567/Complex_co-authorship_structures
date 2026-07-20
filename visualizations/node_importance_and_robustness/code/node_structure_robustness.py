# -*- coding: utf-8 -*-
"""
plot_robustness_structural_final_v10.py

功能：
    绘制节点结构鲁棒性分析图（主图：LCC大小，嵌入图：连通分量数）。

修正点（本次）：
    1. 【透明度高级控制】：
       - 线条和Marker的彩色部分：半透明 (alpha=0.75)。
       - 空心Marker的白色填充 & 实心Marker的白色描边：完全不透明 (alpha=1.0)，确保遮挡背景。
    2. 【嵌入图独立配置】：
       - 嵌入图的 线宽 和 Marker边缘宽度 单独设置，不再依赖主图比例。
    3. 【基础保持】：保留了之前的去刻度、内朝向、框线保留等样式。

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
# 【新增】导入颜色转换工具，用于精确控制颜色的透明度通道 (RGBA)
from matplotlib.colors import to_rgba
# 导入警告管理模块
import warnings

# 忽略代码运行过程中的非致命警告，保持控制台输出整洁
warnings.filterwarnings("ignore")

# =========================
# 1) 全局路径与绘图风格设置
# =========================

# 定义输入数据和图片输出目录，统一放在 importance 下的“数据”“图像”文件夹中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORTANCE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(IMPORTANCE_DIR, "data")
OUTPUT_FIG_DIR = os.path.join(IMPORTANCE_DIR, "figures")
# 创建输出目录，如果目录已存在则不报错（exist_ok=True）
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

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

# 设置 Matplotlib 的全局字体为 Arial，这是学术论文中通用的无衬线字体
plt.rcParams["font.family"] = "Arial"
# 设置全局文本的基础颜色，与 fractal_rg_flow.py 保持一致
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
# 统一使用正常字重，避免图片中文字加粗
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"
# 设置绘图的分辨率为 600 DPI，确保输出图片高清，满足印刷要求
plt.rcParams["figure.dpi"] = DPI

# =========================
# 2) 攻击策略配置（增加 Marker 个性化配置）
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
    # 策略1：随机攻击（作为基准线）
    {"key": "random", "label": "Random", "color": "#999999", "style": "--", "file": "node_random_mean_curve.csv",
     "marker": "o", "ms": 4.5, "hollow": False},
    # 策略2：度中心性攻击
    {"key": "degree", "label": "Degree", "color": "#79B96C", "style": "-", "file": "node_degree_curve.csv",
     "marker": "s", "ms": 4, "hollow": False},
    # 策略3：强度（加权度）攻击
    {"key": "strength", "label": "Strength", "color": "#F46D43", "style": "-", "file": "node_strength_curve.csv",
     "marker": "^", "ms": 5.5, "hollow": False},
    # 策略4：介数中心性攻击（空心圆）
    {"key": "betweenness", "label": "Betweenness", "color": "#F8AD20", "style": "-",
     "file": "node_betweenness_curve.csv", "marker": "o", "ms": 4.5, "hollow": True},
    # 策略5：紧密中心性攻击（空心菱形）
    {"key": "closeness", "label": "Closeness", "color": "#4D97C6", "style": "-", "file": "node_closeness_curve.csv",
     "marker": "D", "ms": 3.0, "hollow": True},
    # 策略6：结构洞有效规模攻击
    {"key": "effective_size", "label": "Effective size", "color": "#9467bd", "style": "-",
     "file": "node_effective_size_curve.csv", "marker": "X", "ms": 5, "hollow": False},
]

# =========================
# 3) 绘图参数配置（独立控制区）
# =========================

# --- A. 全局透明度 ---
# 控制线条和彩色 Marker 部分的透明度 (0~1)，白色填充不受此影响
GLOBAL_ALPHA = 0.85

# --- B. 主图 (Main Plot) 配置 ---
# 主图坐标轴刻度线的长度
MAIN_TICK_LENGTH = 3
# 主图曲线的线条宽度
MAIN_LINE_WIDTH = 1.2
# 主图空心 Marker 的边缘线宽（稍粗，以便在白色填充下看清轮廓）
MAIN_HOLLOW_EDGE_WIDTH = 1.2
# 主图实心 Marker 的白色描边线宽（稍细，仅用于区分重叠）
MAIN_SOLID_EDGE_WIDTH = 0.5

# --- C. 嵌入图 (Inset Plot) 独立配置 ---
# 嵌入图坐标轴刻度线的长度
INSET_TICK_LENGTH = 2
# 嵌入图曲线的线条宽度 (独立设置，通常比主图细)
INSET_LINE_WIDTH = 1.0
# 嵌入图标记点的缩放比例（相对于主图标记大小）
INSET_MARKER_SCALE = 0.5
# 嵌入图空心 Marker 的边缘线宽 (独立设置，通常比主图细)
INSET_HOLLOW_EDGE_WIDTH = 0.8
# 嵌入图实心 Marker 的白色描边线宽 (独立设置)
INSET_SOLID_EDGE_WIDTH = 0.3


# =========================
# 4) 数据处理辅助函数
# =========================

def _get_removed_frac(df: pd.DataFrame) -> np.ndarray:
    """
    功能：从 DataFrame 中提取表示'移除比例'的列。
    兼容不同的列名写法（如 removed_frac, removed_percent）。
    """
    # 检查是否存在标准的 'removed_frac' 列
    if "removed_frac" in df.columns:
        # 转换为数值类型，将非法值（NaN）填充为0，并转为 numpy 数组
        x = pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).to_numpy()
        return x

    # 如果没找到，尝试检查其他可能的列名变体
    for c in ["removed_percent", "removal_percentage", "removed_percentage"]:
        if c in df.columns:
            # 同样转换为数值并填充0
            x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            # 如果数值范围是 0-100（百分比），则除以 100 归一化为 0-1
            if np.nanmax(x) > 1.0:
                x = x / 100.0
            return x

    # 如果所有列名都匹配失败，抛出异常提示用户
    raise ValueError("CSV 中找不到表示移除比例的列 (如 removed_frac)。")


def load_data(strategies, base_dir):
    """
    功能：根据策略配置，批量读取 CSV 文件。
    返回：一个字典，key 为策略名，value 为对应的 DataFrame。
    """
    data_dict = {}  # 初始化数据字典
    print(f"[加载数据] 正在从 {base_dir} 读取...")

    count = 0  # 成功加载的文件计数器
    for strat in strategies:
        # 拼接完整的文件路径
        path = os.path.join(base_dir, strat["file"])
        # 检查文件是否存在
        if os.path.exists(path):
            try:
                # 读取 CSV 文件
                data_dict[strat["key"]] = pd.read_csv(path)
                count += 1  # 计数加一
            except Exception:
                pass  # 如果读取出错（如文件损坏），跳过该文件

    print(f"   成功加载 {count} 个文件。\n")
    return data_dict


# =========================
# 5) 轴美化工具（核心修改：恢复框线）
# =========================

def beautify_axis_box(ax, tick_length=4, tick_width=0.8, tick_pad=3):
    """
    功能：美化坐标轴。
    修正：保留四周边框线，但只在左侧和下方显示刻度线。
    """
    # 设置刻度参数：
    # direction='in': 刻度线朝内
    # top=False, right=False: 【关键修改】关闭上方和右侧的刻度线（小短线）
    # left=True, bottom=True: 开启左侧和下方的刻度线
    # length: 刻度线长度
    # width: 刻度线宽度
    ax.tick_params(direction='in', top=False, right=False, left=True, bottom=True,
                   length=tick_length, width=tick_width, colors=TEXT_COLOR,
                   pad=tick_pad)

    # 设置所有边框线（Spines）的颜色
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['right'].set_color(TEXT_COLOR)
    ax.spines['top'].set_color(TEXT_COLOR)
    ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['right'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['top'].set_linewidth(AXIS_LINE_WIDTH)

    # 确保所有边框线（矩形框）都是可见的
    # 这样就保留了“框”，但没有了上方和右侧的“刻度”
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)


# =========================
# 6) 核心绘图逻辑
# =========================

def plot_node_robustness_clean(strategies, data_dict, filename):
    """
    功能：执行具体的绘图操作，包括主图和嵌入图。
    """
    print(f"[绘制] {filename}")

    # 创建图形画布，设置尺寸为 5x4 英寸，启用约束布局自动调整间距
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_position(AX_POSITION)

    # 调用美化函数，设置主图为全框线风格，刻度长度为配置值
    beautify_axis_box(ax, tick_length=MAIN_TICK_LENGTH, tick_width=MAIN_TICK_WIDTH,
                      tick_pad=MAIN_TICK_PAD)

    # 设置主图 X 轴显示范围 (0 到 1)
    ax.set_xlim(0.0, 1.0)
    # 设置主图 Y 轴显示范围 (略微扩展以容纳 0 和 1 的标记)
    ax.set_ylim(-0.05, 1.0)

    # --- 主图绘图循环 ---
    # 遍历每个策略配置进行绘图
    for strat in strategies:
        key = strat["key"]
        # 如果该策略的数据未加载，则跳过
        if key not in data_dict: continue

        # 获取 DataFrame
        df = data_dict[key]

        # 获取 X 轴数据，并进行降采样（[::2] 每隔一个取一个点，使标记不拥挤）
        x = _get_removed_frac(df)[::2]
        # 获取 Y 轴数据 (LCC比例)，同样降采样
        y = pd.to_numeric(df["lcc_ratio"], errors="coerce").fillna(np.nan).to_numpy()[::2]

        # === 透明度与颜色处理 ===
        # 1. 计算带透明度的基础颜色 (用于线条和彩色Marker部分)
        # to_rgba 可以将十六进制颜色转换为 (R, G, B, Alpha)
        transparent_base_color = to_rgba(strat['color'], alpha=GLOBAL_ALPHA)

        # 2. Marker 样式逻辑 (区分空心/实心)
        if strat['hollow']:
            # 空心 Marker 配置：
            # face_col='white': 强制白色不透明，遮挡背景网格
            face_col = 'white'
            # edge_col: 边缘颜色使用半透明的策略颜色
            edge_col = transparent_base_color
            # edge_width: 使用【主图】配置的空心边缘线宽
            edge_width = MAIN_HOLLOW_EDGE_WIDTH
        else:
            # 实心 Marker 配置：
            # face_col: 填充颜色使用半透明的策略颜色
            face_col = transparent_base_color
            # edge_col='white': 边缘使用白色不透明描边，增加对比度
            edge_col = 'white'
            # edge_width: 使用【主图】配置的实心边缘线宽
            edge_width = MAIN_SOLID_EDGE_WIDTH

        # 执行绘图命令
        ax.plot(
            x, y,
            label=strat["label"],       # 图例标签
            color=transparent_base_color, # 线条颜色 (带透明度)
            linestyle=strat["style"],   # 线条样式
            # 线宽：随机攻击线略粗 (基准线)，其他略细
            linewidth=MAIN_LINE_WIDTH if key != "random" else MAIN_LINE_WIDTH + 0.5,
            # 【重要】移除全局 alpha 参数，否则 'white' 也会变成半透明
            # alpha=1.0,
            marker=strat["marker"],     # 标记形状
            markersize=strat["ms"],     # 标记大小
            markevery=1,                # 每个采样点都绘制标记
            markerfacecolor=face_col,   # 标记填充色 (已处理透明度)
            markeredgecolor=edge_col,   # 标记边缘色 (已处理透明度)
            markeredgewidth=edge_width, # 标记边缘线宽 (使用主图配置)
            zorder=10                   # 绘图层级设为 10，确保覆盖网格线
        )

    # 设置主图 X 轴标签，颜色深灰，距离轴线 8 像素
    ax.set_xlabel("Fraction of nodes removed", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_X_LABEL_PAD)
    # 设置主图 Y 轴标签
    ax.set_ylabel("Relative size of LCC", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_Y_LABEL_PAD)

    # 设置图例
    legend = ax.legend(
        loc='lower right',  # 位置：右下角
        bbox_to_anchor=(0.99, 0.17),  # 精确坐标微调，保持与嵌入图标签分离
        frameon=False,  # 关闭图例边框
        fontsize=MAIN_LEGEND_SIZE,  # 图例字号
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.35,
        borderaxespad=0.0,
        markerscale=0.85
    )
    # 遍历图例文本，强制设置为深灰色
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # --- 嵌入图绘图循环 ---

    # 创建嵌入坐标轴对象 [x, y, width, height] (相对于主图坐标系的比例)
    ax_ins = ax.inset_axes(INSET_POS)

    # 对嵌入图也应用美化函数（恢复框线，刻度朝内，但无右上刻度）
    beautify_axis_box(ax_ins, tick_length=INSET_TICK_LENGTH,
                      tick_width=INSET_TICK_WIDTH, tick_pad=INSET_TICK_PAD)

    # 设置嵌入图背景为不透明白色，防止下方主图线条透出干扰
    ax_ins.set_facecolor("white")
    # 设置嵌入图 X 轴范围
    ax_ins.set_xlim(0.0, 1.0)

    # 遍历策略绘制嵌入图曲线
    for strat in strategies:
        key = strat["key"]
        if key not in data_dict: continue
        df = data_dict[key]

        # 获取数据并降采样
        x = _get_removed_frac(df)[::2]
        # 获取连通分量数数据
        y2 = pd.to_numeric(df["num_components"], errors="coerce").fillna(np.nan).to_numpy()[::2]

        # === 颜色与透明度处理 (逻辑同主图) ===
        transparent_base_color = to_rgba(strat['color'], alpha=GLOBAL_ALPHA)

        # 嵌入图 Marker 样式逻辑
        if strat['hollow']:
            face_col = 'white'
            edge_col = transparent_base_color
            # 【关键修改】使用独立的【嵌入图】空心边缘线宽
            edge_width = INSET_HOLLOW_EDGE_WIDTH
        else:
            face_col = transparent_base_color
            edge_col = 'white'
            # 【关键修改】使用独立的【嵌入图】实心边缘线宽
            edge_width = INSET_SOLID_EDGE_WIDTH

        # 绘制嵌入图曲线
        ax_ins.plot(
            x, y2,
            color=transparent_base_color, # 线条颜色 (带透明度)
            linestyle=strat["style"],
            # 【关键修改】使用独立的【嵌入图】线宽
            linewidth=INSET_LINE_WIDTH,
            # 移除全局 alpha
            marker=strat["marker"],  # 标记形状
            # 标记大小按比例缩小，适应较小的嵌入图
            markersize=strat["ms"] * INSET_MARKER_SCALE,
            markevery=1,
            markerfacecolor=face_col,   # 填充色
            markeredgecolor=edge_col,   # 边缘色
            markeredgewidth=edge_width, # 边缘线宽 (使用嵌入图独立配置)
            zorder=10
        )

    inset_ymin, inset_ymax = ax_ins.get_ylim()
    ax_ins.set_ylim(inset_ymin, inset_ymax * 1.12)

    # 设置嵌入图 X 轴标签
    ax_ins.set_xlabel("Fraction removed", fontsize=INSET_LABEL_SIZE, labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)
    # 设置嵌入图 Y 轴标签
    ax_ins.set_ylabel("Components", fontsize=INSET_LABEL_SIZE, labelpad=INSET_LABEL_PAD, color=TEXT_COLOR)

    # 设置嵌入图 X 轴刻度：只显示 0.0、0.5、1.0
    ax_ins.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax_ins.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # 限制嵌入图 Y 轴最多显示 4 个刻度
    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_ins.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    # 调整嵌入图刻度参数：字号和颜色
    ax_ins.tick_params(axis="both", which="major", direction="in",
                       top=False, right=False, bottom=True, left=True,
                       length=INSET_TICK_LENGTH, width=INSET_TICK_WIDTH,
                       pad=INSET_TICK_PAD,
                       labelsize=INSET_TICK_LABEL_SIZE, labelcolor=TEXT_COLOR)

    # --- 8. 保存与显示 (支持多格式输出) ---

    # 定义要保存的格式列表
    # 'png': 通用预览
    # 'pdf': LaTeX 排版最佳（矢量）
    # 'svg': 网页或后续用 Adobe Illustrator/Inkscape 编辑（矢量）
    # 'tiff': 很多老牌期刊要求的高清位图
    formats = ['png', 'pdf', 'svg', 'tiff', 'jpg']

    # 移除文件名的后缀（如果有），以便我们添加自己的后缀
    base_filename = os.path.splitext(filename)[0]

    for fmt in formats:
        # 拼接完整路径
        save_file = f"{base_filename}.{fmt}"
        save_path = os.path.join(OUTPUT_FIG_DIR, save_file)

        ax.set_position(AX_POSITION)
        # 保存图片
        # dpi=600: 满足绝大多数期刊对位图(tiff/png)的要求 (通常要求 300-600)
        # bbox_inches='tight': 自动剪裁白边
        # format=fmt: 显式指定格式
        # pil_kwargs: 针对 tiff 的压缩选项（可选，防止文件过大）
        if fmt == 'tiff':
            plt.savefig(save_path, dpi=DPI, format=fmt,
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(save_path, dpi=DPI, format=fmt)

        print(f"   已保存 ({fmt}): {save_path}")

    # 屏幕显示（使用默认后端）
    plt.show()
    # 关闭图形释放内存
    plt.close()
    print("   绘图全部完成。\n")


def main():
    # 1. 加载所有数据
    node_data = load_data(NODE_STRATEGIES, INPUT_DIR)
    # 2. 执行绘图函数
    plot_node_robustness_clean(NODE_STRATEGIES, node_data, "Node_Robustness_Clean_Style_v10.png")


# 程序入口检查
if __name__ == "__main__":
    main()

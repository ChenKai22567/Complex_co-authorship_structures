# -*- coding: utf-8 -*-
"""
Plot Script: RWCP Profile + Heatmap (Final Optimized with Scatter Types & Shared Zero)
绘图脚本：RWCP (Rich-Club/Core-Periphery) 轮廓图 + 块模型热力图 (最终优化版-定制散点样式-共用0刻度)

================================================================================
【功能概述】
本脚本用于可视化网络的核-边缘 (Core-Periphery) 结构特征，主要包含两部分：
1. 主图 (Line Plot)：展示 Coreness Profile ($\alpha_k$) 随节点移除比例 ($k/n$) 的变化趋势。
   - 指标：$\alpha_k$ (归一化的富人俱乐部系数/核心度)。
   - 计算方法：计算前 $k$ 个最高度节点内部的边权重总和，通过零模型归一化。
   - 包含加权 (Weighted) 与未加权 (Unweighted) 两种曲线。
2. 嵌入图 (Inset Heatmap)：展示不同层级 (Core, Semi-periphery, Periphery) 间的连接密度或强度。
   - 指标：块模型矩阵 $M_{ij}$。
   - 计算方法：统计属于层级 $i$ 和层级 $j$ 的节点之间的边权重总和 (或密度)。

【本次修改内容】
1. 散点样式定制：蓝色曲线使用空心圆圈，橙色曲线使用三角形。
2. 保持 20 个等距散点标记。
3. **坐标轴优化**：隐藏 X/Y 轴起点的 0.0，并在左下角手动添加共用 "0"。
4. 全文详细中文注释。
================================================================================

Inputs:
- Node CSV (metrics): 包含节点层级归属 (Core/Periphery 等)。
- Edgelist CSV: 网络的边列表 (u, v, weight)。
- Profile CSV: 预计算好的 Profile 曲线数据 (k_over_n, alpha)。
"""

# =======================
# 0) 参数配置区域 (PARAMETERS)
# =======================
import os

# --- 输入文件路径 (请根据实际情况修改路径) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPLEMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SUPPLEMENT_DIR, "data")
FIGURE_DIR = os.path.join(SUPPLEMENT_DIR, "figures")
NODE_CSV_PATH = os.path.join(DATA_DIR, "RWCP_3layer_node_roles_with_metrics.csv")
EDGELIST_PATH = os.path.join(DATA_DIR, "filtered_graph_edgelist.csv")
PROFILE_CSV_PATH = os.path.join(DATA_DIR, "RWCP_profile_alpha_k_weighted_vs_unweighted.csv")

# --- 输出设置 ---
# 输出目录 (若为空则自动生成)
OUT_DIR = FIGURE_DIR
# 图片导出 DPI (分辨率)
EXPORT_DPI = 600
# 导出格式列表
EXPORT_FORMATS = ['png', 'pdf', 'svg', 'tiff', 'jpg']
# 是否在运行后弹窗显示图片
SHOW_PLOT_WINDOW = False

# --- 列名配置 (需与 CSV 文件头一致) ---
NODE_AUTHOR_COL = "author"  # 节点名称列
NODE_LAYER_COL = "cp_layer"  # 节点所属层级列
EDGE_U_COL = "u"  # 起点列
EDGE_V_COL = "v"  # 终点列
EDGE_W_COL = "weight"  # 权重列
PROFILE_X_COL = "k_over_n"  # X轴数据列：节点移除比例
PROFILE_W_COL = "alpha_weighted"  # Y轴数据列：加权 Alpha 值
PROFILE_B_COL = "alpha_unweighted"  # Y轴数据列：未加权 Alpha 值

# --- 绘图逻辑配置 ---
# 层级顺序 (用于矩阵排序)
LAYER_ORDER = ["core", "semi_core", "periphery"]
# 热力图计算模式 ('sum' 为权重和, 'density' 为密度)
HEATMAP_MODE = "sum"
# 热力图是否对数据做 log(x+1) 处理以增强对比度
HEATMAP_LOG1P = True
# 是否绘制加权曲线
PLOT_WEIGHTED_PROFILE = True
# 是否绘制未加权曲线
PLOT_UNWEIGHTED_PROFILE = True

# --- 视觉样式配置 ---
FONT_FAMILY = 'Arial'  # 全局字体
TEXT_COLOR = "#444444"
FIG_SIZE = (4.25, 3.12)
AX_POSITION = [0.17, 0.25, 0.508, 0.692]
AXIS_LINE_WIDTH = 0.8
MAIN_TICK_WIDTH = 0.8
INSET_TICK_WIDTH = 0.4
MAIN_LABEL_SIZE = 12
MAIN_TICK_LABEL_SIZE = 10
MAIN_LEGEND_SIZE = 6.5
INSET_TICK_LABEL_SIZE = 7
MAIN_X_LABEL_PAD = 7
MAIN_Y_LABEL_PAD = 10
MAIN_TICK_PAD = 3
MAIN_X_TICK_PAD = 5
COLOR_AXIS_TEXT = TEXT_COLOR
COLOR_SPINE = TEXT_COLOR
COLOR_LEGEND_TEXT = TEXT_COLOR
PALE_BLUE_BG = "#f4f9fd"  # 热力图背景极浅蓝色

# 区域网格颜色与边界 (背景划分)
ZONE_BOUNDARIES = [0, 0.35, 0.85, 1.0]  # 背景区域的X轴分割点
ZONE_COLORS = ["#1f77b4", "#BBBBBB", "#ed7d2f"]  # 对应区域的颜色 (蓝, 灰, 橙)
ZONE_BOUNDARY_ALPHA = 0.55
ZONE_BOUNDARY_LINE_WIDTH = 0.9
ZONE_BOUNDARY_DASHES = (3, 2)

# 嵌入图 (Inset) 的位置与大小 [x, y, width, height] (归一化坐标)
INSET_RECT = (0.16, 0.65, 0.30, 0.30)

# =======================
# 1) 导入与全局设置
# =======================
from pathlib import Path  # 用于处理文件路径
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图主库
import matplotlib.ticker as ticker  # 【新增】刻度格式化工具
import matplotlib.colors as mcolors  # 颜色处理模块
import matplotlib.patches as mpatches  # 图形块模块
import matplotlib.patheffects as pe  # 路径效果模块

# --- Matplotlib 全局参数设置 ---
plt.rcParams['font.family'] = FONT_FAMILY  # 设置字体
plt.rcParams['text.color'] = COLOR_AXIS_TEXT  # 设置文本颜色
plt.rcParams['axes.labelcolor'] = COLOR_AXIS_TEXT  # 设置轴标签颜色
plt.rcParams['xtick.color'] = COLOR_AXIS_TEXT  # 设置X轴刻度颜色
plt.rcParams['ytick.color'] = COLOR_AXIS_TEXT  # 设置Y轴刻度颜色
plt.rcParams['axes.edgecolor'] = COLOR_SPINE  # 设置轴边框颜色
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
# 2) 数据处理函数
# =======================
def read_and_collapse_edgelist(path: str) -> pd.DataFrame:
    """
    读取并压缩边列表。
    功能：
    1. 读取 CSV，统一列名为小写。
    2. 过滤掉缺失值和权重非正的边。
    3. 去除自环 (Self-loops)。
    4. 将多重边 (Multi-edges) 的权重合并相加 (无向图逻辑)。
    """
    df = pd.read_csv(path)  # 读取文件
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})  # 清洗列名

    # 检查必要列是否存在
    req = {EDGE_U_COL, EDGE_V_COL, EDGE_W_COL}
    if not req.issubset(df.columns):
        raise ValueError(f"Edgelist columns missing. Need {req}")  # 报错提示

    # 数据清洗：去除空值
    df = df.dropna(subset=[EDGE_U_COL, EDGE_V_COL, EDGE_W_COL]).copy()
    # 转换为字符串并去空格
    df[EDGE_U_COL] = df[EDGE_U_COL].astype(str).str.strip()
    df[EDGE_V_COL] = df[EDGE_V_COL].astype(str).str.strip()
    # 转换权重为数值
    df[EDGE_W_COL] = pd.to_numeric(df[EDGE_W_COL], errors="coerce")
    # 过滤掉权重<=0的边
    df = df[df[EDGE_W_COL] > 0].copy()
    # 过滤掉自环 (起点=终点)
    df = df[df[EDGE_U_COL] != df[EDGE_V_COL]].copy()

    # 无向化处理：对每一行的 u, v 进行排序，确保 (A, B) 和 (B, A) 视为同一条边
    uv = np.sort(df[[EDGE_U_COL, EDGE_V_COL]].to_numpy(), axis=1)
    df["a"], df["b"] = uv[:, 0], uv[:, 1]

    # 聚合：按起点终点分组，权重求和
    return df.groupby(["a", "b"], as_index=False)[EDGE_W_COL].sum().rename(columns={"a": EDGE_U_COL, "b": EDGE_V_COL})


def compute_block_matrix(df_edges, author_to_layer, layer_order, mode="sum"):
    """
    计算块模型矩阵 (Block Matrix)。
    功能：统计各层级之间 (如 Core-Core, Core-Periphery) 的连接强度。
    参数：
    - df_edges: 边列表 DataFrame
    - author_to_layer: 节点到层级的映射字典
    - layer_order: 层级排序列表
    - mode: 'sum' (总权重) 或 'density' (连接密度)
    """
    layers = layer_order
    L = len(layers)
    # 建立层级名称到索引的映射
    layer_to_idx = {l: i for i, l in enumerate(layers)}
    # 统计各层级的节点数量
    sizes = {l: 0 for l in layers}
    for lay in author_to_layer.values():
        if lay in sizes: sizes[lay] += 1

    # 初始化 L x L 的零矩阵
    M = np.zeros((L, L), dtype=float)
    used = 0  # 计数器 (调试用)

    # 遍历每一条边，累加权重到对应矩阵块
    for u, v, w in zip(df_edges[EDGE_U_COL], df_edges[EDGE_V_COL], df_edges[EDGE_W_COL]):
        lu, lv = author_to_layer.get(u), author_to_layer.get(v)
        # 确保边的两个端点都在已知的层级中
        if (lu in layer_to_idx) and (lv in layer_to_idx):
            i, j = layer_to_idx[lu], layer_to_idx[lv]
            M[i, j] += float(w)
            # 矩阵对称化 (无向图)
            if i != j: M[j, i] += float(w)
            used += 1

    # 如果模式是密度，则除以该块可能存在的最大边数
    if mode == "density":
        for i, li in enumerate(layers):
            for j, lj in enumerate(layers):
                # 计算分母：对角线块为 n*(n-1)/2，非对角线块为 n1*n2
                denom = (sizes[li] * (sizes[li] - 1) / 2.0) if i == j else (sizes[li] * sizes[lj])
                M[i, j] = (M[i, j] / denom) if denom > 0 else 0.0
    return M, sizes, used


# =======================
# 3) 核心绘图函数
# =======================
def plot_styled_profile(df_profile, M, layer_labels, out_base_path, log1p=True):
    """
    绘制主图：Profile 曲线图 + 嵌入热力图。
    """
    # 创建画布，指定特定尺寸 (单位英寸)
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=EXPORT_DPI)
    ax.set_position(AX_POSITION)

    # --- A. 坐标轴与边框样式 ---
    # 去掉上方和右侧的刻度线，保留边框
    ax.tick_params(direction='in', top=False, right=False,
                   colors=COLOR_AXIS_TEXT, labelcolor=COLOR_AXIS_TEXT,
                   width=MAIN_TICK_WIDTH, length=3, pad=MAIN_TICK_PAD,
                   labelsize=MAIN_TICK_LABEL_SIZE)
    ax.tick_params(axis='x', pad=MAIN_X_TICK_PAD)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_SPINE)  # 设置边框颜色
        spine.set_linewidth(AXIS_LINE_WIDTH)
        spine.set_zorder(100)

    # --- B. 绘制背景区域 (网格纹理) ---
    # Draw compact dashed region boundaries instead of textured backgrounds.

    # Core and periphery boundary markers.
    for x_boundary, y_start, color in [
        (ZONE_BOUNDARIES[1], 0.0, ZONE_COLORS[0]),
        (ZONE_BOUNDARIES[2], 0.18, ZONE_COLORS[2]),
    ]:
        ax.plot([x_boundary, x_boundary], [y_start, x_boundary],
                linestyle='--', dashes=ZONE_BOUNDARY_DASHES,
                color=color, linewidth=ZONE_BOUNDARY_LINE_WIDTH,
                alpha=ZONE_BOUNDARY_ALPHA, zorder=4)

    # 绘制参考对角线 (y=x)，灰色虚线
    ax.plot([0, 1], [0, 1], linestyle='--', color='#777777', linewidth=1.2, alpha=0.6, zorder=5)

    # --- C. 绘制主数据曲线与新增散点 ---
    # 获取 X 轴数据 (节点移除比例)
    x = df_profile[PROFILE_X_COL].to_numpy(dtype=float)

    # 【新增功能】 生成20个等距的X轴坐标点，用于绘制散点
    num_scatter_points = 20
    x_scatter_target = np.linspace(0, 1.0, num_scatter_points)

    # 1. 绘制 Weighted 曲线 (加权)
    if PLOT_WEIGHTED_PROFILE and (PROFILE_W_COL in df_profile.columns):
        y_w = df_profile[PROFILE_W_COL].to_numpy(dtype=float)
        # 绘制连线
        ax.plot(x, y_w, label="Weighted", color="#1f77b4", linewidth=1.2, alpha=0.8, zorder=10)

        # 【新增功能】 计算 Weighted 对应的插值点并绘制散点
        # 使用 np.interp 确保点在 X 轴上严格等距
        y_scatter_w = np.interp(x_scatter_target, x, y_w)

        # 修改：蓝色使用空心圆圈
        # facecolors='none': 内部无填充, edgecolors: 边缘颜色
        ax.scatter(x_scatter_target, y_scatter_w, edgecolors="#1f77b4",
                   facecolors='w', marker='s', s=20, linewidths=1.5, alpha=0.9, zorder=12)

    # 2. 绘制 Unweighted 曲线 (未加权)
    if PLOT_UNWEIGHTED_PROFILE and (PROFILE_B_COL in df_profile.columns):
        y_b = df_profile[PROFILE_B_COL].to_numpy(dtype=float)
        # 绘制连线 (虚线)
        ax.plot(x, y_b, label="Unweighted", color="#ff7f0e", linewidth=1.2, linestyle="--", alpha=0.8, zorder=11)

        # 【新增功能】 计算 Unweighted 对应的插值点并绘制散点
        # 使用 np.interp 确保点在 X 轴上严格等距
        y_scatter_b = np.interp(x_scatter_target, x, y_b)

        # 修改：橙色使用三角形
        # marker='^': 上三角
        ax.scatter(x_scatter_target, y_scatter_b, facecolors='w', color="#ff7f0e", marker='o', s=25, alpha=0.9,
                   zorder=11)

    # --- 【新增】坐标轴刻度与共用原点 "0" ---

    # 1. 定义格式化函数：如果数值接近0，返回空字符串；否则返回保留1位小数的字符串
    def no_zero_fmt(x, pos):
        if np.isclose(x, 0): return ""  # 隐藏 0.0
        return f"{x:.1f}"

    # 2. 应用格式化器到 X 和 Y 轴
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(no_zero_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(no_zero_fmt))

    # 3. 手动添加共用的 "0"
    # (-0.03, -0.03) 是数据坐标，稍稍偏离原点
    ax.text(-0.01, -0.01, "0", ha='right', va='top', fontsize=MAIN_TICK_LABEL_SIZE,
            color=COLOR_AXIS_TEXT, zorder=200)

    # 设置轴标签 (支持 LaTeX 格式)
    ax.set_xlabel("Proportion of nodes removed ($k/n$)", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_X_LABEL_PAD)
    ax.set_ylabel(r"Coreness profile, $\alpha_k$", fontsize=MAIN_LABEL_SIZE,
                  color=TEXT_COLOR, labelpad=MAIN_Y_LABEL_PAD)
    # 设置轴范围
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # 添加图例
    leg = ax.legend(loc='lower right', frameon=False, fontsize=MAIN_LEGEND_SIZE)
    for text in leg.get_texts():
        text.set_color(COLOR_LEGEND_TEXT)  # 设置图例文字颜色

    # --- D. 嵌入热力图 (Inset Heatmap) ---
    # 创建嵌入轴对象
    axins = ax.inset_axes(INSET_RECT)
    axins.set_facecolor(PALE_BLUE_BG)  # 设置嵌入图背景色

    # 自定义色板 (极浅蓝 -> 深蓝)
    cmap_colors = [PALE_BLUE_BG, "#d0e1f0", "#4d97cd"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("CustomBlues", cmap_colors)

    # 数据预处理：是否进行 log(x+1) 变换
    H = np.log1p(M) if log1p else M
    # 绘制热力图矩阵
    im = axins.imshow(H, aspect="equal", interpolation="nearest", cmap=custom_cmap)

    # 嵌入图样式设置
    axins.tick_params(direction='in', color=COLOR_SPINE, labelcolor=COLOR_AXIS_TEXT,
                      width=INSET_TICK_WIDTH, length=2, pad=3,
                      top=False, right=True, left=False, bottom=True,
                      labelsize=INSET_TICK_LABEL_SIZE)
    for spine in axins.spines.values():
        spine.set_edgecolor(COLOR_SPINE)
        spine.set_linewidth(AXIS_LINE_WIDTH)

    # 处理轴标签 (替换下划线，改名)
    labels_clean = []
    for l in layer_labels:
        clean_l = l.replace('_', '-')
        if clean_l == 'semi-core':
            clean_l = 'semi-periphery'
        clean_l = clean_l[:1].upper() + clean_l[1:]
        labels_clean.append(clean_l)

    # 设置刻度位置
    axins.set_xticks(range(len(labels_clean)))
    axins.set_yticks(range(len(labels_clean)))

    # 将 Y 轴标签移至右侧
    axins.yaxis.tick_right()

    # 设置刻度文字及旋转角度
    axins.set_xticklabels(labels_clean, rotation=30, ha="right")
    axins.set_yticklabels(labels_clean, rotation=0, va="center")

    # --- E. Colorbar (调整位置与距离) ---
    # [位置参数] [x, y, width, height] (相对于 axins 的坐标)
    # x=-0.12 使其紧贴热力图左侧
    cax = axins.inset_axes([-0.12, 0, 0.05, 1])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')

    # Colorbar 样式微调
    cbar.outline.set_visible(False)  # 去掉外框
    cax.yaxis.set_ticks_position('left')  # 刻度在左侧
    cbar.ax.tick_params(direction='in', size=3, width=INSET_TICK_WIDTH,
                        labelsize=INSET_TICK_LABEL_SIZE, labelcolor=COLOR_AXIS_TEXT)

    # --- F. 多格式导出 ---
    print("-" * 30)
    for fmt in EXPORT_FORMATS:
        save_path = out_base_path.with_suffix(f'.{fmt}')  # 构造保存路径
        ax.set_position(AX_POSITION)
        if fmt == "tiff":
            plt.savefig(save_path, dpi=EXPORT_DPI, format=fmt,
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(save_path, dpi=EXPORT_DPI, format=fmt)
        print(f"Saved [{fmt}]: {save_path.name}")  # 打印保存信息
    print("-" * 30)

    # 显示或关闭窗口
    if SHOW_PLOT_WINDOW:
        plt.show()
    else:
        plt.close(fig)


# =======================
# 4) 主程序入口
# =======================
def main():
    """
    主执行逻辑。
    """
    node_csv = Path(NODE_CSV_PATH)  # 转换路径对象
    if not node_csv.exists():
        print(f"Error: {node_csv} not found.")  # 文件检查
        return

    # 设置输出目录
    out_dir = Path(OUT_DIR) if OUT_DIR else (node_csv.parent / "PLOTS_profile_styled_final")
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

    # 1. 加载数据
    print("Reading data...")
    df_nodes = pd.read_csv(node_csv)
    # 转换为字符串防止 ID 错乱
    df_nodes[NODE_AUTHOR_COL] = df_nodes[NODE_AUTHOR_COL].astype(str).str.strip()
    df_nodes[NODE_LAYER_COL] = df_nodes[NODE_LAYER_COL].astype(str).str.strip()
    # 建立 Author -> Layer 的映射字典
    author_to_layer = dict(zip(df_nodes[NODE_AUTHOR_COL], df_nodes[NODE_LAYER_COL]))

    # 读取边列表和 Profile 数据
    df_edges = read_and_collapse_edgelist(EDGELIST_PATH)
    df_prof = pd.read_csv(PROFILE_CSV_PATH)

    # 2. 计算矩阵
    print("Computing matrix...")
    M, _, _ = compute_block_matrix(df_edges, author_to_layer, LAYER_ORDER, mode=HEATMAP_MODE)

    # 保存计算出的矩阵数据 CSV，以备检查
    matrix_path = Path(DATA_DIR) / "3block_matrix.csv"
    pd.DataFrame(M, index=LAYER_ORDER, columns=LAYER_ORDER).to_csv(matrix_path, encoding="utf-8-sig")

    # 3. 绘图
    print("Plotting...")
    # 设置输出文件名基座
    out_base = out_dir / "RWCP_Profile_MultiFormat_v9_Grid_Scatter_Shapes"
    # 调用绘图函数
    plot_styled_profile(df_prof, M, LAYER_ORDER, out_base, log1p=HEATMAP_LOG1P)

    print("Done.")


# 脚本入口判断
if __name__ == "__main__":
    main()

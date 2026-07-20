import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress, t
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import warnings

# 忽略运行过程中的警告信息（如除零警告等），保持控制台输出清洁
warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局配置 (Configuration) - 基础设置区
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

script_dir = os.path.dirname(os.path.abspath(__file__))
fractal_dir = os.path.dirname(script_dir)
input_dir = os.path.join(fractal_dir, "data")
output_dir = os.path.join(fractal_dir, "figures")

# 输入数据文件路径 (必须是包含 BoxSize 和 MinBoxes 列的 CSV)
INPUT_FILE = os.path.join(input_dir, "fractal_data_continuous.csv")
OUTPUT_DIR = output_dir

# 基础文件名 (不带后缀)
FILENAME_BASE = "figure_fractal_final"
# 【样式字典】
# 定义颜色、形状和拟合范围。
# 注意：点的大小(size)和线宽(width)不在这里，而在 main 函数中单独控制。
STYLES = {
    "Overall": {
        "marker": "s",  # 方块
        "color": "#888888",  # 灰色
        "label": "Overall",
        "mfc": "#888888",  # 标记填充色 (实心)
        "fit_start": 1,  # 拟合起点 (Box Size >= 4)
        "fit_limit": 10  # 拟合终点 (None代表直到最后)
    },
    "Active": {
        "marker": "o",  # 圆点
        "color": "#0172be",  # 蓝色
        "label": "Active",
        "mfc": "w",  # 【关键】白色填充 (空心效果)
        "fit_start": 1,
        "fit_limit": 9
    },
    "Backbone": {
        "marker": "^",  # 三角
        "color": "#ed7d2f",  # 橙色
        "label": "Backbone",
        "mfc": "#ed7d2f",  # 标记填充色 (实心)
        "fit_start": 2,
        "fit_limit": None
    }
}


# ==========================================
# 2. 核心计算与绘图函数 (Core Plotting Logic)
# ==========================================
def fit_and_plot(ax, x, y, style_dict, scatter_size, line_width,
                 plot_ci=True, intercept_shift=0, scatter_edge_width=1):
    """
    通用绘图引擎：处理数据筛选、回归计算、拟合线平移、CI计算及最终绘图。

    参数详解：
    - scatter_size: 散点大小 (s)
    - line_width: 拟合线宽度 (linewidth)
    - plot_ci: 是否绘制置信区间 (True/False)
    - intercept_shift: 【微调参数】截距偏移量。
      正数(如0.5)会使拟合线向下平移；0表示不平移。用于避免线遮挡点。
    - scatter_edge_width: 【微调参数】散点描边宽度。
      设为 0 去除描边(适合实心点)；设为 >0 增加描边(适合空心点)。
    """

    # --- Step A: 数据筛选 (Data Filtering) ---
    upper = style_dict.get('fit_limit')
    lower = style_dict.get('fit_start')

    # 创建掩码筛选出在 [fit_start, fit_limit] 范围内的数据
    mask = np.ones(len(x), dtype=bool)
    if upper: mask &= (x <= upper)
    if lower: mask &= (x >= lower)

    x_fit, y_fit = x[mask], y[mask]

    # 点太少无法拟合，直接返回
    if len(x_fit) < 3: return None

    # --- Step B: 线性回归 (Linear Regression in Log-Log) ---
    X_log, Y_log = np.log(x_fit), np.log(y_fit)
    slope, intercept, r_val, _, _ = linregress(X_log, Y_log)
    dB, r2 = -slope, r_val ** 2

    # --- Step C: 生成平滑拟合线 (Line Generation) ---
    # 生成 100 个平滑点用于画线
    X_smooth_log = np.linspace(X_log.min(), X_log.max(), 100)

    # 【关键逻辑】：应用截距偏移 (intercept_shift)
    # y_new = slope*x + (intercept - shift)
    y_smooth_log = slope * X_smooth_log + (intercept - intercept_shift)

    # 还原回线性坐标系
    x_smooth, y_smooth = np.exp(X_smooth_log), np.exp(y_smooth_log)

    color = style_dict['color']

    # --- Step D: 绘制置信区间 (Confidence Interval) ---
    if plot_ci:
        n = len(X_log)
        # 误差计算基于原始截距 (未偏移的)
        y_pred = slope * X_log + intercept
        mse = np.sum((Y_log - y_pred) ** 2) / (n - 2)
        x_mean = np.mean(X_log)
        Sxx = np.sum((X_log - x_mean) ** 2)
        se = np.sqrt(mse * (1 / n + (X_smooth_log - x_mean) ** 2 / Sxx))
        t_v = t.ppf(0.975, n - 2)  # 95% CI

        # 上下界基于平滑线 (如果线平移了，CI带也跟着平移)
        upper = np.exp(y_smooth_log + t_v * se)
        lower = np.exp(y_smooth_log - t_v * se)

        # 绘制半透明阴影
        ax.fill_between(x_smooth, lower, upper, color=color, alpha=0.15, linewidth=0, zorder=1)

    # --- Step E: 实际绘图 (Drawing) ---
    # 1. 画拟合线 (虚线)
    ax.plot(x_smooth, y_smooth, color=color, linestyle='--',
            linewidth=line_width, alpha=0.8, zorder=1.5)

    # 2. 画散点
    # 使用传入的 scatter_size 和 scatter_edge_width
    ax.scatter(x, y, s=scatter_size, c=style_dict['mfc'], edgecolor=color,
               linewidth=scatter_edge_width,
               marker=style_dict['marker'], zorder=10)

    return {'dB': dB, 'R2': r2}


# ==========================================
# 3. 坐标轴格式化函数 (Axis Formatting)
# ==========================================
def format_axis_main(ax):
    """ 主图坐标轴设置 """
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 【刻度样式】：向内 (direction='in')
    # top=False, right=False: 去掉上、右边框的刻度线
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, bottom=True, left=True,
                   labelsize=11)

    # 【数值格式】：科学计数法 (10^x)
    fmt = LogFormatterSciNotation()
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # 强制显示主刻度
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))


def format_axis_inset(ax):
    """ 嵌入图坐标轴设置 - 重点修复了刻度向内 """
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 【关键修复】：显式设置 direction='in' 并应用到 both (major & minor)
    # labelsize=8: 字体与参考图的嵌入图保持一致
    # width=0.4: 刻度线变细
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, bottom=True, left=True,
                   labelsize=8, width=0.4, length=2)

    # 只显示左侧和下侧刻度，隐藏次级刻度标签
    ax.tick_params(axis='both', which='minor', labelsize=0)

    # 隐藏网格线，保持小图清爽
    ax.grid(False)

    # 设置背景透明度
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.85)

    # 边框变细
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Y轴标签 (可根据需要注释掉)
    ax.set_ylabel(r"$N_B$", fontsize=9, labelpad=1, color=BASE_COLOR)


# ==========================================
# 4. 主程序 (Main Execution)
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. 加载数据
    df = pd.read_csv(INPUT_FILE)
    df['BoxSize'] = pd.to_numeric(df['BoxSize'])
    df['MinBoxes'] = pd.to_numeric(df['MinBoxes'])

    # 统一截断 (Unified Cutoff)
    cutoff_lb = df.groupby("Network")['BoxSize'].max().min()
    print(f">>> Unified Cutoff: lb <= {cutoff_lb}")
    df = df[df['BoxSize'] <= cutoff_lb].copy()

    # 2. 初始化画布
    fig, ax = plt.subplots(figsize=(4.55, 3.34), dpi=600)
    ax.set_position([0.189231, 0.220848, 0.571477, 0.707526])
    legend_handles = []  # 用于收集图例句柄

    # ---------------------------------------------------------
    # PART 1: 绘制主图 (Active)
    # ---------------------------------------------------------
    print("Plotting Main: Active...")
    name = "Active"
    data = df[df["Network"] == name]
    style = STYLES[name]

    # 【微调区域 1 - 主图 Active】
    res = fit_and_plot(ax, data['BoxSize'].values, data['MinBoxes'].values, style,
                       scatter_size=35,  # 点的大小
                       line_width=1.5,  # 线宽
                       plot_ci=False,  # 关闭置信区间 (由您要求)
                       intercept_shift=1.5,  # 下移拟合线 (正值向下)
                       scatter_edge_width=1.5)  # 描边宽度 (空心点需要描边)

    # 生成 Active 图例
    label = f"{style['label']}: $d_B={res['dB']:.2f}$ ($R^2={res['R2']:.2f}$)"
    h_active = Line2D([0], [0], marker=style['marker'], color='w',
                      markerfacecolor=style['mfc'], markeredgecolor=style['color'],
                      markersize=8, label=label)
    legend_handles.append(h_active)

    # ---------------------------------------------------------
    # PART 2: 绘制嵌入图 (Overall)
    # ---------------------------------------------------------
    # 设置嵌入图位置 [left, bottom, width, height] (相对坐标)
    ax_ins = ax.inset_axes([0.60, 0.60, 0.35, 0.35])

    print("Plotting Inset: Overall...")
    name = "Overall"
    data = df[df["Network"] == name]
    style = STYLES[name]

    # 【微调区域 2 - 嵌入图 Overall】
    res = fit_and_plot(ax_ins, data['BoxSize'].values, data['MinBoxes'].values, style,
                       scatter_size=10,  # 点极小
                       line_width=0.8,  # 线极细
                       plot_ci=True,  # 保留 CI
                       intercept_shift=0,  # 不平移
                       scatter_edge_width=0)  # 无描边 (实心点更清晰)

    # 生成 Overall 图例
    label = f"{style['label']}: $d_B={res['dB']:.2f}$ ($R^2={res['R2']:.2f}$)"
    h_overall = Line2D([0], [0], marker=style['marker'], color='w',
                       markerfacecolor=style['mfc'], markeredgecolor=style['color'],
                       markersize=8, label=label)
    legend_handles.append(h_overall)

    # ---------------------------------------------------------
    # PART 3: 绘制嵌入图 (Backbone)
    # ---------------------------------------------------------
    print("Plotting Inset: Backbone...")
    name = "Backbone"
    data = df[df["Network"] == name]
    style = STYLES[name]

    # 【微调区域 3 - 嵌入图 Backbone】
    res = fit_and_plot(ax_ins, data['BoxSize'].values, data['MinBoxes'].values, style,
                       scatter_size=12,  # 点稍大一点点
                       line_width=0.8,
                       plot_ci=True,
                       intercept_shift=0,
                       scatter_edge_width=0)  # 无描边

    # 生成 Backbone 图例
    label = f"{style['label']}: $d_B={res['dB']:.2f}$ ($R^2={res['R2']:.2f}$)"
    h_backbone = Line2D([0], [0], marker=style['marker'], color='w',
                        markerfacecolor=style['mfc'], markeredgecolor=style['color'],
                        markersize=8, label=label)
    legend_handles.append(h_backbone)

    # ---------------------------------------------------------
    # PART 4: 最终格式化与保存
    # ---------------------------------------------------------
    # 应用坐标轴样式 (包含刻度向内修复)
    format_axis_main(ax)
    format_axis_inset(ax_ins)

    # 轴标签
    ax.set_xlabel(r"Box size ($l_B$)", fontsize=13, labelpad=5)
    ax.set_ylabel(r"Min boxes ($N_B$)", fontsize=13, labelpad=8)

    # 图例 (左下角)
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower left')

    # 调整留白 (主图上方留出空间给嵌入图)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], ylims[1] * 3)  # 上限扩大3倍
    ax.set_xlim(xlims[0], xlims[1] * 1.1)

    ax.set_position([0.189231, 0.220848, 0.571477, 0.707526])
    formats = ['png', 'svg', 'tiff', 'jpg']

    print("\n>>> Saving figures in multiple formats...")
    for ext in formats:
        save_path = os.path.join(OUTPUT_DIR, f"{FILENAME_BASE}.{ext}")
        try:
            # dpi=600 确保栅格图(PNG/TIFF)足够清晰，对矢量图(SVG/PDF)无效但无害
            fig.savefig(save_path, format=ext, dpi=600)
            print(f"    [Success] Saved: {save_path}")
        except Exception as e:
            print(f"    [Error] Failed to save {ext}: {e}")
    plt.show()

if __name__ == "__main__":
    main()

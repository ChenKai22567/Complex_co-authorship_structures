import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 全局风格设置
# ==========================================
FONT_FAMILY = 'Arial'  # 设置全局字体为 Arial
BASE_COLOR = '#444444' # 设置基础深灰色，比纯黑更柔和
TITLE_COLOR = '#666666'

# 更新 matplotlib 的全局配置 (rcParams)，确保所有图表元素的风格统一
plt.rcParams['font.family'] = FONT_FAMILY      # 字体
plt.rcParams['text.color'] = BASE_COLOR        # 文本颜色
plt.rcParams['axes.labelcolor'] = BASE_COLOR   # 坐标轴标签颜色
plt.rcParams['xtick.color'] = BASE_COLOR       # X轴刻度颜色
plt.rcParams['ytick.color'] = BASE_COLOR       # Y轴刻度颜色
plt.rcParams['axes.edgecolor'] = BASE_COLOR    # 边框颜色
plt.rcParams['font.size'] = 11                 # 基础字号

# ==========================================
# 2. 配置与数据处理
# ==========================================
# Locate data and output folders relative to community/代码.
script_dir = os.path.dirname(os.path.abspath(__file__))
community_dir = os.path.dirname(script_dir)
input_dir = os.path.join(community_dir, "data")
output_dir = os.path.join(community_dir, "figures")

# 定义 CSV 数据文件路径
csv_path = os.path.join(input_dir, "network_evolution_stats_fixed2.csv")
# 定义输出文件名的基础部分
filename_base = "3plot_growth_types_stacked_v37_textures"

# [关键配置] 定义链接类型的绘图属性
# 包含：数据列名 (col), 图例标签 (label), 边框颜色 (color), 填充纹理 (hatch)
LINK_CONFIG = [
    # 1. Duplication (重复边): 使用右斜线纹理 (//////)，橙色边框
    {'col': 'link_repeated', 'label': 'Duplication', 'color': '#ed7d2f', 'hatch': '//////'},
    # 2. Intra (组件内连接): 使用点状纹理 (......)，浅橙色边框
    {'col': 'link_intra_comp', 'label': 'Intra-component', 'color': '#fdbf6f', 'hatch': '......'},
    # 3. Merge (组件合并): 使用十字纹理 (++++++)，深绿色边框
    {'col': 'link_merge', 'label': 'Component-merging', 'color': '#228B22', 'hatch': '++++++'},
    # 4. Growth (组件增长): 使用左斜线纹理 (\\\\\\)，浅蓝色边框
    {'col': 'link_growth', 'label': 'Component-growing link', 'color': '#9ecae1', 'hatch': '\\\\\\\\\\\\'},
    # 5. New Cluster (新组件): 使用斜网格纹理 (xxxxxx)，深蓝色边框
    {'col': 'link_new_cluster', 'label': 'New component', 'color': '#4292c6', 'hatch': 'xxxxxx'}
]

# 定义三个子图对应的 Pipeline ID 和显示标题
SUBPLOTS_CONFIG = [
    {'id': 'S2', 'title': 'Overall network'},      # 总体网络
    {'id': 'S5-S2', 'title': 'Active network'},    # 活跃网络
    {'id': 'S5-S3-S2', 'title': 'Backbone network'} # 骨干网络
]

try:
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # 如果文件未找到，生成模拟数据用于演示
    print("Error: File not found. Creating dummy data.")
    years = range(2006, 2026)
    data = []
    for y in years:
        # 生成模拟数据字典
        data.append({'year_end': y, 'pipeline': 'S2', 'method': 'newman',
                     'link_repeated': y * 2, 'link_intra_comp': y, 'link_merge': y / 2, 'link_growth': y * 3,
                     'link_new_cluster': y})
        data.append({'year_end': y, 'pipeline': 'S5-S2', 'method': 'newman',
                     'link_repeated': y, 'link_intra_comp': y / 2, 'link_merge': y / 4, 'link_growth': y * 1.5,
                     'link_new_cluster': y / 2})
        data.append({'year_end': y, 'pipeline': 'S5-S3-S2', 'method': 'newman',
                     'link_repeated': y / 2, 'link_intra_comp': y / 4, 'link_merge': y / 8, 'link_growth': y,
                     'link_new_cluster': y / 4})
    df = pd.DataFrame(data)

# 数据过滤：只保留 'newman' 方法的数据
df_plot = df[df["method"] == "newman"]
# 数据过滤：剔除 2025 年及以后的数据
df_plot = df_plot[df_plot["year_end"] < 2025]

# 获取年份列表并排序
years = sorted(df_plot['year_end'].unique())
# 设置柱子的宽度
bar_width = 0.82

# Match the physical width of community_degree_time.py's main plot plus colorbar.
FIGSIZE = (4.3, 4.35)
REFERENCE_FIGSIZE = (4.2, 3.34)
REFERENCE_AX_POSITION = (0.205000, 0.220848, 0.648370, 0.707526)

plot_left = REFERENCE_AX_POSITION[0] * REFERENCE_FIGSIZE[0] / FIGSIZE[0]
plot_bottom = REFERENCE_AX_POSITION[1] * REFERENCE_FIGSIZE[1] / FIGSIZE[1]
plot_width = REFERENCE_AX_POSITION[2] * REFERENCE_FIGSIZE[0] / FIGSIZE[0]
plot_height = REFERENCE_AX_POSITION[3] * REFERENCE_FIGSIZE[1] / FIGSIZE[1]
plot_right = plot_left + plot_width
plot_top = plot_bottom + plot_height

# ==========================================
# 3. 绘图
# ==========================================
# 创建画布：3 行 1 列，大小 4x3.5 英寸，DPI 600，共享 X 轴
fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, dpi=600, sharex=True)

# 调整子图布局
# hspace=0.08: 子图垂直间距很小
# top=0.88: 顶部预留空间给图例
# bottom=0.12: 底部预留空间给X轴标签
# left=0.18: 左侧预留空间给Y轴标签
plt.subplots_adjust(hspace=0.08, top=plot_top, bottom=plot_bottom, left=plot_left, right=plot_right)

# --- 遍历绘制三个子图 ---
for idx, (ax, config) in enumerate(zip(axes, SUBPLOTS_CONFIG)):
    pipe_id = config['id']
    title = config['title']

    # 提取当前 Pipeline 的数据并按年份重索引 (防止缺失年份导致错位)
    subset = df_plot[df_plot["pipeline"] == pipe_id].set_index('year_end')
    subset = subset.reindex(years, fill_value=0)

    # 初始化底部高度数组 (用于堆叠)
    bottom = np.zeros(len(years))

    # 循环绘制每一种边类型的层
    for link_item in LINK_CONFIG:
        col_name = link_item['col']
        values = subset[col_name].values  # 获取当前类型的数值
        c = link_item['color']            # 获取颜色
        h = link_item['hatch']            # 获取纹理

        # [核心绘图] 绘制柱状图
        # color='white': 背景设为白色，避免纯色填充
        # edgecolor=c: 边框和纹理线使用指定颜色
        # hatch=h: 应用纹理图案
        # label=...: 仅在第一个子图 (idx==0) 添加图例标签，避免重复
        ax.bar(years, values, width=bar_width, bottom=bottom,
               label=link_item['label'] if idx == 0 else "",
               color='white', edgecolor=c, hatch=h, linewidth=0.6)

        # 更新底部高度，以便下一层堆叠在当前层之上
        bottom += values

    # --- 子图样式设置 ---
    # 仅在中间子图 (idx==1) 显示 Y 轴标签
    if idx == 1:
        ax.set_ylabel("Annual new edges", fontsize=13, labelpad=8)
    else:
        ax.set_ylabel("")

    # 添加子图标题 (放置在图内左上角)
    ax.text(0.05, 0.85, title, transform=ax.transAxes,
            fontsize=12, fontweight='normal', color=TITLE_COLOR, # 灰色非加粗字体
            ha='left', va='top')

    # 设置刻度样式：朝内，仅显示底部和左侧刻度
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, bottom=True, left=True,
                   labelsize=11)

    # 移除网格线
    ax.grid(False)
    # 设置 Y 轴上限为最大值的 1.1 倍，留出一点顶部空间。
    # 左侧刻度只保留 0 和中间值，避免相邻子图边界处显示顶部刻度。
    y_top = bottom.max() * 1.1
    ax.set_ylim(0, y_top)
    middle_ticks = {
        'S2': 15000,
        'S5-S2': 5000,
        'S5-S3-S2': 350
    }
    ax.set_yticks([0, middle_ticks[pipe_id]])

# --- X 轴设置 (针对最下方的子图) ---
axes[-1].set_xlabel("Year", fontsize=13, labelpad=5)
axes[-1].set_xticks(np.arange(2008, 2025, 4)) # 四年一显示
axes[-1].set_xlim(2005.4, 2024.6) # 设置 X 轴范围
axes[-1].tick_params(axis='x', rotation=30) # 年份标签倾斜 30 度
for label in axes[-1].get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

# ==========================================
# 4. 图例 (顶部)
# ==========================================
# 获取第一个子图的句柄和标签
handles, labels = axes[0].get_legend_handles_labels()
legend_order = [
    'Duplication',
    'Intra-component',
    'Component-growing link',
    'Component-merging',
    'New component'
]
legend_lookup = dict(zip(labels, handles))
handles = [legend_lookup[label] for label in legend_order]
labels = legend_order

# 创建全局图例
# loc='upper center': 基准点为上方中心
# bbox_to_anchor=(0.50, 0.900241): 将图例到主绘图区的距离继续缩短 20%
# ncol=2: 分 2 列排列
# handlelength=1.8: 增加图例图标长度，以便清楚显示纹理
leg = fig.legend(handles, labels, loc='upper center',
                 bbox_to_anchor=(0.52, 0.900241),
                 ncol=2,
                 frameon=False, fontsize=11,
                 columnspacing=-2, labelspacing=0.35,
                 handlelength=1.2, handletextpad=0.3)

# ==========================================
# 5. 保存与显示
# ==========================================
# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 循环保存为 png, svg, tiff, jpg 格式
for ext in ['png', 'svg', 'tiff', 'jpg']:
    save_path = os.path.join(output_dir, f"{filename_base}.{ext}")
    try:
        # 保存图片，去除多余白边 (bbox_inches='tight' 会改变 figsize，这里未启用以保持精确尺寸控制，但布局调整已确保完整)
        # 注意：如果需要严格保持 figsize，不建议使用 bbox_inches='tight'，而是依赖 subplots_adjust
        fig.savefig(save_path, format=ext, dpi=600)
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error saving {ext}: {e}")

plt.show()

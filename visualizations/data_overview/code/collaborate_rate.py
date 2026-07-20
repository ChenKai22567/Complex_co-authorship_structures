import pandas as pd                             # 导入 pandas，用于读取和处理 CSV 数据
import matplotlib.pyplot as plt                 # 导入 matplotlib.pyplot，简称 plt，用于绘图
from matplotlib.ticker import FuncFormatter     # 导入 FuncFormatter，用于自定义刻度标签格式
import numpy as np                              # 导入 NumPy，简称 np，用于数值计算
from pathlib import Path

# ————— 全局字体与尺寸设置 —————
plt.rcParams['font.family']    = 'Arial'       # 全局无衬线字体设置为 Arial
plt.rcParams['axes.titlesize'] = 12            # 坐标轴标题字号设为 12
plt.rcParams['axes.labelsize'] = 12            # 坐标轴标签字号设为 12
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['legend.labelcolor'] = '#333333'

# ————— 1. 读取 CSV 文件 —————
data_dir = Path(__file__).resolve().parents[1]
csv_path = data_dir / 'data' / 'year.csv'
df = pd.read_csv(csv_path, encoding='utf-8')    # 将 CSV 读入 DataFrame

# ————— 2. 提取年份与合作比例数据 —————
years        = df['publication_year'].astype(int)            # 出版年份（整数）
inst_rate    = df['col_cross_institution_rate'] * 100        # 跨机构合作率 → 百分比
country_rate = df['col_cross_country_rate']    * 100         # 跨国家合作率 → 百分比

# ————— 3. 创建绘图画布 —————
fig, ax = plt.subplots(
    figsize=(4, 3),  # 画布宽 4 英寸，高 3 英寸
    dpi=600          # 分辨率 600 DPI
)

# ————— 4. 绘制“上下对称”柱状图 —————
ax.bar(
    years,
    inst_rate,
    width=0.8,
    color='#82a7d1',
    label='Institutional collaboration'
)
ax.bar(
    years,
    -country_rate,   # 负值使柱子向下
    width=0.8,
    color='#f2AB6A',
    label='International collaboration'
)

# ————— 5. 设置 X 轴刻度间隔，每 10 年一个 —————
start_year = int(years.min())
end_year   = int(years.max())
year_ticks = np.arange(start_year, end_year + 1, 10)
if year_ticks[-1] != end_year:
    year_ticks = np.append(year_ticks, end_year)
ax.set_xticks(year_ticks)      # X 轴刻度从起始到结束，每隔 10 年，并保留末尾年份
ax.set_xticklabels(
    [str(y) for y in year_ticks],
    rotation=0,                                             # 旋转 45° 便于阅读
    fontsize=8                                               # 字体大小设为 8
)
ax.tick_params(axis='x', direction='in', length=1.8, width=1.2, labelsize=9)  # X 轴刻度线缩短 40%

# ————— 6. 将 X 轴移到上方，仅保留“上下”两条边框 —————
ax.xaxis.set_label_position('top')                          # X 轴标签放顶端
ax.xaxis.tick_top()                                         # X 轴刻度放顶端
# 隐藏左、右两条脊柱（其余框线）
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# 确保上、下脊柱可见（默认即可），无需额外设置

# ————— 7. Y 轴刻度格式化与样式 —————
ax.yaxis.set_label_position('right')                          # X 轴标签放顶端
ax.yaxis.tick_right()                                         # X 轴刻度放顶端
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f"{abs(y):.0f}%")         # 绝对值百分比，无小数
)
ax.tick_params(axis='y', direction='in', length=1.8, width=1.2, labelsize=9)  # Y 轴刻度线缩短 40%

# ————— 8. 轴标签与图例（图例置于底部） —————
ax.set_xlabel('Year',    labelpad=10)            # X 轴标签
ax.set_ylabel('Collaboration rate (%)', labelpad=10)       # Y 轴标签
leg = ax.legend(
    ncol=2,
    frameon=False,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),                             # 图例在下方，并留出空白
    fontsize=7,
    columnspacing=0.7
)
leg.set_title('')                                           # 清空图例标题

# ————— 9. 添加 2006 年竖直虚线，缩短长度并留底部空隙 —————
ax.axvline(
    x=2006,
    ymin=0,            # 从画布下方 5% 处开始绘制
    ymax=1,            # 到画布上方 95% 处结束
    linestyle='--',
    color='#c8e4d2',
    linewidth=1
)

# ————— 10. 优化布局并显示 —————
plt.tight_layout()

# 将框内宽度统一缩小 10%，保持中心位置和高度不变。
current_position = ax.get_position()
plot_width = current_position.width * 0.9
plot_left = current_position.x0 + (current_position.width - plot_width) / 2
ax.set_position([
    plot_left,
    current_position.y0,
    plot_width,
    current_position.height
])

# ————— 13. 保存多种格式图片 —————
output_dir = data_dir / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
filename_base = 'cross_bar'                       # 自定义：修改为图片文件的基础名称
for ext in ['png', 'jpg', 'svg', 'tiff']:                       # 循环保存为 png、jpg、svg、tiff 四种格式
    save_path = output_dir / f"{filename_base}.{ext}"           # 拼接完整文件路径
    fig.savefig(save_path, format=ext, dpi=600)                 # 保存图片，格式由 ext 决定，分辨率设为 300 dpi

plt.close(fig)

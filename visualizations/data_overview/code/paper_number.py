from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# ————— 全局字体与尺寸设置 —————
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['legend.labelcolor'] = '#333333'

# ————— 1. 读取 CSV 文件 —————
data_dir = Path(__file__).resolve().parents[1]
csv_path = data_dir / 'data' / 'year.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

# ————— 2. 提取年份、论文数量与独著率 —————
years = df['publication_year'].astype(int)
single_papers = df['single_author_papers']
multi_papers = df['multi_author_papers']
single_rate = df['single_rate'] * 100

# ————— 3. 创建绘图画布与双 Y 轴 —————
fig, ax = plt.subplots(figsize=(4.8, 3), dpi=600)
ax_rate = ax.twinx()

# ————— 4. 绘制堆叠柱状图 —————
ax.bar(
    years,
    single_papers,
    width=0.8,
    color='#f2AB6A',
    label='Single-author papers'
)
ax.bar(
    years,
    multi_papers,
    width=0.8,
    bottom=single_papers,
    color='#82a7d1',
    label='Multi-author papers'
)

# ————— 5. 绘制独著率虚线 —————
ax_rate.plot(
    years,
    single_rate,
    color='#c8e4d2',
    linestyle='--',
    linewidth=1.2,
    label='Single-author rate'
)

# ————— 6. 设置顶部 X 轴刻度 —————
start_year = int(years.min())
end_year = int(years.max())
year_ticks = np.arange(start_year, end_year + 1, 10)
if year_ticks[-1] != end_year:
    year_ticks = np.append(year_ticks, end_year)
ax.set_xticks(year_ticks)
ax.set_xticklabels([str(year) for year in year_ticks], rotation=0, fontsize=8)
ax.tick_params(
    axis='x',
    direction='in',
    length=1.8,
    width=1.2,
    labelsize=9
)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# ————— 7. 设置左侧论文数量轴 —————
ax.yaxis.set_label_position('left')
ax.yaxis.tick_left()
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:,.0f}'))
ax.tick_params(
    axis='y',
    direction='in',
    length=1.8,
    width=1.2,
    labelsize=9
)
paper_top = (single_papers + multi_papers).max() * 1.05
bottom_margin_ratio = 0.04
ax.set_ylim(-paper_top * bottom_margin_ratio, paper_top)
ax.set_yticks(np.arange(0, np.floor(paper_top / 1000) * 1000 + 1, 1000))

# ————— 8. 设置右侧独著率轴 —————
ax_rate.yaxis.set_label_position('right')
ax_rate.yaxis.tick_right()
ax_rate.yaxis.set_major_formatter(
    FuncFormatter(lambda value, pos: f'{value:.0f}%')
)
ax_rate.tick_params(
    axis='y',
    direction='in',
    length=1.8,
    width=1.2,
    labelsize=9
)
rate_top = single_rate.max() * 1.04
ax_rate.set_ylim(-rate_top * bottom_margin_ratio, rate_top)
ax_rate.set_yticks(np.arange(0, np.floor(rate_top / 20) * 20 + 1, 20))

# ————— 9. 设置边框与坐标轴标签 —————
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax_rate.spines['left'].set_visible(False)
ax_rate.spines['bottom'].set_visible(False)
ax_rate.spines['top'].set_visible(False)
for spine in [ax.spines['left'], ax.spines['top'], ax.spines['bottom'],
              ax_rate.spines['right']]:
    spine.set_linewidth(0.8)

ax.set_xlabel('Year', labelpad=10)
ax.set_ylabel('Number of papers', labelpad=10)
ax_rate.set_ylabel('Single-author rate (%)', labelpad=10)

# ————— 10. 合并柱状图与折线图图例 —————
bar_handles, bar_labels = ax.get_legend_handles_labels()
line_handles, line_labels = ax_rate.get_legend_handles_labels()
legend = ax.legend(
    bar_handles + line_handles,
    bar_labels + line_labels,
    ncol=3,
    frameon=False,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),
    fontsize=7,
    columnspacing=0.7
)
legend.set_title('')

# ————— 11. 优化布局并统一坐标框显示区域 —————
plt.tight_layout()

# 与 collaborate_rate.py 的坐标框物理宽、高和纵向位置完全一致。
plot_width_inches = 4 * 0.7763020833333333 * 0.9
plot_width = plot_width_inches / fig.get_figwidth()
plot_height = 0.6598703703703703
plot_left = (1 - plot_width) / 2
plot_bottom = 0.14929629629629637
plot_position = [plot_left, plot_bottom, plot_width, plot_height]
ax.set_position(plot_position)
ax_rate.set_position(plot_position)

# ————— 12. 保存多种格式图片 —————
output_dir = data_dir / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
filename_base = 'author_stack_line'
for ext in ['png', 'jpg', 'svg', 'tiff']:
    save_path = output_dir / f'{filename_base}.{ext}'
    fig.savefig(save_path, format=ext, dpi=600)

plt.close(fig)

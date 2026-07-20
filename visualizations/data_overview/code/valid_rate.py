import matplotlib.pyplot as plt               # 导入 matplotlib.pyplot 模块并简写为 plt，用于绘图
from matplotlib.ticker import FuncFormatter   # 从 matplotlib.ticker 模块导入 FuncFormatter，用于自定义刻度格式
import numpy as np                            # 导入 NumPy 库并简写为 np，用于数值计算
import pandas as pd                           # 导入 pandas，用于读取和处理 CSV 数据
from pathlib import Path

# ————— 全局字体与尺寸设置 —————
plt.rcParams['font.family']    = 'Arial'     # 设置全局字体族为无衬线字体（Arial）
plt.rcParams['axes.titlesize'] = 12          # 设置坐标轴标题（title）的默认字号为 12
plt.rcParams['axes.labelsize'] = 12          # 设置坐标轴标签（xlabel/ylabel）的默认字号为 12
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['legend.labelcolor'] = '#333333'

# ————— 1. 读取数据 —————
data_dir = Path(__file__).resolve().parents[1]
csv_path = data_dir / 'data' / 'year.csv'
# 使用 pandas 的 read_csv 自动处理 BOM 和编码问题
df = pd.read_csv(csv_path, encoding='utf-8')  # 将 CSV 文件读入 DataFrame

# 如果不确定列名，可取消下一行注释临时打印
# print(df.columns)  # 查看 DataFrame 中所有列的名称

# ————— 2. 提取并转换数据 —————
years     = df['publication_year'].astype(int)  # 提取“publication_year”列，表示出版年份
addr_rate = df['valid_address_rate'] * 100   # 将“valid_address_rate”乘以 100 转为百分比
kw_rate   = df['valid_keyword_rate'] * 100   # 将“valid_keyword_rate”乘以 100 转为百分比

# ————— 3. 创建绘图画布 —————
# figsize=(4,3) 指定画布宽 4 英寸、高 3 英寸；dpi=600 指定每英寸 600 像素
fig, ax = plt.subplots(figsize=(4.8, 3), dpi=600)


# ————— 4. 绘制折线图（调整线宽和点大小） —————
ax.plot(
    years, addr_rate,
    marker='o',                # 数据点形状为圆点
    markersize=2,              # 设置标记点大小为 6
    linewidth=1.2,               # 设置折线宽度为 2
    label='Valid address rate',
    color='#82a7d1'
)
ax.plot(
    years, kw_rate,
    #marker='^',                # 数据点形状为方块
    marker='s',
    markersize=1.5,              # 设置标记点大小为 6
    linewidth=1.2,               # 设置折线宽度为 2
    label='Valid keyword rate',
    color='#f2AB6A'
)
# ————— 5. 设置年份刻度与百分比刻度 —————
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
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x:.0f}%")  # 将刻度值格式化为无小数的百分比形式
)
ax.tick_params(
    axis='y',
    direction='in',
    length=1.8,
    width=1.2,
    labelsize=9
)

# ————— 6. 设置坐标轴标签与标题 —————
ax.set_xlabel('Year', labelpad=10)      # 设置 x 轴标签，并与轴之间留 10 点间距
ax.set_ylabel('Extraction rate (%)', labelpad=10)   # 设置 y 轴标签，并与轴之间留 10 点间距
ax.set_title('')
                                                    # 设置图表标题

# ————— 11. 去除标题与图例标题，并缩小两列间距 —————
ax.set_title('')  # 清空图表标题
leg = ax.legend(
    ncol=2,                    # 将图例分为两列
    frameon=False,             # 不显示图例边框
    loc='lower center',        # 将图例放到图框下方
    bbox_to_anchor=(0.5, -0.15),  # 通过锚点微调位置
    title='',                  # 清空图例标题
    fontsize=7,                # 图例文字大小
    columnspacing=0.7          # 缩小两列之间的水平间距（默认 2，数值越小间距越紧凑）
)
leg.set_title('')  # 确保图例标题为空

# ————— 5. 于 2006 年处添加竖直虚线 —————
ax.axvline(
    x=2006,                    # 虚线对应的 x 轴位置：2006 年
    linestyle='--',            # 虚线样式
    color='#c8e4d2',           # 虚线颜色，十六进制格式
    linewidth=1                # 虚线宽度
)

# ————— 9. 优化布局并显示 —————
plt.tight_layout()  # 自动调整子图参数，避免标签或标题与画布边缘重叠

# 与前两个文件的坐标框物理大小完全一致，并保留左、下坐标轴。
plot_width_inches = 4 * 0.7763020833333333 * 0.9
plot_width = plot_width_inches / fig.get_figwidth()
plot_height = 0.6598703703703703
plot_left = (1 - plot_width) / 2
plot_bottom = 0.14929629629629637
plot_position = [plot_left, plot_bottom, plot_width, plot_height]
ax.set_position(plot_position)

# ————— 13. 保存多种格式图片 —————
output_dir = data_dir / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
filename_base = 'valid_lines'                       # 自定义：修改为图片文件的基础名称
for ext in ['png', 'jpg', 'svg', 'tiff']:                       # 循环保存为四种格式
    save_path = output_dir / f"{filename_base}.{ext}"           # 拼接完整文件路径
    fig.savefig(save_path, format=ext, dpi=600)                 # 保存图片，格式由 ext 决定，分辨率设为 300 dpi

plt.close(fig)

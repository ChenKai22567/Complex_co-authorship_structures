import matplotlib.pyplot as plt               # 导入 matplotlib.pyplot 模块并简写为 plt，用于绘图
from matplotlib.ticker import FuncFormatter   # 从 matplotlib.ticker 模块导入 FuncFormatter，用于自定义刻度标签格式
import numpy as np                            # 导入 NumPy 库并简写为 np，用于数值计算
from pathlib import Path


# ————— 全局字体与尺寸设置 —————
plt.rcParams['font.family']     = 'Arial'                  # 设置全局字体族为无衬线字体
# ————— 调整画布物理尺寸和分辨率 —————


plt.rcParams['axes.titlesize']  = 12                             # 设置坐标轴标题字号
plt.rcParams['axes.labelsize']  = 12                             # 设置坐标轴标签字号
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['legend.labelcolor'] = '#333333'

# ————— 1. 输入数据 —————
labels               = ['1', '2', '3', '4', '5', '≥6']           # 定义作者数量的分组标签
counts_1900_2025     = np.array([78002, 32646, 21478, 11716, 5598, 6649])   # 定义 1900–2025 时期各分组的论文计数
counts_2006_2025     = np.array([17292, 19246, 16911, 10089, 4890, 5865])   # 定义 2006–2025 时期各分组的论文计数

# ————— 2. 计算百分比 —————
pct_1900 = counts_1900_2025 / counts_1900_2025.sum() * 100      # 计算 1900–2025 时期各分组论文数占比（百分比）
pct_2006 = counts_2006_2025 / counts_2006_2025.sum() * 100      # 计算 2006–2025 时期各分组论文数占比（百分比）

# ————— 3. 翻转顺序，让“≥6”在最上面 —————
labels            = labels[::-1]                                # 将标签列表逆序，使“≥6”排在最前
pct_1900          = pct_1900[::-1]                              # 将 1900–2025 百分比数组逆序
pct_2006          = pct_2006[::-1]                              # 将 2006–2025 百分比数组逆序
counts_1900_2025  = counts_1900_2025[::-1]                      # 将 1900–2025 计数数组逆序
counts_2006_2025  = counts_2006_2025[::-1]                      # 将 2006–2025 计数数组逆序

# ————— 4. 创建画布 —————
fig, ax = plt.subplots(figsize=(4.8, 3), dpi=600)
y        = np.arange(len(labels))                               # 生成 y 轴位置数组，长度等于标签数量
bar_h    = 0.6                                                  # 设置水平条形图的高度为 0.6

# ————— 5. 绘制镜像条形图 —————
ax.barh(y, -pct_1900, bar_h,                                    # 绘制左侧水平条形图，值为负数以镜像展示
        color='#82a7d1',                                      # 设置条形颜色为钢蓝色
        label='1900-2025 period')                               # 添加图例标签
ax.barh(y,  pct_2006, bar_h,                                    # 绘制右侧水平条形图，值为正数
        color='#f2AB6A',                                          # 设置条形颜色为珊瑚色
        label='2006-2025 period')                               # 添加图例标签

# ————— 6. 顶部百分比刻度 —————
ax.xaxis.set_label_position('top')                              # 将 x 轴标签移到图形顶部
ax.xaxis.tick_top()                                             # 将 x 轴刻度线移到图形顶部
ax.xaxis.set_major_formatter(                                   # 自定义 x 轴刻度的显示格式
    FuncFormatter(lambda x, pos: f"{abs(x):.0f}%")             # 刻度标签显示为绝对值百分比，无小数
)

# ————— 7. 将纵轴移到右侧并设置边框 —————
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_linewidth(0.8)
ax.spines['top'].set_linewidth(0.8)

# ————— 8. 平均值虚线 —————
ax.axvline(-pct_1900.mean(),                                    # 在左侧条形图上绘制平均值虚线
           color='#82a7d1', linestyle='--', alpha=0.6)        # 线型为虚线，透明度 0.6，与条形颜色一致
ax.axvline( pct_2006.mean(),                                    # 在右侧条形图上绘制平均值虚线
           color='#f2AB6A',     linestyle='--', alpha=0.6)        # 线型为虚线，透明度 0.6，与条形颜色一致

ax.text(
    -pct_1900.mean() - 0.9,
    len(labels) / 2 - 0.75,
    'Mean',
    color='#82a7d1',
    fontsize=7,
    ha='right',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.4),
    zorder=5
)
ax.text(
    pct_2006.mean() + 0.9,
    len(labels) / 2 - 0.75,
    'Mean',
    color='#f2AB6A',
    fontsize=7,
    ha='left',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.4),
    zorder=5
)

# ————— 9. 条内简写数值 —————
for i, (c1, c2) in enumerate(zip(counts_1900_2025, counts_2006_2025)):  # 遍历每个分组的计数及其索引
    ax.text(-pct_1900[i]/2, i, f"{c1/1000:.1f}k",              # 在左侧条形中心添加计数值文本，单位“k”
            va='center', ha='center',                          # 垂直水平居中对齐文本
            color='white', fontsize=7)                        # 文本颜色为白色，字号 10
    ax.text( pct_2006[i]/2, i, f"{c2/1000:.1f}k",              # 在右侧条形中心添加计数值文本，单位“k”
            va='center', ha='center',                          # 垂直水平居中对齐文本
            color='white', fontsize=7)                        # 文本颜色为白色，字号 10

# ————— 10. 坐标轴标签与刻度字号调整 —————
ax.set_yticks(y)                                                # 设置 y 轴刻度位置
ax.set_yticklabels(labels, fontsize=9)                         # 设置 y 轴刻度标签及其字号
ax.set_ylabel('Number of authors', fontsize=12)                # 设置 y 轴标签及其字号
ax.tick_params(
    axis='y',
    direction='in',
    labelsize=9,
    width=1.2,
    length=1.8
)

ax.tick_params(
    axis='x',
    direction='in',
    labelsize=9,
    width=1.2,
    length=1.8
)
ax.set_xlabel('Percentage of publications (%)',                 # 设置 x 轴标签
              fontsize=12,                                      # x 轴标签字号
              labelpad=10)                                      # x 轴标签与坐标轴的距离

# ————— 11. 去除标题与图例标题 —————
ax.set_title('')                                                # 清空图表标题
leg = ax.legend(ncol=2,                                        # 创建图例，分两列显示
                frameon=False,                                 # 不显示图例边框
                loc='lower center',                            # 图例位置设为下方中央
                bbox_to_anchor=(0.5, -0.15),                   # 通过锚点微调位置
                title='',
                fontsize=7,)                                      # 清空图例标题
leg.set_title('')                                               # 确保图例标题为空

# ————— 12. 布局与显示 —————
plt.tight_layout()                                              # 自动调整子图参数，使之填充整个图像区域

# 与前 3 个文件的坐标框物理大小完全一致。
plot_width_inches = 4 * 0.7763020833333333 * 0.9
plot_width = plot_width_inches / fig.get_figwidth()
plot_height = 0.6598703703703703
plot_left = (1 - plot_width) / 2
plot_bottom = 0.14929629629629637
ax.set_position([plot_left, plot_bottom, plot_width, plot_height])

# ————— 13. 保存多种格式图片 —————
data_dir = Path(__file__).resolve().parents[1]
output_dir = data_dir / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
filename_base = 'author_count_mirror_bar'                       # 自定义：修改为图片文件的基础名称
for ext in ['png', 'jpg', 'svg', 'tiff']:
    save_path = output_dir / f"{filename_base}.{ext}"           # 拼接完整文件路径
    fig.savefig(save_path, format=ext, dpi=600)                 # 保存图片，格式由 ext 决定，分辨率设为 300 dpi

plt.close(fig)

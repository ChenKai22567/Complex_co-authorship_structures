from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['legend.labelcolor'] = '#333333'

labels = ['1', '2', '3', '4', '5', '≥6']
counts_1966_2025 = np.array([63864, 32015, 21384, 11699, 5574, 6613])
pct_1966 = np.array([45.2458, 22.6817, 15.1499, 8.2884, 3.9490, 4.6851])

counts_2006_2025 = np.array([17292, 19246, 16911, 10089, 4890, 5865])
pct_2006 = counts_2006_2025 / counts_2006_2025.sum() * 100

labels = labels[::-1]
counts_1966_2025 = counts_1966_2025[::-1]
counts_2006_2025 = counts_2006_2025[::-1]
pct_1966 = pct_1966[::-1]
pct_2006 = pct_2006[::-1]

fig, ax = plt.subplots(figsize=(4.8, 3), dpi=600)
y = np.arange(len(labels))
bar_h = 0.6

ax.barh(
    y,
    -pct_1966,
    bar_h,
    color='#82a7d1',
    label='1966-2025 period',
)
ax.barh(
    y,
    pct_2006,
    bar_h,
    color='#f2AB6A',
    label='2006-2025 period',
)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{abs(value):.0f}%'))

ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_linewidth(0.8)
ax.spines['top'].set_linewidth(0.8)

ax.axvline(-pct_1966.mean(), color='#82a7d1', linestyle='--', alpha=0.6)
ax.axvline(pct_2006.mean(), color='#f2AB6A', linestyle='--', alpha=0.6)

ax.text(
    -pct_1966.mean() - 0.9,
    len(labels) / 2 - 0.75,
    'Mean',
    color='#82a7d1',
    fontsize=7,
    ha='right',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.4),
    zorder=5,
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
    zorder=5,
)

for i, (c1, c2) in enumerate(zip(counts_1966_2025, counts_2006_2025)):
    if labels[i] in {'5', '≥6'}:
        left_label_x = -pct_1966[i] - 0.8
        left_label_color = '#333333'
        left_label_ha = 'right'
    else:
        left_label_x = -pct_1966[i] / 2
        left_label_color = 'white'
        left_label_ha = 'center'

    ax.text(
        left_label_x,
        i,
        f'{c1 / 1000:.1f}k',
        va='center',
        ha=left_label_ha,
        color=left_label_color,
        fontsize=7,
    )
    ax.text(
        pct_2006[i] / 2,
        i,
        f'{c2 / 1000:.1f}k',
        va='center',
        ha='center',
        color='white',
        fontsize=7,
    )

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_ylabel('Number of authors', fontsize=12)
ax.tick_params(axis='y', direction='in', labelsize=9, width=1.2, length=1.8)
ax.tick_params(axis='x', direction='in', labelsize=9, width=1.2, length=1.8)
ax.set_xlabel('Percentage of publications (%)', fontsize=12, labelpad=10)
ax.set_title('')

leg = ax.legend(
    ncol=2,
    frameon=False,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),
    title='',
    fontsize=7,
)
leg.set_title('')

plt.tight_layout()

plot_width_inches = 4 * 0.7763020833333333 * 0.9
plot_width = plot_width_inches / fig.get_figwidth()
plot_height = 0.6598703703703703
plot_left = (1 - plot_width) / 2
plot_bottom = 0.14929629629629637
ax.set_position([plot_left, plot_bottom, plot_width, plot_height])

data_dir = Path(__file__).resolve().parents[1]
output_dir = data_dir / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
filename_base = 'author_count_mirror_bar_1966'
for ext in ['png', 'jpg', 'svg', 'tiff']:
    save_path = output_dir / f'{filename_base}.{ext}'
    fig.savefig(save_path, format=ext, dpi=600)

plt.close(fig)

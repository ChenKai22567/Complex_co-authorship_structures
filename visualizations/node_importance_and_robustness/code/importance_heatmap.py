# 导入所需的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import os  # 文件和路径操作
import matplotlib.pyplot as plt  # 绘图
import matplotlib.colors as mcolors  # 颜色处理
import matplotlib.patches as mpatches  # 图形补丁（如多边形）
import seaborn as sns  # 热力图绘制
import warnings  # 警告管理

# =========================
# 0) 全局设置
# =========================
warnings.filterwarnings('ignore')  # 忽略所有警告
plt.rcParams['font.family'] = 'Arial'  # 设置全局字体为 Arial
plt.rcParams['font.weight'] = 'normal'  # 设置默认字体粗细为正常
plt.rcParams['axes.labelweight'] = 'normal'  # 设置坐标轴标签的粗细
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学文本字体集为 STIX（类似 LaTeX）

print("绘制：颜色修正(#333333) + 字体去粗版...")  # 打印脚本开始执行的信息

# =========================
# 1) 读取数据 (保持不变)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORTANCE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(IMPORTANCE_DIR, "data")
OUTPUT_FIG_DIR = os.path.join(IMPORTANCE_DIR, "figures")
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# 定义相关性矩阵 CSV 文件的路径
CORR_ALL_CSV = os.path.join(INPUT_DIR, "corr_all.csv")
# 定义 P 值矩阵 CSV 文件的路径
P_ALL_CSV = os.path.join(INPUT_DIR, "p_all.csv")

# 检查文件是否存在，不存在则抛出异常
if not os.path.exists(CORR_ALL_CSV):
    raise FileNotFoundError(f"未找到 corr 文件：{CORR_ALL_CSV}")
if not os.path.exists(P_ALL_CSV):
    raise FileNotFoundError(f"未找到 p 文件：{P_ALL_CSV}")

# 读取 CSV 文件，将第一列作为索引
corr_df = pd.read_csv(CORR_ALL_CSV, index_col=0)
p_df = pd.read_csv(P_ALL_CSV, index_col=0)

# =========================
# 1.5) 变量重命名 (保持不变)
# =========================
# 定义旧变量名到新变量名的映射字典
rename_dict = {
    "Degree": "Degree",
    "Strength": "Weighted degree",
    "Constraint": "Constraint",
    "EffSize": "Effective size",
    "CoreHD": "CoreHD",
    "ViralRank": "ViralRank",
    "Betweenness": "Betweenness",
    "W_Betweenness": "W. betweenness",
    "Closeness": "Closeness",
    "W_Closeness": "W. closeness",
    "LeaderRank": "LeaderRank",
    "Katz": "Katz centrality",
    "TotalCites": "Total citations",
    "G-index": "$g$-index",
    "ActiveSpan": "Active years"
}

# 获取 CSV 中实际存在的列名
exist_cols = list(corr_df.columns)
# 过滤 rename_dict，只保留实际存在的列名
valid_rename = {k: v for k, v in rename_dict.items() if k in exist_cols}
# 对 DataFrame 的索引和列名进行重命名
corr_df.rename(index=valid_rename, columns=valid_rename, inplace=True)
p_df.rename(index=valid_rename, columns=valid_rename, inplace=True)

# =========================
# 2) 分组配置 (保持不变)
# =========================
# 定义变量的分组字典
group_dict = {
    "Neighbors-based": ["Degree", "Weighted degree", "Constraint", "Effective size"],
    "Control\noptimization": ["CoreHD"],
    "Path-based": ["ViralRank", "Betweenness", "Closeness"],
    "Eigenvector-based": ["LeaderRank", "Katz centrality"],
    "Impact-based": ["Total citations", "$g$-index", "Active years"],
}

# 获取所有变量名
excluded_vars = {"W. betweenness", "W. closeness"}
all_vars = [v for v in corr_df.columns if v not in excluded_vars]
order = []  # 存储排序后的变量名
bounds = []  # 存储分组边界信息 (组名, 开始索引, 结束索引)
used = set()  # 记录已处理的变量
idx = 0  # 当前索引计数器

# 遍历分组字典，确定变量顺序和分组边界
for gname, members in group_dict.items():
    # 找出当前组中实际存在的变量
    present = [v for v in members if v in all_vars and v not in used]
    if len(present) == 0: continue  # 如果组内无变量，跳过
    start = idx  # 记录当前组开始索引
    order.extend(present)  # 添加到排序列表
    idx += len(present)  # 更新索引计数器
    end = idx - 1  # 记录当前组结束索引
    bounds.append((gname, start, end))  # 添加分组边界信息
    used.update(present)  # 标记变量为已处理

# 处理未被分组的剩余变量
rest = [v for v in all_vars if v not in used]
if len(rest) > 0:
    start = idx
    order.extend(rest)
    idx += len(rest)
    end = idx - 1
    bounds.append(("Others", start, end))

# 根据排序结果重新排列 DataFrame
corr_matrix = corr_df.loc[order, order].copy()
p_matrix = p_df.loc[order, order].copy()
cols = corr_matrix.columns  # 获取排好序的列名
n = len(cols)  # 变量总数

# =========================
# 3) 绘图配色配置 (保持不变)
# =========================
# 定义颜色列表（红-黄-蓝）
colors_list = ['#A50026', '#F46D43', '#FFFFBF', '#74ADD1', '#313695']
# 创建自定义的线性分段色彩映射
cmap = mcolors.LinearSegmentedColormap.from_list("Original_Replication", colors_list, N=256)

# 定义各部分的字体大小
fs = {
    'cell': 8.5,
    'tick': 9,
    'group': 11,
    'cbar_tick': 9,
    'cbar_title': 10.5
}

# =========================
# 4) 核心绘图流程
# =========================
fig, ax = plt.subplots(figsize=(7.01, 7.01))  # 创建画布和坐标轴
# 创建上三角掩码（用于隐藏热力图的上三角部分）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# 绘制热力图
sns.heatmap(corr_matrix, mask=mask, cmap=cmap,
            vmin=0, vmax=1, center=0.5,  # 设置数值范围 0-1
            square=True, linewidths=0, linecolor='white', alpha=0.7,  # 设置方格形状、线宽、颜色和透明度
            cbar=False, annot=False, ax=ax)  # 不显示颜色条和自动标注

# --- 修改1：单元格文字去粗 ---
for i in range(n):
    for j in range(n):
        if i >= j:  # 只处理下三角部分（包括对角线）
            val = float(corr_matrix.iloc[i, j])  # 获取相关系数
            p_val = float(p_matrix.iloc[i, j])  # 获取 P 值

            # 设置文本颜色：深色背景用白色，浅色背景用深灰(#333333)
            txt_col = 'white' if val > 0.6 or val < 0.2 else '#333333'
            txt_weight = 'bold' if txt_col == 'white' else 'normal'
            txt = f"{val:.2f}"  # 格式化数值为两位小数

            # 添加显著性星号（对角线除外）
            if i != j:
                if p_val <= 0.001:
                    txt += "\n***"
                elif p_val <= 0.01:
                    txt += "\n**"
                elif p_val <= 0.05:
                    txt += "\n*"

            # 在单元格中心添加文本
            ax.text(j + 0.5, i + 0.6, txt, ha='center', va='center',
                    color=txt_col, fontsize=fs['cell'], fontweight=txt_weight)

# --- 修改2：XY轴标签 颜色变更为 #333333 且 去粗 ---
ax.set_xticks(np.arange(n) + 0.5)  # 设置 X 轴刻度位置
ax.set_yticks(np.arange(n) + 0.5)  # 设置 Y 轴刻度位置

# X轴标签设置：旋转40度，右对齐，字体大小，颜色#333333，粗体
ax.set_xticklabels(cols, rotation=40, ha='right', fontsize=fs['tick'],
                   color='#333333', fontweight='normal')

# Y轴标签设置：水平，垂直居中，字体大小，颜色#333333，粗体
ax.set_yticklabels(cols, rotation=0, va='center', fontsize=fs['tick'],
                   color='#333333', fontweight='normal')

ax.tick_params(length=0)  # 隐藏刻度线

# =========================
# 5) 绘制分组框线 (颜色保持 #333333)
# =========================
# 将边界信息转换为列表
groups = [(label, s, e) for (label, s, e) in bounds]

for label, s, e in groups:
    offset_dist = 0.8  # 线条偏移距离
    leg_len = 0.2  # 线条末端长度
    gap = 0.15  # 线条断开间隙

    # 计算框线的坐标点
    x1 = s + offset_dist
    y1 = s - offset_dist
    x2 = e + 1 + offset_dist
    y2 = e + 1 - offset_dist

    rx1 = x1 + gap;
    ry1 = y1 + gap
    rx2 = x2 - gap;
    ry2 = y2 - gap

    lx1 = rx1 - leg_len;
    ly1 = ry1 + leg_len
    lx2 = rx2 - leg_len;
    ly2 = ry2 + leg_len

    path_x = [lx1, rx1, rx2, lx2]
    path_y = [ly1, ry1, ry2, ly2]

    # 绘制分组框线，颜色#333333
    ax.plot(path_x, path_y, color='#333333', lw=1.2, clip_on=False)

    text_offset = 0.6  # 文本偏移
    tx = (x1 + x2) / 2 + text_offset
    ty = (y1 + y2) / 2 - text_offset

    # 添加分组标签，颜色#333333
    ax.text(tx, ty, label, rotation=-45, ha='center', va='center',
            fontsize=fs['group'], color='#333333')

# =========================
# 6) 自定义平行 Colorbar
# =========================
# 设置 Colorbar 的起始和结束位置
cb_start = np.array([0.44 * n, 0])
cb_end = np.array([1.0 * n, 0.56 * n])
cb_vec = cb_end - cb_start  # Colorbar 向量
n_seg = 250  # 颜色分段数
width = 0.6  # Colorbar 宽度
w_vec = np.array([1, -1]) / np.sqrt(2) * width  # 宽度向量

# 绘制 Colorbar 的每一小段
for k in range(n_seg):
    c = cmap(k / n_seg)  # 获取颜色
    t = k / n_seg  # 当前位置比例
    p = cb_start + cb_vec * t  # 当前段起始点
    t_next = (k + 1) / n_seg  # 下一段位置比例
    p_next = cb_start + cb_vec * t_next  # 当前段结束点
    # 定义多边形顶点
    pts = np.array([p, p_next, p_next + w_vec, p + w_vec])
    # 添加多边形补丁
# --- 修改点：添加 alpha=0.7 参数 ---
    ax.add_patch(mpatches.Polygon(pts, color=c, alpha=0.7, ec=None, clip_on=False))

# --- Colorbar 刻度与标签 ---
tick_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 刻度值
tick_len_scale = 0.2  # 刻度线长度比例

for val in tick_vals:
    t_pos = cb_start + cb_vec * val  # 刻度位置
    tick_start = t_pos
    tick_end = t_pos - (w_vec * tick_len_scale)  # 刻度线终点

    # 绘制刻度线，颜色 #333333
    ax.plot([tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]],
            color='#333333', lw=1.2, clip_on=False)

    label_pos = tick_end - (w_vec * 0.5)  # 标签位置
    # 添加刻度标签，颜色 #333333
    ax.text(label_pos[0], label_pos[1], f"{val:.1f}",
            rotation=-45, ha='center', va='center',
            fontsize=fs['cbar_tick'], color='#333333')

# --- 修改3：Colorbar 标题颜色 #333333 ---
title_offset = w_vec * 2.5  # 标题偏移
title_pos = (cb_start + cb_end) / 2 + title_offset  # 标题位置

# 添加 Colorbar 标题，颜色 #333333
ax.text(title_pos[0], title_pos[1], r"Spearman correlation ($\rho$)",
        rotation=-45, ha='center', va='center',
        fontsize=fs['cbar_title'], color='#333333')

# =========================
# 7) 保存 (多格式：SVG, JPEG, TIFF, PNG)
# =========================
out_dir = OUTPUT_FIG_DIR
os.makedirs(out_dir, exist_ok=True)  # 确保目录存在

plt.subplots_adjust(bottom=0.15, left=0.15)  # 调整边距

# 定义文件名前缀
base_filename = "Fig_Spearman_ALL_FixedColors_NormalFont"
# 定义需要保存的所有格式
formats = ['png', 'svg', 'tiff', 'jpg']

print("--------------------------------------------------")
for fmt in formats:
    # 拼接完整路径
    save_path = os.path.join(out_dir, f"{base_filename}.{fmt}")
    print(f"正在保存 {fmt} 格式...")

    # 关键设置：
    # dpi=600: 高分辨率
    # bbox_inches='tight': 自动裁剪白边
    # facecolor='white': 确保背景为白色（防止 JPEG 变黑）
    try:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"   [成功] {save_path}")
    except Exception as e:
        print(f"   [失败] 无法保存 {fmt} 格式: {e}")

print("--------------------------------------------------")
plt.show()  # 显示图片

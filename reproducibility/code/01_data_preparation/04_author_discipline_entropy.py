# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
根据已生成的 author_address_disambiguation_full_new.csv
回查 wos_excels 文件夹中的 WoS Categories，
生成“作者层面学科分布 Shannon 熵”统计表。

核心说明：
1. 先构造 author_id - paper_id 唯一关系
2. 再从 WoS Excel 提取 paper_id -> disciplines
3. 将每位作者参与过的论文映射到学科分布
4. 计算作者层面的 Shannon entropy:
       H = - sum(p_i * ln(p_i))
   其中 p_i 为作者在第 i 个学科上的占比
5. 同时输出标准化 Shannon 熵：
       H_norm = H / ln(S)
   其中 S 为作者涉及的学科类别数（权重大于0的类别数）

可选计数方式：
- COUNT_MODE = 'fractional'
    一篇论文若属于 k 个学科，则每个学科给该作者贡献 1/k
    （更适合做学科分布概率）
- COUNT_MODE = 'full'
    一篇论文若属于 k 个学科，则每个学科都贡献 1
    （会放大多分类论文的贡献）

缺失 WoS Categories 的处理：
- 若 INCLUDE_NO_WOS_AS_ONE = True
  则无 WoS Categories 的论文记入一个占位学科 [NO_WOS_CATEGORY]
- 若 INCLUDE_NO_WOS_AS_ONE = False
  则该论文不进入 Shannon 熵计算
"""

import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd

# =========================
# 1. 路径配置
# =========================
FOLDER_WOS = str(_INPUT_ROOT / "01_data_preparation" / "wos_2000_2025")
PATH_AUTHOR_FILE = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_metrics" / "author_discipline_entropy.csv")

# =========================
# 2. 参数配置（可改）
# =========================
# 可选：'fractional' 或 'full'
COUNT_MODE = 'full'

# 若论文无 WoS Categories，是否按 1 个占位学科计入
INCLUDE_NO_WOS_AS_ONE = True

# 缺失 WoS Categories 时使用的占位类别名称
NO_WOS_PLACEHOLDER = '[NO_WOS_CATEGORY]'

# 是否输出保留小数位数
ROUND_DIGITS = 6

# =========================
# 3. 候选列名
# =========================
PAPER_COL_CANDIDATES = ['UT (Unique WOS ID)', 'UT']
WOS_CAT_COL_CANDIDATES = ['WoS Categories', 'Web of Science Categories', 'WC']


# =========================
# 4. 工具函数
# =========================
def detect_column(df: pd.DataFrame, candidates: list) -> str:
    """自动识别列名（不区分大小写）"""
    for cand in candidates:
        for col in df.columns:
            if str(cand).strip().lower() == str(col).strip().lower():
                return col
    return ''


def safe_str(x) -> str:
    if pd.isna(x):
        return ''
    s = str(x).strip()
    return '' if s.lower() == 'nan' else s


def split_disciplines(cat_text: str) -> list:
    """
    按分号拆分 WoS Categories
    例如：
    'Computer Science; Information Science & Library Science'
    -> ['Computer Science', 'Information Science & Library Science']
    """
    text = safe_str(cat_text)
    if not text:
        return []
    return [p.strip() for p in text.split(';') if p.strip()]


def read_excel_auto(file_path: str) -> pd.DataFrame:
    """
    根据扩展名自动选择读取引擎
    .xls -> xlrd
    .xlsx -> openpyxl
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.xls':
        try:
            return pd.read_excel(file_path, dtype=str, engine='xlrd')
        except ImportError:
            raise ImportError(
                "读取 .xls 需要安装 xlrd。请先运行：pip install xlrd>=2.0.1 或 conda install xlrd"
            )
    elif ext == '.xlsx':
        return pd.read_excel(file_path, dtype=str, engine='openpyxl')
    else:
        raise ValueError(f'不支持的文件类型: {file_path}')


def pick_author_name(df_author: pd.DataFrame) -> pd.DataFrame:
    """每个 author_id 选出现频次最高的 raw_name 作为 author_name"""
    if 'raw_name' not in df_author.columns:
        return pd.DataFrame(columns=['author_id', 'author_name'])

    name_df = (
        df_author.groupby(['author_id', 'raw_name'])
                 .size()
                 .reset_index(name='cnt')
                 .sort_values(['author_id', 'cnt', 'raw_name'], ascending=[True, False, True])
                 .drop_duplicates('author_id', keep='first')
                 [['author_id', 'raw_name']]
                 .rename(columns={'raw_name': 'author_name'})
    )
    return name_df


def shannon_entropy_from_weights(weight_dict: dict) -> tuple:
    """
    根据 {discipline: weight} 计算：
    - Shannon entropy
    - normalized Shannon entropy
    - discipline_count（正权重类别数）
    - total_weight（总权重）
    """
    weights = np.array([v for v in weight_dict.values() if v > 0], dtype=float)

    if len(weights) == 0:
        return 0.0, 0.0, 0, 0.0

    total_weight = weights.sum()
    if total_weight <= 0:
        return 0.0, 0.0, 0, 0.0

    probs = weights / total_weight
    H = -np.sum(probs * np.log(probs))

    S = len(weights)
    if S > 1:
        H_norm = H / np.log(S)
    else:
        H_norm = 0.0

    return float(H), float(H_norm), int(S), float(total_weight)


# =========================
# 5. 从 WoS Excel 提取 paper_id -> disciplines
# =========================
print('>>> 开始读取 WoS Excel 并提取 WoS Categories ...')

file_paths = glob.glob(os.path.join(FOLDER_WOS, '*.xls')) + glob.glob(os.path.join(FOLDER_WOS, '*.xlsx'))
print(f'检测到 Excel 文件数：{len(file_paths)}')

paper_disc_map = defaultdict(set)

for file_path in file_paths:
    try:
        df = read_excel_auto(file_path)
        print(f'√ 已读取：{file_path}')
    except Exception as e:
        print(f'! 无法读取 {file_path}（{e}），已跳过。')
        continue

    paper_col = detect_column(df, PAPER_COL_CANDIDATES)
    wos_col = detect_column(df, WOS_CAT_COL_CANDIDATES)

    print(f'  识别到论文ID列: {paper_col if paper_col else "未找到"}')
    print(f'  识别到WoS学科列: {wos_col if wos_col else "未找到"}')

    if not paper_col:
        print(f'! 文件缺少论文ID列（UT）：{file_path}，已跳过。')
        print(f'  实际列名示例：{list(df.columns)[:20]}')
        continue

    if not wos_col:
        print(f'! 文件缺少 WoS Categories 列：{file_path}，已跳过。')
        print(f'  实际列名示例：{list(df.columns)[:20]}')
        continue

    sub = df[[paper_col, wos_col]].copy()

    for _, row in sub.iterrows():
        pid = safe_str(row[paper_col])
        if not pid:
            continue

        disc_list = split_disciplines(row[wos_col])
        for d in disc_list:
            paper_disc_map[pid].add(d)

paper_rows = []
for pid, disc_set in paper_disc_map.items():
    disc_list_sorted = sorted(disc_set)
    paper_rows.append({
        'paper_id': pid,
        'paper_discipline_list': '; '.join(disc_list_sorted),
        'paper_discipline_count': len(disc_list_sorted)
    })

paper_disc_df = pd.DataFrame(
    paper_rows,
    columns=['paper_id', 'paper_discipline_list', 'paper_discipline_count']
)

print(f'提取到带 WoS Categories 的论文数：{len(paper_disc_df)}')

# =========================
# 6. 读取作者消歧结果，并整理作者-论文唯一关系
# =========================
print('\n>>> 开始读取作者消歧结果并整理作者参与论文 ...')

author_df = pd.read_csv(PATH_AUTHOR_FILE, dtype=str, encoding='utf-8-sig')

required_cols = ['author_id', 'paper_id']
for col in required_cols:
    if col not in author_df.columns:
        raise ValueError(f'缺少必要列：{col}')

author_df['author_id'] = author_df['author_id'].fillna('').astype(str).str.strip()
author_df['paper_id'] = author_df['paper_id'].fillna('').astype(str).str.strip()

author_df = author_df[(author_df['author_id'] != '') & (author_df['paper_id'] != '')].copy()

# 保留当前 CSV 原始行顺序
author_df['_row_order'] = np.arange(len(author_df))

# 同一论文内，同一 author_id 可能有多条记录（如 multiple_addresses）
# 按 paper_id + author_id 去重，保留最先出现的一条
paper_author_unique = (
    author_df.sort_values('_row_order')
             .drop_duplicates(subset=['paper_id', 'author_id'], keep='first')
             .copy()
)

print(f'作者-论文唯一记录数：{len(paper_author_unique)}')

# 合并论文学科信息
author_paper_disc = paper_author_unique.merge(paper_disc_df, on='paper_id', how='left')

author_paper_disc['paper_discipline_list'] = author_paper_disc['paper_discipline_list'].fillna('')
author_paper_disc['paper_discipline_count'] = pd.to_numeric(
    author_paper_disc['paper_discipline_count'], errors='coerce'
).fillna(0).astype(int)

# 是否该论文有 WoS Categories
author_paper_disc['has_wos_categories'] = author_paper_disc['paper_discipline_count'].apply(
    lambda n: 1 if n > 0 else 0
)

# =========================
# 7. 构造作者层面的学科权重分布
# =========================
print('\n>>> 开始构造作者层面的学科分布 ...')

# author_id -> {discipline: weight}
author_discipline_weights = defaultdict(lambda: defaultdict(float))

for row in author_paper_disc.itertuples(index=False):
    author_id = safe_str(row.author_id)
    disc_text = safe_str(row.paper_discipline_list)
    disc_list = split_disciplines(disc_text)

    # 若该论文无 WoS Categories
    if len(disc_list) == 0:
        if INCLUDE_NO_WOS_AS_ONE:
            author_discipline_weights[author_id][NO_WOS_PLACEHOLDER] += 1.0
        continue

    # 有 WoS Categories
    k = len(disc_list)

    if COUNT_MODE.lower() == 'fractional':
        weight = 1.0 / k
        for d in disc_list:
            author_discipline_weights[author_id][d] += weight
    elif COUNT_MODE.lower() == 'full':
        for d in disc_list:
            author_discipline_weights[author_id][d] += 1.0
    else:
        raise ValueError("COUNT_MODE 只能是 'fractional' 或 'full'")

print(f'已构造作者学科分布数：{len(author_discipline_weights)}')

# =========================
# 8. 计算作者层面 Shannon 熵
# =========================
print('>>> 开始计算作者层面 Shannon 熵 ...')

author_shannon_rows = []

for author_id, weight_dict in author_discipline_weights.items():
    H, H_norm, discipline_count, total_weight = shannon_entropy_from_weights(weight_dict)

    author_shannon_rows.append({
        'author_id': author_id,
        'discipline_count_in_entropy': discipline_count,
        'total_discipline_weight': total_weight,
        'shannon_entropy': round(H, ROUND_DIGITS),
        'shannon_entropy_norm': round(H_norm, ROUND_DIGITS)
    })

author_shannon_df = pd.DataFrame(author_shannon_rows)

# =========================
# 9. 汇总作者基础统计
# =========================
print('>>> 开始汇总作者基础统计 ...')

author_base_summary = (
    author_paper_disc.groupby('author_id')
    .agg(
        participated_paper_count=('paper_id', 'nunique'),
        papers_with_wos_categories=('has_wos_categories', 'sum')
    )
    .reset_index()
)

author_base_summary['participated_paper_count'] = (
    pd.to_numeric(author_base_summary['participated_paper_count'], errors='coerce')
    .fillna(0).astype(int)
)

author_base_summary['papers_with_wos_categories'] = (
    pd.to_numeric(author_base_summary['papers_with_wos_categories'], errors='coerce')
    .fillna(0).astype(int)
)

# 补作者名
name_df = pick_author_name(author_df)

# 合并
author_summary = author_base_summary.merge(author_shannon_df, on='author_id', how='left')
author_summary = author_summary.merge(name_df, on='author_id', how='left')

# 对于没有进入熵计算的作者（如全部论文无学科且你设置 INCLUDE_NO_WOS_AS_ONE=False）
for col in ['discipline_count_in_entropy', 'total_discipline_weight', 'shannon_entropy', 'shannon_entropy_norm']:
    if col in author_summary.columns:
        author_summary[col] = pd.to_numeric(author_summary[col], errors='coerce').fillna(0)

author_summary['discipline_count_in_entropy'] = author_summary['discipline_count_in_entropy'].astype(int)
author_summary['total_discipline_weight'] = author_summary['total_discipline_weight'].round(ROUND_DIGITS)
author_summary['shannon_entropy'] = author_summary['shannon_entropy'].round(ROUND_DIGITS)
author_summary['shannon_entropy_norm'] = author_summary['shannon_entropy_norm'].round(ROUND_DIGITS)

# =========================
# 10. 列顺序与排序
# =========================
output_cols = [
    'author_id',
    'author_name',
    'participated_paper_count',
    'papers_with_wos_categories',
    'discipline_count_in_entropy',
    'total_discipline_weight',
    'shannon_entropy',
    'shannon_entropy_norm'
]

author_summary = author_summary[output_cols].copy()

author_summary = author_summary.sort_values(
    by=[
        'shannon_entropy_norm',
        'shannon_entropy',
        'participated_paper_count',
        'author_id'
    ],
    ascending=[False, False, False, True]
).reset_index(drop=True)

# =========================
# 11. 输出
# =========================
author_summary.to_csv(PATH_OUTPUT, index=False, encoding='utf-8-sig')

print(f'\n[OK] 完成！作者层面 Shannon 熵结果已保存至：{PATH_OUTPUT}')
print(f'共输出作者数：{len(author_summary)}')
print(f'当前 COUNT_MODE = {COUNT_MODE}')
print(f'当前 INCLUDE_NO_WOS_AS_ONE = {INCLUDE_NO_WOS_AS_ONE}')
if INCLUDE_NO_WOS_AS_ONE:
    print(f'当前无 WoS Categories 占位类别 = {NO_WOS_PLACEHOLDER}')

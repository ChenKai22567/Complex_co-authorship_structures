# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
根据已生成的 author_address_disambiguation_full_new.csv
回查 wos_excels 文件夹中的 WoS Categories，
生成“作者参与过的文章”的平均学科数统计表。

规则：
- 若论文有 WoS Categories，则学科数 = 实际 WoS Categories 数
- 若论文无 WoS Categories，则学科数按 1 计
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
PATH_OUTPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_metrics" / "author_discipline_count.csv")

# =========================
# 2. 候选列名
# =========================
PAPER_COL_CANDIDATES = ['UT (Unique WOS ID)', 'UT']
WOS_CAT_COL_CANDIDATES = ['WoS Categories', 'Web of Science Categories', 'WC']

# =========================
# 3. 工具函数
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


# =========================
# 4. 从 WoS Excel 提取 paper_id -> disciplines
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
# 5. 读取作者消歧结果，并整理作者-论文唯一关系
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

# 若无 WoS Categories，则学科数按 1 计
author_paper_disc['paper_discipline_count_adjusted'] = author_paper_disc['paper_discipline_count'].apply(
    lambda n: n if n > 0 else 1
)

# 是否该论文有 WoS Categories
author_paper_disc['has_wos_categories'] = author_paper_disc['paper_discipline_count'].apply(
    lambda n: 1 if n > 0 else 0
)

# =========================
# 6. 统计“作者参与过的文章”的平均学科数
# =========================
print('>>> 开始汇总作者参与论文的平均学科数 ...')

author_summary = (
    author_paper_disc.groupby('author_id')
    .agg(
        participated_paper_count=('paper_id', 'nunique'),
        papers_with_wos_categories=('has_wos_categories', 'sum'),
        total_discipline_count_adjusted=('paper_discipline_count_adjusted', 'sum')
    )
    .reset_index()
)

author_summary['avg_participated_paper_discipline_count'] = (
    author_summary['total_discipline_count_adjusted'] / author_summary['participated_paper_count']
)

author_summary['avg_participated_paper_discipline_count'] = (
    pd.to_numeric(author_summary['avg_participated_paper_discipline_count'], errors='coerce')
    .fillna(0)
    .round(4)
)

author_summary['participated_paper_count'] = (
    pd.to_numeric(author_summary['participated_paper_count'], errors='coerce')
    .fillna(0)
    .astype(int)
)

author_summary['papers_with_wos_categories'] = (
    pd.to_numeric(author_summary['papers_with_wos_categories'], errors='coerce')
    .fillna(0)
    .astype(int)
)

# 补作者名
name_df = pick_author_name(author_df)
author_summary = author_summary.merge(name_df, on='author_id', how='left')

# 列顺序
output_cols = [
    'author_id',
    'author_name',
    'participated_paper_count',
    'papers_with_wos_categories',
    'avg_participated_paper_discipline_count'
]
author_summary = author_summary[output_cols].copy()

# 排序
author_summary = author_summary.sort_values(
    by=[
        'avg_participated_paper_discipline_count',
        'participated_paper_count',
        'author_id'
    ],
    ascending=[False, False, True]
).reset_index(drop=True)

# 输出
author_summary.to_csv(PATH_OUTPUT, index=False, encoding='utf-8-sig')
print(f'[OK] 完成！作者参与论文平均学科数已保存至：{PATH_OUTPUT}')
print(f'共输出作者数：{len(author_summary)}')

# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
计算作者影响力是否高于其平均合作者影响力（1/0）

输入文件：
1. author_attribute.csv
   需要至少包含：
   - Id
   - totalcitation

2. edge.csv
   需要至少包含：
   - Source
   - Target
   - Weight（本脚本默认不使用，仅保留以备扩展）

输出文件：
- author_influence_vs_avg_collaborator.csv

规则：
- 影响力指标使用 totalcitation
- 对每位作者，找到其所有合作者
- 计算这些合作者的 totalcitation 平均值
- 若作者自己的 totalcitation > 平均合作者 totalcitation，则记为 1，否则为 0
- 若作者没有可用合作者（如孤立点，或合作者不在 author_attribute 中），默认记为 0
"""

import os
import pandas as pd
import numpy as np


# =========================
# 1. 路径配置
# =========================
BASE_DIR = str(_INPUT_ROOT / "01_data_preparation" / "author_influence")
PATH_AUTHOR = os.path.join(BASE_DIR, 'author.csv')
PATH_EDGE = os.path.join(BASE_DIR, 'edge.csv')
PATH_OUTPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_metrics" / "author_neighbor_influence.csv")


# =========================
# 2. 参数配置
# =========================
AUTHOR_ID_COL = 'Id'
INFLUENCE_COL = 'totalcitation'

EDGE_SOURCE_COL = 'Source'
EDGE_TARGET_COL = 'Target'
EDGE_WEIGHT_COL = 'Weight'   # 当前脚本默认不加权，仅保留

# 当作者没有可用合作者时，二元结果赋值为何
# 可选 0 或 np.nan
NO_COLLAB_RESULT = 0

# 是否去除重复无向边
# 若 edge.csv 中每对作者只出现一次，保持 True 即可
DEDUP_UNDIRECTED_EDGES = True


# =========================
# 3. 工具函数
# =========================
def read_csv_auto(path):
    """自动尝试不同编码读取 CSV"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin1']
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


def standardize_str_col(series):
    """统一转为字符串并去空格"""
    return series.fillna('').astype(str).str.strip()


# =========================
# 4. 读取作者属性表
# =========================
print('>>> 读取 author_attribute.csv ...')
author_df = read_csv_auto(PATH_AUTHOR)

required_author_cols = [AUTHOR_ID_COL, INFLUENCE_COL]
for col in required_author_cols:
    if col not in author_df.columns:
        raise ValueError(f'author_attribute.csv 缺少必要列：{col}')

author_df = author_df.copy()
author_df[AUTHOR_ID_COL] = standardize_str_col(author_df[AUTHOR_ID_COL])
author_df[INFLUENCE_COL] = pd.to_numeric(author_df[INFLUENCE_COL], errors='coerce')

# 去除无 author_id 的记录
author_df = author_df[author_df[AUTHOR_ID_COL] != ''].copy()

# 若同一个作者重复出现，保留第一条；也可改成聚合
author_df = author_df.drop_duplicates(subset=[AUTHOR_ID_COL], keep='first').copy()

print(f'作者数：{len(author_df)}')


# =========================
# 5. 读取边表
# =========================
print('>>> 读取 edge.csv ...')
edge_df = read_csv_auto(PATH_EDGE)

required_edge_cols = [EDGE_SOURCE_COL, EDGE_TARGET_COL]
for col in required_edge_cols:
    if col not in edge_df.columns:
        raise ValueError(f'edge.csv 缺少必要列：{col}')

edge_df = edge_df.copy()
edge_df[EDGE_SOURCE_COL] = standardize_str_col(edge_df[EDGE_SOURCE_COL])
edge_df[EDGE_TARGET_COL] = standardize_str_col(edge_df[EDGE_TARGET_COL])

# 删除空端点和自环
edge_df = edge_df[
    (edge_df[EDGE_SOURCE_COL] != '') &
    (edge_df[EDGE_TARGET_COL] != '') &
    (edge_df[EDGE_SOURCE_COL] != edge_df[EDGE_TARGET_COL])
].copy()

# 若有 Weight 列则转数值；没有也不影响
if EDGE_WEIGHT_COL in edge_df.columns:
    edge_df[EDGE_WEIGHT_COL] = pd.to_numeric(edge_df[EDGE_WEIGHT_COL], errors='coerce')

# 无向边去重：A-B 和 B-A 视为同一条
if DEDUP_UNDIRECTED_EDGES:
    edge_df['node_min'] = edge_df[[EDGE_SOURCE_COL, EDGE_TARGET_COL]].min(axis=1)
    edge_df['node_max'] = edge_df[[EDGE_SOURCE_COL, EDGE_TARGET_COL]].max(axis=1)
    edge_df = edge_df.drop_duplicates(subset=['node_min', 'node_max'], keep='first').copy()
    edge_df = edge_df.drop(columns=['node_min', 'node_max'])

print(f'有效无向边数：{len(edge_df)}')


# =========================
# 6. 构造“作者-合作者”对应表（双向展开）
# =========================
print('>>> 构造作者-合作者对应关系 ...')

pairs_1 = edge_df[[EDGE_SOURCE_COL, EDGE_TARGET_COL]].copy()
pairs_1.columns = ['author_id', 'coauthor_id']

pairs_2 = edge_df[[EDGE_TARGET_COL, EDGE_SOURCE_COL]].copy()
pairs_2.columns = ['author_id', 'coauthor_id']

author_coauthor_df = pd.concat([pairs_1, pairs_2], ignore_index=True)

# 去重，避免重复合作者关系
author_coauthor_df = author_coauthor_df.drop_duplicates(subset=['author_id', 'coauthor_id']).copy()

print(f'作者-合作者关系数：{len(author_coauthor_df)}')


# =========================
# 7. 合并合作者影响力
# =========================
print('>>> 合并合作者影响力 ...')

author_infl = author_df[[AUTHOR_ID_COL, INFLUENCE_COL]].copy()
author_infl.columns = ['author_id_ref', 'author_influence']

coauthor_infl = author_df[[AUTHOR_ID_COL, INFLUENCE_COL]].copy()
coauthor_infl.columns = ['coauthor_id', 'coauthor_influence']

# 合作者影响力
merged_df = author_coauthor_df.merge(
    coauthor_infl,
    on='coauthor_id',
    how='left'
)

# 作者自身影响力
merged_df = merged_df.merge(
    author_infl,
    left_on='author_id',
    right_on='author_id_ref',
    how='left'
)

merged_df = merged_df.drop(columns=['author_id_ref'])

print(f'可用于计算的作者-合作者记录数：{len(merged_df)}')


# =========================
# 8. 计算每位作者的平均合作者影响力
# =========================
print('>>> 计算平均合作者影响力 ...')

summary_df = (
    merged_df.groupby('author_id', as_index=False)
    .agg(
        collaborator_count=('coauthor_id', 'nunique'),
        valid_coauthor_count=('coauthor_influence', lambda x: x.notna().sum()),
        avg_coauthor_influence=('coauthor_influence', 'mean')
    )
)

# 合并作者自身影响力
result_df = author_df[[AUTHOR_ID_COL, INFLUENCE_COL]].copy()
result_df.columns = ['author_id', 'author_influence']

result_df = result_df.merge(summary_df, on='author_id', how='left')

# 缺失填充
result_df['collaborator_count'] = pd.to_numeric(
    result_df['collaborator_count'], errors='coerce'
).fillna(0).astype(int)

result_df['valid_coauthor_count'] = pd.to_numeric(
    result_df['valid_coauthor_count'], errors='coerce'
).fillna(0).astype(int)

result_df['avg_coauthor_influence'] = pd.to_numeric(
    result_df['avg_coauthor_influence'], errors='coerce'
)

# =========================
# 9. 生成 1/0 指标
# =========================
print('>>> 生成二元指标 ...')

def compare_influence(row):
    """
    若作者影响力 > 平均合作者影响力，则记为1，否则为0
    若没有可用合作者，则返回 NO_COLLAB_RESULT
    """
    if row['valid_coauthor_count'] <= 0 or pd.isna(row['avg_coauthor_influence']):
        return NO_COLLAB_RESULT
    if pd.isna(row['author_influence']):
        return 0
    return 1 if row['author_influence'] > row['avg_coauthor_influence'] else 0

result_df['influence_higher_than_avg_coauthor'] = result_df.apply(compare_influence, axis=1)

# 可选：顺便输出差值，便于检查
result_df['influence_minus_avg_coauthor'] = (
    result_df['author_influence'] - result_df['avg_coauthor_influence']
)

# 排序
result_df = result_df.sort_values(
    by=['influence_higher_than_avg_coauthor', 'influence_minus_avg_coauthor', 'author_id'],
    ascending=[False, False, True]
).reset_index(drop=True)

# =========================
# 10. 输出
# =========================
output_cols = [
    'author_id',
    'author_influence',
    'collaborator_count',
    'valid_coauthor_count',
    'avg_coauthor_influence',
    'influence_minus_avg_coauthor',
    'influence_higher_than_avg_coauthor'
]

result_df = result_df[output_cols].copy()
result_df.to_csv(PATH_OUTPUT, index=False, encoding='utf-8-sig')

print(f'[OK] 完成！结果已输出到：{PATH_OUTPUT}')
print(f'输出作者数：{len(result_df)}')

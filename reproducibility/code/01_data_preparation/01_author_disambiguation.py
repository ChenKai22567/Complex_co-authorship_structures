# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
完整作者地址提取 + 消歧脚本

本脚本针对 Web of Science 导出的多份 .xls/.xlsx 文件，
批量抽取、清洗并消歧作者信息，最终生成行级和作者级两套输出表。

新增功能：
 - 按每篇文章的作者数量统计分布，分箱为 ['1', '2', '3', '4', '5', '≥6']，
   并输出 author_count_distribution.csv。

主要功能：
 1. 批量读取指定文件夹下所有 WOS 导出的 .xls/.xlsx 文件；
    - 使用 pandas + xlrd/openpyxl 读取；若 Addresses 或 Keywords 字段为真正的 NaN
      或字符串 "nan"，会被视为空。
 2. 对每篇论文记录逐行解析，抽取以下字段：
    - UT (Unique WOS ID)           ：论文唯一标识
    - Author Full Names            ：作者姓名列表（分号分隔）
    - Addresses                    ：原始地址，NaN/“nan”视为空
    - Publication Year             ：发表年份（取前4位）
    - Author Keywords              ：关键词列表（分号分隔，NaN/“nan”视为空）
    - Times Cited, WoS Core        ：被引次数（整数，非数字视为0）
 3. 对 Addresses 做多格式分类与解析，提取 raw_affiliation 和简化 original_address：
    - bracketed          ：“[作者1; 作者2] 机构A; …” → 一一映射
    - single             ：单一机构，所有作者共用
    - multiple_addresses ：单作者对应多机构，拆分为多条记录
    - ambiguous          ：多作者 & 多机构，拆分每个机构并简化
    - no_address         ：Addresses 为空 → category='no_address'，original_address=''
    - other              ：不符合以上规则的特殊格式，同 no_address 处理并收集示例
 4. 控制台输出每类原始论文行数统计，展示部分 other 示例，辅助手动优化
 5. 扁平化生成“作者实例”表，字段包括：
    paper_id, raw_name, raw_affiliation, original_address,
    category, publication_year, keywords, times_cited
 6. 文本清洗与特征提取：
    - name_clean：小写、去标点、归一空格后的姓名
    - aff_clean : 小写、去标点、归一空格后的机构 raw_affiliation
    - initials  : name_clean 每词首字母拼接，用于分块(Group)加速匹配
 7. 构建合著网络 co_graph：
    - 节点：每个作者实例（DataFrame 行索引）
    - 边  ：同一篇 paper_id 下两实例共同出现
    - 控制台输出网络节点数和边数
 8. 生成候选对 candidate_pairs（含 direct_merges & remaining_candidates）：
    8.1 先用 high_name_match + (关键词交集 or 简化地址相似度 ≥0.9) + 地址非空 →
        direct_merges（高置信度直接合并对）
    8.2 否则若 (姓名相似度 ≥0.8 & 地址相似度 ≥0.5) or (姓名相似度 ≥0.9) →
        remaining_candidates（待网络验证）
    - 排除同篇论文合著者
    - 导出最终 remaining_candidates 到 author_merge_candidates.csv
    - 控制台输出 direct_merges 与 remaining_candidates 数量
 9. 迭代网络验证合并（Union–Find）：
    - 初始化并查集 UnionFind(all instances)
    - 7.1 合并所有 direct_merges 对
    - 7.2 反复遍历 remaining_candidates，
          若候选对在 co_graph 上有任何公共邻居且该邻居已同源合并→ union(i,j)
          直至本轮无新合并（changed=False）
    - 控制台输出网络验证合并对数
 10. 根据并查集根分配唯一 author_id：
    - root 出现时分配 “author_1, author_2, …”
    - 每实例映射到其根对应的 author_id
 11. 保存行级消歧结果：
    - author_address_disambiguation_full.csv（所有实例及其 author_id）
 12. 保存“验证通过”作者实例：
    - validated_authors.csv（仅 direct_merges + network_validated 对应实例）
      字段：instance_id, paper_id, raw_name, original_address, author_id
 13. 作者级汇总 author_summary.csv：
    - 聚合指标：
      • paper_ids         ：去重排序的 UT 列表
      • publication_years ：去重排序的年份列表
      • keyword_list      ：拆分、过滤na后的关键词列表
      • total_times_cited ：times_cited 求和
      • affiliation_list  ：去重、过滤空后的简化地址列表
      • top_affiliation   ：出现频次最高的简化地址
      • paper_count       ：论文数量
      • author_name       ：按 raw_name 出现频次最高者
 14. 控制台依序打印进度与统计信息，方便监控

使用指南：
 1) 修改脚本顶部 folder_path 为你的 WOS 导出文件夹路径；
 2) 安装依赖：pandas, networkx, xlrd, openpyxl；
 3) 运行脚本，关注控制台输出；
 4) 在 output_csv 所在目录下获取：
    - author_address_disambiguation_full.csv
    - author_merge_candidates.csv
    - validated_authors.csv
    - author_summary.csv
"""

import os
import glob
import re
from itertools import combinations
from difflib import SequenceMatcher
import pandas as pd
import networkx as nx
from collections import Counter
from functools import lru_cache

# ———— 配置区 ———— #
# 1) 存放所有 .xls 文件的文件夹路径
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = str(_INPUT_ROOT / "01_data_preparation" / "wos_2000_2025")
# 2) 最终 CSV 输出路径
output_dir = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation")
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'author_disambiguation_records.csv')

# ———— 地址简化函数 ———— #
def simplify_address(addr: str) -> str:
    """
    将一般长地址简化为“首段, 尾段”格式。
    例如：
      "Vrije Univ Amsterdam, Dept Commun Sci, ... , Netherlands"
    → "Vrije Univ Amsterdam, Netherlands"
    """
    # 按逗号拆分并去除空项
    parts = [p.strip() for p in addr.split(',') if p.strip()]
    # 至少保留第一段和最后一段
    if len(parts) >= 2:
        return f"{parts[0]}, {parts[-1]}"
    # 不足两段则返回原始
    return addr

# ———— 正则和统计容器 ———— #
# 匹配 “[作者1; 作者2] 机构” 的模式
pattern = re.compile(r'\[([^\]]+)\]\s*([^;\[\]]+)')
# 各类别论文行数统计
category_counts = {
    'bracketed': 0,
    'single': 0,
    'multiple_addresses': 0,
    'ambiguous': 0,
    'no_address': 0,
    'other': 0
}
# 文章作者数量分布统计，统计单位是原始论文行
AUTHOR_COUNT_BINS = ['1', '2', '3', '4', '5', '≥6']
author_count_distribution = {label: 0 for label in AUTHOR_COUNT_BINS}
missing_author_count = 0
# 存储 other 类别样本以便打印
other_records = []
# 汇总所有作者实例的列表
all_records = []

# ———— 批量遍历文件夹中的 .xls 文件 ———— #
file_paths = glob.glob(os.path.join(folder_path, '*.xls'))
for file_path in file_paths:
    try:
        # 读取 Excel (.xls) 文件，使用 xlrd 引擎
        df = pd.read_excel(file_path, dtype=str, engine='xlrd')
        print(f'√ 已读取 Excel：{file_path}')
    except Exception as e:
        # 若读取失败，则跳过此文件
        print(f'! 无法读取 {file_path}（{e}），已跳过。')
        continue

    # 遍历该表的每一行（对应一篇论文）
    for idx, row in df.iterrows():
        # 提取论文唯一标识 UT
        pid = str(row.get('UT (Unique WOS ID)', '')).strip()
        # 拆分“Author Full Names”字段，分号分隔多个作者
        authors = [a.strip() for a in str(row.get('Author Full Names', '')).split(';') if a.strip()]
        author_count = len(authors)
        if author_count >= 6:
            author_count_distribution['≥6'] += 1
        elif author_count >= 1:
            author_count_distribution[str(author_count)] += 1
        else:
            missing_author_count += 1
        # 原始 Addresses 文本
        raw_addr = row.get('Addresses', '')
        if pd.isna(raw_addr) or str(raw_addr).strip().lower() == 'nan':
            addr_text = ''    # 视作无地址
        else:
            addr_text = str(raw_addr).strip()
        # 提取“Publication Year”前4位
        year_raw = str(row.get('Publication Year', '')).strip()
        pub_year = year_raw[:4] if len(year_raw) >= 4 else year_raw
        # 原始 Author Keywords 文本
        raw_kw = row.get('Author Keywords', '')
        if pd.isna(raw_kw) or str(raw_kw).strip().lower() == 'nan':
            keywords = ''
        else:
            keywords = str(raw_kw).strip()
        # —— 新增：提取 Times Cited 字段 —— #
        # 如果字段名不是完全一致，请根据你的表头做相应修改
        times_cited_raw = str(row.get('Times Cited, WoS Core', '')).strip()
        # 将空值或非数字视作 0
        try:
            times_cited = int(times_cited_raw)
        except:
            times_cited = 0

        # — 分类判断 — #
        if not addr_text:
            cat = 'no_address'
        elif pattern.search(addr_text):
            cat = 'bracketed'
        else:
            inst_list = [i.strip() for i in addr_text.split(';') if i.strip()]
            if len(authors) == 1 and len(inst_list) > 1:
                cat = 'multiple_addresses'
            elif len(inst_list) == 1:
                cat = 'single'
            elif len(authors) > 1 and len(inst_list) > 1:
                cat = 'ambiguous'
            else:
                cat = 'other'
        # 累加该类别的论文行数
        category_counts[cat] += 1

        # 如果是 other 类别，保存示例以便后续打印
        if cat == 'other':
            other_records.append({
                'file': os.path.basename(file_path),
                'row_index': idx,
                'UT': pid,
                'Authors': authors,
                'Addresses': addr_text
            })

        # — 根据类别提取 raw_affiliation 和 original_address — #
        if cat == 'bracketed':
            # “[作者] 机构” → 构建作者到机构的映射
            aff_map = {}
            for m in pattern.finditer(addr_text):
                names = [n.strip() for n in m.group(1).split(';') if n.strip()]
                inst_full = m.group(2).strip()
                for nm in names:
                    aff_map[nm] = inst_full
            # 为每个作者生成一条记录
            for au in authors:
                inst_full = aff_map.get(au, '')
                all_records.append({
                    'paper_id': pid,
                    'raw_name': au,
                    'raw_affiliation': inst_full,
                    # original_address 用简化后的机构名
                    'original_address': simplify_address(inst_full),
                    'category': cat,
                    'publication_year': pub_year,
                    'keywords': keywords,
                    'times_cited': times_cited
                })

        elif cat == 'multiple_addresses':
            # 单作者，多机构 → 每个机构一条记录
            au = authors[0]
            inst_list = [i.strip() for i in addr_text.split(';') if i.strip()]
            for inst_full in inst_list:
                all_records.append({
                    'paper_id': pid,
                    'raw_name': au,
                    'raw_affiliation': inst_full,
                    'original_address': simplify_address(inst_full),
                    'category': cat,
                    'publication_year': pub_year,
                    'keywords': keywords,
                    'times_cited': times_cited
                })

        elif cat == 'single':
            # 单一机构 → 所有作者同一机构
            inst_full = addr_text
            for au in authors:
                all_records.append({
                    'paper_id': pid,
                    'raw_name': au,
                    'raw_affiliation': inst_full,
                    'original_address': simplify_address(inst_full),
                    'category': cat,
                    'publication_year': pub_year,
                    'keywords': keywords,
                    'times_cited': times_cited
                })

        elif cat == 'ambiguous':
            # 多作者 & 多机构 → 标记 ambiguous 并保留原始地址
            for au in authors:
                all_records.append({
                    'paper_id': pid,
                    'raw_name': au,
                    'raw_affiliation': 'ambiguous',
                    'original_address': simplify_address(inst_full),
                    'category': cat,
                    'publication_year': pub_year,
                    'keywords': keywords,
                    'times_cited': times_cited
                })

        else:  # no_address & other
            for au in authors:
                all_records.append({
                    'paper_id': pid,
                    'raw_name': au,
                    'raw_affiliation': cat,  # 'no_address' 或 ''
                    'original_address': '',
                    'category': cat,
                    'publication_year': pub_year,
                    'keywords': keywords,
                    'times_cited': times_cited
                })

# ———— 将所有记录装入 DataFrame ———— #
records_df = pd.DataFrame(all_records)

# ———— 控制台输出统计信息 & other 类别示例 ———— #
print(f'\n共处理 {len(file_paths)} 个文件')
total_rows = sum(category_counts.values())
print(f'总论文行数：{total_rows}')
print('各类别原始论文行数统计：')
for k, v in category_counts.items():
    print(f'  {k}: {v}')

author_count_df = pd.DataFrame({
    'author_count_bin': AUTHOR_COUNT_BINS,
    'paper_count': [author_count_distribution[label] for label in AUTHOR_COUNT_BINS]
})
author_count_path = os.path.join(output_dir, 'author_count_distribution.csv')
author_count_df.to_csv(author_count_path, index=False, encoding='utf-8-sig')
print('\n文章作者数量分布：')
for _, row in author_count_df.iterrows():
    print(f"  {row['author_count_bin']}: {row['paper_count']}")
if missing_author_count:
    print(f"  missing_author_full_names: {missing_author_count}")
print(f"[OK] 已保存文章作者数量分布到：{author_count_path}")

print('\n【OTHER 类别示例】（最多前10条）')
for rec in other_records[:10]:
    print(f"文件 {rec['file']} 行 {rec['row_index']} | UT={rec['UT']} | Authors={rec['Authors']} | Addresses={rec['Addresses']}")

if records_df.empty:
    raise SystemExit(
        "No author records were read. Check that data/01_data_preparation/wos_2000_2025 contains readable WOS .xls files "
        "and that xlrd is installed in the active Python environment."
    )

# ———— 4. 文本清洗 & 特征提取 ———— #
def clean_text(s: str) -> str:
    """统一小写、去除非字母数字字符、多空格归一"""
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

records_df['name_clean'] = records_df['raw_name'].apply(clean_text)
records_df['aff_clean']  = records_df['raw_affiliation'].apply(clean_text)
records_df['initials']   = records_df['name_clean'].apply(lambda x: ''.join(w[0] for w in x.split()))

# ———— 5. 构建合著网络 co_graph ————
#   节点：每个作者实例（records_df 的行索引）
#   边：同在一篇 paper_id 下出现的两位作者实例，表示共同合著
co_graph = nx.Graph()
co_graph.add_nodes_from(records_df.index)
for pid, grp in records_df.groupby('paper_id'):
    idxs = grp.index.tolist()
    for u, v in combinations(idxs, 2):
        co_graph.add_edge(u, v)

# 输出合著网络的基本信息
print(f"合著网络节点数：{co_graph.number_of_nodes()}，边数：{co_graph.number_of_edges()}")

# —— 5.X. 关键词拆分函数 —— #
def split_keywords(kw_str: str) -> list:
    """
    将拼接的关键词按分号拆分，去除空串和'nan'，返回清洗后的列表
    """
    parts = [p.strip() for p in kw_str.split(';') if p.strip() and p.strip().lower() != 'nan']
    return parts

# ———— 6. 生成“候选对” candidate_pairs ————
#   条件1：姓名相似度 ≥ 0.8 且 机构相似度 ≥ 0.9
#   条件2：或 姓名相似度 ≥ 0.9（机构可忽略）
#  sim() 定义
# ———— 6. 生成候选对 & 直接高置信度合并 ————
def sim(a: str, b: str) -> float:
    """最长公共子序列相似度"""
    a, b = a or '', b or ''
    return SequenceMatcher(None, a, b).ratio()

def addr_sim(i, j):
    """计算两条记录的 simplified address 的最大相似度（支持多地址拆分）"""
    ai = records_df.at[i, 'original_address']
    aj = records_df.at[j, 'original_address']
    # 这里 ai, aj 本身已是 simplify_address 结果
    return sim(ai, aj)

@lru_cache(maxsize=None)
def addr_value_sim(ai: str, aj: str) -> float:
    """Cached similarity for repeated simplified-address comparisons."""
    return sim(ai, aj)

def high_name_match(a: str, b: str) -> bool:
    """
    绝对匹配：姓名 token 完全一致、或顺序颠倒、或集合相等
    """
    toks_a = a.split()
    toks_b = b.split()
    return toks_a == toks_b \
        or toks_a == toks_b[::-1] \
        or set(toks_a) == set(toks_b)

# 容器与计数
direct_merges = []         # 高置信度直接合并对
remaining_candidates = []  # 后续网络验证候选对
direct_merge_count = 0

# 缓存逐实例字段，避免在数千万次候选比较中反复访问 DataFrame/拆分关键词
paper_ids = records_df['paper_id'].to_dict()
addresses = records_df['original_address'].to_dict()
keyword_sets = records_df['keywords'].apply(lambda kw: set(split_keywords(kw))).to_dict()

# 遍历每个 initials 分组；先比较唯一姓名，再展开到实例对，保持原有阈值逻辑
for init, grp in records_df.groupby('initials'):
    name_to_indices = {
        name: subgrp.index.tolist()
        for name, subgrp in grp.groupby('name_clean')
        if name
    }
    names = list(name_to_indices)

    for name_pos, name_i in enumerate(names):
        idxs_i = name_to_indices[name_i]

        for name_j in names[name_pos:]:
            idxs_j = name_to_indices[name_j]
            same_name = name_i == name_j
            high_match = same_name or high_name_match(name_i, name_j)
            name_similarity = 1.0 if same_name else sim(name_i, name_j)

            # 原逻辑中非直接合并候选实际等价于 name_similarity >= 0.8
            if not high_match and name_similarity < 0.8:
                continue

            pair_iter = combinations(idxs_i, 2) if same_name else ((i, j) for i in idxs_i for j in idxs_j)
            for i, j in pair_iter:
                # 跳过同篇论文合著者
                if paper_ids[i] == paper_ids[j]:
                    continue

                addr_i = addresses[i]
                addr_j = addresses[j]

                # —— 高置信度直接合并 —— #
                if (
                    high_match
                    and addr_i
                    and addr_j
                    and ((keyword_sets[i] & keyword_sets[j]) or addr_value_sim(addr_i, addr_j) >= 0.9)
                ):
                    direct_merges.append((i, j))
                    direct_merge_count += 1
                elif name_similarity >= 0.8:
                    remaining_candidates.append((i, j))

# 输出统计
print(f"直接高置信度合并对数量：{direct_merge_count}")
print(f"后续候选对数（待网络验证）：{len(remaining_candidates)}")

# 保存所有候选对（不含直接合并）到 author_merge_candidates.csv
merge_df = pd.DataFrame(remaining_candidates, columns=['inst_i','inst_j'])
merge_df['name_i'] = merge_df['inst_i'].map(records_df['raw_name'])
merge_df['addr_i'] = merge_df['inst_i'].map(records_df['original_address'])
merge_df['name_j'] = merge_df['inst_j'].map(records_df['raw_name'])
merge_df['addr_j'] = merge_df['inst_j'].map(records_df['original_address'])
merge_mapping_path = os.path.join(output_dir, 'author_merge_candidates.csv')
merge_df.to_csv(merge_mapping_path, index=False, encoding='utf-8-sig')
print(f"[OK] 已保存后续候选对到 {merge_mapping_path}，共 {len(merge_df)} 对")

# ———— 7. 迭代网络验证合并（Union–Find） ————
class UnionFind:
    """
    并查集实现：支持 find、union 操作，用于动态合并作者实例
    """
    def __init__(self, elements):
        self.parent = {x: x for x in elements}   # 每个元素初始父节点为自身
        self.rank   = dict.fromkeys(elements, 0) # 秩（rank）初始化为 0

    def find(self, x):
        """查找 x 的根节点，并做路径压缩"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """按秩合并 a、b 所在集合，返回是否发生合并"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 保证低秩树挂到高秩树下
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

# 初始化并查集，节点为所有作者实例的索引
uf = UnionFind(records_df.index)

# 7.1 合并所有高置信度直接合并对
for i, j in direct_merges:
    uf.union(i, j)

# ———— 在第 7 步之前，先缓存每个节点的邻居集 ———
# 避免下面循环里频繁调用 co_graph.neighbors
neighbors = {n: set(co_graph[n]) for n in co_graph.nodes()}

# ———— 7.2 迭代网络验证合并 remaining_candidates ———
#    只要 i、j 在合著网络中通过已合并的公共邻居间接相连，就合并；迭代至收敛
network_validated = []   # 记录通过网络验证真正合并的对
changed = True
while changed:
    changed = False
    for i, j in remaining_candidates:
        # 若 i,j 已在同一集合，则跳过
        if uf.find(i) == uf.find(j):
            continue

        # 取出两者的邻居集（已缓存）
        nei_i = neighbors[i]
        nei_j = neighbors[j]

        # 计算各自邻居在并查集中的根集合
        roots_i = {uf.find(x) for x in nei_i}
        roots_j = {uf.find(x) for x in nei_j}

        # 若根集合有交集，说明存在已合并的公共邻居 → 合并 i,j
        if roots_i & roots_j:
            uf.union(i, j)
            network_validated.append((i, j))
            changed = True
    # 一轮结束，无任何新合并时退出

# 输出网络验证合并对数
print(f"网络验证合并对数量：{len(network_validated)}")

# 7.3 根据并查集根分配 author_id
mapping = {}
for inst in records_df.index:
    root = uf.find(inst)
    if root not in mapping:
        mapping[root] = f'author_{len(mapping) + 1}'
    mapping[inst] = mapping[root]

records_df['author_id'] = records_df.index.map(mapping)
print("[OK] 基于迭代网络验证合并完成并分配 author_id")


# ———— 9. 保存“验证通过”作者实例 ————
# 收集所有参与过合并的实例：包括 direct_merges 与 network_validated
validated_ids = set()
for i, j in direct_merges:
    validated_ids.update([i, j])
for i, j in network_validated:
    validated_ids.update([i, j])

# 提取并保存实例
valid_df = records_df.loc[sorted(validated_ids)].copy()
valid_df.index.name = 'instance_id'
valid_output = valid_df.reset_index()[[
    'instance_id', 'paper_id', 'raw_name', 'original_address', 'author_id'
]]
validated_path = os.path.join(output_dir, 'validated_author_records.csv')
valid_output.to_csv(validated_path, index=False, encoding='utf-8-sig')
print(f"[OK] 已保存验证通过的作者实例到 {validated_path}，共 {len(valid_output)} 条")

# ———— 10. 导出行级消歧全量结果 ————
records_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"[OK] 完成！行级消歧结果已保存至：{output_csv}")


# ———— 11. 作者级汇总 ————
# 使用 records_df 中的 author_id 聚合各项指标：
# - paper_ids: 去重排序的论文列表
# - publication_years: 去重排序的年份列表
# - keyword_list: 拆分去重后的关键词列表
# - total_times_cited: 引文次数求和
# - affiliation_list: 去重排序的简化后地址列表
# - paper_count: 论文数量
# - author_name: 出现频次最高的 raw_name

def split_keywords(kw_str: str) -> list:
    # 将拼接的关键词按分号拆分并去重
    return [p.strip() for p in kw_str.split(';') if p.strip()]

# 聚合
author_summary = records_df.groupby('author_id').agg({
    'paper_id':         lambda ids: sorted(set(ids)),
    'publication_year': lambda yrs: sorted({y for y in yrs if y}),
    'keywords':         lambda kws: split_keywords(';'.join(kws)),
    'times_cited':      'sum',
    'original_address': lambda addrs: [a for a in sorted(set(addrs)) if a]
}).reset_index().rename(columns={
    'paper_id':         'paper_ids',
    'publication_year': 'publication_years',
    'keywords':         'keyword_list',
    'times_cited':      'total_times_cited',
    'original_address': 'affiliation_list'
})
# 新增一列 top_affiliation：出现次数最高的那个地址
# 这里直接对原始所有地址（含重复）计数
top_affs = (
     records_df
       .groupby('author_id')['original_address']
       # 如果分组后 Series 非空，取出现频次最高的那个地址，否则 ''
       .apply(lambda addrs: Counter(addrs).most_common(1)[0][0]
                                 if not addrs.empty else '')
       .rename('top_affiliation')
       .reset_index()
)
# 论文数量
author_summary['paper_count'] = author_summary['paper_ids'].apply(len)
author_summary = author_summary.merge(top_affs, on='author_id')

# 代表性 author_name：取出现最频繁的 raw_name
name_counts = (
    records_df
      .groupby(['author_id','raw_name'])
      .size()
      .rename('cnt')
      .reset_index()
      .sort_values(['author_id','cnt'], ascending=[True,False])
      .drop_duplicates('author_id', keep='first')
      .loc[:, ['author_id','raw_name']]
      .rename(columns={'raw_name':'author_name'})
)
author_summary = author_summary.merge(name_counts, on='author_id')

# 保存作者级汇总
summary_path = os.path.join(os.path.dirname(output_csv), 'author_disambiguation_summary.csv')
author_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(f"[OK] 作者级汇总（含 paper_count）已保存至：{summary_path}，共 {len(author_summary)} 位作者")

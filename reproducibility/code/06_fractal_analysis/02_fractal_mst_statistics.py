# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
【程序功能】
科研合作网络分形特征全维度统计工具 (精确计算版)。

【已包含的指标与计算方法】
1. 活跃网络提取: 筛选 TARGET_YEAR 以前的数据，过滤发文量 < MIN_PAPER_COUNT 的作者，提取 LCC。
2. 边介数中心性 (EBC): 精确计算网络中所有最短路径经过某条边的频率，用于衡量边重要性。
3. 确定性 MST 骨架: 基于 Kruskal 算法，按 EBC 权重降序排序；并列时按节点 ID 字典序排序，确保骨架唯一。
4. 图 b - 骨架分支率: 统计从骨架 Hub 节点出发，各层级 d 的平均分支数 <b(d)>。
5. 图 c - 短路长度分布: 识别不属于骨架的“捷径”边 (Shortcuts)，计算其两端点在骨架上的最短距离 ds，统计 Ps(ds)。
"""

import os  # 操作系统模块
import pandas as pd  # 数据处理模块
import numpy as np  # 数值计算模块
import networkx as nx  # 网络分析模块
from collections import defaultdict  # 字典增强模块

# ==========================================
# 1. 全局参数配置
# ==========================================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
TARGET_YEAR = 2025  # 截止年份
OUTPUT_DIR = str(_OUTPUT_ROOT / "06_fractal_analysis" / "fractal_mst_statistics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_PAPER_COUNT = 2  # 活跃阈值


# ==========================================
# 2. 基础算法：确定性最大生成树 (MST)
# ==========================================
def maximum_spanning_tree_deterministic(Gw: nx.Graph, weight: str = "weight") -> nx.Graph:
    """构造确定性最大生成树，消除 EBC 并列时的不确定性"""
    parent = {n: n for n in Gw.nodes()}  # 并查集初始化
    rank = {n: 0 for n in Gw.nodes()}  # 并查集秩初始化

    def find(x):  # 带路径压缩的查找
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):  # 按秩合并
        ra, rb = find(a), find(b)
        if ra == rb: return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra; rank[ra] += 1
        return True

    edges = []  # 提取所有边信息
    for u, v, d in Gw.edges(data=True):
        w = d.get(weight, 0.0)  # 获取 EBC 权重
        a, b = (u, v) if u <= v else (v, u)  # 节点字典序规范化
        edges.append((w, a, b))  # 存储

    # 排序规则：权重降序 -> 节点 A 升序 -> 节点 B 升序
    edges.sort(key=lambda x: (-x[0], x[1], x[2]))

    T = nx.Graph()  # 初始化骨架图
    T.add_nodes_from(Gw.nodes())  # 添加节点
    for w, a, b in edges:  # Kruskal 核心步骤
        if union(a, b): T.add_edge(a, b, **{weight: w})  # 无环则添加
    return T  # 返回骨架


# ==========================================
# 3. 核心统计模块
# ==========================================
def run_full_statistics(G_lcc):
    """执行图 b 与 图 c 的精确统计"""
    print("    -> [1/3] 计算精确边介数中心性 (EBC)...")  # 进度提示
    ebc = nx.edge_betweenness_centrality(G_lcc, normalized=True)  # 精确计算每条边的介数

    Gw = G_lcc.copy()  # 建立带权图副本
    for u, v in Gw.edges():  # 遍历所有边
        Gw[u][v]["weight"] = float(ebc.get((u, v), ebc.get((v, u), 0.0)))  # 注入权重

    print("    -> [2/3] 提取骨架并计算分支率 (图 b)...")  # 进度提示
    skeleton = maximum_spanning_tree_deterministic(Gw, weight="weight")  # 提取确定性骨架

    # 图 b 统计：分支率
    skel_deg = dict(skeleton.degree())  # 骨架节点度数
    root = max(skel_deg.items(), key=lambda x: x[1])[0]  # 选取 Hub 为根
    lengths = nx.single_source_shortest_path_length(skeleton, root)  # 计算到根距离
    dist_map_b = defaultdict(list)  # 存储分支数
    for node, d in lengths.items():  # 遍历节点
        b = skel_deg[node] if node == root else max(0, skel_deg[node] - 1)  # 分支率定义
        dist_map_b[d].append(b)  # 归类

    # 图 c 统计：短路长度分布
    print("    -> [3/3] 识别捷径并计算短路长度 (图 c)...")  # 进度提示
    all_edges = set(tuple(sorted(e)) for e in G_lcc.edges())  # 原图所有边
    skel_edges = set(tuple(sorted(e)) for e in skeleton.edges())  # 骨架所有边
    shortcuts = list(all_edges - skel_edges)  # 剩余链接集合

    ds_values = []  # 存储 ds 值
    for i, (u, v) in enumerate(shortcuts):  # 遍历所有捷径
        d_s = nx.shortest_path_length(skeleton, source=u, target=v)  # 计算骨架上的最短路径 ds
        ds_values.append(d_s)  # 记录
        if (i + 1) % 1000 == 0: print(f"       已处理 {i + 1}/{len(shortcuts)} 条捷径...")

    ds_counts = defaultdict(int)  # 统计 ds 频次
    for d in ds_values: ds_counts[d] += 1  # 累加
    total_s = len(ds_values)  # 捷径总数

    # 汇总结果
    res_b = pd.DataFrame([{"d": d, "mean_b": np.mean(dist_map_b[d])} for d in sorted(dist_map_b.keys())])
    res_c = pd.DataFrame([{"ds": d, "Ps": ds_counts[d] / total_s} for d in sorted(ds_counts.keys())])

    return res_b, res_c, len(skeleton), root  # 返回数据帧及元数据


# ==========================================
# 4. 数据加载与主程序
# ==========================================
def load_data_lcc(path, year):
    """加载并清洗数据"""
    if not os.path.exists(path): return None
    df = pd.read_csv(path, dtype=str)  # 读取
    y_col = next((c for c in df.columns if 'year' in c.lower() or 'PY' in c), None)  # 找年份列
    if y_col:  # 时间过滤
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int)
        df = df[df[y_col] <= year]

    G = nx.Graph()  # 构建网络
    groups = df.groupby("paper_id")["author_id"].apply(list)  # 聚合作者
    for authors in groups:  # 建立合作连边
        authors = list(set(authors))
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)): G.add_edge(authors[i], authors[j])

    counts = df["author_id"].value_counts().to_dict()  # 统计发文量
    active = [n for n in G.nodes() if counts.get(n, 0) >= MIN_PAPER_COUNT]  # 过滤
    G_active = G.subgraph(active)  # 诱导活跃子图
    return G_active.subgraph(max(nx.connected_components(G_active), key=len)).copy()  # 提取 LCC


if __name__ == "__main__":
    G_main = load_data_lcc(PATH_INPUT, TARGET_YEAR)  # 运行加载
    if G_main:  # 如果加载成功
        df_b, df_c, nodes, root_id = run_full_statistics(G_main)  # 运行统计
        df_b.to_csv(os.path.join(OUTPUT_DIR, "fractal_mst_branching.csv"), index=False)  # 保存图 b 数据
        df_c.to_csv(os.path.join(OUTPUT_DIR, "fractal_shortcut_lengths.csv"), index=False)  # 保存图 c 数据
        print(f"\n统计完成！\n骨架规模: {nodes}\n根节点: {root_id}\n数据已保存。")  # 结束提示

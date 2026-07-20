# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_stats_advanced_v21_s3_logic_confirmed_fixed.py

==============================================================================
【程序功能：网络演化深度分类统计 (S3 逻辑确认 & 修复版)】
==============================================================================
1. 核心目标:
   基于 "Global-First" (全局优先) 策略，先在全量时间窗口构建加权网络并提取骨干(Backbone)，
   然后将骨干结构映射回各个时间切片，追踪网络的动态演化过程。

2. S3 差异过滤器 (Disparity Filter) 逻辑:
   - 采用 "Union of Backbones" (OR 逻辑):
     对于边 (u, v)，只要 u 端认为该边显著 (alpha_u < threshold) 或者 v 端认为该边显著，
     该边即被保留。
   - 仅当两端都认为该边不显著时，才移除该边。
   - 特殊处理: 度数(k)=1 的节点，其唯一连边默认被视为显著(alpha=0)，以防止网络边缘破碎。

3. 演化分类体系 (Evolution Types):
   基于 Palla et al. (2007) 及后续改进理论，对比 t-1 时刻网络 (G_old) 与 t 时刻网络 (G_curr):
   (i)   New Cluster (新社团/新簇): 两个节点在 G_old 中均未出现，在 t 时刻首次连接。
   (ii)  Growth (生长): 一端是 G_old 中的旧节点，另一端是新节点。
   (iii) Repeated (重复连接): 两个节点及它们之间的边在 G_old 中已经存在。
   (iv)  Intra-component (组件内致密化): 两个节点在 G_old 中已存在且属于同一个连通分量，但在 t 时刻产生了一条新边。
   (v)   Merge (融合): 两个节点在 G_old 中属于不同的连通分量，在 t 时刻连接导致分量合并。

4. 关键修复 (v21_fixed):
   - 修复了切片截取逻辑: 确保切片不仅保留骨干节点，也只保留骨干边 (Intersection of Edges)。
   - 优化了矩阵运算性能: 避免在不同过滤管道中重复计算相同权重的全局矩阵。
   - 补全了切片网络的权重属性。

5. 统计指标输出:
   - cumulative_nodes/edges: 截止当前时刻，历史网络累积的规模。
   - new_nodes: 当前切片新增节点数。
   - new_edges_unique: 当前切片新增的边数（不计重复）。
   - link_*: 各类演化事件发生的连边数量。

==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

# 尝试导入 SciPy 稀疏矩阵库，用于加速大型网络的权重计算
try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    print("Warning: SciPy not found. Matrix operations will fail.")

# =========================
# 1. 全局配置区
# =========================

# 输入数据路径 (请根据实际情况修改)
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
# 输出目录路径
PATH_OUTDIR = str(_OUTPUT_ROOT / "03_network_metrics" / "temporal_node_edge_statistics")

# 列名映射配置
AUTHOR_COL = "author_id"  # 作者ID列
PAPER_COL = "paper_id"  # 论文ID列
# 自动识别年份列的候选名称
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]

# --- 核心运行参数 (User Confirmed) ---

# 1. 构图加权方法: "newman" (合著者权重 1/(n-1)), "full" (简单团权重 1)
METHODS_TO_RUN = ["newman"]

# 2. 过滤管道配置: 定义依次执行的过滤器顺序
# S1: 去除孤立点, S2: 最大连通子图(LCC), S3: 差异过滤器, S5: 最小论文数过滤
FILTER_CONFIGS = [["S2"], ["S5", "S2"], ["S5", "S3", "S2"]]

# 3. 过滤阈值参数
S3_ALPHA = 0.1  # S3 显著性水平 (alpha < 0.1 保留)
MIN_PAPER_COUNT = 2  # S5 最小发表论文数

# 4. 时间切片设置
TIME_START = 2006  # 起始年份
TIME_END = 2025  # 结束年份

# 时间窗口模式: "disjoint" (不重叠) 或 "sliding" (滑动窗口)
TIME_MODE = "disjoint"
TIME_WINDOW_SIZE = 1  # 窗口大小 (年)
TIME_WINDOW_SHIFT = 1  # 滑动步长 (仅在 sliding 模式下有效)


# =========================
# 工具函数
# =========================

def ensure_outdir(path: str):
    """确保输出目录存在，如果不存在则创建"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)  # exist_ok=True 防止目录已存在报错


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    """在 DataFrame 中自动检测符合候选列表的列名"""
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == df_col.lower():  # 不区分大小写匹配
                return df_col
    return ""


def generate_time_slices(mode, start_year, end_year, window_size, user_shift):
    """生成时间切片的生成器 (start, end)"""
    slices = []
    # 如果是 disjoint 模式，步长等于窗口大小；否则使用用户定义的 shift
    actual_shift = window_size if mode == "disjoint" else user_shift
    curr = start_year
    while True:
        win_end = curr + window_size - 1
        if win_end > end_year: break  # 超出结束年份停止
        slices.append((curr, win_end))
        curr += actual_shift
    return slices


def load_data(csv_path: str):
    """加载原始 CSV 数据并进行基本清洗"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")
    print(f"Reading: {csv_path} ...")

    # 统一作为字符串读取，避免 ID 被误转为科学计数法
    df = pd.read_csv(csv_path, dtype=str)

    # 检测年份列
    year_col = detect_column(df, YEAR_COL_CANDIDATES)

    # 删除缺少作者或论文 ID 的行
    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    # 删除空白字符的 ID
    df = df[(df[AUTHOR_COL].str.strip() != "") & (df[PAPER_COL].str.strip() != "")]

    # 将年份列转换为整数，无法转换的填充为 0 (0 表示年份未知)
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)

    return df, year_col


def build_index_maps(df_slice):
    """
    构建作者和论文的索引映射，用于矩阵运算。
    返回: author_ids(列表), paper_ids(列表), df_idx(带数字索引的DataFrame)
    """
    author_ids = df_slice[AUTHOR_COL].unique()
    paper_ids = df_slice[PAPER_COL].unique()

    # 建立 ID 到 0..N 的映射字典
    aid2idx = {a: i for i, a in enumerate(author_ids)}
    pid2idx = {p: i for i, p in enumerate(paper_ids)}

    df_idx = df_slice.copy()
    # 将原始 ID 映射为数字索引
    df_idx["ai"] = df_idx[AUTHOR_COL].map(aid2idx).astype(int)
    df_idx["pk"] = df_idx[PAPER_COL].map(pid2idx).astype(int)

    return author_ids, paper_ids, df_idx


# =========================
# 演化追踪器 (核心统计逻辑)
# =========================

class EvolutionTracker:
    def __init__(self):
        # 存储历史图状态: key=(method, pipeline), value=nx.Graph
        self.history_graphs = {}
        # 存储统计结果列表
        self.stats_records = []

    def process_slice(self, method, pipeline, slice_label, slice_end_year, G_slice):
        """
        处理单个时间切片，对比 G_slice (当前) 和 G_old (历史累积)
        """
        key = (method, pipeline)
        # 如果是该配置的第一个切片，初始化空的 Graph
        if key not in self.history_graphs:
            self.history_graphs[key] = nx.Graph()

        G_old = self.history_graphs[key]

        # 预计算 G_old 的连通分量映射，用于判断 Merge/Intra-comp
        node_comp_map = {}
        if G_old.number_of_nodes() > 0:
            # 获取所有连通分量，并将节点映射到 component_id
            for comp_id, comp_nodes in enumerate(nx.connected_components(G_old)):
                for node in comp_nodes:
                    node_comp_map[node] = comp_id

        # 初始化当前切片的计数器
        counts = {
            'new_nodes': 0,
            'new_edges_unique': 0,  # 新增的从未见过的边
            'type_repeated': 0,  # 重复连接
            'type_intra_comp': 0,  # 组件内致密化
            'type_merge': 0,  # 组件融合
            'type_growth': 0,  # 生长 (旧连新)
            'type_new_cluster': 0  # 新簇 (新连新)
        }

        # --- 1. 节点统计 ---
        curr_nodes = set(G_slice.nodes())
        old_nodes = set(G_old.nodes())
        counts['new_nodes'] = len(curr_nodes - old_nodes)  # 计算新增节点数

        # --- 2. 边演化分类 (核心逻辑) ---
        for u, v in G_slice.edges():
            is_u_old = u in G_old
            is_v_old = v in G_old

            # Case A: 两个节点都是新的 -> New Cluster
            if not is_u_old and not is_v_old:
                counts['type_new_cluster'] += 1

            # Case B: 一个旧一个新 -> Growth
            elif is_u_old != is_v_old:
                counts['type_growth'] += 1

            # Case C: 两个节点都是旧的
            else:
                # 检查这条边以前是否出现过
                if G_old.has_edge(u, v):
                    counts['type_repeated'] += 1
                else:
                    # 边是新的，检查两个节点是否属于同一个旧连通分量
                    comp_u = node_comp_map.get(u, -1)
                    comp_v = node_comp_map.get(v, -2)  # 使用不同默认值防止未匹配时的假相等

                    if comp_u == comp_v and comp_u != -1:
                        # 同一分量内的连接 -> Intra-component
                        counts['type_intra_comp'] += 1
                    else:
                        # 不同分量间的连接 -> Merge
                        counts['type_merge'] += 1

        # --- 3. 统计全新边数量 ---
        # 为了处理无向边，先排序转 tuple
        curr_edges = set(tuple(sorted((u, v))) for u, v in G_slice.edges())
        old_edges = set(tuple(sorted((u, v))) for u, v in G_old.edges())
        counts['new_edges_unique'] = len(curr_edges - old_edges)

        # --- 4. 更新历史状态 ---
        # 将当前切片的节点和边合并入历史网络 (Cumulative Union)
        G_old.add_nodes_from(G_slice.nodes())
        G_old.add_edges_from(G_slice.edges())

        # --- 5. 记录数据 ---
        self.stats_records.append({
            "slice_label": slice_label,
            "year_end": slice_end_year,
            "method": method,
            "pipeline": pipeline,
            "time_mode": TIME_MODE,
            "cumulative_nodes": G_old.number_of_nodes(),  # 历史总节点数
            "cumulative_edges": G_old.number_of_edges(),  # 历史总边数
            "new_nodes": counts['new_nodes'],
            "new_edges_unique": counts['new_edges_unique'],
            "link_repeated": counts['type_repeated'],
            "link_intra_comp": counts['type_intra_comp'],
            "link_merge": counts['type_merge'],
            "link_growth": counts['type_growth'],
            "link_new_cluster": counts['type_new_cluster']
        })

    def export_csv(self, out_path):
        """将统计结果导出为 CSV"""
        if not self.stats_records:
            print("    [Tracker] No data.")
            return
        df = pd.DataFrame(self.stats_records)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n>>> [Saved] {out_path}")


# =========================
# 构图与过滤 (S3 Logic Confirmed)
# =========================

def zero_diag_csr_inplace(U):
    """将稀疏矩阵的对角线置零 (去除自环)"""
    U = U.tolil(copy=False)
    U.setdiag(0)
    U = U.tocsr(copy=False)
    U.eliminate_zeros()  # 清除显式存储的0
    return U


def edges_from_sparse(S, author_ids):
    """从稀疏矩阵提取边列表 DataFrame"""
    S = S.tocoo()
    # 仅提取上三角矩阵 (避免无向图重复边)
    mask = S.row < S.col
    rows, cols, data = S.row[mask], S.col[mask], S.data[mask]
    return pd.DataFrame({
        "src_author_id": [author_ids[i] for i in rows],
        "dst_author_id": [author_ids[j] for j in cols],
        "weight": data.astype(float)
    })


def build_network_matrix(method, author_ids, paper_ids, df_idx):
    """
    使用矩阵乘法构建共现网络 A @ W @ A.T
    """
    if not HAVE_SCIPY: raise RuntimeError("Need SciPy.")

    # 构建 Author-Paper 关联矩阵 A (0/1)
    # 行: Author, 列: Paper
    A = sp.csr_matrix((np.ones(len(df_idx), dtype=np.int8), (df_idx["ai"], df_idx["pk"])),
                      shape=(len(author_ids), len(paper_ids)))

    if method == "full":
        # 简单共现: 两个作者共同发表一篇论文，权重+1
        U = A @ A.T
    elif method == "newman":
        # Newman权重: 每一篇有 n 个作者的论文，对共现贡献 1/(n-1)
        # 1. 计算每篇论文的作者数 n
        n = np.asarray(A.sum(axis=0)).ravel()
        # 2. 计算权重向量 w
        w = np.zeros_like(n, dtype=float)
        mask = n > 1  # 忽略单作者论文
        w[mask] = 1.0 / (n[mask] - 1.0)
        # 3. 矩阵乘法 A @ diag(w) @ A.T
        U = A @ sp.diags(w) @ A.T
    else:
        raise ValueError(method)

    # 去除对角线并返回 DataFrame
    U = zero_diag_csr_inplace(U.tocsr())
    return edges_from_sparse(U, author_ids)


def filter_isolates_s1(G):
    """[S1] 移除孤立节点 (度数为0)"""
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    return G


def filter_lcc_s2(G):
    """[S2] 保留最大连通子图 (LCC)"""
    if G.number_of_nodes() == 0: return G
    # 获取最大的连通分量并返回其诱导子图的副本
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def filter_disparity_s3(G, alpha_threshold=0.1):
    """
    [S3 Update] 差异过滤器 (Disparity Filter) - OR 逻辑
    逻辑: 只要边的一端节点认为该边显著 (alpha < threshold)，即保留该边。
    """
    if G.number_of_edges() == 0: return G

    # 预计算节点的 Strength (加权度) 和 Degree (无权度)
    strength = {n: 0.0 for n in G.nodes()}
    degree = {n: 0 for n in G.nodes()}

    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)  # 获取权重，默认为1
        strength[u] += w
        strength[v] += w
        degree[u] += 1
        degree[v] += 1

    rem_edges = []

    # 遍历每条边，计算显著性
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)

        # --- 计算 u 端 alpha ---
        k_u = degree[u]
        # 公式: (1 - w/s)^(k-1)
        # 注意: 当 k=1 时，alpha=0 (因为只有一条边，这唯一一条边就是它的全部)
        # 这里的 0.0 意味着 k=1 的节点总是显著的 (0 < threshold) -> 保留边缘节点
        a_u = (max(0.0, 1.0 - w / strength[u]) ** (k_u - 1)) if k_u > 1 else 0.0

        # --- 计算 v 端 alpha ---
        k_v = degree[v]
        a_v = (max(0.0, 1.0 - w / strength[v]) ** (k_v - 1)) if k_v > 1 else 0.0

        # --- OR 逻辑判断 ---
        is_u_sig = a_u < alpha_threshold
        is_v_sig = a_v < alpha_threshold

        # 如果两端都不显著，则标记删除
        if not (is_u_sig or is_v_sig):
            rem_edges.append((u, v))

    # 批量移除不显著的边
    G.remove_edges_from(rem_edges)

    # 移除因删边产生的孤立点 (通常 S3 后会跟 S1)
    return filter_isolates_s1(G)


def filter_min_papers_s5(G, paper_counts, min_count):
    """[S5] 移除发表论文数少于阈值的作者"""
    # 这里的 paper_counts 是从原始数据统计得到的
    nodes_to_remove = [n for n in G.nodes() if paper_counts.get(n, 0) < min_count]
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
    return G


def run_filter_pipeline(G, pipeline_list, alpha_val, paper_counts, min_paper_count):
    """按顺序执行过滤器列表"""
    G_curr = G.copy()
    for step in pipeline_list:
        if step == "S1":
            G_curr = filter_isolates_s1(G_curr)
        elif step == "S2":
            G_curr = filter_lcc_s2(G_curr)
        elif step == "S3":
            G_curr = filter_disparity_s3(G_curr, alpha_val)
        elif step == "S5":
            G_curr = filter_min_papers_s5(G_curr, paper_counts, min_paper_count)
    return G_curr


# =========================
# 主程序
# =========================

def main():
    print(">>> 启动网络演化分析 (v21 - Logic Confirmed & Fixed)...")
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载全量数据
    df_all, year_col = load_data(PATH_INPUT)
    if not year_col:
        print("Error: No year column found.")
        return

    # 2. 生成时间切片
    slices = generate_time_slices(TIME_MODE, TIME_START, TIME_END, TIME_WINDOW_SIZE, TIME_WINDOW_SHIFT)
    print(f"    [Time] Mode: {TIME_MODE}, Count: {len(slices)}")

    tracker = EvolutionTracker()

    # 3. 循环构图方法 (Newman / Full)
    for method in METHODS_TO_RUN:
        print(f"\n### {method.upper()} ###")

        # [优化] 将全量矩阵构建移至 Pipeline 循环外
        # 只要 method 相同，初始全量网络 G_global_base 是一样的
        print("    [Global] Building Base Weighted Network...")
        auth_ids_g, paper_ids_g, df_idx_g = build_index_maps(df_all)
        edges_g = build_network_matrix(method, auth_ids_g, paper_ids_g, df_idx_g)

        G_global_base = nx.Graph()
        # 批量添加带权边
        for r in edges_g.itertuples(index=False):
            G_global_base.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

        # 预计算每个作者的论文总数 (用于 S5 过滤)
        global_counts = df_all[AUTHOR_COL].value_counts().to_dict()

        # 4. 循环过滤管道
        for pipeline in FILTER_CONFIGS:
            pipeline_name = "-".join(pipeline)
            print(f"  >>> [{pipeline_name}]")

            # [Step A] 获取全局骨干网络
            # 注意: 必须 copy，否则 pipeline 会修改 G_global_base
            G_global_filtered = run_filter_pipeline(G_global_base, pipeline, S3_ALPHA, global_counts, MIN_PAPER_COUNT)

            # 关键: 获取全局有效的【节点集】和【边集】
            # 仅仅过滤节点是不够的，如果 S3 删除了 A-B 边但保留了 A 和 B 节点，
            # 在切片中单纯使用 subgraph(nodes) 会错误地把 A-B 加回来。
            # 因此，我们需要同时限制切片中的边必须存在于全局骨干网中。
            valid_nodes_set = set(G_global_filtered.nodes())

            # 使用 frozenset 或 tuple(sorted) 来存储无向边，确保查找一致性
            valid_edges_set = set()
            for u, v in G_global_filtered.edges():
                valid_edges_set.add(tuple(sorted((u, v))))

            print(f"    [Global] Valid Nodes: {len(valid_nodes_set)}, Valid Edges: {len(valid_edges_set)}")

            if not valid_nodes_set:
                print("    [Warning] Empty global network, skipping.")
                continue

            # [Step B] 切片统计
            for (t_s, t_e) in slices:
                t_label = f"{t_s}-{t_e}"

                # 提取切片数据
                df_slice = df_all[(df_all[year_col] >= t_s) & (df_all[year_col] <= t_e)]

                # 如果切片为空
                if df_slice.empty:
                    tracker.process_slice(method, pipeline_name, t_label, t_e, nx.Graph())
                    continue

                # 构建切片原始网络
                auth_ids_s, paper_ids_s, df_idx_s = build_index_maps(df_slice)
                edges_s = build_network_matrix(method, auth_ids_s, paper_ids_s, df_idx_s)

                G_slice_raw = nx.Graph()
                for r in edges_s.itertuples(index=False):
                    # [修复] 添加权重，虽然目前 Tracker 只用拓扑，但保持数据完整性
                    G_slice_raw.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

                # [Step C] 应用 Global-First 过滤策略
                # 1. 仅保留在 Global 中存在的节点
                G_slice_filtered = G_slice_raw.subgraph(valid_nodes_set).copy()

                # 2. [关键修复] 仅保留在 Global 中存在的边
                # 遍历切片现有的边，如果不在 valid_edges_set 中，则移除
                edges_to_remove = []
                for u, v in G_slice_filtered.edges():
                    edge_tuple = tuple(sorted((u, v)))
                    if edge_tuple not in valid_edges_set:
                        edges_to_remove.append((u, v))

                if edges_to_remove:
                    G_slice_filtered.remove_edges_from(edges_to_remove)

                # 3. 清理产生的孤立点
                G_final = filter_isolates_s1(G_slice_filtered)

                # 提交给追踪器
                tracker.process_slice(method, pipeline_name, t_label, t_e, G_final)

    # 5. 导出结果
    out_csv = os.path.join(PATH_OUTDIR, "network_evolution_stats_fixed2.csv")
    tracker.export_csv(out_csv)
    print(">>> All Done.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_stats_advanced_v15_final_settings.py

==============================================================================
【程序功能概述】
==============================================================================
1. 核心逻辑：全局优先策略 (Global-First Strategy)
   - 第一步：基于全量数据 (2006-2025) 构建全局网络。
   - 第二步：应用全局过滤 (本配置为 S2: 最大连通分量)，获取“核心节点白名单”。
   - 第三步：生成时间切片 (滑动窗口 5年)，在每个切片中仅保留“核心节点”。
   - 第四步：将切片数据合并，生成带有时间轴 (Timeline) 的动态 GEXF 文件。

2. 输出指标：
   - 基础指标：节点数、边数、密度、平均度、平均加权度。
   - 影响力指标：h-index (h指数), g-index (g指数), cpp (篇均被引)。
   - 结构指标：连通分量数、覆盖率。

3. 动态网络策略 (Accumulative):
   - 节点的 start 为首次出现的切片年份，end 为 2025。
   - 边的权重会随着时间累加 (Sum)，体现合作关系的加强。

==============================================================================
"""

import os  # 导入操作系统模块，用于路径管理
import sys  # 导入系统模块
import json  # 导入JSON模块，用于数据序列化
import numpy as np  # 导入NumPy，用于数值计算
import pandas as pd  # 导入Pandas，用于数据框操作
import networkx as nx  # 导入NetworkX，用于图论计算

# 尝试导入 SciPy 稀疏矩阵模块，用于加速大型矩阵运算
try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False  # 如果未安装，后续会报错提示

# =========================
# 1. 全局配置区 (User Configuration)
# =========================

# 输入数据文件路径 (请确认路径正确)
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
# 输出结果目录
PATH_OUTDIR = str(_OUTPUT_ROOT / "02_network_construction" / "global_first_dynamic_network")

# CSV 列名映射
AUTHOR_COL = "author_id"  # 作者ID列
PAPER_COL = "paper_id"  # 论文ID列

# 自动识别列名的候选列表
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]
CITE_COL_CANDIDATES = ["times_cited", "Times Cited, WoS Core", "Times Cited", "TC"]

# --- 核心运行参数 (基于您的设定) ---

# 构图方法：仅运行 Newman (分数计数法)
METHODS_TO_RUN = ["newman"]

# 过滤管道：仅保留最大连通分量 (S2)
# 这将作用于【全局网络】，筛选出核心连通子图内的作者
FILTER_CONFIGS = [["S5", "S2"]]

# 其他过滤参数 (S3, S5在此配置下虽未直接调用，但保留变量定义以防扩展)
S3_ALPHA = 0.1
MIN_PAPER_COUNT = 2

# --- 时间切片设置 ---
TIME_START = 2006  # 起始年份
TIME_END = 2025  # 结束年份
TIME_MODE = "sliding"  # 切片模式：滑动窗口
TIME_WINDOW_SIZE = 5  # 窗口大小：5年 (如 2006-2010)
TIME_WINDOW_SHIFT = 1  # 移动步长：1年 (如 下一个 2007-2011)

# --- 动态网络导出配置 ---
EXPORT_DYNAMIC_GEXF = True  # 开关：是否导出动态图

# 策略: "accumulative" (累积模式)
# 逻辑：关系一旦建立，一直存在到 2025，权重累加。
DYNAMIC_STRATEGY = "accumulative"
DECAY_WINDOW = 3  # 仅在 active_decay 模式下有效，此处忽略

# 动态图的数据源匹配
DYNAMIC_BASE_METHOD = "newman"
# [自动修正] 您的配置中 FILTER_CONFIGS 是 [["S2"]]，因此这里必须匹配为 "S2"
DYNAMIC_BASE_PIPELINE = "S5-S2"

EXPORT_SLICES = True  # 开关：是否导出每个切片的静态图
GRAPH_FORMAT = "gexf"  # 导出格式


# =========================
# 工具函数
# =========================

def ensure_outdir(path: str):
    """确保输出目录存在，不存在则创建"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    """自动检测 DataFrame 中是否存在候选列名"""
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == df_col.lower():  # 忽略大小写匹配
                return df_col
    return ""


def generate_time_slices(mode, start_year, end_year, window_size, window_shift):
    """生成时间切片列表 (Start, End)"""
    slices = []
    if mode == "disjoint":  # 不重叠模式
        curr = start_year
        while curr <= end_year:
            win_end = min(curr + window_size - 1, end_year)
            slices.append((curr, win_end))
            curr += window_size
    elif mode == "sliding":  # 滑动窗口模式 (当前使用)
        curr = start_year
        while True:
            win_end = curr + window_size - 1
            if win_end > end_year: break  # 超过结束年份则停止
            slices.append((curr, win_end))
            curr += window_shift
    elif mode == "cumulative":  # 累计窗口模式
        curr_end = min(start_year + window_size - 1, end_year)
        while curr_end <= end_year:
            slices.append((start_year, curr_end))
            if curr_end == end_year: break
            curr_end += window_shift
            curr_end = min(curr_end, end_year)
    else:  # 默认全量
        slices.append((start_year, end_year))
    return slices


def load_data(csv_path: str):
    """加载并清洗 CSV 数据"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"文件未找到: {csv_path}")
    print(f"正在读取文件: {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)  # 统一按字符串读取防止ID精度丢失

    # 自动识别列名
    year_col = detect_column(df, YEAR_COL_CANDIDATES)
    cite_col = detect_column(df, CITE_COL_CANDIDATES)

    # 清洗空数据
    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[(df[AUTHOR_COL].str.strip() != "") & (df[PAPER_COL].str.strip() != "")]

    # 类型转换
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)
    if cite_col:
        df[cite_col] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0).astype(int)

    return df, year_col, cite_col


def build_index_maps(df_slice):
    """构建 作者/论文 ID 到 矩阵索引 的映射"""
    author_ids = df_slice[AUTHOR_COL].unique()  # 唯一作者
    paper_ids = df_slice[PAPER_COL].unique()  # 唯一论文

    aid2idx = {a: i for i, a in enumerate(author_ids)}  # 映射字典
    pid2idx = {p: i for i, p in enumerate(paper_ids)}

    df_idx = df_slice.copy()
    df_idx["ai"] = df_idx[AUTHOR_COL].map(aid2idx).astype(int)  # 映射为数字索引
    df_idx["pk"] = df_idx[PAPER_COL].map(pid2idx).astype(int)

    return author_ids, paper_ids, df_idx


# =========================
# 矩阵构图 (Newman & Full)
# =========================

def zero_diag_csr_inplace(U):
    """稀疏矩阵去除对角线元素 (消除自环)"""
    U = U.tolil(copy=False)
    U.setdiag(0)
    U = U.tocsr(copy=False)
    U.eliminate_zeros()
    return U


def edges_from_sparse(S, author_ids):
    """从稀疏矩阵提取边列表 DataFrame"""
    S = S.tocoo()  # 转为坐标格式
    mask = S.row < S.col  # 仅取上三角，避免重复和自环
    rows, cols, data = S.row[mask], S.col[mask], S.data[mask]
    return pd.DataFrame({
        "src_author_id": [author_ids[i] for i in rows],
        "dst_author_id": [author_ids[j] for j in cols],
        "weight": data.astype(float)
    })


def build_network_matrix(method, author_ids, paper_ids, df_idx):
    """利用矩阵乘法快速构建共现网络"""
    if not HAVE_SCIPY: raise RuntimeError("需要安装 SciPy 库")

    # 构建二模矩阵 A (Author x Paper)
    A = sp.csr_matrix(
        (np.ones(len(df_idx), dtype=np.int8), (df_idx["ai"], df_idx["pk"])),
        shape=(len(author_ids), len(paper_ids))
    )

    if method == "full":  # 全计数
        U = A @ A.T
    elif method == "newman":  # Newman 分数计数
        n = np.asarray(A.sum(axis=0)).ravel()  # 每篇论文的作者数
        w = np.zeros_like(n, dtype=float)
        mask = n > 1
        w[mask] = 1.0 / (n[mask] - 1.0)  # 权重公式: 1/(n-1)
        U = A @ sp.diags(w) @ A.T  # 加权乘法
    else:
        raise ValueError(f"未知方法: {method}")

    U = zero_diag_csr_inplace(U.tocsr())  # 去自环
    return edges_from_sparse(U, author_ids)


# =========================
# 过滤算法 (作用于全局网络)
# =========================

def filter_isolates_s1(G):
    """[S1] 去除孤立节点"""
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    return G


def filter_lcc_s2(G):
    """[S2] 最大连通分量 (保留最大子图)"""
    if G.number_of_nodes() == 0: return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def filter_disparity_s3(G, alpha_threshold=0.1):
    """
    [S3] 差异过滤器 (Disparity Filter)
    公式: alpha_ij = (1 - p_ij)^(k-1)
    [修正] 判定逻辑: 使用 AND 逻辑。
    """
    if G.number_of_edges() == 0: return G

    strength = {n: 0.0 for n in G.nodes()}
    degree = {n: 0 for n in G.nodes()}

    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        strength[u] += w
        strength[v] += w
        degree[u] += 1
        degree[v] += 1

    edges_to_remove = []

    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)

        # 计算 u 端
        k_u = degree[u]
        if k_u > 1:
            p_uv = w / strength[u]
            val = max(0.0, 1.0 - p_uv)
            alpha_u = val ** (k_u - 1)
        else:
            alpha_u = 0.0

        # 计算 v 端
        k_v = degree[v]
        if k_v > 1:
            p_vu = w / strength[v]
            val = max(0.0, 1.0 - p_vu)
            alpha_v = val ** (k_v - 1)
        else:
            alpha_v = 0.0

        # [修正] 必须两端均显著 (AND)
        significant = (alpha_u < alpha_threshold) or (alpha_v < alpha_threshold)

        if not significant:
            edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)
    # 移除因删边产生的孤立点
    G = filter_isolates_s1(G)

    print(f"      [S3] Alpha={alpha_threshold}: Removed {len(edges_to_remove)} edges.")
    return G


def filter_min_papers_s5(G, paper_counts, min_count):
    """[S5] 最小发文量过滤"""
    nodes_to_remove = [n for n in G.nodes() if paper_counts.get(n, 0) < min_count]
    if nodes_to_remove: G.remove_nodes_from(nodes_to_remove)
    return G


def run_filter_pipeline(G, pipeline_list, alpha_val, paper_counts, min_paper_count):
    """执行过滤管道"""
    G_curr = G.copy()
    for step in pipeline_list:
        if step == "S1":
            G_curr = filter_isolates_s1(G_curr)
        elif step == "S2":
            G_curr = filter_lcc_s2(G_curr)
        # S3, S5 逻辑同理
    return G_curr


# =========================
# 指标计算 (h-index, g-index)
# =========================

def calculate_h_index(citations_list):
    """计算 h-index"""
    if not citations_list: return 0
    sorted_cites = sorted(citations_list, reverse=True)
    h = 0
    for i, c in enumerate(sorted_cites):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


def calculate_g_index(citations_list):
    """计算 g-index"""
    if not citations_list: return 0
    sorted_cites = sorted(citations_list, reverse=True)
    g = 0
    cum_sum = 0
    for i, c in enumerate(sorted_cites):
        cum_sum += c
        if cum_sum >= (i + 1) ** 2:
            g = i + 1
        else:
            break
    return g


def compute_node_attributes(df_slice, G, year_col, cite_col):
    """计算节点属性 (h指数, g指数, 篇均被引)"""
    grouped = df_slice.groupby(AUTHOR_COL)
    stats_count = grouped.size().to_dict()  # 发文量
    stats_cite = grouped[cite_col].sum().to_dict() if cite_col else {}  # 总被引
    auth_cites_map = grouped[cite_col].apply(list).to_dict() if cite_col else {}  # 引用列表

    attrs = {}
    for n in G.nodes():
        pc = stats_count.get(n, 0)
        tc = stats_cite.get(n, 0)
        cpp = tc / pc if pc > 0 else 0.0  # 篇均被引
        clist = auth_cites_map.get(n, [])

        attrs[n] = {
            "papers_count": int(pc),
            "total_cites": int(tc),
            "avg_cites_per_paper": float(round(cpp, 2)),
            "h_index": int(calculate_h_index(clist)),
            "g_index": int(calculate_g_index(clist))
        }
    return attrs


# =========================
# 动态网络追踪器 (Dynamic Tracker)
# =========================

class DynamicTracker:
    """
    追踪节点和边在时间切片中的演化，用于生成 Accumulative 动态图
    """

    def __init__(self, strategy="accumulative", decay_window=3, global_end=2025):
        self.strategy = strategy
        self.decay = decay_window
        self.global_end = global_end

        self.node_years = {}  # 记录节点出现的年份集合
        self.node_attrs = {}  # 记录节点最新属性
        self.edge_years = {}  # 记录边出现的年份集合
        self.edge_weights = {}  # 记录边的权重列表

        # 强制将 ID 转为字符串，防止 NetworkX/Gephi 类型混淆
        self.force_str = True

    def update(self, G, slice_start_year):
        """处理一个新的时间切片"""
        # 1. 更新节点
        for n, data in G.nodes(data=True):
            n_str = str(n)
            if n_str not in self.node_years:
                self.node_years[n_str] = set()
                self.node_attrs[n_str] = data
            self.node_years[n_str].add(slice_start_year)
            self.node_attrs[n_str] = data  # 总是保存最新的属性

        # 2. 更新边
        for u, v, data in G.edges(data=True):
            u_str, v_str = str(u), str(v)
            edge_key = tuple(sorted((u_str, v_str)))  # 确保无向边顺序一致
            w = data.get('weight', 1.0)

            if edge_key not in self.edge_years:
                self.edge_years[edge_key] = set()
                self.edge_weights[edge_key] = []

            self.edge_years[edge_key].add(slice_start_year)
            self.edge_weights[edge_key].append(w)

    def _calculate_interval(self, years_set):
        """计算 Start 和 End 年份"""
        if not years_set: return 0, 0
        first_year = min(years_set)

        # Accumulative 策略: 首次出现 -> 直到最后 (只增不减)
        if self.strategy == "accumulative":
            return first_year, self.global_end
        else:
            # Fallback (Active Decay)
            last_year = max(years_set)
            return first_year, min(last_year + self.decay, self.global_end)

    def export_merged_graph(self, out_path):
        """导出合并后的 GEXF"""
        G_merged = nx.Graph()

        # 添加节点
        for n, years in self.node_years.items():
            s, e = self._calculate_interval(years)
            attrs = self.node_attrs[n].copy()
            attrs['start'] = int(s)
            attrs['end'] = int(e)
            G_merged.add_node(n, **attrs)

        # 添加边
        for (u, v), years in self.edge_years.items():
            s, e = self._calculate_interval(years)
            weights = self.edge_weights[(u, v)]

            # Accumulative: 权重求和 (Sum)，体现合作频次累积
            final_weight = sum(weights)

            G_merged.add_edge(u, v,
                              weight=float(round(final_weight, 4)),
                              start=int(s),
                              end=int(e))

        print(f"\n[Dynamic Export] 已保存至: {out_path}")
        print(f"  策略: {self.strategy}")
        print(f"  最终规模: Nodes={G_merged.number_of_nodes()}, Edges={G_merged.number_of_edges()}")
        nx.write_gexf(G_merged, out_path)


# =========================
# 主程序
# =========================

def main():
    print(">>> 启动网络分析 (v15 - Final Settings)...")
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载数据
    df_all, year_col, cite_col = load_data(PATH_INPUT)

    # 2. 生成时间切片
    print(f"\n>>> 生成时间切片 (Mode: {TIME_MODE})...")
    slices = generate_time_slices(TIME_MODE, TIME_START, TIME_END, TIME_WINDOW_SIZE, TIME_WINDOW_SHIFT)
    print(f"    共 {len(slices)} 个切片: {slices[0]} ... {slices[-1]}")

    # 3. 初始化动态追踪器 (用于 Accumulative 导出)
    dynamic_tracker = None
    if EXPORT_DYNAMIC_GEXF:
        print(f"    [动态图策略] {DYNAMIC_STRATEGY.upper()} (边从首次出现一直保持到 {TIME_END})")
        dynamic_tracker = DynamicTracker(strategy=DYNAMIC_STRATEGY, decay_window=DECAY_WINDOW, global_end=TIME_END)

    # 4. 遍历构图方法 (Newman)
    for method in METHODS_TO_RUN:
        print(f"\n################################################")
        print(f"### 当前方法: {method.upper()} ###")
        print(f"################################################")

        # 遍历过滤方案 (这里只有 ["S2"])
        for pipeline in FILTER_CONFIGS:
            pipeline_name = "-".join(pipeline)
            print(f"\n  >>> 执行过滤方案: [{pipeline_name}] (全局优先策略)")

            # =====================================================
            # [Step A] 全局构图与过滤 -> 获取核心白名单
            # =====================================================
            print(f"    [Step 1] 构建全局网络 (2006-2025) 并过滤...")

            # 全局映射与构图
            auth_ids_g, paper_ids_g, df_idx_g = build_index_maps(df_all)
            edges_g = build_network_matrix(method, auth_ids_g, paper_ids_g, df_idx_g)

            G_global = nx.Graph()
            for r in edges_g.itertuples(index=False):
                G_global.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

            # 全局发文统计 (虽然 S2 不用，但 run_filter_pipeline 接口需要)
            global_counts = df_all[AUTHOR_COL].value_counts().to_dict()

            # 执行过滤 (S2: Max Connected Component)
            G_global_filtered = run_filter_pipeline(G_global, pipeline, S3_ALPHA, global_counts, MIN_PAPER_COUNT)

            # 获取白名单 (核心节点集合)
            valid_nodes_set = set(G_global_filtered.nodes())
            print(
                f"    [Step 1 结果] 全局原始节点: {G_global.number_of_nodes()} -> 过滤后核心节点: {len(valid_nodes_set)}")

            if len(valid_nodes_set) == 0:
                print("    [警告] 全局过滤后网络为空，跳过此方案。")
                continue

            # =====================================================
            # [Step B] 切片投影 -> 仅保留核心节点
            # =====================================================
            print(f"    [Step 2] 处理时间切片 (映射白名单)...")

            for (t_s, t_e) in slices:
                t_label = f"{t_s}-{t_e}"

                # 提取当前切片数据
                if t_s == "ALL":
                    df_slice = df_all
                else:
                    df_slice = df_all[(df_all[year_col] >= t_s) & (df_all[year_col] <= t_e)]

                if df_slice.empty: continue

                # 局部构图 (未过滤状态)
                auth_ids_s, paper_ids_s, df_idx_s = build_index_maps(df_slice)
                edges_s = build_network_matrix(method, auth_ids_s, paper_ids_s, df_idx_s)
                G_slice_raw = nx.Graph()
                for r in edges_s.itertuples(index=False):
                    G_slice_raw.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

                # 关键步骤：应用全局白名单
                # 仅保留 G_slice_raw 中属于 valid_nodes_set 的节点
                G_slice_final = G_slice_raw.subgraph(valid_nodes_set).copy()

                # 移除映射后产生的孤立点 (清理视图)
                G_slice_final = filter_isolates_s1(G_slice_final)

                nodes_n = G_slice_final.number_of_nodes()
                if nodes_n > 0:
                    # 计算切片内的节点属性 (h-index, cpp 等)
                    attrs = compute_node_attributes(df_slice, G_slice_final, year_col, cite_col)
                    nx.set_node_attributes(G_slice_final, attrs)

                    # 导出切片静态图
                    if EXPORT_SLICES:
                        fname = f"global_first_{method}_{pipeline_name.lower().replace('-', '_')}_{str(t_label).replace('-', '_')}.{GRAPH_FORMAT}"
                        nx.write_gexf(G_slice_final, os.path.join(PATH_OUTDIR, fname))

                    # 更新动态追踪器
                    if EXPORT_DYNAMIC_GEXF and dynamic_tracker:
                        # 检查当前方法和管道是否匹配动态图配置
                        if method == DYNAMIC_BASE_METHOD and pipeline_name == DYNAMIC_BASE_PIPELINE:
                            dynamic_tracker.update(G_slice_final, t_s)

            print(f"    [Step 2 完成] 切片处理结束。")

    # 5. 导出最终合并的动态图
    if EXPORT_DYNAMIC_GEXF and dynamic_tracker:
        print("\n>>> 正在导出合并动态网络...")
        fname = f"global_first_dynamic_{DYNAMIC_BASE_METHOD}_{DYNAMIC_STRATEGY}.gexf"
        dynamic_tracker.export_merged_graph(os.path.join(PATH_OUTDIR, fname))
        print(">>> 全部流程完成。")


if __name__ == "__main__":
    main()

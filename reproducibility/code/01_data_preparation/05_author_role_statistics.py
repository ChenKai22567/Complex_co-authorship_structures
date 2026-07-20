# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
author_single_corresponding_first_stats.py

功能：
1. 统计每位作者是否有独著文章，以及独著文章数量
2. 统计每位作者是否担任过通讯作者，以及担任通讯作者的论文数量
3. 统计每位作者担任第一作者的论文数量
4. 统计每位作者担任“第一作者或通讯作者”的论文数量
   （若同一篇论文同时为第一作者和通讯作者，只记 1 篇）
5. 最终只输出 1 个 CSV 表格

输出列：
- author_id
- has_corresponding_author_role
- corresponding_author_paper_count
- has_first_author_role
- first_author_paper_count
- has_first_or_corresponding_author_role
- first_or_corresponding_author_paper_count
- has_single_paper
- single_paper_count

独著定义：
- 按 paper_id 分组
- 在同一篇论文内，先去掉同一作者因多个地址造成的重复记录
- 若该 paper_id 下仅有 1 个唯一 author_id，则该论文为独著论文

通讯作者识别：
- 从 WoS 原始文件的 Authors 列读取作者简称顺序
- 从 Reprint Addresses 列提取 "(corresponding author)" 标记的作者简称
- 按同一篇论文内作者顺序，将简称映射回消歧后的 author_id

第一作者识别：
- 基于消歧文件中同一篇论文内作者首次出现顺序
- 去掉同一作者在同一篇论文中的重复地址记录后
- author_order == 1 视为第一作者
"""

import os
import re
import glob
import pandas as pd
import numpy as np


# =========================
# 1. 路径配置
# =========================
PATH_INPUT_DISAMB = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_WOS_DIR = str(_INPUT_ROOT / "01_data_preparation" / "wos_2000_2025")
PATH_OUTPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_metrics" / "author_role_statistics.csv")


# =========================
# 2. 列名配置
# =========================
AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"

WOS_PAPER_KEY_CANDIDATES = [
    "UT (Unique WOS ID)",
    "UT",
    "paper_id",
    "Accession Number",
    "accession number",
    "UT (Unique WOS Item)"
]

WOS_AUTHORS_COL_CANDIDATES = [
    "Authors",
    "AU"
]

WOS_REPRINT_COL_CANDIDATES = [
    "Reprint Addresses",
    "Reprint Address",
    "RP"
]


# =========================
# 3. 工具函数
# =========================
def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for cand in candidates:
        for col in df.columns:
            if str(col).strip().lower() == cand.lower():
                return col
    return ""


def safe_strip_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key(x) -> str:
    return norm_text(x).upper()


def norm_abbrev(x) -> str:
    """
    将作者简称规范化，例如：
    Morgan, DL
    Morgan, D L
    Morgan, D.L.
    -> MORGAN, DL
    """
    s = norm_text(x).upper()
    s = s.replace(".", "")
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ;,.")


def first_nonempty(series: pd.Series) -> str:
    for v in series:
        s = norm_text(v)
        if s != "":
            return s
    return ""


def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
            try:
                return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
            except Exception:
                continue
        raise ValueError(f"无法读取 CSV：{path}")

    if ext == ".xls":
        return pd.read_excel(path, dtype=str, engine="xlrd")

    if ext == ".xlsx":
        return pd.read_excel(path, dtype=str)

    raise ValueError(f"不支持的文件格式：{path}")


def split_wos_authors(authors_text: str):
    """
    解析 WoS Authors 列，例如：
    Morgan, DL
    Hohmann, MH; Barnett, AG; King, N; Connell, SD
    """
    s = norm_text(authors_text)
    if s == "":
        return []
    parts = [norm_text(x) for x in s.split(";")]
    return [x for x in parts if x != ""]


def extract_corresponding_abbrevs(reprint_text: str):
    """
    从 Reprint Addresses 中提取通讯作者简称，例如：
    Morgan, DL (corresponding author), ...
    Connell, SD (corresponding author), ...
    """
    s = norm_text(reprint_text)
    if s == "":
        return []

    pattern = re.compile(
        r'([^;.\n]+?,\s*[^()]+?)\s*\((?:corresponding|reprint)\s+author\)',
        flags=re.I
    )
    matches = pattern.findall(s)

    out = []
    seen = set()
    for m in matches:
        m2 = norm_text(m).strip(" ;,.")
        if m2 != "" and m2 not in seen:
            seen.add(m2)
            out.append(m2)
    return out


def load_wos_folder_tables(folder_path: str) -> pd.DataFrame:
    """
    读取 wos_excels 文件夹中的所有 csv/xls/xlsx，
    提取：
    - 论文键
    - Authors
    - Reprint Addresses
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"WoS 文件夹不存在：{folder_path}")

    files = []
    for ext in ["*.xls", "*.xlsx", "*.csv"]:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        raise FileNotFoundError(f"文件夹中未找到 WoS 文件：{folder_path}")

    all_rows = []

    for fp in files:
        try:
            df = read_any_table(fp)
            print(f"√ 已读取：{fp}")
        except Exception as e:
            print(f"! 无法读取 {fp}（{e}），已跳过。")
            continue

        paper_col = detect_column(df, WOS_PAPER_KEY_CANDIDATES)
        authors_col = detect_column(df, WOS_AUTHORS_COL_CANDIDATES)
        reprint_col = detect_column(df, WOS_REPRINT_COL_CANDIDATES)

        if not paper_col or not authors_col:
            print(f"! 跳过 {fp}：缺少论文键列或 Authors 列")
            continue

        tmp = pd.DataFrame({
            "wos_paper_key_raw": df[paper_col],
            "wos_authors_raw": df[authors_col],
            "wos_reprint_raw": df[reprint_col] if reprint_col else ""
        }).copy()

        tmp["wos_paper_key_raw"] = safe_strip_series(tmp["wos_paper_key_raw"])
        tmp["wos_authors_raw"] = safe_strip_series(tmp["wos_authors_raw"])
        tmp["wos_reprint_raw"] = safe_strip_series(tmp["wos_reprint_raw"])
        tmp["paper_key_norm"] = tmp["wos_paper_key_raw"].map(norm_key)

        tmp = tmp[tmp["paper_key_norm"] != ""].copy()
        if not tmp.empty:
            all_rows.append(tmp)

    if not all_rows:
        raise ValueError("未能从 wos_excels 文件夹中读取到有效 WoS 数据。")

    wos_all = pd.concat(all_rows, ignore_index=True)

    # 同一篇论文可能重复出现，取非空值
    wos_papers = (
        wos_all.groupby("paper_key_norm", as_index=False)
        .agg(
            wos_paper_key_raw=("wos_paper_key_raw", first_nonempty),
            wos_authors_raw=("wos_authors_raw", first_nonempty),
            wos_reprint_raw=("wos_reprint_raw", first_nonempty)
        )
    )

    return wos_papers


# =========================
# 4. 主程序
# =========================
def main():
    if not os.path.isfile(PATH_INPUT_DISAMB):
        raise FileNotFoundError(f"消歧文件不存在：{PATH_INPUT_DISAMB}")

    print(f"正在读取消歧文件：{PATH_INPUT_DISAMB}")
    df_raw = pd.read_csv(PATH_INPUT_DISAMB, dtype=str, encoding="utf-8-sig", low_memory=False)

    for col in [AUTHOR_COL, PAPER_COL]:
        if col not in df_raw.columns:
            raise ValueError(f"缺少必要列：{col}")

    # 基础清洗
    df = df_raw.copy()
    df[AUTHOR_COL] = safe_strip_series(df[AUTHOR_COL])
    df[PAPER_COL] = safe_strip_series(df[PAPER_COL])
    df["_row_order"] = np.arange(len(df))

    df = df[(df[AUTHOR_COL] != "") & (df[PAPER_COL] != "")].copy()
    if df.empty:
        raise ValueError("清洗后 author_id / paper_id 非空的数据为空。")

    # 作者主表
    author_master = (
        df.sort_values("_row_order")
          .drop_duplicates(subset=[AUTHOR_COL], keep="first")[[AUTHOR_COL]]
          .copy()
    )

    # 同一论文内，同一作者去重（去掉地址重复）
    df_author_paper = (
        df.sort_values("_row_order")
          .drop_duplicates(subset=[PAPER_COL, AUTHOR_COL], keep="first")
          .copy()
    )

    # 论文内作者顺序：用于与 WoS Authors 顺序对齐
    df_author_paper = df_author_paper.sort_values([PAPER_COL, "_row_order"]).copy()
    df_author_paper["author_order"] = df_author_paper.groupby(PAPER_COL).cumcount() + 1
    df_author_paper["paper_key_norm"] = df_author_paper[PAPER_COL].map(norm_key)

    # =========================
    # A. 统计独著论文
    # =========================
    paper_author_count_df = (
        df_author_paper.groupby(PAPER_COL, as_index=False)[AUTHOR_COL]
        .nunique()
        .rename(columns={AUTHOR_COL: "paper_unique_author_count"})
    )

    single_papers_df = paper_author_count_df[
        paper_author_count_df["paper_unique_author_count"] == 1
    ][[PAPER_COL]].copy()

    single_author_paper_df = df_author_paper.merge(single_papers_df, on=PAPER_COL, how="inner")

    single_count_df = (
        single_author_paper_df.groupby(AUTHOR_COL, as_index=False)[PAPER_COL]
        .nunique()
        .rename(columns={PAPER_COL: "single_paper_count"})
    )

    # =========================
    # B. 统计第一作者论文
    # =========================
    first_author_papers_df = df_author_paper[
        df_author_paper["author_order"] == 1
    ].copy()

    first_count_df = (
        first_author_papers_df.groupby(AUTHOR_COL, as_index=False)[PAPER_COL]
        .nunique()
        .rename(columns={PAPER_COL: "first_author_paper_count"})
    )

    # =========================
    # C. 读取 WoS 原始文件并提取通讯作者
    # =========================
    print(f"\n正在读取 WoS 文件夹：{PATH_WOS_DIR}")
    wos_papers = load_wos_folder_tables(PATH_WOS_DIR)

    wos_author_rows = []

    for _, row in wos_papers.iterrows():
        paper_key_norm = row["paper_key_norm"]
        authors_raw = row["wos_authors_raw"]
        reprint_raw = row["wos_reprint_raw"]

        author_abbrevs = split_wos_authors(authors_raw)
        corr_abbrevs_raw = extract_corresponding_abbrevs(reprint_raw)
        corr_abbrevs_norm_set = set(
            norm_abbrev(x) for x in corr_abbrevs_raw if norm_abbrev(x) != ""
        )

        for i, abbr in enumerate(author_abbrevs, start=1):
            abbr_norm = norm_abbrev(abbr)
            wos_author_rows.append({
                "paper_key_norm": paper_key_norm,
                "author_order": i,
                "is_corresponding_author_on_wos": 1 if abbr_norm in corr_abbrevs_norm_set else 0
            })

    wos_author_order_df = pd.DataFrame(wos_author_rows)

    # 按 (paper_key_norm, author_order) 映射通讯作者标记
    author_paper_corr_map = df_author_paper.merge(
        wos_author_order_df,
        on=["paper_key_norm", "author_order"],
        how="left"
    )

    author_paper_corr_map["is_corresponding_author_on_wos"] = (
        pd.to_numeric(author_paper_corr_map["is_corresponding_author_on_wos"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    corr_author_papers_df = author_paper_corr_map[
        author_paper_corr_map["is_corresponding_author_on_wos"] == 1
    ].copy()

    corr_count_df = (
        corr_author_papers_df.groupby(AUTHOR_COL, as_index=False)[PAPER_COL]
        .nunique()
        .rename(columns={PAPER_COL: "corresponding_author_paper_count"})
    )

    # =========================
    # D. 统计“第一作者或通讯作者”论文数量
    #    同一篇同时满足两种角色，只记1篇
    # =========================
    author_paper_corr_map["is_first_author"] = np.where(
        author_paper_corr_map["author_order"] == 1, 1, 0
    )

    author_paper_corr_map["is_first_or_corresponding_author"] = np.where(
        (author_paper_corr_map["is_first_author"] == 1) |
        (author_paper_corr_map["is_corresponding_author_on_wos"] == 1),
        1, 0
    )

    first_or_corr_papers_df = author_paper_corr_map[
        author_paper_corr_map["is_first_or_corresponding_author"] == 1
    ].copy()

    first_or_corr_count_df = (
        first_or_corr_papers_df.groupby(AUTHOR_COL, as_index=False)[PAPER_COL]
        .nunique()
        .rename(columns={PAPER_COL: "first_or_corresponding_author_paper_count"})
    )

    # =========================
    # E. 合并为最终作者表
    # =========================
    result = (
        author_master
        .merge(corr_count_df, on=AUTHOR_COL, how="left")
        .merge(first_count_df, on=AUTHOR_COL, how="left")
        .merge(first_or_corr_count_df, on=AUTHOR_COL, how="left")
        .merge(single_count_df, on=AUTHOR_COL, how="left")
    )

    result["corresponding_author_paper_count"] = (
        pd.to_numeric(result["corresponding_author_paper_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    result["first_author_paper_count"] = (
        pd.to_numeric(result["first_author_paper_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    result["first_or_corresponding_author_paper_count"] = (
        pd.to_numeric(result["first_or_corresponding_author_paper_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    result["single_paper_count"] = (
        pd.to_numeric(result["single_paper_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    result["has_corresponding_author_role"] = np.where(
        result["corresponding_author_paper_count"] > 0, "Yes", "No"
    )
    result["has_first_author_role"] = np.where(
        result["first_author_paper_count"] > 0, "Yes", "No"
    )
    result["has_first_or_corresponding_author_role"] = np.where(
        result["first_or_corresponding_author_paper_count"] > 0, "Yes", "No"
    )
    result["has_single_paper"] = np.where(
        result["single_paper_count"] > 0, "Yes", "No"
    )

    result = result[
        [
            AUTHOR_COL,
            "has_corresponding_author_role",
            "corresponding_author_paper_count",
            "has_first_author_role",
            "first_author_paper_count",
            "has_first_or_corresponding_author_role",
            "first_or_corresponding_author_paper_count",
            "has_single_paper",
            "single_paper_count"
        ]
    ].copy()

    result = result.sort_values(
        by=[
            "has_first_or_corresponding_author_role",
            "first_or_corresponding_author_paper_count",
            "has_first_author_role",
            "first_author_paper_count",
            "has_corresponding_author_role",
            "corresponding_author_paper_count",
            "has_single_paper",
            "single_paper_count",
            AUTHOR_COL
        ],
        ascending=[False, False, False, False, False, False, False, False, True]
    ).reset_index(drop=True)

    # =========================
    # F. 输出
    # =========================
    result.to_csv(PATH_OUTPUT, index=False, encoding="utf-8-sig")

    print("\n========== 完成 ==========")
    print(f"作者总数：{len(result)}")
    print(f"输出文件：{PATH_OUTPUT}")


if __name__ == "__main__":
    main()

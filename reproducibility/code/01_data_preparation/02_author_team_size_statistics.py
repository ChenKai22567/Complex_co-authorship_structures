# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
统计 Web of Science 文章作者团队规模。

功能：
1. 读取归档 `data/01_data_preparation/wos_1900_2025` 中的 .xls/.xlsx 文件。
2. 按 Publication Year 筛选统计窗口；默认预设窗口为 1966-2025。
3. 统计每篇文章的作者数量分布，分箱为 ['1', '2', '3', '4', '5', '≥6']。
4. 输出两个 CSV：
   - author_team_size_distribution_<start>_<end>.csv：窗口内总体分布
   - author_team_size_by_year_<start>_<end>.csv：窗口内逐年分布

使用示例：
  python author_team_size_stats.py
  python author_team_size_stats.py --preset 1966-2025
  python author_team_size_stats.py --start-year 1990 --end-year 2025
"""

import argparse
import glob
import os
import re
from typing import Optional

import pandas as pd


TEAM_SIZE_BINS = ['1', '2', '3', '4', '5', '≥6']
PRESET_WINDOWS = {
    '1966-2025': (1966, 2025),
}


def parse_year(value) -> Optional[int]:
    """Extract a four-digit publication year from a WOS field value."""
    if pd.isna(value):
        return None
    match = re.search(r'\d{4}', str(value))
    if not match:
        return None
    return int(match.group(0))


def count_authors(value) -> int:
    """Count authors from WOS 'Author Full Names', separated by semicolons."""
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return 0
    return len([part.strip() for part in text.split(';') if part.strip()])


def team_size_bin(author_count: int) -> Optional[str]:
    """Map author count to ['1', '2', '3', '4', '5', '≥6']; return None for missing."""
    if author_count <= 0:
        return None
    if author_count >= 6:
        return '≥6'
    return str(author_count)


def read_wos_files(input_dir: str) -> pd.DataFrame:
    """Read required WOS columns from every Excel file in input_dir."""
    file_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.xls'))
        + glob.glob(os.path.join(input_dir, '*.xlsx'))
    )
    if not file_paths:
        raise FileNotFoundError(f'No .xls/.xlsx files found in: {input_dir}')

    frames = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        engine = 'xlrd' if ext == '.xls' else 'openpyxl'
        try:
            df = pd.read_excel(
                file_path,
                dtype=str,
                engine=engine,
                usecols=['Publication Year', 'Author Full Names'],
            )
        except Exception as exc:
            print(f'! 无法读取 {file_path}（{exc}），已跳过。')
            continue

        df['source_file'] = os.path.basename(file_path)
        frames.append(df)
        print(f'√ 已读取 Excel：{file_path}，{len(df)} 行')

    if not frames:
        raise RuntimeError('No readable WOS Excel files were loaded.')

    return pd.concat(frames, ignore_index=True)


def build_distribution(df: pd.DataFrame, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build aggregate and yearly team-size distributions for the selected window."""
    work = df.copy()
    work['publication_year'] = work['Publication Year'].apply(parse_year)
    work['author_count'] = work['Author Full Names'].apply(count_authors)
    work['team_size_bin'] = work['author_count'].apply(team_size_bin)

    in_window = work[
        (work['publication_year'] >= start_year)
        & (work['publication_year'] <= end_year)
    ].copy()

    total_papers = len(in_window)
    missing_author_count = int(in_window['team_size_bin'].isna().sum())
    valid = in_window.dropna(subset=['team_size_bin'])

    counts = valid['team_size_bin'].value_counts().reindex(TEAM_SIZE_BINS, fill_value=0)
    aggregate = pd.DataFrame({
        'team_size_bin': TEAM_SIZE_BINS,
        'paper_count': [int(counts[label]) for label in TEAM_SIZE_BINS],
    })
    valid_total = int(aggregate['paper_count'].sum())
    aggregate['percentage'] = aggregate['paper_count'].apply(
        lambda count: round(count / valid_total * 100, 4) if valid_total else 0.0
    )

    yearly = (
        valid
        .groupby(['publication_year', 'team_size_bin'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TEAM_SIZE_BINS, fill_value=0)
        .reset_index()
        .rename(columns={'publication_year': 'year'})
    )
    all_years = pd.DataFrame({'year': list(range(start_year, end_year + 1))})
    yearly = all_years.merge(yearly, on='year', how='left').fillna(0)
    for label in TEAM_SIZE_BINS:
        yearly[label] = yearly[label].astype(int)
    yearly['total_papers'] = yearly[TEAM_SIZE_BINS].sum(axis=1)

    summary = {
        'start_year': start_year,
        'end_year': end_year,
        'total_rows_in_window': total_papers,
        'valid_author_count_rows': valid_total,
        'missing_author_count_rows': missing_author_count,
    }
    return aggregate, yearly, summary


def resolve_window(args: argparse.Namespace) -> tuple[int, int, str]:
    """Resolve preset or custom year window from CLI arguments."""
    if args.start_year is not None or args.end_year is not None:
        if args.start_year is None or args.end_year is None:
            raise ValueError('--start-year and --end-year must be used together.')
        start_year, end_year = args.start_year, args.end_year
        label = f'{start_year}-{end_year}'
    else:
        if args.preset not in PRESET_WINDOWS:
            available = ', '.join(sorted(PRESET_WINDOWS))
            raise ValueError(f'Unknown preset {args.preset!r}. Available presets: {available}')
        start_year, end_year = PRESET_WINDOWS[args.preset]
        label = args.preset

    if start_year > end_year:
        raise ValueError('start year cannot be greater than end year.')
    return start_year, end_year, label


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='统计 WOS 文章作者团队规模分布。')
    parser.add_argument(
        '--input-dir',
        default=str(_INPUT_ROOT / "01_data_preparation" / "wos_1900_2025"),
        help='WOS Excel 文件夹，默认归档data/01_data_preparation/wos_1900_2025。',
    )
    parser.add_argument(
        '--output-dir',
        default=str(_OUTPUT_ROOT / "01_data_preparation" / "team_size_statistics"),
        help='输出文件夹，默认归档results/01_data_preparation/author_team_size_statistics。',
    )
    parser.add_argument(
        '--preset',
        default='1966-2025',
        choices=sorted(PRESET_WINDOWS),
        help='预设统计窗口，默认 1966-2025。',
    )
    parser.add_argument('--start-year', type=int, help='自定义统计窗口起始年份。')
    parser.add_argument('--end-year', type=int, help='自定义统计窗口结束年份。')
    args = parser.parse_args()

    start_year, end_year, window_label = resolve_window(args)
    os.makedirs(args.output_dir, exist_ok=True)

    df = read_wos_files(args.input_dir)
    aggregate, yearly, summary = build_distribution(df, start_year, end_year)

    safe_label = window_label.replace('-', '_')
    aggregate_path = os.path.join(args.output_dir, f'author_team_size_distribution_{safe_label}.csv')
    yearly_path = os.path.join(args.output_dir, f'author_team_size_by_year_{safe_label}.csv')
    summary_path = os.path.join(args.output_dir, f'author_team_size_summary_{safe_label}.csv')

    aggregate.to_csv(aggregate_path, index=False, encoding='utf-8-sig')
    yearly.to_csv(yearly_path, index=False, encoding='utf-8-sig')
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding='utf-8-sig')

    print(f'\n统计窗口：{start_year}-{end_year}')
    print('作者团队规模总体分布：')
    for _, row in aggregate.iterrows():
        print(f"  {row['team_size_bin']}: {row['paper_count']} ({row['percentage']}%)")
    print(f"有效记录数：{summary['valid_author_count_rows']}")
    print(f"缺失作者字段记录数：{summary['missing_author_count_rows']}")
    print(f'[OK] 已保存总体分布：{aggregate_path}')
    print(f'[OK] 已保存逐年分布：{yearly_path}')
    print(f'[OK] 已保存摘要：{summary_path}')


if __name__ == '__main__':
    main()

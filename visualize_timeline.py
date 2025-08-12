#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化：
  左侧：按时间轴展示每个请求的总耗时拆分（堆叠柱状图）
    - 下段：首 token 时延
    - 上段：token 增量耗时（若提供逐 token 时延，则求和显示总增量耗时）
  右侧：每个请求的输入输出tokens数量（分组条形图）

同时统计关键列的指标：均值、最大值、最小值、P90、样本量，并在左图叠加均值线。
- 统计列：Request tokens(input_tokens)、Response tokens(output_tokens)、first_token_time、decode_token_time(使用decode_time，总和)、total_time(ms)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("excel", help="原始 Excel/CSV 文件")
    parser.add_argument("-s", "--save", help="保存为 html 文件名", default="timeline_with_tokens.html")
    return parser.parse_args()


def read_dataframe(input_path: str) -> pd.DataFrame:
    file_path = Path(input_path)
    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(input_path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def to_datetime_safe(series: pd.Series) -> pd.Series:
    # 按你的脚本：不做 errors='coerce'，严格解析
    return pd.to_datetime(series)


def compute_decode_time_ms(row: pd.Series) -> float:
    value = row.get("decode_token_time")
    output_tokens = row.get("output_tokens", 0) or 0

    if pd.isna(value):
        return 0.0

    if isinstance(value, list):
        try:
            return float(sum(float(x) for x in value))
        except Exception:
            return 0.0

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                arr = json.loads(stripped)
                return float(sum(float(x) for x in arr))
            except Exception:
                return 0.0
        try:
            avg_per_token_ms = float(value)
            return avg_per_token_ms * float(output_tokens)
        except Exception:
            return 0.0

    try:
        avg_per_token_ms = float(value)
        return avg_per_token_ms * float(output_tokens)
    except Exception:
        return 0.0


def calculate_statistics(data: List[float]) -> Dict:
    """计算统计指标：均值、最大值、最小值、P90值"""
    if not data:
        return {"mean": None, "max": None, "min": None, "p90": None, "count": 0}
    arr = np.array(list(data), dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "count": int(arr.size),
    }


def build_plot(df: pd.DataFrame, output_html: Path) -> None:
    # 时间列处理
    df = df.copy()
    df["arrive_timestamp"] = to_datetime_safe(df["arrive_timestamp"])  # 可能为 NaT

    # 计算 decode_time（毫秒）与 total_time（秒 -> 毫秒）
    df["decode_time"] = df.apply(compute_decode_time_ms, axis=1)
    df["total_time"] = df["total_time"] * 1000.0

    # 排序
    df = df.sort_values('arrive_timestamp').reset_index(drop=True)

    # y 轴标签：按时间分桶+序号
    df['s_time'] = df['arrive_timestamp'].dt.floor('S')
    df['rank'] = df.groupby('s_time').cumcount()
    df['y_label'] = (
        df['arrive_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f') + '_' + df['rank'].astype(str)
    )

    # 统计指标（按你的字段命名约定）
    # Request tokens -> input_tokens, Response tokens -> output_tokens
    def sanitize_numeric(series: pd.Series) -> List[float]:
        values: List[float] = []
        for v in series.tolist():
            try:
                if pd.isna(v):
                    continue
                fv = float(v)
                # 与示例逻辑一致：0 和 -1 不纳入统计
                if fv in (0.0, -1.0):
                    continue
                values.append(fv)
            except Exception:
                continue
        return values

    stats_inputs = calculate_statistics(sanitize_numeric(df['input_tokens']))
    stats_outputs = calculate_statistics(sanitize_numeric(df['output_tokens']))
    stats_first = calculate_statistics(sanitize_numeric(df['first_token_time']))
    stats_decode = calculate_statistics(sanitize_numeric(df['decode_time']))  # 使用总增量耗时
    stats_total = calculate_statistics(sanitize_numeric(df['total_time']))

    # 打印统计结果
    print("\n===== 统计结果 =====")
    def print_stats(title: str, stats: Dict):
        print(f"\n{title}:")
        print(f"  有效数据量: {stats['count']}")
        print(f"  均值: {stats['mean']}")
        print(f"  最大值: {stats['max']}")
        print(f"  最小值: {stats['min']}")
        print(f"  P90值: {stats['p90']}")

    print_stats('Request tokens', stats_inputs)
    print_stats('Response tokens', stats_outputs)
    print_stats('first_token_time (ms)', stats_first)
    print_stats('decode_token_time -> decode_time 总和 (ms)', stats_decode)
    print_stats('total_time (ms)', stats_total)

    # 构建图表
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("请求耗时分解", "输入输出Tokens数量"),
        column_widths=[0.7, 0.3],
    )

    # 左侧：首 token 时延
    fig.add_trace(
        go.Bar(
            base=0,
            x=df["first_token_time"],
            y=df["y_label"],
            orientation="h",
            name="首token时延",
            marker=dict(color="#ff7f0e", line=dict(color="black", width=0.5)),
            customdata=df[["arrive_timestamp", "first_token_time", "total_time"]],
            hovertemplate=(
                "到达时间: %{customdata[0]}<br>"
                "首token时延: %{customdata[1]:.2f} ms<br>"
                "总耗时: %{customdata[2]:.2f} ms<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # 左侧：token 增量耗时（总和）
    fig.add_trace(
        go.Bar(
            base=df["first_token_time"],
            x=df["decode_time"],
            y=df["y_label"],
            orientation="h",
            name="token增量耗时",
            marker=dict(color="#1f77b4", line=dict(color="black", width=0.5)),
            customdata=df[["arrive_timestamp", "decode_time", "total_time"]],
            hovertemplate=(
                "到达时间: %{customdata[0]}<br>"
                "增量耗时: %{customdata[1]:.2f} ms<br>"
                "总耗时: %{customdata[2]:.2f} ms<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # 左侧：均值线（首token均值、总耗时均值）
    if stats_first["mean"] is not None:
        fig.add_vline(
            x=stats_first["mean"], line_dash="dash", line_color="#ff7f0e",
            annotation_text="首token均值", annotation_font_color="#ff7f0e",
            row=1, col=1,
        )
    if stats_total["mean"] is not None:
        fig.add_vline(
            x=stats_total["mean"], line_dash="dash", line_color="#000000",
            annotation_text="总耗时均值", annotation_font_color="#000000",
            row=1, col=1,
        )

    # 右侧：input_tokens
    fig.add_trace(
        go.Bar(
            x=df["input_tokens"],
            y=df["y_label"],
            orientation="h",
            name="输入tokens",
            marker=dict(color="#2ca02c", line=dict(color="black", width=0.5)),
            customdata=df[["arrive_timestamp", "input_tokens"]],
            hovertemplate=(
                "到达时间: %{customdata[0]}<br>"
                "输入tokens: %{customdata[1]}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # 右侧：output_tokens
    fig.add_trace(
        go.Bar(
            x=df["output_tokens"],
            y=df["y_label"],
            orientation="h",
            name="输出tokens",
            marker=dict(color="#d62728", line=dict(color="black", width=0.5)),
            customdata=df[["arrive_timestamp", "output_tokens"]],
            hovertemplate=(
                "到达时间: %{customdata[0]}<br>"
                "输出tokens: %{customdata[1]}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="请求耗时与Tokens分析",
        height=max(600, len(df) * 20),
        margin=dict(l=200, r=50, t=80, b=50),
        template="plotly_white",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="耗时 (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Tokens数量", row=1, col=2)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)


def main():
    args = parse_args()
    df = read_dataframe(args.excel)
    build_plot(df, Path(args.save))
    print(f"可视化已生成：{Path(args.save).resolve()}")


if __name__ == "__main__":
    main()
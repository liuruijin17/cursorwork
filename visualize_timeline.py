#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化：
  左侧：按时间轴展示每个请求的总耗时拆分（堆叠柱状图）
    - 下段：首 token 时延
    - 上段：token 增量耗时（若提供逐 token 时延，则求和显示总增量耗时）
  右侧：每个请求的输入输出tokens数量（分组条形图）
"""

import argparse
import json
from pathlib import Path

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
    # 尝试将字符串解析为 datetime（新数据中可能不是 epoch 时间，转换会产生 NaT）
    return pd.to_datetime(series, errors="coerce")


def compute_decode_time_ms(row: pd.Series) -> float:
    value = row.get("decode_token_time")
    output_tokens = row.get("output_tokens", 0) or 0

    # 已是缺失
    if pd.isna(value):
        return 0.0

    # 若是列表（例如来自 CSV 被正确解析为对象，或我们前置处理过）
    if isinstance(value, list):
        try:
            return float(sum(float(x) for x in value))
        except Exception:
            return 0.0

    # 若是字符串，尝试当作 JSON 数组解析
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                arr = json.loads(stripped)
                return float(sum(float(x) for x in arr))
            except Exception:
                return 0.0
        # 退化为数值：视为“平均每 token 时延（毫秒）”，乘以输出 token 数得总增量时长
        try:
            avg_per_token_ms = float(value)
            return avg_per_token_ms * float(output_tokens)
        except Exception:
            return 0.0

    # 退化：将其视为平均值 * 输出 tokens
    try:
        avg_per_token_ms = float(value)
        return avg_per_token_ms * float(output_tokens)
    except Exception:
        return 0.0


def build_plot(df: pd.DataFrame, output_html: Path) -> None:
    # 时间列处理
    df = df.copy()
    df["arrive_dt"] = to_datetime_safe(df["arrive_timestamp"])  # 可能为 NaT
    # 也尝试解析为数值（新数据中可能是单调时钟值）
    df["arrive_num"] = pd.to_numeric(df["arrive_timestamp"], errors="coerce")

    # 计算 decode_time（毫秒）与 total_time（秒 -> 毫秒）
    df["decode_time"] = df.apply(compute_decode_time_ms, axis=1)
    df["total_time"] = df["total_time"] * 1000.0

    # 对显示的到达时间做字符串格式化，兼容 NaT/数值
    def format_arrive_display(r: pd.Series) -> str:
        if pd.notna(r["arrive_dt"]):
            return r["arrive_dt"].strftime("%Y-%m-%d %H:%M:%S.%f")
        if pd.notna(r["arrive_num"]):
            return f"start_time={r['arrive_num']:.6f}"
        # 兜底：保留原始字符串
        return str(r.get("arrive_timestamp", ""))

    df["arrive_display"] = df.apply(format_arrive_display, axis=1)

    # 排序：优先按 datetime，其次按数值，最后不变
    if df["arrive_dt"].notna().any():
        df = df.sort_values("arrive_dt")
    elif df["arrive_num"].notna().any():
        df = df.sort_values("arrive_num")
    df = df.reset_index(drop=True)

    # y 轴标签：使用 arrive_display + 序号
    df["s_time"] = df["arrive_display"].str.slice(0, 26)  # 限长避免过长
    df["rank"] = df.groupby("s_time").cumcount()
    df["y_label"] = df["arrive_display"] + "_" + df["rank"].astype(str)

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
            customdata=df[["arrive_display", "first_token_time", "total_time"]],
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
            customdata=df[["arrive_display", "decode_time", "total_time"]],
            hovertemplate=(
                "到达时间: %{customdata[0]}<br>"
                "增量耗时: %{customdata[1]:.2f} ms<br>"
                "总耗时: %{customdata[2]:.2f} ms<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # 右侧：input_tokens
    fig.add_trace(
        go.Bar(
            x=df["input_tokens"],
            y=df["y_label"],
            orientation="h",
            name="输入tokens",
            marker=dict(color="#2ca02c", line=dict(color="black", width=0.5)),
            customdata=df[["arrive_display", "input_tokens"]],
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
            customdata=df[["arrive_display", "output_tokens"]],
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
import json
import pandas as pd
from datetime import datetime
import argparse
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from statistics import mean


def parse_arguments():
    parser = argparse.ArgumentParser(description='逐行处理JSON文件并转换为XLSX')
    parser.add_argument('json_file', help='JSON文件路径（每行一个JSON对象）')
    parser.add_argument('--model', required=True, help='模型名称')
    parser.add_argument('--output', required=True, help='输出XLSX文件路径')
    return parser.parse_args()


def build_row_from_old_format(item: dict, model_name: str, line_num: int):
    required_fields = [
        "event_id",
        "timestamp",
        "prompt_token_num",
        "completion_token_num",
        "first_chunk_time",
        "record_time",
    ]
    missing = [f for f in required_fields if f not in item]
    if missing:
        print(f"警告：第{line_num}行缺少字段{missing}，已跳过")
        return None

    try:
        arrive_ts = datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        print(f"警告：第{line_num}行时间戳转换失败，已跳过（{str(e)}）")
        return None

    completion_num = item["completion_token_num"] or 0
    if completion_num:
        decode_time_ms = (
            (item["record_time"] - item["timestamp"] - item["first_chunk_time"]) / completion_num * 1000
        )
    else:
        decode_time_ms = 0

    total_time_s = item["record_time"] - item["timestamp"]

    return {
        "request_id": item["event_id"],
        "model_name": model_name,
        "arrive_timestamp": arrive_ts,
        "input_tokens": item["prompt_token_num"],
        "output_tokens": item["completion_token_num"],
        "first_token_time": item["first_chunk_time"] * 1000,
        "decode_token_time": decode_time_ms,
        "total_time": total_time_s,
    }


def build_rows_from_new_format(obj: dict, model_name: str, line_num: int):
    rows = []

    # 期望结构：{"<request_id>": {"input_len": int, "output_len": int, "latency": [..], ...}, ...}
    for request_id, record in obj.items():
        if not isinstance(record, dict):
            print(f"警告：第{line_num}行中请求{request_id}的记录不是对象，已跳过")
            continue

        input_len = record.get("input_len")
        output_len = record.get("output_len")
        latency_list = record.get("latency") or []
        start_time = record.get("start_time")
        end_time = record.get("end_time")
        req_latency_ms = record.get("req_latency")

        # 基础校验
        if input_len is None or output_len is None or not isinstance(latency_list, list) or len(latency_list) == 0:
            print(f"警告：第{line_num}行请求{request_id}缺少必要字段(input_len/output_len/latency)，已跳过")
            continue

        first_token_time_ms = float(latency_list[0])
        decode_latencies_ms = [float(x) for x in latency_list[1:]]
        decode_token_time_ms = mean(decode_latencies_ms) if len(decode_latencies_ms) > 0 else 0.0

        # total_time 按旧脚本保持“秒”为单位
        if req_latency_ms is not None:
            total_time_s = float(req_latency_ms) / 1000.0
        elif start_time is not None and end_time is not None:
            try:
                total_time_s = float(end_time) - float(start_time)
            except Exception:
                total_time_s = 0.0
        else:
            total_time_s = 0.0

        # 旧脚本的 arrive_timestamp 来源于 epoch 秒的 timestamp；新数据给的是单调时钟，无法转换
        # 这里保留列但使用 start_time 的原始数值字符串，若不存在则为空
        arrive_ts_str = f"{start_time}" if start_time is not None else ""

        rows.append({
            "request_id": str(request_id),
            "model_name": model_name,
            "arrive_timestamp": arrive_ts_str,
            "input_tokens": int(input_len),
            "output_tokens": int(output_len),
            "first_token_time": first_token_time_ms,
            "decode_token_time": decode_token_time_ms,
            "total_time": total_time_s,
        })

    return rows


def process_line(line: str, model_name: str, line_num: int):
    stripped_line = line.strip()
    if not stripped_line:
        return []

    try:
        obj = json.loads(stripped_line)
    except json.JSONDecodeError as e:
        print(f"警告：第{line_num}行JSON格式错误，已跳过（{str(e)}）")
        return []

    # 兼容两种格式：
    # 1) 旧格式：单个对象，包含 event_id/timestamp 等字段
    # 2) 新格式：顶层为 {request_id: {..}, ...}
    if isinstance(obj, dict) and "event_id" in obj:
        row = build_row_from_old_format(obj, model_name, line_num)
        return [row] if row else []

    if isinstance(obj, dict):
        return build_rows_from_new_format(obj, model_name, line_num)

    print(f"警告：第{line_num}行JSON顶层不是对象，已跳过")
    return []


def main():
    args = parse_arguments()

    headers = [
        "request_id",
        "model_name",
        "arrive_timestamp",
        "input_tokens",
        "output_tokens",
        "first_token_time",
        "decode_token_time",
        "total_time",
    ]

    data_rows = []
    processed_count = 0

    with open(args.json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            rows = process_line(line, args.model, line_num)
            if rows:
                data_rows.extend(rows)
                processed_count += len(rows)

    if data_rows:
        df = pd.DataFrame(data_rows)
        df_sorted = df.sort_values(by="request_id")

        wb = Workbook()
        ws = wb.active
        ws.append(headers)
        for row in dataframe_to_rows(df_sorted, index=False, header=False):
            ws.append(row)

        wb.save(args.output)
        print(f"处理完成！共处理{processed_count}条有效请求，已按request_id排序并保存至{args.output}")
    else:
        print("没有有效数据可处理，未生成Excel文件")


if __name__ == "__main__":
    main()
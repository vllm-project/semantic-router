"""Read native Jev records and print a review table; never call a model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def cell(value):
    if value is None:
        return "—"
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("|", "&#124;")
            .replace("\r", " ").replace("\n", " "))


def summarize(record):
    """Interpret recorded outcomes, without revalidating provider probabilities."""
    schema = record.get("schema")
    expected = record.get("expected")
    if "expected" not in record or (expected is not None and not isinstance(expected, str)):
        raise ValueError("expected must be an explicit string or null")
    attempts = record.get("attempts")
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a non-negative integer")
    if schema == "jev-research-not-executed.v1":
        if attempts != 0 or record.get("execution_status") != "not_executed" or not record.get("reason"):
            raise ValueError("inconsistent not-executed record")
        if any(k in record for k in ("correct", "contract_valid", "raw_response", "elapsed_ms", "http_status", "request")):
            raise ValueError("not-executed record contains measurements")
        return ["未执行", expected, None, "未检查", "不评分", attempts, record["reason"]]
    if schema != "jev-research-record.v1":
        raise ValueError("unknown record schema")
    valid = record.get("contract_valid")
    if type(valid) is not bool:
        raise ValueError("contract_valid must be an explicit boolean")
    try:
        raw = json.loads(record["raw_response"])
        prediction = raw["answers"]["intent"]["choice"]
        if not isinstance(prediction, str):
            raise ValueError("choice must be a string")
    except (KeyError, TypeError, ValueError):
        if valid:
            raise ValueError("valid record has no readable raw prediction") from None
        prediction = None
    if not valid:
        if "correct" in record:
            raise ValueError("failed record must not contain correctness")
        return ["失败", expected, prediction, "未通过", "不评分", attempts,
                record.get("error_kind") or "unspecified_failure"]
    if attempts < 1 or record.get("error") or record.get("error_kind"):
        raise ValueError("valid record has inconsistent attempts/error fields")
    if expected is None:
        if "correct" in record:
            raise ValueError("diagnostic record must omit correctness")
        return ["诊断", None, prediction, "通过", "不评分", attempts, "无参考答案"]
    correct = record.get("correct")
    if type(correct) is not bool or correct != (prediction == expected):
        raise ValueError("correctness is missing or disagrees with the recorded labels")
    return ["可评分", expected, prediction, "通过", "正确" if correct else "错误", attempts, None]


def render(data, mode):
    if mode not in ("mock", "live"):
        raise ValueError("mode must be mock or live")
    rows, seen, identity = [], set(), None
    for line, text in enumerate(data.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            record = json.loads(text)
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            case_id = record.get("id")
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise ValueError("case ID missing or duplicated")
            source = tuple(record.get(k) for k in ("revision", "dataset_sha256", "question_sha256"))
            if any(not isinstance(v, str) or not v for v in source):
                raise ValueError("source identity is missing")
            if identity is not None and source != identity:
                raise ValueError("mixed source revisions or input/question hashes")
            identity = source
            seen.add(case_id)
            rows.append([case_id, *summarize(record), line])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {line}: {exc}") from exc
    if not rows:
        raise ValueError("no records")
    title = "MOCK 演示：不是 Jev 实验结果" if mode == "mock" else "Jev 记录审阅表（来源模式由操作者声明）"
    header = [f"# {title}", "", f"源 JSONL SHA-256：`{hashlib.sha256(data).hexdigest()}`", "",
              "仅展示记录中的判定，未重新校验概率。失败行的预测仅供观察，不可视为有效分类。",
              "不计算准确率，不验证计划题目是否齐全；不是三方对比报告。请保留原文件和运行说明。", "",
              "| ID | 状态 | 参考答案 | 原始预测 | 契约记录 | 正确性 | 尝试次数 | 原因 | 源文件行 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    return "\n".join(header + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--mode", required=True, choices=("mock", "live"), help="Operator-declared source; not inferred or verified")
    args = parser.parse_args()
    try:
        report = render(args.records.read_bytes(), args.mode)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot render records: {exc}\n")
    sys.stdout.write(report)


if __name__ == "__main__":
    main()

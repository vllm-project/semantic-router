"""Measure forced feedback classifications on ordinary new user questions.

These original development negatives have no valid label in the four-class
taxonomy. They are never assigned SAT, used for training, or counted as accuracy.
Each EN/ZH translation pair is one semantic scenario, not two independent cases.
"""

# Native punctuation and optional heavy CLI dependencies are intentional.
# ruff: noqa: RUF001, PLC0415

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .data_contract import ID2LABEL, LABEL2ID

CASES = [
    ("What is the capital of Peru?", "秘鲁的首都是哪里？"),
    (
        "Write a short invitation for a birthday party.",
        "帮我写一份简短的生日聚会邀请。",
    ),
    ("How do I reset the password on my own account?", "我怎样重置自己账号的密码？"),
    ("Explain how photosynthesis works.", "解释一下光合作用的原理。"),
    (
        "What does this Python error mean: KeyError?",
        "Python 的 KeyError 错误是什么意思？",
    ),
    (
        "Give me three different vegetarian dinner ideas.",
        "给我三个不同的素食晚餐方案。",
    ),
    ("Translate 'good morning' into Italian.", "把“早上好”翻译成意大利语。"),
    ("Summarize the history of public libraries.", "总结一下公共图书馆的历史。"),
    ("Can you help me plan a weekend hiking trip?", "你能帮我规划一次周末徒步吗？"),
    ("Which is larger, three eighths or two fifths?", "八分之三和五分之二哪个大？"),
    ("List the main causes of coral bleaching.", "列出珊瑚白化的主要原因。"),
    (
        "Find the mistake in this equation: 7 + 5 = 13.",
        "找出这个等式的错误：7 + 5 = 13。",
    ),
    ("Make a comparison table for trains and buses.", "制作一张火车与公交车的对比表。"),
    (
        "I don't understand recursion. Teach me the basics.",
        "我不理解递归，请教我基础知识。",
    ),
    (
        "What are some alternatives to a traditional résumé?",
        "传统简历之外还有什么替代形式？",
    ),
    (
        "Write a polite email asking for a project update.",
        "写一封礼貌询问项目进展的邮件。",
    ),
    ("How many minutes are there in a week?", "一周有多少分钟？"),
    (
        "Compare an encoder with a decoder in simple terms.",
        "用简单的话比较编码器和解码器。",
    ),
    ("Suggest a name for my new gardening club.", "给我新成立的园艺俱乐部起个名字。"),
    ("Show me how to sort a list of dates.", "演示如何给日期列表排序。"),
]


def records():
    return [
        {
            "id": f"feedback-applicability:{index}:{language}",
            "group_id": f"feedback-applicability:{index}",
            "text": text,
            "language": language,
            "source": "original_non_feedback_development_queries",
            "has_preceding_assistant_turn": False,
            "feedback_label": None,
        }
        for index, pair in enumerate(CASES)
        for language, text in zip(["en", "zh"], pair, strict=True)
    ]


def summarize(predictions, thresholds=(0.5, 0.8, 0.95)):
    if not predictions:
        raise ValueError("No applicability predictions")
    return {
        "rows": len(predictions),
        "semantic_scenarios": len({row["group_id"] for row in predictions}),
        "accuracy": None,
        "reason": "No-feedback is outside the four-class output taxonomy",
        "predicted_labels": dict(Counter(row["prediction"] for row in predictions)),
        "thresholds": {
            str(threshold): {
                "high_confidence_non_sat_fraction": sum(
                    row["prediction"] != "SAT" and row["confidence"] >= threshold
                    for row in predictions
                )
                / len(predictions),
                "high_confidence_by_label": dict(
                    Counter(
                        row["prediction"]
                        for row in predictions
                        if row["confidence"] >= threshold
                    )
                ),
            }
            for threshold in thresholds
        },
    }


def main():
    import torch

    from ..sequence_repair.model import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite applicability evidence")
    adapter_path = Path(args.adapter) / "adapter_model.safetensors"
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    model, tokenizer, labels, _ = load_model(args.base, args.contract, args.adapter)
    if labels != LABEL2ID:
        raise ValueError("Unexpected feedback label contract")
    if not 0 < args.max_length <= model.config.max_position_embeddings:
        raise ValueError("Invalid explicit token budget")
    torch.set_num_threads(8)
    model.eval()
    rows = records()
    inputs = tokenizer(
        [row["text"] for row in rows],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    if inputs["input_ids"].shape[1] > args.max_length:
        raise ValueError("Applicability input exceeds explicit budget")
    with torch.inference_mode():
        probabilities = model(**inputs).logits.float().softmax(-1).tolist()
    predictions = []
    for row, scores in zip(rows, probabilities, strict=True):
        label_id = max(range(len(scores)), key=scores.__getitem__)
        predictions.append(
            {
                **row,
                "scores": scores,
                "prediction": ID2LABEL[label_id],
                "confidence": scores[label_id],
            }
        )
    if hashlib.sha256(adapter_path.read_bytes()).hexdigest() != adapter_sha256:
        raise ValueError("Adapter changed during evaluation; use a frozen snapshot")
    payload = {
        "summary": summarize(predictions),
        "predictions": predictions,
        "precision": "float32_cpu",
        "query_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest(),
        "adapter_sha256": adapter_sha256,
        "base": args.base,
        "selection_use": "development diagnostic; no valid four-class gold labels",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

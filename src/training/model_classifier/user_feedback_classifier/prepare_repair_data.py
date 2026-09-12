"""Materialize an explicit, family-separated feedback repair curriculum.

These are authored examples, not human-annotated production conversations. Old
pseudo-labels are deliberately excluded: they included assistant messages and
non-feedback statements. Final evaluation must use independently frozen data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data_contract import LABEL2ID, parse_example, validate_splits

FAMILIES = Path(__file__).parent / "data" / "repair-v2-families.json"
TRAIN_TOPICS = (
    ("the monthly budget", "月度预算"),
    ("the travel itinerary", "旅行行程"),
    ("the report layout", "报告排版"),
    ("the soup recipe", "汤的食谱"),
    ("the garden plan", "花园规划"),
    ("the meeting agenda", "会议议程"),
    ("the data backup", "数据备份"),
    ("the spreadsheet formula", "表格公式"),
    ("the history summary", "历史摘要"),
    ("the customer email", "客户邮件"),
    ("the project timeline", "项目时间表"),
    ("the exercise routine", "锻炼安排"),
    ("the algebra problem", "代数问题"),
    ("the software installation", "软件安装"),
    ("the school presentation", "学校演示"),
    ("the office seating", "办公室座位"),
    ("the product comparison", "产品对比"),
    ("the book recommendations", "书籍推荐"),
    ("the interview preparation", "面试准备"),
    ("the translation", "翻译"),
    ("the lesson outline", "课程提纲"),
    ("the image description", "图像描述"),
    ("the function parameters", "函数参数"),
    ("the dinner menu", "晚餐菜单"),
    ("the event announcement", "活动通知"),
    ("the file conversion", "文件转换"),
    ("the storage arrangement", "收纳安排"),
    ("the poem", "诗歌"),
    ("the public transport route", "公共交通路线"),
    ("the schedule conflict", "日程冲突"),
)
VALIDATION_TOPICS = (
    ("the aquarium maintenance", "鱼缸维护"),
    ("the exhibition signage", "展览标识"),
    ("the telescope settings", "望远镜设置"),
    ("the music rehearsal", "音乐排练"),
    ("the ferry reservation", "轮渡预订"),
    ("the camera comparison", "相机比较"),
    ("the newsletter headline", "通讯标题"),
    ("the ceramic workshop", "陶艺工作坊"),
)


def build_examples(families: dict) -> dict[str, list[dict]]:
    output = {}
    for split, topics in (("train", TRAIN_TOPICS), ("validation", VALIDATION_TOPICS)):
        rows = []
        for label in LABEL2ID:
            templates = families[split][label]
            for index, template in enumerate(templates):
                chinese = any("\u4e00" <= char <= "\u9fff" for char in template)
                for topic_index, topic in enumerate(topics):
                    text = template.format(topic=topic[int(chinese)])
                    row = parse_example(
                        {
                            "text": text,
                            "label_name": label,
                            "source": "authored-feedback-repair-v2",
                            "group_id": f"{split}-{label}-family-{index % (len(templates) // 2)}",
                            "sample_id": f"{split}-{label}-{index}-{topic_index}",
                            "id": f"{split}-{label}-{index}-{topic_index}",
                            "language": "zh" if chinese else "en",
                            "evidence_scope": "authored curriculum; paired translations and topic substitutions are not independent natural examples",
                        }
                    )
                    rows.append(row)
        output[split] = rows
    return output


def prepare(output_dir: Path, families_path: Path = FAMILIES) -> dict:
    families = json.loads(families_path.read_text(encoding="utf-8"))
    examples = build_examples(families)
    manifest = validate_splits(examples["train"], examples["validation"])
    manifest.update(
        recipe="feedback-repair-v2",
        family_source_sha256=hashlib.sha256(families_path.read_bytes()).hexdigest(),
        semantics="classify the current user's feedback; do not infer unavailable conversation context",
        exclusions="old pseudo-labelled corpus, historical SAT templates and independently frozen final tests are not training inputs",
        validation_scope="held-out expression families and topics within this authored curriculum; not production accuracy",
        final_test_policy="evaluate an independently frozen external/authored set only after checkpoint selection",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        target = output_dir / f"{split}.jsonl"
        target.write_text(payload, encoding="utf-8")
        manifest["splits"][split]["file_sha256"] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir), indent=2))

"""Create a bounded context curriculum from training-only source records.

Original source groups and supervision (including hazard masks) are retained.
The background is independently authored and is different from development
stress probes. Payloads are never truncated or relabelled by a classifier.
"""

# The Chinese prose uses its native punctuation.
# ruff: noqa: RUF001

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .sequence_repair.context import insert_payload

BACKGROUND = {
    "en": """The regional music festival takes place beside a lake. Each morning a small group of volunteers arranges tables for instrument makers and checks the printed programme. The programme contains short descriptions of each performance, together with notes about the composers and the history of their instruments. Visitors often arrive early to see the woodworkers demonstrate how a violin is shaped.

The exhibition hall contains photographs of rehearsals from previous years. One series follows the preparation of a brass ensemble, while another records a choir practising in a school auditorium. The captions describe the work of conductors, librarians and stage crews. An audio station lets visitors compare different interpretations of the same melody.

In the afternoon, families gather under a canvas canopy for a concert. The weather changes slowly from bright sunshine to a light breeze. Between performances the organisers talk about the scenery around the lake and the walking paths that connect nearby villages. A display of hand-drawn maps shows the location of bridges, picnic areas and the railway station.

When the final concert ends, musicians pack their instruments and discuss the next day's schedule. Volunteers collect the programmes left on empty chairs and sort the reusable materials. The festival newsletter will contain photographs, interviews and a short account of the preparations, recognising the many ordinary contributions that made the event possible.""",
    "zh": """地方音乐节在湖边举行。每天早上，志愿者为乐器制作师布置展示桌，并核对印刷好的演出手册。手册介绍每场演出的作品，也记录作曲家和乐器的历史。来访者常常提前到场，观看木工怎样制作小提琴的弧面和琴身。

展厅收藏了往年排练的照片。一组照片记录铜管乐团的准备过程，另一组展示合唱团在学校礼堂练习的场景。图片说明介绍指挥、乐谱管理员和舞台工作人员各自承担的工作。观众还可以在试听区域比较同一段旋律的不同演绎方式。

下午，家庭观众来到帆布篷下欣赏音乐。湖边的阳光逐渐柔和，微风吹过草地。演出间隙，主持人介绍周围村庄的风景，以及连接各个步行路线的桥梁。旁边的手绘地图标出了野餐区、车站和适合观察候鸟的地点。

最后一场演出结束后，乐手收好乐器，交流第二天的安排。志愿者收集空椅子上的节目单，把可以再次使用的材料分类保存。音乐节简报将刊登照片、访谈和准备工作的记录，让人们了解这场活动背后许多普通而细致的贡献。""",
}


def main():
    from transformers import AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label-language", type=int, default=2)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 32768]
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite curriculum")
    if args.per_label_language <= 0:
        raise ValueError("Positive source-family budget required")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(text):
        return len(
            tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        )

    candidates = defaultdict(list)
    for path in args.input:
        for line in Path(path).read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label_name", row["label"])
            language = row.get("language", "en")
            if language not in BACKGROUND:
                language = "en"
            if count(row["text"]) <= min(args.budgets):
                candidates[(label, language)].append(row)
    rows = []
    for (label, language), items in candidates.items():
        groups = set()
        for row in sorted(
            items, key=lambda item: hashlib.sha256(item["text"].encode()).hexdigest()
        ):
            if row["group_id"] in groups:
                continue
            groups.add(row["group_id"])
            for budget in args.budgets:
                for position in ["head", "middle", "tail"]:
                    built = insert_payload(
                        row["text"], BACKGROUND[language], position, budget, count
                    )
                    rows.append(
                        {
                            **row,
                            "id": f"vela-curriculum:{row.get('id',row.get('sample_id'))}:{budget}:{position}",
                            "text": built["text"],
                            "label": label,
                            "source": "authored_training_context_on_source_payload",
                            "source_payload_id": row.get("id", row.get("sample_id")),
                            "language": language,
                            "length_bucket": str(budget),
                            "position": position,
                            "actual_tokens": built["actual_tokens"],
                            "payload_start": built["payload_start"],
                            "payload_end": built["payload_end"],
                        }
                    )
            if len(groups) >= args.per_label_language:
                break
    if not rows:
        raise ValueError("No eligible training payloads")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    )
    manifest = {
        "scope": "training curriculum from training-only groups; original multi-paragraph background",
        "inputs": [
            {
                "file": Path(path).name,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in args.input
        ],
        "rows": len(rows),
        "source_groups": len({row["group_id"] for row in rows}),
        "budgets": args.budgets,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

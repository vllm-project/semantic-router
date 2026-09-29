"""Tiny synthetic snapshots, one writer per HT-DEV source.

`WRITERS[key](root, flip)` writes a snapshot whose items are the same for both values
of `flip`; only the gold fields change (or, for Moral Stories, which action is the
moral one). All text is invented.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path


def parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def table(path: Path, rows: list[dict], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def nycc(root: Path, flip: bool) -> None:
    folds = ("ranking", "ranking_1", "ranking_2", "ranking_3", "ranking_4")
    for f, fold in enumerate(folds):
        rows = []
        for i in range(2):
            n = f * 10 + i
            label = "AB"[(n + flip) % 2]
            rows.append(
                {
                    "contest_number": str(100 + n),
                    "image_location": f"an office {n}",
                    "image_description": f"A man sits at desk number {n}.",
                    "image_uncanny_description": "The desk is floating.",
                    "entities": ["https://en.wikipedia.org/wiki/Office_chair"],
                    "caption_choices": [f"First caption {n}.", f"Second one {n}!"],
                    "label": label,
                    "instance_id": f"inst{n}",
                }
            )
        parquet(root / f"{fold}/test-00000-of-00001.parquet", rows)


def circa(root: Path, flip: bool) -> None:
    labels = [0, 1, 2, 3, 4] if not flip else [1, 0, 3, 2, 4]
    parquet(
        root / "data/train-00000-of-00001.parquet",
        [
            {
                "context": "X and Y are friends.",
                "question-X": f"Do you like tea {i}?",
                "canquestion-X": "I like tea.",
                "answer-Y": f"I drink it every morning {i}.",
                "judgements": "Yes#Yes#No#Yes#Yes",
                "goldstandard1": 0,
                "goldstandard2": label,
            }
            for i, label in enumerate(labels)
        ],
    )


def figqa(root: Path, flip: bool) -> None:
    for name in ("dev.csv", "train.csv"):
        table(
            root / name,
            [
                {
                    "startphrase": f"Her smile was a lighthouse {i}",
                    "ending1": "She looked welcoming.",
                    "ending2": f"She looked cold {i}.",
                    "labels": str((i + flip) % 2),
                    "valid": "1",
                }
                for i in range(3)
            ],
        )


def ukpconvarg(root: Path, flip: bool) -> None:
    table(
        root / "data/UKPConvArg1Strict-CSV/ban-toy-things_yes-ban-them.csv",
        [
            {
                "#id": f"arg{i}_arg{i + 10}",
                "label": "a1" if (i + flip) % 2 else "a2",
                "a1": f"Toys hurt people {i}.<br/>Really.",
                "a2": f"Toys are fine and make children happy {i}.",
            }
            for i in range(3)
        ],
        "\t",
    )


def pubhealth(root: Path, flip: bool) -> None:
    for split in ("train", "validation", "test"):
        parquet(
            root / f"default/{split}/0000.parquet",
            [
                {
                    "claim_id": f"{split}{i}",
                    "claim": f"Vitamin {i} cures colds.",
                    "date_published": "May 1, 2020",
                    "explanation": "The claim is false because of the trial.",
                    "fact_checkers": "Some Checker",
                    "main_text": "Long article.",
                    "sources": "https://example.org",
                    "label": [0, 1, 2][(i + flip) % 3] if i < 3 else 3,
                    "subjects": "Health",
                }
                for i in range(4)
            ],
        )


def brighter(root: Path, flip: bool) -> None:
    emotions = ("anger", "fear", "joy", "sadness", "surprise")
    for split in ("train", "dev", "test"):
        rows = []
        for i in range(4):
            hot = (i + flip) % 4
            row = {"id": f"eng_{split}_{i}", "text": f"I saw the storm coming {i}."}
            for j, e in enumerate(emotions):
                row[e] = 1 if j == hot and hot < 3 else 0
            row["disgust"] = None
            row["emotions"] = []
            rows.append(row)
        rows.append({**rows[0], "id": f"eng_{split}_multi", "anger": 1, "fear": 1})
        parquet(root / f"eng/{split}-00000-of-00001.parquet", rows)


def claim_stance(root: Path, flip: bool) -> None:
    for name in ("test.csv", "train.csv"):
        table(
            root / name,
            [
                {
                    "topicId": str(i % 2),
                    "topicText": f"This house would ban toys {i % 2}",
                    "claims.claimId": f"{name[:2]}{i}",
                    "claims.stance": "PRO" if (i + flip) % 2 else "CON",
                    "claims.claimCorrectedText": f"Toys cause harm {i}.",
                    "claims.claimSentiment": "-1",
                }
                for i in range(3)
            ],
        )


def hyperpartisan(root: Path, flip: bool) -> None:
    parts = {
        "test": ("20181207", "articles-test-byarticle", "ground-truth-test-byarticle"),
        "train": (
            "20181122",
            "articles-training-byarticle",
            "ground-truth-training-byarticle",
        ),
    }
    long_body = "".join(f"<p>Sentence number {k} of the body.</p>" for k in range(150))
    for split, (date, articles, truth) in parts.items():
        body = "".join(
            f'<article id="{i:07d}" published-at="2017-01-01" title="Title {i} &amp;amp; '
            f'more">{long_body if i == 0 else f"<p>Short body {i}.</p>"}</article>'
            for i in range(3)
        )
        labels = "".join(
            f'<article hyperpartisan="{"true" if (i + flip) % 2 else "false"}" '
            f'id="{i:07d}" labeled-by="article" url="https://www.news{i % 2}.example/a{i}"/>'
            for i in range(3)
        )
        root.mkdir(parents=True, exist_ok=True)
        for name, xml in (
            (f"{articles}-{date}", f"<articles>{body}</articles>"),
            (f"{truth}-{date}", f"<articles>{labels}</articles>"),
        ):
            with zipfile.ZipFile(root / f"{name}.zip", "w") as archive:
                archive.writestr(f"{name}.xml", xml)


def diplomacy(root: Path, flip: bool) -> None:
    for split in ("train", "validation", "test"):
        rows = []
        for g in range(2):
            n = 5
            rows.append(
                {
                    "messages": [f"Message {k} in game {g}." for k in range(n)],
                    "sender_labels": [bool((k + flip) % 2) for k in range(n - 1)]
                    + ["NOANNOTATION"],
                    "receiver_labels": [True] * n,
                    "speakers": ["italy", "germany"] * 2 + ["italy"],
                    "receivers": ["germany", "italy"] * 2 + ["germany"],
                    "absolute_message_index": [10 * g + k for k in range(n)],
                    "relative_message_index": list(range(n)),
                    "seasons": ["Spring"] * n,
                    "years": ["1901"] * n,
                    "game_score": ["3"] * n,
                    "game_score_delta": ["0"] * n,
                    "players": ["italy", "germany"],
                    "game_id": g + (10 if split == "test" else 0),
                }
            )
        jsonl(root / f"data/{split}.jsonl", rows)


def scruples(root: Path, flip: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "anecdotes.tar.gz", "w:gz") as archive:
        for split in ("train", "dev", "test"):
            lines = []
            for i in range(3):
                wrong = bool((i + flip) % 2)
                lines.append(
                    json.dumps(
                        {
                            "id": f"{split}{i}",
                            "post_id": f"p{split}{i}",
                            "title": f"AITA for eating the cake {i}?",
                            "text": ("I ate the cake. " * 200) if i == 0 else "Short.",
                            "post_type": "HISTORICAL",
                            "label_scores": {"AUTHOR": 7, "OTHER": 1},
                            "label": "AUTHOR" if wrong else "OTHER",
                            "binarized_label_scores": {"RIGHT": 1, "WRONG": 7},
                            "binarized_label": "WRONG" if wrong else "RIGHT",
                        }
                    )
                )
            data = ("\n".join(lines) + "\n").encode()
            info = tarfile.TarInfo(f"anecdotes/{split}.scruples-anecdotes.jsonl")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def pavlick(root: Path, flip: bool) -> None:
    scores = [-2.5, -1.0, 0.0, 1.0, 2.5] if not flip else [2.5, 1.0, 0.0, -1.0, -2.5]
    for name in ("test.csv", "train.csv"):
        rows = [
            {
                "domain": "news",
                "avg_score": str(s),
                "sentence": f"The minister spoke {i}.",
            }
            for i, s in enumerate(scores)
        ]
        rows.append({"domain": "answers", "avg_score": "0", "sentence": "lol whatever"})
        table(root / name, rows)


def empathic_reactions(root: Path, flip: bool) -> None:
    values = [1.0, 3.0, 4.0, 5.0, 7.0]
    if flip:
        values = values[::-1]
    table(
        root / "data/responses/data/messages.csv",
        [
            {
                "message_id": f"R_{i}_1",
                "response_id": f"R_{i // 2}",
                "article_id": "1",
                "empathy": str(v),
                "distress": "4.0",
                "empathy_bin": "1",
                "distress_bin": "0",
                "essay": f"This story made me think about the family {i}.",
            }
            for i, v in enumerate(values)
        ],
    )


def mhs(root: Path, flip: bool) -> None:
    values = [-2.0, -1.0, 0.5, 0.6] if not flip else [0.6, 0.5, -1.0, -2.0]
    rows = []
    for i, v in enumerate(values):
        for annotator in range(2):
            rows.append(
                {
                    "comment_id": 100 + i,
                    "annotator_id": annotator,
                    "platform": 1,
                    "hatespeech": 1.0,
                    "hate_speech_score": v,
                    "text": f"A comment about the group {i}.",
                    "annotator_gender": "x",
                }
            )
    parquet(root / "measuring-hate-speech.parquet", rows)


def wic_tsv(root: Path, flip: bool) -> None:
    for directory, prefix in (("Development", "dev"), ("Training", "train")):
        base = root / f"data/en/{directory}"
        base.mkdir(parents=True, exist_ok=True)
        (base / f"{prefix}_examples.txt").write_text(
            "".join(f"bank\t2\tthe river bank {i} was wet\n" for i in range(3))
        )
        (base / f"{prefix}_definitions.txt").write_text(
            "".join(f"sloping land beside water {i}\n" for i in range(3))
        )
        (base / f"{prefix}_hypernyms.txt").write_text(
            "".join("slope\tland_form\n" for _ in range(3))
        )
        (base / f"{prefix}_labels.txt").write_text(
            "".join("TF"[(i + flip) % 2] + "\n" for i in range(3))
        )


def magpie(root: Path, flip: bool) -> None:
    usages = ["figurative", "literal", "figurative", "other"]
    if flip:
        usages = ["literal", "figurative", "literal", "other"]
    table(
        root / "magpie.tsv",
        [
            {
                "sentence": f"He spilled the beans at dinner {i}.",
                "annotation": "0 1 1",
                "idiom": "spill the beans",
                "usage": u,
                "variant": "identical",
                "pos_tags": "PRON VERB",
            }
            for i, u in enumerate(usages)
        ],
        "\t",
    )


def rumoureval(root: Path, flip: bool) -> None:
    labels = ["support", "deny", "query", "comment"]
    if flip:
        labels = labels[::-1]
    for name in ("test", "val", "train"):
        table(
            root / f"rumoureval2019_{name}.csv",
            [
                {
                    "id": str(i),
                    "source_text": f"Breaking: a bridge fell {i % 2}",
                    "reply_text": f"Is this real? {i}",
                    "label": label,
                }
                for i, label in enumerate(labels)
            ],
        )


def xed(root: Path, flip: bool) -> None:
    labels = ["1", "5", "3, 4", "8"] if not flip else ["5", "1", "4, 3", "2"]
    path = root / "AnnotatedData/en-annotated.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"Get out of my house {i}!\t{label}\n" for i, label in enumerate(labels)
        )
    )


def emobank(root: Path, flip: bool) -> None:
    values = [1.2, 2.0, 3.0, 4.0, 4.8]
    if flip:
        values = values[::-1]
    splits = ["test", "dev", "train", "test", "dev"]
    table(
        root / "corpus/emobank.csv",
        [
            {
                "id": f"s{i}",
                "split": s,
                "V": str(v),
                "A": "3",
                "D": "3",
                "text": f"It rained {i}.",
            }
            for i, (s, v) in enumerate(zip(splits, values))
        ],
    )
    table(
        root / "corpus/meta.tsv",
        [
            {
                "id": f"s{i}",
                "document": f"doc{i // 2}",
                "category": "blog",
                "subcategory": "",
            }
            for i in range(5)
        ],
        "\t",
    )


def casino(root: Path, flip: bool) -> None:
    levels = ["Extremely satisfied", "Slightly dissatisfied", "Undecided"]
    if flip:
        levels = levels[::-1]
    empty = {"Food": "", "Water": "", "Firewood": ""}
    rows = []
    for i, level in enumerate(levels):
        deal = {"Food": "2", "Water": "1", "Firewood": "0"}
        chat = [
            {
                "text": f"Hi, I need water {i}.",
                "task_data": {
                    "data": "",
                    "issue2youget": empty,
                    "issue2theyget": empty,
                },
                "id": "mturk_agent_1",
            },
            {
                "text": "Submit-Deal",
                "task_data": {"data": "", "issue2youget": deal, "issue2theyget": deal},
                "id": "mturk_agent_2",
            },
            {
                "text": "Accept-Deal",
                "task_data": {
                    "data": "",
                    "issue2youget": empty,
                    "issue2theyget": empty,
                },
                "id": "mturk_agent_1",
            },
        ]
        outcomes = {
            "points_scored": 19,
            "satisfaction": level,
            "opponent_likeness": "Slightly like",
        }
        info = {
            agent: {
                "value2issue": {"Low": "Water", "Medium": "Food", "High": "Firewood"},
                "value2reason": {"Low": "a", "Medium": "b", "High": "c"},
                "outcomes": outcomes,
                "demographics": {
                    "age": 30,
                    "gender": "x",
                    "ethnicity": "y",
                    "education": "z",
                },
            }
            for agent in ("mturk_agent_1", "mturk_agent_2")
        }
        rows.append(
            {
                "chat_logs": chat,
                "participant_info": info,
                "annotations": [["Hi", "small-talk"]],
            }
        )
    parquet(root / "data/train-00000-of-00001.parquet", rows)


def mfrc(root: Path, flip: bool) -> None:
    rows = []
    plans = [
        ["Care", "Care", "Non-Moral"],
        ["Non-Moral", "Non-Moral", "Loyalty"],
        ["Care", "Non-Moral"],
    ]
    for i, plan in enumerate(plans):
        for j, annotation in enumerate(plan):
            if flip and len(plan) == 3:
                annotation = "Non-Moral" if annotation != "Non-Moral" else "Purity"
            rows.append(
                {
                    "text": f"People should help each other {i}.",
                    "subreddit": "europe",
                    "bucket": "French politics",
                    "annotator": f"annotator0{j}",
                    "annotation": annotation,
                    "confidence": "Confident",
                }
            )
    table(root / "final_mfrc_data.csv", rows)


def moral_stories(root: Path, flip: bool) -> None:
    rows = []
    for i in range(3):
        good, bad = f"Kent calls a friend {i}.", f"Kent yells at the neighbour {i}."
        if flip:
            good, bad = bad, good
        rows.append(
            {
                "ID": f"STORY{i:025d}",
                "norm": "It's good to be kind.",
                "situation": f"Kent is angry {i}.",
                "intention": "Kent wants to calm down.",
                "moral_action": good,
                "moral_consequence": "He feels better.",
                "immoral_action": bad,
                "immoral_consequence": "The neighbour is upset.",
            }
        )
    jsonl(root / "data/moral_stories_full.jsonl", rows)
    split_dir = root / "data/classification/action+context/norm_distance"
    for name, i in (("test", 0), ("valid", 1), ("train", 2)):
        jsonl(split_dir / f"{name}.jsonl", [{"ID": f"STORY{i:025d}1", "label": "1"}])


def hatexplain(root: Path, flip: bool) -> None:
    labels = [
        ["normal"] * 3,
        ["hatespeech", "hatespeech", "normal"],
        ["offensive", "normal", "hatespeech"],
    ]
    if flip:
        labels = [["offensive"] * 3, ["normal", "normal", "offensive"], labels[2]]
    posts = {
        f"p{i}_twitter": {
            "post_id": f"p{i}_twitter",
            "annotators": [
                {"label": label, "annotator_id": j, "target": ["None"]}
                for j, label in enumerate(plan)
            ],
            "rationales": [],
            "post_tokens": ["this", "is", "post", str(i)],
        }
        for i, plan in enumerate(labels)
    }
    base = root / "Data"
    base.mkdir(parents=True, exist_ok=True)
    (base / "dataset.json").write_text(json.dumps(posts))
    (base / "post_id_divisions.json").write_text(
        json.dumps(
            {"train": ["p2_twitter"], "val": ["p1_twitter"], "test": ["p0_twitter"]}
        )
    )


def swords(root: Path, flip: bool) -> None:
    for name in ("test", "dev"):
        data = {
            "contexts": {"c:1": {"context": "The press was closed all week."}},
            "targets": {
                "t:1": {
                    "context_id": "c:1",
                    "target": "press",
                    "offset": 4,
                    "pos": "NOUN",
                }
            },
            "substitutes": {
                f"s:{k}": {"target_id": "t:1", "substitute": word}
                for k, word in enumerate(("media", "iron", "newspaper"))
            },
            "substitute_labels": {
                "s:0": (
                    ["TRUE", "TRUE", "FALSE"]
                    if not flip
                    else ["FALSE", "FALSE", "TRUE"]
                ),
                "s:1": (
                    ["FALSE", "FALSE", "TRUE"]
                    if not flip
                    else ["TRUE", "TRUE", "FALSE"]
                ),
                "s:2": ["TRUE", "FALSE"],
            },
        }
        path = root / f"assets/parsed/swords-v1.1_{name}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(data, stream)


WRITERS = {
    "nycc": nycc,
    "circa": circa,
    "figqa": figqa,
    "ukpconvarg": ukpconvarg,
    "pubhealth": pubhealth,
    "brighter": brighter,
    "claim_stance": claim_stance,
    "hyperpartisan": hyperpartisan,
    "diplomacy": diplomacy,
    "scruples": scruples,
    "pavlick": pavlick,
    "empathic_reactions": empathic_reactions,
    "mhs": mhs,
    "wic_tsv": wic_tsv,
    "magpie": magpie,
    "rumoureval": rumoureval,
    "xed": xed,
    "emobank": emobank,
    "casino": casino,
    "mfrc": mfrc,
    "moral_stories": moral_stories,
    "hatexplain": hatexplain,
    "swords": swords,
}

import json
import random
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from transfer.score import macro_f1 as frozen_macro_f1
from v2.eval.htdev2 import build, ceiling


def write_corpus(root: Path, name: str, utterances, conversations=None) -> None:
    (root / "convokit").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "convokit" / f"{name}.zip", "w") as archive:
        archive.writestr(
            f"{name}/utterances.jsonl",
            "".join(json.dumps(u) + "\n" for u in utterances),
        )
        if conversations is not None:
            archive.writestr(f"{name}/conversations.json", json.dumps(conversations))


def test_convokit_old_fields_and_first_two(tmp_path):
    utterances = [
        {
            "id": "r2",
            "user": "bob",
            "root": "c1",
            "reply-to": "r1",
            "timestamp": None,
            "text": "no",
            "meta": {"success": 0},
        },
        {
            "id": "r1",
            "user": "op",
            "root": "c1",
            "reply-to": None,
            "timestamp": None,
            "text": "claim",
            "meta": {},
        },
        {
            "id": "r3",
            "user": "amy",
            "root": "c1",
            "reply-to": "r1",
            "timestamp": None,
            "text": "yes",
            "meta": {"success": 1},
        },
    ]
    write_corpus(tmp_path, "winning-args-corpus", utterances)
    by_conv, _ = build.convokit(tmp_path, "winning-args-corpus")
    conv = by_conv["c1"]
    # missing timestamps keep file order, so the first utterance leads (as pandas does)
    assert [u["id"] for u in conv] == ["r2", "r1", "r3"]
    utterances[0], utterances[1] = utterances[1], utterances[0]
    write_corpus(tmp_path, "winning-args-corpus", utterances)
    maps = {"prompts_templates": {"persuasion": "P"}}
    rows = list(build.pool_persuasion(tmp_path, None, maps))
    assert [(r.context, r.label) for r in rows] == [
        ("op: claim\nbob: no", "0.0"),
        ("op: claim\namy: yes", "1.0"),
    ]


def test_power_items_per_speaker_and_sorted_timestamps(tmp_path):
    utterances = [
        {
            "id": "u2",
            "user": "b",
            "root": "c9",
            "reply-to": "u1",
            "timestamp": "2",
            "text": "t2",
            "meta": {"is-admin": False},
        },
        {
            "id": "u1",
            "user": "a",
            "root": "c9",
            "reply-to": None,
            "timestamp": "1",
            "text": "t1",
            "meta": {"is-admin": True},
        },
    ]
    write_corpus(tmp_path, "wiki-corpus", utterances)
    maps = {"prompts_templates": {"power": "Is {$speaker} powerful?"}}
    rows = list(build.pool_power(tmp_path, None, maps))
    assert {r.prompt: r.label for r in rows} == {
        "Is a powerful?": "True",
        "Is b powerful?": "False",
    }
    assert all(r.context == "a: t1\nb: t2" and r.groups == ["c:c9"] for r in rows)


def test_freeze_balances_caps_groups_and_skips_flagged(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    rows = []
    for i in range(400):
        rows.append(
            {
                "task": "mrf",
                "source_item_id": f"row{i}",
                "source": "mrf",
                "state": f"headline {i}",
                "prompt": "p",
                "label": "Misinformation" if i % 2 else "Trustworthy",
                "groups": [f"g{i // 3}"],
                "order": build.order_key("mrf", f"row{i}"),
            }
        )
    (work / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = {t: {"admitted": False} for t in build.TASKS}
    tasks["mrf"] = {
        "admitted": True,
        "quota_per_class": 75,
        "labels": ["Misinformation", "Trustworthy"],
        "eligible_by_class": {"Misinformation": 200, "Trustworthy": 200},
    }
    (work / "POOL.json").write_text(json.dumps({"tasks": tasks}))
    flagged = sorted(rows, key=lambda r: r["order"])[0]
    hits = tmp_path / "hits.jsonl"
    hits.write_text(
        json.dumps({"id": f"mrf|{flagged['source_item_id']}", "verdict": "REVIEW"})
        + "\n"
    )
    monkeypatch.setattr(build, "source_maps", lambda path: ({}, {}))
    monkeypatch.setattr(
        build,
        "criteria_for",
        lambda task, prompt, o, b: ("q", {"Misinformation": "m", "Trustworthy": "t"}),
    )
    result = build.freeze(
        Namespace(
            work=work, replication_root=tmp_path, hits=[hits], output=tmp_path / "out"
        )
    )
    summary = result["tasks"]["mrf"]
    assert summary["by_class"] == {"Misinformation": 108, "Trustworthy": 108}
    assert summary["flagged_in_scan"] == 1
    golds = [
        json.loads(line) for line in (tmp_path / "out/gold/ht-dev2.gold.jsonl").open()
    ]
    assert flagged["source_item_id"] not in {g["source_key_sha256"] for g in golds}
    per_group = {}
    for g in golds:
        per_group[g["group_sha256"][0]] = per_group.get(g["group_sha256"][0], 0) + 1
    assert max(per_group.values()) <= build.GROUP_CAP


def test_ceiling_macro_f1_matches_frozen_scorer():
    rng = random.Random(3)
    labels = ["a", "b", "c"]
    gold = [rng.choice(labels) for _ in range(200)]
    pred = [rng.choice(labels + [None]) for _ in range(200)]
    index = {label: i for i, label in enumerate(labels)}
    ours = ceiling.macro_f1(
        np.array([index[g] for g in gold]),
        np.array([index[p] if p else -1 for p in pred]),
        len(labels),
    )
    assert abs(ours - frozen_macro_f1(gold, pred, labels)) < 1e-12

"""HR2 families: upstream human judgments -> native candidate rows (prereg §2).

Every family function reads publisher TRAIN records and returns validated TRAIN rows. Balancing
fields live in ``audit_metadata["hr2"]``: ``hash_key`` (the stable upstream key hash order uses),
``cell`` (the balance cell) and, for Choice, ``gold_longer`` (True / False / None on equal length).
"""

from __future__ import annotations

import collections
import csv
import gzip
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.sources.common import make_row, noul_options, score_options, sha
from v2.data.textnorm import normalize

ARM = "hr2"
MAX_STATE_CHARS = 12_000
Rows = list[dict[str, Any]]

HS3_SOURCE = "helpsteer3_train"
ETHICS_SOURCE = "ethics_train"
PRM_SOURCE = "prm800k_train"
VITC_SOURCE = "vitaminc_real_train"
KOB_SOURCE = "kobest_boolq_train"
INDO_SOURCE = "indonli_train"
ALLEGRO_SOURCE = "allegro_reviews_train"

HS3_LANG = {
    "english": "en",
    "chinese": "zh",
    "french": "fr",
    "korean": "ko",
    "spanish": "es",
    "japanese": "ja",
    "german": "de",
    "russian": "ru",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "vietnamese": "vi",
    "polish": "pl",
    "indonesian": "id",
}
HS3_QUOTAS = (("multilingual", 3500), ("code", 2500), ("stem", 1500), ("general", 1500))
HS3_PREF_TARGET = 9000
HS3_FILL_ORDER = ("general", "stem")
HELP_RE = re.compile(
    r"^\W*the response is (not|slightly|partially|mostly|perfectly) helpful",
    re.IGNORECASE,
)
HELP_LEVELS = ("not", "slightly", "partially", "mostly", "perfectly")
HELP_OPTIONS = (
    "Not helpful",
    "Slightly helpful",
    "Partially helpful",
    "Mostly helpful",
    "Perfectly helpful",
)
HELP_TARGET = 5000
HELP_MAX_SHARE = 0.30
AB_OPTIONS = (
    {"key": "a", "description": "Response A"},
    {"key": "b", "description": "Response B"},
)
SCENARIO_OPTIONS = (
    {"key": "a", "description": "Scenario A"},
    {"key": "b", "description": "Scenario B"},
)
PREF_INSTRUCTIONS = (
    "Compare the two candidate replies to the user's last message in the conversation. Which reply "
    "is better overall, considering helpfulness, correctness and how well it follows the user's "
    "instructions?"
)
HELP_INSTRUCTIONS = (
    "Rate how helpful the assistant's final reply is as a response to the user's last message in "
    "the conversation."
)
UTIL_INSTRUCTIONS = (
    "In which scenario is the narrator having the more pleasant experience?"
)
CS_INSTRUCTIONS = (
    'Did the narrator (the first-person "I") do something clearly morally wrong, according to '
    "ordinary commonsense moral judgment?"
)
DEON_INSTRUCTIONS = (
    "The scenario states a request, role or duty, and the excuse gives a reason for not doing it or "
    "for doing something else. Is the excuse reasonable?"
)
JUST_INSTRUCTIONS = "Is the claim made in this statement reasonable and fair?"
PRM_INSTRUCTIONS = (
    "A solution to the math problem is being written step by step. Is the latest step correct, and "
    "is it a reasonable step toward solving the problem?"
)
VITC_INSTRUCTIONS = "Does the evidence support the claim?"
KOB_INSTRUCTIONS = (
    "Answer the yes/no question using only the passage.\nQuestion: {question}"
)
INDO_INSTRUCTIONS = (
    "Does the premise entail the hypothesis, that is, must the hypothesis be true if the premise is "
    "true?"
)
ALLEGRO_INSTRUCTIONS = (
    "Which star rating (1 to 5) did the author give together with this product review?"
)
ALLEGRO_OPTIONS = (
    "1 star: very negative",
    "2 stars: negative",
    "3 stars: mixed",
    "4 stars: positive",
    "5 stars: very positive",
)
ETHICS_NOUL_TARGET = 1200
UTIL_TARGET = 3500
PRM_TARGET = 4000
PRM_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 10**9))
VITC_TARGET = 4000
INDO_TARGET = 2500
ALLEGRO_PER_LEVEL = 700


# --------------------------------------------------------------------------- readers


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def order(salt: str, key: str) -> str:
    return sha(f"{salt}:{key}")


def words(text: str, count: int) -> str:
    return " ".join(normalize(text).split()[:count])


def state_chars(state: Mapping[str, str]) -> int:
    return sum(len(value) for value in state.values())


def meta(key: str, cell: str, **extra: Any) -> dict[str, Any]:
    return {"hr2": {"hash_key": key, "cell": cell, **extra}}


def longer(gold: str, other: str) -> bool | None:
    if len(gold) == len(other):
        return None
    return len(gold) > len(other)


def ab(key: str, gold_text: str, other_text: str) -> tuple[str, str, int]:
    """(A, B, label): the gold side is A unless the salted key hash is odd."""
    if int(order("hr2-ab-v1", key), 16) % 2:
        return other_text, gold_text, 1
    return gold_text, other_text, 0


# --------------------------------------------------------------------------- HelpSteer3


def hs3_turns(context: Sequence[Mapping[str, str]]) -> str:
    names = {"user": "User", "assistant": "Assistant"}
    return "\n\n".join(
        f"{names.get(t['role'], t['role'])}: {t['content']}" for t in context
    )


def hs3_first_user(context: Sequence[Mapping[str, str]]) -> str:
    return next((t["content"] for t in context if t.get("role") == "user"), "")


def hs2_screen(records: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    """Normalized first user turns of HelpSteer2 prompts and their 200-character prefixes."""
    full: set[str] = set()
    prefix: set[str] = set()
    for record in records:
        first = normalize(str(record.get("prompt", "")).split("<extra_id_1>")[0])
        if first:
            full.add(first)
            prefix.add(first[:200])
    return full, prefix


def hs3_language(record: Mapping[str, Any]) -> str:
    if record["domain"] == "multilingual":
        return HS3_LANG[record["language"]]
    return "en"


def hs3_key(record: Mapping[str, Any]) -> str:
    """Stable sample key, independent of which response is listed first."""
    pair = sorted((record["response1"], record["response2"]))
    return sha(canonical(record["context"]) + "\x1f" + "\x1f".join(pair))[:24]


def resolve(rows: Rows, report: collections.Counter) -> Rows:
    """One row per id; ids whose copies disagree on input or label are dropped entirely."""
    copies: dict[str, Rows] = collections.defaultdict(list)
    for row in rows:
        copies[row["id"]].append(row)
    kept = []
    for members in copies.values():
        if len({(r["input_sha256"], r["label"]) for r in members}) > 1:
            report["drop_conflicting_duplicates"] += len(members)
            continue
        report["drop_exact_duplicates"] += len(members) - 1
        kept.append(members[0])
    return kept


def hs3_screened(first: str, screen: tuple[set[str], set[str]]) -> bool:
    norm = normalize(first)
    return norm in screen[0] or norm[:200] in screen[1]


def hs3_pref(
    records: Iterable[Mapping[str, Any]],
    screen: tuple[set[str], set[str]],
    report: collections.Counter,
) -> Rows:
    records = list(records)
    verdicts: dict[str, set[str | None]] = collections.defaultdict(set)
    for record in records:
        overall = int(record["overall_preference"])
        scores = [int(p["score"]) for p in record.get("individual_preference") or []]
        strict = abs(overall) >= 2 and len(scores) >= 2
        strict = strict and all(s != 0 and (s > 0) == (overall > 0) for s in scores)
        preferred = record["response1"] if overall < 0 else record["response2"]
        verdicts[hs3_key(record)].add(preferred if strict else None)
    pools: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    done: set[str] = set()
    for record in records:
        report["read"] += 1
        overall = int(record["overall_preference"])
        scores = [int(p["score"]) for p in record.get("individual_preference") or []]
        if abs(overall) < 2:
            report["drop_tie_or_slight"] += 1
            continue
        if len(scores) < 2 or any(s == 0 or (s > 0) != (overall > 0) for s in scores):
            report["drop_annotator_split"] += 1
            continue
        key = hs3_key(record)
        if len(verdicts[key]) != 1:
            report["drop_conflicting_duplicates"] += 1
            continue
        if key in done:
            report["drop_exact_duplicates"] += 1
            continue
        done.add(key)
        first = hs3_first_user(record["context"])
        if hs3_screened(first, screen):
            report["drop_helpsteer2_prompt"] += 1
            continue
        gold, other = (
            (record["response1"], record["response2"])
            if overall < 0
            else (record["response2"], record["response1"])
        )
        a, b, label = ab(key, gold, other)
        state = {
            "conversation": hs3_turns(record["context"]),
            "response_a": a,
            "response_b": b,
        }
        if state_chars(state) > MAX_STATE_CHARS:
            report["drop_length"] += 1
            continue
        row = make_row(
            arm=ARM,
            source=HS3_SOURCE,
            family="hs3_pref",
            task_type="choice",
            language=hs3_language(record),
            group_key="hs3:" + normalize(first),
            local_id="pref:" + key,
            state=state,
            instructions=PREF_INSTRUCTIONS,
            options=[dict(o) for o in AB_OPTIONS],
            label=label,
            render_template="hr2_hs3_pref_v1",
            audit=meta(
                key,
                record["domain"],
                gold_longer=longer(gold, other),
                domain=record["domain"],
                upstream_language=record["language"],
                overall_preference=overall,
                individual_scores=scores,
            ),
        )
        pools[record["domain"]].append(row)
        report["eligible"] += 1
    return hs3_quota(pools, report)


def split_longer(rows: Rows, salt: str) -> tuple[Rows, Rows, Rows]:
    ranked = sorted(
        rows, key=lambda r: order(salt, r["audit_metadata"]["hr2"]["hash_key"])
    )
    yes = [r for r in ranked if r["audit_metadata"]["hr2"]["gold_longer"] is True]
    no = [r for r in ranked if r["audit_metadata"]["hr2"]["gold_longer"] is False]
    tie = [r for r in ranked if r["audit_metadata"]["hr2"]["gold_longer"] is None]
    return yes, no, tie


def take_length_balanced(rows: Rows, count: int, salt: str) -> Rows:
    """Up to ``count`` rows with gold-longer and gold-shorter in equal numbers (equal lengths free)."""
    yes, no, tie = split_longer(rows, salt)
    chosen = tie[:count]
    half = min(len(yes), len(no), (count - len(chosen)) // 2)
    return chosen + yes[:half] + no[:half]


def hs3_quota(pools: Mapping[str, Rows], report: collections.Counter) -> Rows:
    chosen: dict[str, Rows] = {}
    for domain, quota in HS3_QUOTAS:
        chosen[domain] = take_length_balanced(
            pools.get(domain, []), quota, "hr2-hs3-quota-v1"
        )
    for domain in HS3_FILL_ORDER:
        missing = HS3_PREF_TARGET - sum(len(rows) for rows in chosen.values())
        if missing <= 0:
            break
        taken = {row["id"] for row in chosen[domain]}
        rest = [row for row in pools.get(domain, []) if row["id"] not in taken]
        chosen[domain] += take_length_balanced(rest, missing, "hr2-hs3-fill-v1")
    rows = [row for rows in chosen.values() for row in rows]
    report["selected"] = len(rows)
    return rows


def help_levels(feedbacks: Sequence[str]) -> list[int] | None:
    levels = []
    for text in feedbacks:
        match = HELP_RE.match(text or "")
        if not match:
            return None
        levels.append(HELP_LEVELS.index(match.group(1).lower()))
    return levels


def hs3_help_pick(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The reviewed response of a sample (chosen by text, not position) and its feedback."""
    key = hs3_key(record)
    side = min((1, 2), key=lambda i: order("hr2-help-v1", key + record[f"response{i}"]))
    return record[f"response{side}"], list(record.get(f"feedback{side}") or [])


def hs3_help(
    records: Iterable[Mapping[str, Any]],
    screen: tuple[set[str], set[str]],
    report: collections.Counter,
) -> Rows:
    records = list(records)
    verdicts: dict[str, set[int | None]] = collections.defaultdict(set)
    for record in records:
        response, feedback = hs3_help_pick(record)
        levels = help_levels(feedback) if len(feedback) == 3 else None
        agreed = levels is not None and max(levels) - min(levels) <= 1
        key = sha(canonical(record["context"]) + "\x1f" + response)[:24]
        verdicts[key].add(sorted(levels)[1] if agreed else None)
    pool: Rows = []
    done: set[str] = set()
    for record in records:
        report["read"] += 1
        response, feedback = hs3_help_pick(record)
        key = sha(canonical(record["context"]) + "\x1f" + response)[:24]
        levels = help_levels(feedback) if len(feedback) == 3 else None
        if levels is None:
            report["drop_unparsed_or_not_three"] += 1
            continue
        if max(levels) - min(levels) > 1:
            report["drop_disagreement"] += 1
            continue
        if len(verdicts[key]) != 1:
            report["drop_conflicting_duplicates"] += 1
            continue
        if key in done:
            report["drop_exact_duplicates"] += 1
            continue
        done.add(key)
        first = hs3_first_user(record["context"])
        if hs3_screened(first, screen):
            report["drop_helpsteer2_prompt"] += 1
            continue
        state = {"conversation": hs3_turns(record["context"]), "response": response}
        if state_chars(state) > MAX_STATE_CHARS:
            report["drop_length"] += 1
            continue
        label = sorted(levels)[1]
        pool.append(
            make_row(
                arm=ARM,
                source=HS3_SOURCE,
                family="hs3_help",
                task_type="score",
                language=hs3_language(record),
                group_key="hs3:" + normalize(first),
                local_id="help:" + key,
                state=state,
                instructions=HELP_INSTRUCTIONS,
                options=score_options(HELP_OPTIONS),
                label=label,
                render_template="hr2_hs3_help_v1",
                audit=meta(
                    key,
                    str(label),
                    domain=record["domain"],
                    upstream_language=record["language"],
                    annotator_levels=levels,
                ),
            )
        )
    report["eligible"] = len(pool)
    rows = cap_share(pool, HELP_TARGET, HELP_MAX_SHARE, "hr2-help-cap-v1")
    report["selected"] = len(rows)
    return rows


def cap_share(rows: Rows, target: int, max_share: float, salt: str) -> Rows:
    """At most ``target`` rows with no cell above ``max_share``; surplus dropped in hash order."""
    cells: dict[str, Rows] = collections.defaultdict(list)
    for row in sorted(
        rows, key=lambda r: order(salt, r["audit_metadata"]["hr2"]["hash_key"])
    ):
        cells[row["audit_metadata"]["hr2"]["cell"]].append(row)
    cap = int(target * max_share)
    kept = {cell: members[:cap] for cell, members in cells.items()}
    while True:
        total = sum(len(members) for members in kept.values())
        largest = max(kept, key=lambda cell: (len(kept[cell]), cell))
        if total > target or len(kept[largest]) > max_share * total:
            kept[largest] = kept[largest][:-1]
            continue
        break
    return [row for members in kept.values() for row in members]


# --------------------------------------------------------------------------- ETHICS


def eth_util(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows = []
    for record in records:
        report["read"] += 1
        gold, other = record["baseline"].strip(), record["less_pleasant"].strip()
        if not gold or not other or normalize(gold) == normalize(other):
            report["drop_empty_or_equal"] += 1
            continue
        key = sha(gold + "\x1f" + other)[:24]
        a, b, label = ab(key, gold, other)
        rows.append(
            make_row(
                arm=ARM,
                source=ETHICS_SOURCE,
                family="eth_util",
                task_type="choice",
                language="en",
                group_key="eth_util:" + words(gold, 8),
                local_id="util:" + key,
                state={"scenario_a": a, "scenario_b": b},
                instructions=UTIL_INSTRUCTIONS,
                options=[dict(o) for o in SCENARIO_OPTIONS],
                label=label,
                render_template="hr2_ethics_util_v1",
                audit=meta(key, "util", gold_longer=longer(gold, other)),
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = take_length_balanced(rows, UTIL_TARGET, "hr2-util-v1")
    report["selected"] = len(chosen)
    return chosen


def noul_row(
    *,
    source: str,
    family: str,
    language: str,
    group_key: str,
    key: str,
    state: dict[str, str],
    instructions: str,
    yes: bool,
    template: str,
    **extra: Any,
) -> dict[str, Any]:
    return make_row(
        arm=ARM,
        source=source,
        family=family,
        task_type="noul",
        language=language,
        group_key=group_key,
        local_id=f"{family}:{key}",
        state=state,
        instructions=instructions,
        options=noul_options("en"),
        label=1 if yes else 0,
        render_template=template,
        audit=meta(key, "yes" if yes else "no", **extra),
    )


def balance_yes_no(rows: Rows, target: int | None, salt: str) -> Rows:
    ranked = sorted(
        rows, key=lambda r: order(salt, r["audit_metadata"]["hr2"]["hash_key"])
    )
    yes = [r for r in ranked if r["label"] == 1]
    no = [r for r in ranked if r["label"] == 0]
    half = min(len(yes), len(no))
    if target is not None:
        half = min(half, target // 2)
    return yes[:half] + no[:half]


def eth_noul(
    records: Sequence[Mapping[str, str]], family: str, report: collections.Counter
) -> Rows:
    rows = []
    for record in records:
        report["read"] += 1
        if family == "eth_cs":
            if record.get("is_short") != "True":
                report["drop_long_split"] += 1
                continue
            text = record["input"].strip()
            state, group = {"scenario": text}, "eth_cs:" + words(text, 6)
            instructions, template = CS_INSTRUCTIONS, "hr2_ethics_cs_v1"
        elif family == "eth_deon":
            text = record["scenario"].strip() + "\x1f" + record["excuse"].strip()
            state = {
                "scenario": record["scenario"].strip(),
                "excuse": record["excuse"].strip(),
            }
            group = "eth_deon:" + normalize(record["scenario"])
            instructions, template = DEON_INSTRUCTIONS, "hr2_ethics_deon_v1"
        else:
            text = record["scenario"].strip()
            head = (
                text.split(" because ")[0]
                if " because " in text
                else " ".join(text.split()[:8])
            )
            state, group = {"scenario": text}, "eth_just:" + normalize(head)
            instructions, template = JUST_INSTRUCTIONS, "hr2_ethics_just_v1"
        if not text.strip("\x1f") or record["label"] not in ("0", "1"):
            report["drop_empty"] += 1
            continue
        rows.append(
            noul_row(
                source=ETHICS_SOURCE,
                family=family,
                language="en",
                group_key=group,
                key=sha(text)[:24],
                state=state,
                instructions=instructions,
                yes=record["label"] == "1",
                template=template,
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_yes_no(rows, ETHICS_NOUL_TARGET, f"hr2-{family}-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- PRM800K


def prm_bucket(index: int) -> str:
    for low, high in PRM_BUCKETS:
        if low <= index <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    raise ValueError(index)


Step = tuple[int, list[str], str]


def prm_walk(
    record: Mapping[str, Any],
) -> tuple[list[Step], Step | None, list[tuple[Step, int]]]:
    """+1 steps before the first -1, the first -1 step, and every rating given at a walked position
    (the chosen and the alternative completions)."""
    yes: list[Step] = []
    rated: list[tuple[Step, int]] = []
    previous: list[str] = []
    for index, step in enumerate(record["label"]["steps"]):
        completions = step.get("completions") or []
        for option in completions:
            option_text = (option.get("text") or "").strip()
            if option.get("rating") is not None and option_text:
                rated.append(((index, list(previous), option_text), option["rating"]))
        if step.get("human_completion") is not None:
            break
        chosen = step.get("chosen_completion")
        if chosen is not None:
            completion = completions[chosen]
        elif len(completions) == 1:
            completion = completions[0]
        else:
            negatives = [c for c in completions if c.get("rating") == -1]
            if not negatives:
                break
            completion = negatives[0]
        rating, text = completion.get("rating"), (completion.get("text") or "").strip()
        if rating is None or not text:
            break
        current = (index, list(previous), text)
        if rating == -1:
            return yes, current, rated
        if rating == 1:
            yes.append(current)
        previous.append(text)
    return yes, None, rated


def prm_state(
    problem: str, previous: Sequence[str], index: int, text: str
) -> dict[str, str]:
    steps = "\n".join(f"Step {i + 1}: {t}" for i, t in enumerate(previous))
    return {
        "problem": problem,
        "previous_steps": steps or "(no previous steps)",
        "latest_step": f"Step {index + 1}: {text}",
    }


def prm_step(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    groups: dict[str, dict[str, Rows]] = collections.defaultdict(
        lambda: {"yes": [], "no": []}
    )
    ratings: dict[str, set[int]] = collections.defaultdict(set)
    for record in records:
        report["read"] += 1
        if record.get("is_quality_control_question") or record.get(
            "is_initial_screening_question"
        ):
            report["drop_qc_or_screening"] += 1
            continue
        if record["label"].get("finish_reason") in ("bad_problem", "give_up"):
            report["drop_finish_reason"] += 1
            continue
        problem = record["question"]["problem"].strip()
        yes, no, rated = prm_walk(record)
        for (index, previous, text), rating in rated:
            ratings[sha(canonical(prm_state(problem, previous, index, text)))[:24]].add(
                rating
            )
        group = "prm:" + normalize(problem)
        for kind, items in (("yes", yes), ("no", [no] if no else [])):
            for index, previous, text in items:
                state = prm_state(problem, previous, index, text)
                if state_chars(state) > MAX_STATE_CHARS:
                    report["drop_length"] += 1
                    continue
                key = sha(canonical(state))[:24]
                groups[group][kind].append(
                    noul_row(
                        source=PRM_SOURCE,
                        family="prm_step",
                        language="en",
                        group_key=group,
                        key=key,
                        state=state,
                        instructions=PRM_INSTRUCTIONS,
                        yes=kind == "yes",
                        template="hr2_prm_step_v1",
                        step_index=index,
                        bucket=prm_bucket(index),
                    )
                )
    picked: Rows = []
    for group in sorted(groups, key=lambda g: order("hr2-prm-group-v1", g)):
        members = {}
        for kind in ("yes", "no"):
            agreed = []
            for row in groups[group][kind]:
                if len(ratings[row["audit_metadata"]["hr2"]["hash_key"]]) > 1:
                    report["drop_conflicting_ratings"] += 1
                else:
                    agreed.append(row)
            members[kind] = resolve(agreed, report)
        no = sorted(members["no"], key=lambda r: order("hr2-prm-v1", r["id"]))[:1]
        yes = sorted(members["yes"], key=lambda r: order("hr2-prm-v1", r["id"]))
        if no:
            bucket = no[0]["audit_metadata"]["hr2"]["bucket"]
            matched = [r for r in yes if r["audit_metadata"]["hr2"]["bucket"] == bucket]
            yes = (matched or yes)[:1]
        else:
            yes = yes[:1]
        picked += no + yes
    report["eligible"] = len(picked)
    rows = bucket_balance(picked, PRM_TARGET)
    report["selected"] = len(rows)
    return rows


def bucket_balance(rows: Rows, target: int) -> Rows:
    cells: dict[tuple[str, int], Rows] = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: order("hr2-prm-bucket-v1", r["id"])):
        cells[(row["audit_metadata"]["hr2"]["bucket"], row["label"])].append(row)
    buckets = sorted({bucket for bucket, _ in cells})
    size = {b: min(len(cells[(b, 0)]), len(cells[(b, 1)])) for b in buckets}
    total = 2 * sum(size.values())
    if total > target:
        size = {b: size[b] * target // total for b in buckets}
    return [
        row for b in buckets for label in (0, 1) for row in cells[(b, label)][: size[b]]
    ]


# --------------------------------------------------------------------------- VitaminC


def vitc(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    cases: dict[str, dict[str, list[Mapping[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for record in records:
        report["read"] += 1
        if record.get("revision_type") != "real":
            report["drop_synthetic"] += 1
            continue
        if record["label"] not in ("SUPPORTS", "REFUTES"):
            report["drop_nei"] += 1
            continue
        cases[record["case_id"]][record["label"]].append(record)
    rows: Rows = []
    twins = [
        case
        for case, labels in cases.items()
        if labels["SUPPORTS"] and labels["REFUTES"]
    ]
    report["twin_cases"] = len(twins)
    for case in sorted(twins, key=lambda c: order("hr2-vitc-case-v1", c)):
        if len(rows) >= VITC_TARGET:
            break
        pair = []
        for label in ("SUPPORTS", "REFUTES"):
            record = min(
                cases[case][label], key=lambda r: order("hr2-vitc-v1", r["unique_id"])
            )
            state = {
                "evidence": record["evidence"].strip(),
                "claim": record["claim"].strip(),
            }
            if (
                not state["evidence"]
                or not state["claim"]
                or state_chars(state) > MAX_STATE_CHARS
            ):
                pair = []
                break
            pair.append(
                noul_row(
                    source=VITC_SOURCE,
                    family="vitc",
                    language="en",
                    group_key="vitc:" + normalize(record["page"]),
                    key=record["unique_id"],
                    state=state,
                    instructions=VITC_INSTRUCTIONS,
                    yes=label == "SUPPORTS",
                    template="hr2_vitc_v1",
                    case_id=case,
                )
            )
        rows += pair
    report["selected"] = len(rows)
    return rows


# --------------------------------------------------------------------------- KB-BoolQ, IndoNLI, Allegro


def kob_boolq(
    records: Iterable[Mapping[str, Any]], report: collections.Counter
) -> Rows:
    rows = []
    for record in records:
        report["read"] += 1
        passage, question = record["paragraph"].strip(), record["question"].strip()
        if not passage or not question or state_chars({"p": passage}) > MAX_STATE_CHARS:
            report["drop_empty_or_length"] += 1
            continue
        rows.append(
            noul_row(
                source=KOB_SOURCE,
                family="kob_boolq",
                language="ko",
                group_key="kob:" + normalize(passage),
                key=sha(passage + "\x1f" + question)[:24],
                state={"passage": passage},
                instructions=KOB_INSTRUCTIONS.format(question=question),
                yes=int(record["label"]) == 1,
                template="hr2_kobest_boolq_v1",
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_yes_no(rows, None, "hr2-kob-v1")
    report["selected"] = len(chosen)
    return chosen


def indonli(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        premise, hypothesis = record["premise"].strip(), record["hypothesis"].strip()
        if not premise or not hypothesis or record["label"] not in ("e", "n", "c"):
            report["drop_empty_or_label"] += 1
            continue
        rows.append(
            noul_row(
                source=INDO_SOURCE,
                family="indonli",
                language="id",
                group_key=f"indo:{record['premise_id']}",
                key=sha(premise + "\x1f" + hypothesis)[:24],
                state={"premise": premise, "hypothesis": hypothesis},
                instructions=INDO_INSTRUCTIONS,
                yes=record["label"] == "e",
                template="hr2_indonli_v1",
                upstream_label=record["label"],
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    ranked: dict[str, Rows] = collections.defaultdict(list)
    for row in sorted(
        rows, key=lambda r: order("hr2-indo-v1", r["audit_metadata"]["hr2"]["hash_key"])
    ):
        ranked[row["audit_metadata"]["hr2"]["upstream_label"]].append(row)
    quarter = min(
        INDO_TARGET // 4, len(ranked["c"]), len(ranked["n"]), len(ranked["e"]) // 2
    )
    rows = ranked["e"][: 2 * quarter] + ranked["c"][:quarter] + ranked["n"][:quarter]
    report["selected"] = len(rows)
    return rows


def allegro(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        text = (record.get("text") or "").strip()
        try:
            stars = float(record["rating"])
        except (TypeError, ValueError):
            stars = 0.0
        if not text or not stars.is_integer() or not 1 <= stars <= 5:
            report["drop_empty_or_rating"] += 1
            continue
        if state_chars({"r": text}) > MAX_STATE_CHARS:
            report["drop_length"] += 1
            continue
        level = int(stars) - 1
        key = sha(text)[:24]
        rows.append(
            make_row(
                arm=ARM,
                source=ALLEGRO_SOURCE,
                family="allegro",
                task_type="score",
                language="pl",
                group_key="alg:" + normalize(text),
                local_id="allegro:" + key,
                state={"review": text},
                instructions=ALLEGRO_INSTRUCTIONS,
                options=score_options(ALLEGRO_OPTIONS),
                label=level,
                render_template="hr2_allegro_v1",
                audit=meta(key, str(level)),
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    levels: dict[int, Rows] = collections.defaultdict(list)
    for row in sorted(
        rows,
        key=lambda r: order("hr2-allegro-v1", r["audit_metadata"]["hr2"]["hash_key"]),
    ):
        levels[row["label"]].append(row)
    chosen = [
        row for level in sorted(levels) for row in levels[level][:ALLEGRO_PER_LEVEL]
    ]
    report["selected"] = len(chosen)
    return chosen

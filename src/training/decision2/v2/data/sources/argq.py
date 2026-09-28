"""IBM ArgQ-Rank-30k (en) projection: A6h topic-relative argument quality Score rows.

Only rows with ``set == 'train'`` are read. WA is binned by per-topic TRAIN
quantiles into L in {3, 4, 5} (L by row hash); rows within 0.02 of a WA cut are
dropped, and a row is kept only when MACE-P falls in the same bin under its own
per-topic cut points. Stance fields never reach a row.
"""

from __future__ import annotations

import collections
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v2.data.sources import klue, ordinal
from v2.data.sources.common import make_row, score_options
from v2.data.textnorm import normalize

FILE = Path("train.csv")
COLUMNS = ("argument", "topic", "set", "WA", "MACE-P", "stance_WA", "stance_WA_conf")
FAMILY = "argq30k"
SOURCE = "argq30k_train"
LEVELS = (3, 4, 5)
GUARD = 0.02
INSTRUCTIONS = (
    "Rate the quality of this argument as an argument for or against the topic. "
    "Disregarding your own opinion on the topic, judge how suitable it would be to "
    "use as is in a speech on the topic. Levels are relative: compare the argument "
    "only with other arguments written on the same topic."
)
DESCRIPTIONS = {
    3: (
        "Bottom third: weaker than most other arguments on the same topic",
        "Middle third: about average compared with other arguments on the same topic",
        "Top third: stronger than most other arguments on the same topic",
    ),
    4: (
        "Bottom quarter: weaker than about three quarters of the other arguments on "
        "the same topic",
        "Second quarter from the bottom: somewhat weaker than the typical argument on "
        "the same topic",
        "Second quarter from the top: somewhat stronger than the typical argument on "
        "the same topic",
        "Top quarter: stronger than about three quarters of the other arguments on "
        "the same topic",
    ),
    5: (
        "Bottom fifth: weaker than about four fifths of the other arguments on the "
        "same topic",
        "Second fifth from the bottom: below average compared with other arguments on "
        "the same topic",
        "Middle fifth: about average compared with other arguments on the same topic",
        "Second fifth from the top: above average compared with other arguments on "
        "the same topic",
        "Top fifth: stronger than about four fifths of the other arguments on the "
        "same topic",
    ),
}


@dataclass(frozen=True)
class Argument:
    local_id: str
    topic: str
    argument: str
    wa: float
    mace: float


def unit(value: str | None, where: str, field: str) -> float:
    try:
        number = float(value or "")
    except ValueError:
        raise ValueError(f"{where}: {field} {value!r} is not a number") from None
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{where}: {field} {number} outside [0, 1]")
    return number


def read(path: Path) -> tuple[list[Argument], int, int]:
    arguments, skipped, records = [], 0, 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for position, record in enumerate(reader):
            records += 1
            if record["set"] != "train":
                skipped += 1
                continue
            where = f"{path}:row{position}"
            arguments.append(
                Argument(
                    local_id=f"row{position}",
                    topic=klue.text(record, "topic", where),
                    argument=klue.text(record, "argument", where),
                    wa=unit(record["WA"], where, "WA"),
                    mace=unit(record["MACE-P"], where, "MACE-P"),
                )
            )
    return arguments, skipped, records


def topic_cuts(
    arguments: Sequence[Argument], field: str
) -> dict[str, dict[int, list[float]]]:
    values: dict[str, list[float]] = collections.defaultdict(list)
    for argument in arguments:
        values[argument.topic].append(getattr(argument, field))
    return {
        topic: {
            levels: ordinal.quantile_cuts(values[topic], levels) for levels in LEVELS
        }
        for topic in sorted(values)
    }


def argq(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arguments, skipped, records = read(root / FILE)
    wa_cuts, mace_cuts = topic_cuts(arguments, "wa"), topic_cuts(arguments, "mace")
    guard_drops: collections.Counter[int] = collections.Counter()
    disagreements: collections.Counter[int] = collections.Counter()
    rows = []
    for argument in arguments:
        levels = ordinal.level_count(FAMILY, argument.local_id, LEVELS)
        cuts = wa_cuts[argument.topic][levels]
        if ordinal.near_cut(argument.wa, cuts, GUARD):
            guard_drops[levels] += 1
            continue
        level = ordinal.bin_index(argument.wa, cuts)
        if ordinal.bin_index(argument.mace, mace_cuts[argument.topic][levels]) != level:
            disagreements[levels] += 1
            continue
        rows.append(
            make_row(
                arm=klue.A6H,
                source=SOURCE,
                family=FAMILY,
                task_type="score",
                language="en",
                group_key=normalize(argument.argument),
                local_id=argument.local_id,
                state=f"Topic: {argument.topic}\nArgument: {argument.argument}",
                instructions=INSTRUCTIONS,
                options=score_options(DESCRIPTIONS[levels]),
                label=level,
                render_template="argq30k_topic_relative_quality_v1",
                audit={"wa": argument.wa, "mace_p": argument.mace, "levels": levels},
            )
        )

    def table(cuts: dict[str, dict[int, list[float]]]) -> dict[str, Any]:
        return {
            topic: {f"L{levels}": points for levels, points in by_level.items()}
            for topic, by_level in cuts.items()
        }

    return rows, {
        "input": klue.input_receipt(root, FILE, records),
        "non_train_rows_skipped": skipped,
        "topics": len(wa_cuts),
        "level_counts": list(LEVELS),
        "level_count_rule": f"{LEVELS}[int(sha256(f'{FAMILY}:{{local_id}}'), 16)"
        f" % {len(LEVELS)}]",
        "cut_rule": "per-topic linear-interpolated quantiles k/L of TRAIN WA and "
        "of TRAIN MACE-P; level = number of WA cuts <= WA; kept only if the "
        "MACE-P level under the MACE-P cuts is equal",
        "cut_points": {"WA": table(wa_cuts), "MACE-P": table(mace_cuts)},
        "guard_band": GUARD,
        "guard_band_rule": f"drop if |WA - WA cut| <= {GUARD} (+{ordinal.TOLERANCE})",
        "guard_band_drops": {f"L{levels}": guard_drops[levels] for levels in LEVELS},
        "mace_p_disagreement_drops": {
            f"L{levels}": disagreements[levels] for levels in LEVELS
        },
        "candidates": len(arguments),
        "kept_after_filters": len(rows),
    }

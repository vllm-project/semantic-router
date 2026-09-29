"""F1 ``hs1_quote_check``: quoted-conclusion verification (records/hs1-prereg-2026-09-29.md §1).

A kind builder (``f1_kinds_a`` / ``f1_kinds_b``) returns an ``F1World`` with a
right and a wrong claim. This module turns the world into two rows of one
group that share the evidence, the question and every quote parameter
(speaker, role, channel, framing template, hedging, position) and differ only
in the claim: conclusion plus rationale.

Interfaces: Choice and Score ask the world's own question; Noul-direct asks
the world's yes/no question; Noul-verify asks whether the quoted conclusion is
correct (right row yes, wrong row no) on a world whose base question type is
drawn from the kind's supported bases.
"""

from __future__ import annotations

import random
import re
from typing import Any

from v2.data.hs1 import f1_kinds_a, f1_kinds_b
from v2.data.hs1.core import (
    RecheckMismatch,
    GenerationError,
    Item,
    assemble,
    people,
    pick,
    require_present,
    rng_for,
    sha,
)
from v2.data.hs1.f1_base import Claim, F1World

FAMILY = "hs1_quote_check"
KINDS_TRAIN = (
    "order_sla",
    "expense_total",
    "applicant_screen",
    "project_timeline",
    "directory_route",
    "rubric_pick",
)
KINDS_OOD = ("transit_connection", "subscription_bill")
INTERFACE_SHARES = {
    "choice": 0.40,
    "noul_direct": 0.25,
    "noul_verify": 0.20,
    "score": 0.15,
}
LENGTH_SHARES = {"short": 0.75, "long": 0.25}
BUILDERS = {**f1_kinds_a.BUILDERS, **f1_kinds_b.BUILDERS}
BASES = {**f1_kinds_a.BASES, **f1_kinds_b.BASES}
SCORE_LEVELS = {**f1_kinds_a.SCORE_LEVELS, **f1_kinds_b.SCORE_LEVELS}

INTERFACE_BASE = {"choice": "choice", "noul_direct": "noul", "score": "score"}
TASK_TYPE = {
    "choice": "choice",
    "noul_direct": "noul",
    "noul_verify": "noul",
    "score": "score",
}
SHORT_CHARS = (600, 2000)
LONG_CHARS = (3000, 7500)
MAX_ATTEMPTS = 50

CHANNELS = ("note", "draft recommendation", "review comment", "chat message", "email")
HEDGES = ("confident", "neutral", "hedged")
POSITIONS = ("top", "middle", "end")
_SLUG = {
    "note": "note",
    "draft recommendation": "draft",
    "review comment": "review",
    "chat message": "chat",
    "email": "email",
}

# Framing templates per channel. Slots: speaker, given, role, body, subject.
_FRAMES: dict[str, tuple[str, ...]] = {
    "note": (
        "Note from {speaker} ({role}): {body}",
        "Handwritten note clipped to the file, signed {speaker}, {role}: {body}",
        "Case note added by {speaker}, {role}. {body}",
        "Sticky note on the folder, from {given} ({role}): {body}",
    ),
    "draft recommendation": (
        "Draft recommendation prepared by {speaker} ({role}), not yet signed off: {body}",
        "DRAFT - recommendation from {speaker}, {role}. {body} Please review before this goes out.",
        "Recommendation (draft), author {speaker}, {role}: {body}",
        "Proposed outcome drafted by {given} ({role}) for sign-off: {body}",
    ),
    "review comment": (
        "Review comment from {speaker} ({role}) on this file: {body}",
        "{speaker} ({role}) left a review comment: {body}",
        "Comment by {speaker}, {role}, in the review thread: {body}",
        "Reviewer remark ({speaker}, {role}): {body}",
    ),
    "chat message": (
        "[team chat] {speaker} ({role}): {body}",
        "Chat message from {speaker}, {role}: {body}",
        "{given} ({role}) wrote in the team channel: {body}",
        "Message in the shared chat, {speaker} ({role}): {body}",
    ),
    "email": (
        "From: {speaker} ({role})\nSubject: Re: {subject}\n\nHi all,\n\n{body}\n\nBest,\n{given}",
        "Email from {speaker}, {role}, with the subject line Re: {subject}. {body}",
        "From: {speaker}\nTo: the team\nSubject: {subject}\n\n{body}\n\n{given} ({role})",
        "Forwarded email, sender {speaker} ({role}), about {subject}: {body}",
    ),
}
# Hedge wrappers around the conclusion (prefix) and an optional closing line.
_HEDGE_PREFIX = {
    "confident": (
        "I'm confident that ",
        "It is clear that ",
        "No doubt about it: ",
        "I've checked this, and ",
    ),
    "neutral": ("", "My reading is that ", "As far as I can tell, ", "In short, "),
    "hedged": (
        "I might be missing something, but I think ",
        "If I'm reading this right, ",
        "Tentatively, it looks like ",
        "I'm not fully sure, but I believe ",
    ),
}
_HEDGE_CLOSE = {
    "confident": ("", " This one is straightforward.", " No need to escalate."),
    "neutral": (
        "",
        " Let me know if you need more.",
        " Noting it here for the record.",
    ),
    "hedged": ("", " Happy to be corrected.", " Worth a second look before we reply."),
}
_JOINS = (
    "{conclusion}, because {rationale}.",
    "{conclusion}: {rationale}.",
    "{conclusion}, since {rationale}.",
    "{conclusion}. The reason is that {rationale}.",
)
_VERIFY = (
    "Is {speaker}'s conclusion correct?",
    "Is {given} right that {conclusion}?",
    "Does the evidence support the conclusion in {given}'s {channel}?",
    "Given the records above, is the conclusion {given} reaches in the {channel} correct?",
    "Should {given}'s conclusion be accepted as correct?",
    "Is the {role}'s conclusion about {subject} correct?",
)


def supported(kind: str) -> tuple[str, ...]:
    """Interfaces a kind can express: its bases' interfaces, plus Noul-verify always."""
    bases = BASES[kind]
    return tuple(
        interface
        for interface in INTERFACE_SHARES
        if interface == "noul_verify" or INTERFACE_BASE[interface] in bases
    )


def _draw(seed_key: str, kind: str, label: str, size: int) -> int:
    return int(sha(f"hs1-f1|{seed_key}|{kind}|{label}"), 16) % size


def base_for(seed_key: str, kind: str, interface: str, index: int | None = None) -> str:
    if interface == "noul_verify":
        bases = BASES[kind]
        if index is not None:
            return bases[index % len(bases)]
        return bases[_draw(seed_key, kind, "verify-base", len(bases))]
    return INTERFACE_BASE[interface]


def target_for(
    seed_key: str, kind: str, base: str, index: int | None = None
) -> int | None:
    """The base question's gold: ``index`` modulo the label count, else drawn from the seed."""
    if base == "choice":
        return None
    size = 2 if base == "noul" else SCORE_LEVELS[kind]
    if index is not None:
        return index % size
    return _draw(
        seed_key, kind, "target-noul" if base == "noul" else "target-score", size
    )


def target_index(kind: str, interface: str, index: int | None) -> int | None:
    """The counter that balances a group's target; Noul-verify counts per base, since its base
    cycles with ``index`` (with two bases, ``index % 2`` would fix the Noul target)."""
    if index is None or interface != "noul_verify":
        return index
    return index // len(BASES[kind])


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _quote_params(rng: random.Random, world: F1World, body_text: str) -> dict[str, Any]:
    names_seen = set(re.findall(r"[A-Z][a-z]+", body_text))
    speaker = people(rng, 1, exclude=sorted(names_seen))[0]
    channel = pick(rng, CHANNELS)
    return {
        "speaker": speaker,
        "role": pick(rng, world.roles),
        "channel": channel,
        "frame": rng.randrange(len(_FRAMES[channel])),
        "hedge": pick(rng, HEDGES),
        "hedge_prefix": rng.randrange(4),
        "hedge_close": rng.randrange(3),
        "join": rng.randrange(len(_JOINS)),
        "position": pick(rng, POSITIONS),
        "verify": rng.randrange(len(_VERIFY)),
    }


def render_quote(world: F1World, claim: Claim, params: dict[str, Any]) -> str:
    prefix = _HEDGE_PREFIX[params["hedge"]][params["hedge_prefix"]]
    conclusion = prefix + claim.conclusion if prefix else _cap(claim.conclusion)
    body = _JOINS[params["join"]].format(
        conclusion=conclusion, rationale=claim.rationale
    )
    body += _HEDGE_CLOSE[params["hedge"]][params["hedge_close"]]
    frame = _FRAMES[params["channel"]][params["frame"]]
    return frame.format(
        speaker=params["speaker"],
        given=params["speaker"].split()[0],
        role=params["role"],
        body=_cap(body),
        subject=world.subject,
    )


def _verify_question(world: F1World, claim: Claim, params: dict[str, Any]) -> str:
    return _VERIFY[params["verify"]].format(
        speaker=params["speaker"],
        given=params["speaker"].split()[0],
        conclusion=claim.conclusion,
        channel=params["channel"],
        role=params["role"],
        subject=world.subject,
    )


def _interleave(
    rng: random.Random, evidence: tuple[str, ...], extra: list[str]
) -> list[str]:
    blocks = list(evidence)
    for block in extra:
        blocks.insert(rng.randint(0, len(blocks)), block)
    return blocks


def _body(
    rng: random.Random, world: F1World, length: str, rooms: tuple[int, int]
) -> list[str]:
    """Evidence (+ distractors and filler for long worlds) as blocks; ``rooms`` = (min, max) quote room."""
    small, room = rooms
    if length == "short":
        low, high = SHORT_CHARS[0] - small, SHORT_CHARS[1] - room
        pool = list(world.distractors)
        rng.shuffle(pool)
        wanted = rng.randint(0, min(2, len(pool)))
        for count in [wanted] + [
            c for c in range(min(2, len(pool)) + 1) if c != wanted
        ]:
            blocks = _interleave(rng, world.evidence, pool[:count])
            if low <= len("\n\n".join(blocks)) <= high:
                return blocks
        raise GenerationError(f"{world.kind}: short state outside {SHORT_CHARS}")
    high = LONG_CHARS[1] - room
    goal = rng.randint(LONG_CHARS[0] + 200, LONG_CHARS[1] - 400) - room
    extra = list(world.distractors)
    rng.shuffle(extra)
    core = _interleave(rng, world.evidence, extra)
    while len("\n\n".join(core)) > min(high, goal) and extra:
        core.remove(extra.pop())
    text = assemble(
        rng, core, list(world.filler), max(goal, LONG_CHARS[0] - small), high
    )
    return text.split("\n\n")


def _place(blocks: list[str], quote: str, position: str, rng: random.Random) -> str:
    if position == "top":
        parts = [quote] + blocks
    elif position == "end":
        parts = blocks + [quote]
    else:
        middle = (
            len(blocks) // 2 if len(blocks) < 3 else rng.randint(1, len(blocks) - 1)
        )
        parts = blocks[:middle] + [quote] + blocks[middle:]
    return "\n\n".join(parts)


def _items(
    rng: random.Random, world: F1World, interface: str, length: str
) -> list[Item]:
    body_seed = "\n\n".join(world.evidence + world.distractors + world.filler)
    params = _quote_params(rng, world, body_seed)
    quotes = {
        name: render_quote(world, claim, params)
        for name, claim in (("right", world.right), ("wrong", world.wrong))
    }
    rooms = (
        min(len(text) for text in quotes.values()) + 2,
        max(len(text) for text in quotes.values()) + 2,
    )
    blocks = _body(rng, world, length, rooms)
    layout = rng.getstate()
    task_type = TASK_TYPE[interface]
    items = []
    for name, claim in (("right", world.right), ("wrong", world.wrong)):
        rng.setstate(layout)
        state = _place(blocks, quotes[name], params["position"], rng)
        bounds = SHORT_CHARS if length == "short" else LONG_CHARS
        if not bounds[0] <= len(state) <= bounds[1]:
            raise GenerationError(
                f"{world.kind}: {length} state has {len(state)} chars"
            )
        require_present(state, world.decisive)
        if interface == "noul_verify":
            instructions = _verify_question(world, claim, params)
            gold = int(claim.mechanism == "correct")
            recheck = int(world.recheck == claim.answer)
            choices: tuple[str, ...] = ()
            refs: dict[str, int] = {}
        else:
            instructions = world.question
            gold, recheck = world.gold, world.recheck
            choices = () if task_type == "noul" else world.choices
            refs = {"quoted": claim.answer}
        quote_facts = {
            "speaker": params["speaker"],
            "role": params["role"],
            "channel": params["channel"],
            "frame": params["frame"],
            "hedge": params["hedge"],
            "position": params["position"],
            "join": params["join"],
            "verify_template": params["verify"],
        }
        item = Item(
            task_type=task_type,
            state=state,
            instructions=instructions,
            choices=choices,
            gold=gold,
            recheck=recheck,
            kind=world.kind,
            subtype=world.wrong.mechanism,
            variant=f"{world.variant}/{_SLUG[params['channel']]}{params['frame']}/{params['position']}",
            facts={**dict(world.facts), "quote": quote_facts},
            option_refs=refs,
            meta={
                "interface": interface,
                "length": length,
                "base": world.base,
                "mechanism": world.wrong.mechanism,
                "quote_correct": claim.mechanism == "correct",
                "channel": params["channel"],
                "hedge": params["hedge"],
                "position": params["position"],
                "speaker_role": params["role"],
                "claim_answer": claim.answer,
            },
            probe_view=quotes[name],
        )
        item.check()
        items.append(item)
    return items


def make_group(
    seed_key: str,
    kind: str,
    interface: str,
    length: str,
    *,
    target: int | None = None,
    index: int | None = None,
) -> list[Item]:
    """The [right, wrong] rows of one world; a pure function of the arguments.

    ``index`` is the group's position in its (kind, interface) cell. When
    given, the base question's gold is ``index % 2`` (Noul) or ``index %
    SCORE_LEVELS[kind]`` (Score), and Noul-verify takes the base
    ``BASES[kind][index % len(BASES[kind])]`` and balances that base's gold
    with ``index // len(BASES[kind])``; otherwise both are drawn from the seed.
    ``target`` optionally overrides the gold of the base question.
    """
    if kind not in BUILDERS:
        raise KeyError(f"unknown F1 kind {kind!r}")
    if interface not in supported(kind):
        raise ValueError(f"{kind} does not support interface {interface!r}")
    if length not in LENGTH_SHARES:
        raise ValueError(f"unknown length class {length!r}")
    if index is not None and (
        isinstance(index, bool) or not isinstance(index, int) or index < 0
    ):
        raise ValueError(f"bad index {index!r}")
    base = base_for(seed_key, kind, interface, index)
    if target is None:
        target = target_for(seed_key, kind, base, target_index(kind, interface, index))
    elif base == "choice" or not 0 <= target < (
        2 if base == "noul" else SCORE_LEVELS[kind]
    ):
        raise ValueError(f"bad target {target!r} for base {base}")
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        rng = rng_for(seed_key, "attempt", attempt)
        try:
            world = BUILDERS[kind](rng, base, target, length)
            world.check()
            if world.kind != kind or world.base != base:
                raise GenerationError(
                    f"{kind}: builder returned {world.kind}/{world.base}"
                )
            if target is not None and world.gold != target:
                raise GenerationError(f"{kind}: gold {world.gold} != target {target}")
            return _items(rng_for(seed_key, "quote", attempt), world, interface, length)
        except RecheckMismatch:
            raise
        except GenerationError as exc:
            last = exc
    raise GenerationError(
        f"{kind}/{interface}/{length}: no world after {MAX_ATTEMPTS} attempts ({last})"
    )

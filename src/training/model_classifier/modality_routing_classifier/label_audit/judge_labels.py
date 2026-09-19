#!/usr/bin/env python3
"""Blind LLM-judge audit of the modality-routing labels (AR / DIFFUSION / BOTH).

The judge never sees the current label. Judgments go to an append-only
checkpoint.jsonl (fsync'd per write), so a crash or restart loses nothing and every
command resumes from what is already judged. Rubric: RUBRIC.md (its hash is stored on
every record, so mixed-rubric runs are detectable).

Two ways to produce judgments, same rubric, same checkpoint format:

  In a chat session (cheapest on context: big batches, one call per round):
      python judge_labels.py next --n 150            # blinded rows -> read them
      python judge_labels.py save <<'EOF'            # compact lines: "ID A|D|B [vh] [M|L] [tags]"
      12 A
      13 D
      14 B M
      EOF
      python judge_labels.py status

  Non-interactively via the Claude API (needs `pip install anthropic`, ANTHROPIC_API_KEY):
      python judge_labels.py api --model claude-sonnet-5 --batch-size 50
      python judge_labels.py api --dry-run           # show the request, call nothing

Then compare against the original labels (and optionally re-score model predictions):
      python judge_labels.py report --eval-report ../modality_candidate_eval_report.json
Self-consistency check: `next --rejudge --n 75` / `save --rejudge` judge a seeded random
sample again into checkpoint_rejudge.jsonl, and `report` prints intra-judge agreement.

Independent human check (no API needed):
      python judge_labels.py sheet                   # blinded human_review.tsv: all disagreements + sampled agreements
      python judge_labels.py report --human human_review.tsv   # after filling the last column with A/D/B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "exported_modality_routing_dataset"
RUBRIC = HERE / "RUBRIC.md"
MANIFEST = HERE / "dataset_sha256.json"
CHECKPOINT = HERE / "checkpoint.jsonl"
REJUDGE = HERE / "checkpoint_rejudge.jsonl"

LABELS = ["AR", "DIFFUSION", "BOTH"]
LABEL_CODES = {"A": "AR", "D": "DIFFUSION", "B": "BOTH", "AR": "AR", "DIFFUSION": "DIFFUSION", "BOTH": "BOTH"}
TAG_CODES = {
    "img": "about_images",
    "prm": "prompt_writing",
    "nor": "no_request",
    "frag": "fragment",
    "nen": "non_english",
    "dvis": "deliverable_visual",
    "amb": "ambiguous",
}
# Rows longer than CLIP_LIMIT are shown as head + tail so batches stay cheap; the tooling
# adds the `truncated` tag itself (deterministically) instead of relying on the judge.
CLIP_LIMIT, CLIP_HEAD, CLIP_TAIL = 400, 280, 120
LEGEND = "reply lines: ID A|D|B [vh] [M|L] [" + " ".join(TAG_CODES) + "]"


# ----------------------------------------------------------------------------- data
def check_dataset(path: Path, split: str) -> None:
    """Judgments are keyed by row position, so a re-exported dataset would silently misalign
    them. The export is not reproducible across runs, so pin the file by hash on first use."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    known = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if split not in known:
        known[split] = digest
        MANIFEST.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
    elif known[split] != digest:
        raise SystemExit(f"{path.name} does not match the file the {split} judgments were made on "
                         f"(sha256 {digest[:12]} vs recorded {known[split][:12]}); refusing to continue")


def load_rows(data_dir: Path, split: str) -> list[dict]:
    path = Path(data_dir) / f"{split}.jsonl"
    if path.resolve().parent == DATA_DIR.resolve():  # custom --data-dir (e.g. tests) is not pinned
        check_dataset(path, split)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clip(text: str) -> tuple[str, bool]:
    norm = re.sub(r"\s+", " ", text).strip()
    if len(norm) <= CLIP_LIMIT:
        return norm, False
    cut = len(norm) - CLIP_HEAD - CLIP_TAIL
    return f"{norm[:CLIP_HEAD]} [...{cut} chars cut...] {norm[-CLIP_TAIL:]}", True


def rubric_hash() -> str:
    return hashlib.sha256(RUBRIC.read_bytes()).hexdigest()[:8]


# ------------------------------------------------------------------------ checkpoint
def read_checkpoint(path: Path) -> dict[tuple[str, int], dict]:
    """Last record per (split, id) wins."""
    out: dict[tuple[str, int], dict] = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[(r["split"], r["id"])] = r
    return out


def append_records(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------- parse
def parse_line(line: str) -> tuple[int, str, int, str, list[str]]:
    tok = line.split()
    if len(tok) < 2:
        raise ValueError("need 'ID LABEL'")
    try:
        rid = int(tok[0])
    except ValueError:
        raise ValueError(f"bad id {tok[0]!r}")
    label = LABEL_CODES.get(tok[1].upper())
    if label is None:
        raise ValueError(f"bad label {tok[1]!r}")
    vh, conf, tags = 0, "H", []
    for flag in tok[2:]:
        if flag.lower() == "vh":
            vh = 1
        elif len(flag) == 1 and flag.upper() in "MLH":
            conf = flag.upper()
        elif flag.lower() in TAG_CODES:
            tags.append(TAG_CODES[flag.lower()])
        else:
            raise ValueError(f"unknown flag {flag!r}")
    if vh and label != "AR":
        raise ValueError("vh is only valid with label A")
    return rid, label, vh, conf, sorted(set(tags))


def parse_lines(text: str, allowed: set[int] | None = None):
    """Returns (parsed rows, error strings, count of ignored non-row lines)."""
    parsed, errors, ignored, seen = [], [], 0, set()
    for raw in text.splitlines():
        line = raw.strip().strip("`")
        if not line or line.startswith("#"):
            continue
        if not line[0].isdigit():
            ignored += 1
            continue
        try:
            row = parse_line(line)
        except ValueError as e:
            errors.append(f"{line!r}: {e}")
            continue
        if allowed is not None and row[0] not in allowed:
            errors.append(f"{line!r}: id out of range or not in this batch")
        elif row[0] in seen:
            errors.append(f"{line!r}: duplicate id in input")
        else:
            seen.add(row[0])
            parsed.append(row)
    return parsed, errors, ignored


def make_record(split, parsed, rows, judge, rubric) -> dict:
    rid, label, vh, conf, tags = parsed
    if clip(rows[rid]["text"])[1] and "truncated" not in tags:
        tags = sorted(tags + ["truncated"])
    return {"split": split, "id": rid, "label": label, "vh": vh, "conf": conf,
            "tags": tags, "judge": judge, "rubric": rubric}


# --------------------------------------------------------------------- id selection
def select_ids(split, rows, primary, rejudged, n, rejudge, seed, exclude=frozenset()) -> list[int]:
    if rejudge:  # seeded random sample of rows already judged in the primary pass
        pool = sorted(i for (s, i) in primary if s == split and (s, i) not in rejudged and i not in exclude)
        random.Random(seed).shuffle(pool)
        return pool[:n]
    return [i for i in range(len(rows)) if (split, i) not in primary and i not in exclude][:n]


def target_path(args) -> Path:
    return Path(args.rejudge_checkpoint if args.rejudge else args.checkpoint)


# ---------------------------------------------------------------------------- next
def cmd_next(args) -> int:
    rows = load_rows(args.data_dir, args.split)
    primary = read_checkpoint(args.checkpoint)
    rejudged = read_checkpoint(args.rejudge_checkpoint)
    done = rejudged if args.rejudge else primary
    ids = select_ids(args.split, rows, primary, rejudged, args.n, args.rejudge, args.seed)
    remaining = len(rows) - sum(1 for k in done if k[0] == args.split)
    if not ids:
        print("# DONE: nothing left to judge for this selection")
        return 0
    print(f"# split={args.split} remaining={remaining} showing={len(ids)} | {LEGEND}")
    for i in ids:
        print(f"{i} {clip(rows[i]['text'])[0]}")
    return 0


# ---------------------------------------------------------------------------- save
def save_parsed(args, parsed, rows, judge, path, existing) -> list[dict]:
    rub = rubric_hash()
    fresh = [p for p in parsed if getattr(args, "overwrite", False) or (args.split, p[0]) not in existing]
    recs = [make_record(args.split, p, rows, judge, rub) for p in fresh]
    if recs:
        append_records(path, recs)
    return recs


def cmd_save(args) -> int:
    rows = load_rows(args.data_dir, args.split)
    path = target_path(args)
    existing = read_checkpoint(path)
    parsed, errors, ignored = parse_lines(sys.stdin.read(), allowed=set(range(len(rows))))
    recs = save_parsed(args, parsed, rows, args.judge, path, existing)
    skipped = len(parsed) - len(recs)
    total = len([k for k in read_checkpoint(path) if k[0] == args.split])
    print(f"saved {len(recs)} -> {path.name} | already judged (skipped) {skipped} | "
          f"bad lines {len(errors)} | ignored non-row lines {ignored} | "
          f"{args.split}: {total}/{len(rows)} judged")
    for e in errors[:10]:
        print("  ERR", e)
    return 1 if errors else 0


# -------------------------------------------------------------------------- status
def cmd_status(args) -> int:
    for path, name in ((args.checkpoint, "primary"), (args.rejudge_checkpoint, "rejudge")):
        ck = read_checkpoint(path)
        if not ck:
            continue
        print(f"== {name} ({Path(path).name}) ==")
        for split in sorted({s for s, _ in ck}):
            recs = [r for (s, _), r in ck.items() if s == split]
            try:
                total = len(load_rows(args.data_dir, split))
            except FileNotFoundError:
                total = "?"
            print(f"  {split}: {len(recs)}/{total} judged | labels {dict(Counter(r['label'] for r in recs))} | "
                  f"conf {dict(Counter(r['conf'] for r in recs))} | vh {sum(r['vh'] for r in recs)}")
            tags = Counter(t for r in recs for t in r["tags"])
            print(f"    tags {dict(tags)} | judges {dict(Counter(r['judge'] for r in recs))} | "
                  f"rubric {dict(Counter(r['rubric'] for r in recs))}")
    print(f"current rubric hash: {rubric_hash()}")
    return 0


# ------------------------------------------------------------------------------ api
def make_client():
    import anthropic  # imported lazily: only the api command needs the SDK

    return anthropic.Anthropic()


def sdk_errors():
    try:
        import anthropic

        return anthropic.RateLimitError, anthropic.APIConnectionError
    except ImportError:
        return ()


def build_user_message(rows, ids) -> str:
    lines = "\n".join(f"{i} {clip(rows[i]['text'])[0]}" for i in ids)
    return f"Judge these {len(ids)} rows per the rubric. Reply with exactly one line per row.\n\n{lines}"


def call_model(client, model, system_text, user_text, effort, max_tokens):
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_text}],
    )
    if effort:
        kwargs["output_config"] = {"effort": effort}
    retryable = sdk_errors()
    for attempt in range(4):
        try:
            resp = client.messages.create(**kwargs)
            break
        except retryable as e:  # 429 / connection errors (SDK already retried twice)
            if attempt == 3:
                raise
            wait = 20 * (attempt + 1)
            print(f"  retryable error ({type(e).__name__}); sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("response hit max_tokens; lower --batch-size or raise --max-tokens")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage


def cmd_api(args) -> int:
    rows = load_rows(args.data_dir, args.split)
    system_text = RUBRIC.read_text(encoding="utf-8")
    path = target_path(args)
    primary = read_checkpoint(args.checkpoint)
    rejudged = read_checkpoint(args.rejudge_checkpoint)
    if args.dry_run:
        ids = select_ids(args.split, rows, primary, rejudged, args.batch_size, args.rejudge, args.seed)
        print(f"model={args.model} effort={args.effort} rubric={rubric_hash()} "
              f"system_chars={len(system_text)} batch={len(ids)}")
        print("---- user message (first batch) ----")
        print(build_user_message(rows, ids))
        return 0
    client = make_client()
    skipped: set[int] = set()
    processed = 0
    totals = Counter()
    while args.limit is None or processed < args.limit:
        primary = read_checkpoint(args.checkpoint)
        rejudged = read_checkpoint(args.rejudge_checkpoint)
        n = args.batch_size if args.limit is None else min(args.batch_size, args.limit - processed)
        ids = select_ids(args.split, rows, primary, rejudged, n, args.rejudge, args.seed, exclude=skipped)
        if not ids:
            break
        missing = list(ids)
        for attempt in range(1 + args.retries):
            text, usage = call_model(client, args.model, system_text,
                                     build_user_message(rows, missing), args.effort, args.max_tokens)
            parsed, errors, _ = parse_lines(text, allowed=set(missing))
            recs = save_parsed(args, parsed, rows, args.model, path, read_checkpoint(path))
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                totals[key] += getattr(usage, key, 0) or 0
            got = {r["id"] for r in recs}
            processed += len(got)
            missing = [i for i in missing if i not in got]
            if errors:
                print(f"  {len(errors)} unparseable/unexpected lines, e.g. {errors[0]}", file=sys.stderr)
            if not missing:
                break
        if missing:
            print(f"  giving up on {len(missing)} ids after retries: {missing[:10]}", file=sys.stderr)
            skipped.update(missing)
        done = len([k for k in read_checkpoint(path) if k[0] == args.split])
        print(f"batch done: {done}/{len(rows)} judged | tokens {dict(totals)}")
    if skipped:
        print(f"unjudged (skipped) ids: {sorted(skipped)}")
    return 0


# ---------------------------------------------------------------------------- report
def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def inclusive_label(r: dict) -> str:
    """Policy view: implicit visual benefit upgrades a text-only request to BOTH."""
    return "BOTH" if r["label"] == "BOTH" or (r["label"] == "AR" and r["vh"]) else r["label"]


def load_preds(args, n_rows: int) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    if args.eval_report:
        recs = json.load(open(args.eval_report))["per_example_records"]
        for name in ("published_baseline", "clean_baseline", "candidate"):
            if recs and f"{name}_pred" in recs[0]:
                preds[name] = [r[f"{name}_pred"] for r in recs]
    for spec in args.preds or []:
        name, path = spec.split("=", 1)
        preds[name] = json.load(open(path))["preds"]
    return {k: v for k, v in preds.items() if len(v) == n_rows}


def cmd_sheet(args) -> int:
    """Blinded human-review sheet: every judge/original disagreement plus a random sample of agreements."""
    rows = load_rows(args.data_dir, args.split)
    ck = read_checkpoint(args.checkpoint)
    recs = {i: r for (s, i), r in ck.items() if s == args.split}
    dis = [i for i, r in recs.items() if rows[i]["label_name"] != r["label"]]
    dis_set = set(dis)
    agree = [i for i in recs if i not in dis_set]
    rng = random.Random(args.seed)
    picked = dis + rng.sample(agree, min(args.n_agree, len(agree)))
    rng.shuffle(picked)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# split={args.split}\n")
        f.write("# Blind human labels: put A, D or B in the last column, following RUBRIC.md.\n")
        f.write("# Rows left blank are dropped from the estimate, which biases it toward easy rows, so guess when unsure.\n")
        f.write("# Do not open checkpoint*.jsonl or audit_report_*.txt before finishing: they contain the labels.\n")
        f.write("id\ttext\tlabel\n")
        for i in picked:
            f.write(f"{i}\t{clip(rows[i]['text'])[0]}\t\n")
    print(f"wrote {len(picked)} rows to {args.out} ({len(dis)} disagreements + {len(picked) - len(dis)} sampled agreements, shuffled)")
    return 0


def load_human(path: str, split: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# split=") and line[len("# split="):].strip() != split:
            raise SystemExit(f"{path} was made for split {line[len('# split='):].strip()!r}, "
                             f"but the report is for {split!r}; pass --split")
        if not line.strip() or line.startswith("#") or line.startswith("id\t"):
            continue
        parts = line.split("\t")
        label = LABEL_CODES.get(parts[-1].strip().upper()) if len(parts) >= 3 else None
        if label and parts[0].strip().isdigit():
            out[int(parts[0])] = label
    return out


def print_human_section(path: str, orig: list[str], recs: dict[int, dict], split: str) -> None:
    """Human labels vs original and judge, with a stratified estimate of the true label-error rates."""
    human = {i: h for i, h in load_human(path, split).items() if i in recs}
    dis_all = [i for i in recs if orig[i] != recs[i]["label"]]
    agree_all = [i for i in recs if orig[i] == recs[i]["label"]]
    dis = [i for i in human if orig[i] != recs[i]["label"]]
    agr = [i for i in human if orig[i] == recs[i]["label"]]
    print(f"\n== human review: {len(human)} rows labeled ({len(dis)} disagreement rows, {len(agr)} agreement rows) ==")
    if not dis or not agr:
        print("  need at least one labeled row in each stratum for the estimate")
        return
    side = Counter("judge" if human[i] == recs[i]["label"] else "original" if human[i] == orig[i] else "neither" for i in dis)
    print(f"  on disagreement rows the human sides with: judge {side['judge']} | original {side['original']} | neither {side['neither']}")
    agr_ok = sum(human[i] == orig[i] for i in agr)
    print(f"  on agreement rows the human agrees with the shared label: {agr_ok}/{len(agr)}")
    n = len(recs)
    q_wrong = 1 - agr_ok / len(agr)
    orig_err = (len(dis_all) * (side["judge"] + side["neither"]) / len(dis) + len(agree_all) * q_wrong) / n
    judge_err = (len(dis_all) * (side["original"] + side["neither"]) / len(dis) + len(agree_all) * q_wrong) / n
    print(f"  stratified estimate over all {n} rows: original-label error ~{orig_err:.1%} | judge error ~{judge_err:.1%}"
          f" (small samples: treat as rough)")


def cmd_report(args) -> int:
    rows = load_rows(args.data_dir, args.split)
    orig = [r["label_name"] for r in rows]
    ck = read_checkpoint(args.checkpoint)
    recs = {i: r for (s, i), r in ck.items() if s == args.split}
    if not recs:
        print("no judgments yet")
        return 1
    ids = sorted(recs)
    o = [orig[i] for i in ids]
    j = [recs[i]["label"] for i in ids]
    ji = [inclusive_label(recs[i]) for i in ids]
    agree = sum(a == b for a, b in zip(o, j))
    print(f"judged {len(ids)}/{len(rows)} rows of {args.split} | rubric hashes "
          f"{dict(Counter(r['rubric'] for r in recs.values()))}")
    print(f"agreement original vs judge (strict):    {agree}/{len(ids)} = {agree / len(ids):.4f}  "
          f"kappa={cohen_kappa(o, j):.3f}")
    agree_i = sum(a == b for a, b in zip(o, ji))
    print(f"agreement original vs judge (inclusive): {agree_i}/{len(ids)} = {agree_i / len(ids):.4f}  "
          f"(inclusive = strict, but AR+vh counts as BOTH)")
    cm = Counter(zip(o, j))
    print("\nconfusion, rows=original cols=judge (strict), order AR/DIFFUSION/BOTH:")
    for a in LABELS:
        print(f"  {a:10s}", [cm.get((a, b), 0) for b in LABELS])
    dis = [i for i in ids if orig[i] != recs[i]["label"]]
    print(f"\ndisagreements: {len(dis)} | by judge confidence "
          f"{dict(Counter(recs[i]['conf'] for i in dis))} | vh among judge=AR: "
          f"{sum(recs[i]['vh'] for i in ids)}/{sum(1 for i in ids if recs[i]['label'] == 'AR')}")
    print(f"tags: {dict(Counter(t for r in recs.values() for t in r['tags']))}")

    preds = load_preds(args, len(rows))
    if preds:
        print("\nmodel accuracy on the judged rows: vs original | vs judge strict | vs judge inclusive")
        for name, p in preds.items():
            pj = [p[i] for i in ids]
            f = lambda ref: sum(x == y for x, y in zip(pj, ref)) / len(ids)
            print(f"  {name:22s} {f(o):.4f} | {f(j):.4f} | {f(ji):.4f}")

    if args.human:
        print_human_section(args.human, orig, recs, args.split)

    if Path(args.rejudge_checkpoint).exists():
        rj = {i: r for (s, i), r in read_checkpoint(args.rejudge_checkpoint).items() if s == args.split}
        both = sorted(set(rj) & set(recs))
        if both:
            a = [recs[i]["label"] for i in both]
            b = [rj[i]["label"] for i in both]
            flips = [i for i, x, y in zip(both, a, b) if x != y]
            print(f"\nself-consistency (re-judged {len(both)} rows): agreement "
                  f"{(len(both) - len(flips)) / len(both):.4f} kappa={cohen_kappa(a, b):.3f} | flips {flips[:15]}")

    order = {"H": 0, "M": 1, "L": 2}
    print(f"\n--- disagreements (original -> judge), confident ones first, showing {min(args.show, len(dis))} ---")
    for i in sorted(dis, key=lambda i: (order[recs[i]["conf"]], i))[: args.show]:
        r = recs[i]
        vh = " vh" if r["vh"] else ""
        print(f"{i:4d} {orig[i]}->{r['label']} {r['conf']}{vh} {','.join(r['tags'])} | {clip(rows[i]['text'])[0][:110]}")
    return 0


# ---------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split", default="test", choices=["train", "validation", "test"])
    common.add_argument("--rejudge", action="store_true", help="operate on the self-consistency re-judge pass")
    common.add_argument("--seed", type=int, default=0, help="seed for the re-judge sample")
    common.add_argument("--data-dir", default=str(DATA_DIR))
    common.add_argument("--checkpoint", default=str(CHECKPOINT))
    common.add_argument("--rejudge-checkpoint", default=str(REJUDGE))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("next", parents=[common], help="print the next blinded batch")
    s.add_argument("--n", type=int, default=150)
    s.set_defaults(fn=cmd_next)

    s = sub.add_parser("save", parents=[common], help="read compact judgment lines from stdin")
    s.add_argument("--judge", default=os.environ.get("JUDGE_NAME", "claude-code-session"))
    s.add_argument("--overwrite", action="store_true", help="re-judge rows already in the checkpoint")
    s.set_defaults(fn=cmd_save)

    s = sub.add_parser("status", parents=[common], help="progress and label/tag counts")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("api", parents=[common], help="judge via the Claude API using RUBRIC.md")
    s.add_argument("--model", default="claude-sonnet-5")
    s.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--batch-size", type=int, default=50)
    s.add_argument("--limit", type=int, default=None, help="stop after this many rows (default: all)")
    s.add_argument("--retries", type=int, default=2, help="re-ask for ids missing from a reply")
    s.add_argument("--max-tokens", type=int, default=16000)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_api)

    s = sub.add_parser("report", parents=[common], help="agreement with original labels, model re-scoring")
    s.add_argument("--show", type=int, default=40, help="how many disagreements to list")
    s.add_argument("--eval-report", help="modality_candidate_eval_report.json (per_example_records)")
    s.add_argument("--preds", action="append", metavar="NAME=PATH",
                   help='JSON file with a "preds" list aligned to the split (repeatable)')
    s.add_argument("--human", help="filled-in sheet from `sheet` (id<TAB>text<TAB>label); adds a human-review section")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("sheet", parents=[common], help="write a blinded human-review sheet")
    s.add_argument("--out", default=str(HERE / "human_review.tsv"))
    s.add_argument("--n-agree", type=int, default=44, help="random agreement rows to add to all disagreements")
    s.set_defaults(fn=cmd_sheet)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

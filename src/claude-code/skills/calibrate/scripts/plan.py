"""A calibration run over one plugin id: its options and log, the report, the suggested
thresholds, and applying them to the user's settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "setup" / "scripts"))
import runtime
import settings
from decisions import (
    ACTED_ONLY,
    APPS,
    BINARY,
    BINARY_LABELS,
    ROUTER_KEYS,
    Log,
    acts,
    as_number,
    features,
    implicit_labels,
    level_of,
    read_labels,
    read_log,
)
from fitting import (
    GOALS,
    MAX_STEP,
    MIN_LABELS,
    confusion,
    fit_binary,
    fit_router,
    route_tier,
    router_agreement,
    router_cases,
)
from questions import distance

# The options a calibration may write: the router's lines and each binary app's thresholds.
CALIBRATED = (*ROUTER_KEYS, *(key for spec in BINARY.values() for key in spec.keys))


@dataclass
class Context:
    plugin_id: str
    options: dict
    token: str | None
    log_path: Path
    log: Log
    explicit: dict[str, object]
    implicit: dict[str, object]

    @property
    def labels(self) -> dict[str, object]:
        return {**self.implicit, **self.explicit}

    @property
    def enabled(self) -> list[str]:
        return [app for app in APPS if self.options.get(app)]

    def path(self, name: str) -> Path:
        return self.log_path.with_name(name)


def context(args: argparse.Namespace) -> Context:
    plugin_id = args.id or settings.resolve_id(
        settings.listed_plugins(), args.plugin_root
    )
    options = settings.options_for(plugin_id)
    token = runtime.read_token(args.token_file)
    log_path = (
        Path(args.log).expanduser()
        if args.log
        else settings.log_file(options["logPath"])
    )
    if log_path is None:
        raise settings.SettingsError(
            f"logPath {options['logPath']!r} keeps no log (empty or relative), "
            "so vSR logs no decisions; set logPath or pass --log"
        )
    log = read_log(log_path)
    explicit = read_labels(
        (args.labels and Path(args.labels)) or log_path.with_name("labels.jsonl")
    )
    return Context(
        plugin_id,
        options,
        token,
        log_path,
        log,
        explicit,
        implicit_labels(log.decisions),
    )


def binary_cases(ctx: Context, app: str) -> list[tuple[tuple[float, ...], bool]]:
    cases = []
    for d in ctx.log.decisions:
        if (
            d.app == app
            and d.id in ctx.labels
            and str(ctx.labels[d.id]).lower() in BINARY_LABELS
        ):
            values = features(d)
            if values is not None:
                cases.append((values, BINARY_LABELS[str(ctx.labels[d.id]).lower()]))
    return cases


def histogram(
    values: list[float], top: float, marks: list[float], bins: int = 10, width: int = 30
) -> list[str]:
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, max(0, int(v / top * bins)))] += 1
    biggest = max(counts) or 1
    lines = []
    for i, n in enumerate(counts):
        low, high = i * top / bins, (i + 1) * top / bins
        here = [m for m in marks if low <= m < high or (i == bins - 1 and m == top)]
        bar = "#" * round(n / biggest * width)
        lines.append(
            f"    {low:4.2f}-{high:4.2f} {bar:<{width}} {n:>4}"
            + (f"  <- {', '.join(f'{m:.2f}' for m in here)}" if here else "")
        )
    return lines


def report(ctx: Context) -> str:
    when = [d.at for d in ctx.log.decisions if d.at is not None]
    span = (
        f", {datetime.fromtimestamp(min(when)):%Y-%m-%d} to {datetime.fromtimestamp(max(when)):%Y-%m-%d}"
        if when
        else ""
    )
    out = [
        f"log {ctx.log_path}: {len(ctx.log.decisions)} decisions{span}"
        + (
            f"; {ctx.log.skipped} unreadable line(s) skipped" if ctx.log.skipped else ""
        ),
        f"settings {settings.settings_path()} under pluginConfigs[{ctx.plugin_id!r}]",
    ]
    last = last_calibration(ctx.path("calibrations.jsonl"))
    now = runtime_model(ctx)
    out.append(
        f"runtime model now {now or 'unreachable'}; "
        + (
            f"last calibration {last.get('at')} with model {last.get('model')}"
            if last
            else "no calibration recorded yet"
        )
    )
    for app in APPS:
        mine = [d for d in ctx.log.decisions if d.app == app]
        off = sum(d.verdict == "off" for d in mine)
        state = "on" if ctx.options.get(app) else "off in settings"
        user = sum(1 for d in mine if d.id in ctx.explicit)
        implied = sum(
            1 for d in mine if d.id in ctx.implicit and d.id not in ctx.explicit
        )
        if app == "router":
            keys = ROUTER_KEYS
            values = [v for d in mine for v in [level_of(d)] if v is not None]
            top, near_width = 2.0, 0.2
        else:
            keys = BINARY[app].keys
            values = [f[0] for d in mine for f in [features(d)] if f is not None]
            top, near_width = 1.0, 0.1
        near = sum(
            1
            for d in mine
            for x in [distance(d, ctx.options)]
            if x is not None and x[0] <= near_width
        )
        lines = ", ".join(f"{k} {ctx.options[k]:.2f}" for k in keys)
        out.append(
            f"\n{app} ({state}): {len(mine)} decisions, {off} off, {near} near the line, labels {user} yours + {implied} implied · {lines}"
        )
        if app in ACTED_ONLY and mine:
            out.append(
                "    vSR logs this app only when it acted, so labels here can raise its threshold but not lower it; `probe` checks the other side"
            )
        if values:
            marks = [
                ctx.options[k]
                for k in keys
                if k not in {"routerMinConfidence", "escalateReask"}
            ]
            out.extend(histogram(values, top, marks))
        if app == "escalate":
            reasks = [
                f[1] for d in mine for f in [features(d)] if f is not None and f[1] > 0
            ]
            if reasks:
                out.append("    reask:")
                out.extend(histogram(reasks, 1.0, [ctx.options["escalateReask"]]))
    return "\n".join(out)


def suggestion(ctx: Context, goal: str) -> dict:
    beta = GOALS[goal]
    plan = {"pluginId": ctx.plugin_id, "goal": goal, "changes": {}, "apps": {}}
    for app in ctx.enabled:
        mine = [d for d in ctx.log.decisions if d.app == app]
        if not mine:
            plan["apps"][app] = {"why": "no decisions in the log"}
            continue
        if app == "router":
            cases = router_cases(ctx.log.decisions, ctx.labels)
            currents = tuple(ctx.options[k] for k in ROUTER_KEYS)
            fit = fit_router(cases, *currents)
            entry = {"labels": len(cases), "why": fit["why"], "before": {}, "after": {}}
            for name, values in (("before", currents), ("after", fit["thresholds"])):
                tiers = [0, 0, 0, 0]
                for d in mine:
                    level = level_of(d)
                    if level is None:
                        continue
                    confidence = as_number(d.detail.get("confidence"))
                    tiers[
                        (
                            3
                            if confidence is not None and confidence < values[2]
                            else route_tier(level, values[0], values[1])
                        )
                    ] += 1
                right, routed, kept = router_agreement(cases, *values)
                entry[name] = {
                    "values": dict(zip(ROUTER_KEYS, values, strict=False)),
                    "haiku": tiers[0],
                    "sonnet": tiers[1],
                    "opus": tiers[2],
                    "kept": tiers[3],
                    "agree": f"{right}/{routed} routed right, {kept} kept",
                }
        else:
            spec = BINARY[app]
            cases = binary_cases(ctx, app)
            currents = tuple(ctx.options[k] for k in spec.keys)
            fit = fit_binary(cases, currents, beta, raise_only=app in ACTED_ONLY)
            entry = {"labels": len(cases), "why": fit["why"]}
            for name, values in (("before", currents), ("after", fit["thresholds"])):
                acted = sum(
                    1
                    for d in mine
                    for f in [features(d)]
                    if f is not None and acts(f, values)
                )
                s = confusion(cases, values)
                entry[name] = {
                    "values": dict(zip(spec.keys, values, strict=False)),
                    spec.noun: acted,
                    "agree": f"{s.agree}/{s.n}" if s.n else "-",
                    "falseAlarms": s.fp,
                    "missed": s.fn,
                }
        plan["apps"][app] = entry
        for k, before, after in zip(
            entry["before"]["values"],
            entry["before"]["values"].values(),
            entry["after"]["values"].values(),
            strict=False,
        ):
            if abs(after - before) > 1e-9:
                plan["changes"][k] = {"from": before, "to": after}
    return plan


def table(plan: dict) -> str:
    rows = [
        f"goal: {plan['goal']} · at most {MAX_STEP} per threshold per run · at least {MIN_LABELS} labels per app",
        "",
    ]
    rows.append(
        f"{'app':9} {'threshold':20} {'before':>7} {'after':>7}  {'on the log (before → after)':32} {'labels agree (before → after)':30}"
    )
    for app, entry in plan["apps"].items():
        if "before" not in entry:
            rows.append(f"{app:9} {entry['why']}")
            continue
        for i, (k, before) in enumerate(entry["before"]["values"].items()):
            after = entry["after"]["values"][k]
            mark = "" if abs(after - before) < 1e-9 else " *"
            if i == 0:
                if app == "router":
                    b, a = entry["before"], entry["after"]
                    acted = f"H/S/O/kept {b['haiku']}/{b['sonnet']}/{b['opus']}/{b['kept']} → {a['haiku']}/{a['sonnet']}/{a['opus']}/{a['kept']}"
                else:
                    noun = BINARY[app].noun
                    acted = f"{entry['before'][noun]} → {entry['after'][noun]} {noun}"
                agree = f"{entry['before']['agree']} → {entry['after']['agree']}"
            else:
                acted = agree = ""
            rows.append(
                f"{app if i == 0 else '':9} {k:20} {before:7.2f} {after:7.2f}{mark:2} {acted:32} {agree}"
            )
        rows.append(f"{'':9} {entry['labels']} labels: {entry['why']}")
    if not plan["changes"]:
        rows.append("\nno threshold changes suggested")
    return "\n".join(rows)


def settings_diff(plan: dict) -> str:
    path = settings.settings_path()
    text, current = settings.load(path)
    values = {k: v["to"] for k, v in plan["changes"].items()}
    return settings.diff(
        text, settings.render(settings.merge(current, plan["pluginId"], values)), path
    )


def runtime_model(ctx: Context) -> str | None:
    return runtime.health(ctx.options["runtimeUrl"], ctx.token, timeout=2)["model"]


def last_calibration(path: Path) -> dict | None:
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def checked_changes(ctx: Context, plan: dict) -> dict[str, object]:
    """The plan's new values, refused unless each is a threshold calibration sets, for this
    plugin id, at most MAX_STEP from the value in use now.
    """
    if plan.get("pluginId") != ctx.plugin_id:
        raise settings.SettingsError(
            f"the plan is for {plan.get('pluginId')!r}, not {ctx.plugin_id!r}"
        )
    values = {}
    for name, change in plan.get("changes", {}).items():
        if name not in CALIBRATED or not isinstance(change, dict):
            raise settings.SettingsError(
                f"a calibration plan sets only {', '.join(CALIBRATED)}; not {name!r}"
            )
        value = settings.parse_value(name, change.get("to"))
        if abs(value - ctx.options[name]) > MAX_STEP + 1e-9:
            raise settings.SettingsError(
                f"{name} would move from {ctx.options[name]} to {value}, more than {MAX_STEP} in one run"
            )
        values[name] = value
    return values


def apply(ctx: Context, plan: dict, model: str | None = None) -> str:
    values = checked_changes(ctx, plan)
    if not values:
        return "nothing to apply"
    path = settings.settings_path()
    _text, current = settings.load(path)
    backup = settings.write(
        path, settings.render(settings.merge(current, ctx.plugin_id, values))
    )
    calibrations = ctx.path("calibrations.jsonl")
    calibrations.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pluginId": ctx.plugin_id,
        "goal": plan.get("goal"),
        "model": model,
        "changes": plan["changes"],
        "labels": {
            app: entry.get("labels", 0) for app, entry in plan.get("apps", {}).items()
        },
    }
    with calibrations.open("a") as out:
        out.write(json.dumps(record) + "\n")
    return (
        f"wrote {path}"
        + (f" (previous copy: {backup})" if backup else "")
        + f"; recorded in {calibrations}"
    )

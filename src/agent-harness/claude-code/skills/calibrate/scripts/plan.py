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
import plain
import runtime
import settings
from decisions import (
    ACTED_ONLY,
    APPS,
    BINARY,
    BINARY_LABELS,
    ROUTER_KEYS,
    TIERS,
    Decision,
    Log,
    acts,
    as_number,
    features,
    implicit_labels,
    level_of,
    read_labels,
    read_log,
    tier_of,
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
APP_OF = {
    **dict.fromkeys(ROUTER_KEYS, "router"),
    **{key: app for app, spec in BINARY.items() for key in spec.keys},
}


def parse_apps(text: str | None) -> list[str] | None:
    """The apps a comma list names, in vSR's order; None for no list (every app)."""
    if not text:
        return None
    named = {name.strip().lower() for name in text.split(",") if name.strip()}
    unknown = sorted(named - set(APPS))
    if unknown:
        raise settings.SettingsError(
            f"unknown app {', '.join(map(repr, unknown))}; choose from {', '.join(APPS)}"
        )
    return [app for app in APPS if app in named]


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


def binary_case(ctx: Context, d: Decision) -> tuple[tuple[float, ...], bool] | None:
    label = str(ctx.labels.get(d.id)).lower()
    values = features(d)
    if d.id not in ctx.labels or label not in BINARY_LABELS or values is None:
        return None
    return values, BINARY_LABELS[label]


def binary_cases(ctx: Context, app: str) -> list[tuple[tuple[float, ...], bool]]:
    return [
        case
        for d in ctx.log.decisions
        if d.app == app
        for case in [binary_case(ctx, d)]
        if case is not None
    ]


def answers(ctx: Context, app: str) -> tuple[int, int]:
    """How many of an app's labels a fit can use: the user's own, and the ones only the log implies."""
    usable = [
        d
        for d in ctx.log.decisions
        if d.app == app
        and (
            router_cases([d], ctx.labels)
            if app == "router"
            else binary_case(ctx, d) is not None
        )
    ]
    explicit = sum(d.id in ctx.explicit for d in usable)
    return explicit, len(usable) - explicit


def logged_tiers(mine: list[Decision]) -> dict[str, int]:
    """Where vSR's logged routes went: Haiku, Sonnet, Opus, another model, kept, or off."""
    tiers = dict.fromkeys((*TIERS, "other", "kept", "off"), 0)
    for d in mine:
        if d.verdict == "routed":
            tier = tier_of(d.detail.get("model"))
            tiers[TIERS[tier] if tier is not None else "other"] += 1
        else:
            tiers["kept" if d.verdict == "kept" else "off"] += 1
    return tiers


def plain_app(ctx: Context, app: str) -> list[str]:
    mine = [d for d in ctx.log.decisions if d.app == app]
    head = plain.title(app) + (
        "" if ctx.options.get(app) else " — off in your settings"
    )
    if not mine:
        return [head, "  No decisions logged yet."]
    if app == "router":
        logged = plain.router_logged(logged_tiers(mine))
    else:
        acted = sum(d.verdict == plain.ACTED_VERDICT[app] for d in mine)
        logged = plain.LOGGED[app].format(
            n=len(mine), acted=acted, other=len(mine) - acted
        )
    return [head, f"  {logged} {plain.answered(app, *answers(ctx, app))}"]


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


def report(ctx: Context, apps: list[str] | None = None, detail: bool = False) -> str:
    """The log in plain words for each app in `apps` (default every app); with `detail`, the
    option names, the near-the-line counts and the score histograms after it.
    """
    apps = apps or list(APPS)
    when = [d.at for d in ctx.log.decisions if d.at is not None]
    span = (
        f" from {datetime.fromtimestamp(min(when)):%Y-%m-%d} to {datetime.fromtimestamp(max(when)):%Y-%m-%d}"
        if when
        else ""
    )
    skipped = (
        f" ({plain.plural(ctx.log.skipped, 'unreadable line')} skipped)"
        if ctx.log.skipped
        else ""
    )
    last = last_calibration(ctx.path("calibrations.jsonl"))
    now = runtime_model(ctx)
    out = [
        f"vSR's log holds {plain.plural(len(ctx.log.decisions), 'decision')}{span}{skipped}.",
        f"Decision model now: {now or 'unreachable'}; "
        + (
            f"last calibrated {last.get('at')} with {last.get('model')}."
            if last
            else "not calibrated before."
        ),
    ]
    for app in apps:
        out += ["", *plain_app(ctx, app)]
    if detail:
        out += ["", "Details", *detail_report(ctx, apps)]
    return "\n".join(out)


def detail_report(ctx: Context, apps: list[str]) -> list[str]:
    out = [
        f"log {ctx.log_path}: {len(ctx.log.decisions)} decisions",
        f"settings {settings.settings_path()} under pluginConfigs[{ctx.plugin_id!r}]",
    ]
    for app in apps:
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
    return out


def route_of(d: Decision, values: tuple[float, ...]) -> str | None:
    """Where router lines `values` send a logged subagent: a tier's name, or kept."""
    level = level_of(d)
    if level is None:
        return None
    confidence = as_number(d.detail.get("confidence"))
    if confidence is not None and confidence < values[2]:
        return "kept"
    return TIERS[route_tier(level, values[0], values[1])]


def recommend(labels: int, changed: bool, before: int, after: int) -> str:
    """apply when the change agrees with more of the user's answers; more when too few answers."""
    if labels < MIN_LABELS:
        return "more"
    return "apply" if changed and after > before else "keep"


def router_entry(ctx: Context, mine: list[Decision]) -> dict:
    cases = router_cases(ctx.log.decisions, ctx.labels)
    currents = tuple(ctx.options[k] for k in ROUTER_KEYS)
    fit = fit_router(cases, *currents)
    entry = {"labels": len(cases), "why": fit["why"]}
    for name, values in (("before", currents), ("after", fit["thresholds"])):
        routes = [r for d in mine for r in [route_of(d, values)] if r is not None]
        right, routed, kept = router_agreement(cases, *values)
        entry[name] = {
            "values": dict(zip(ROUTER_KEYS, values, strict=False)),
            **{tier: routes.count(tier) for tier in (*TIERS, "kept")},
            "agree": f"{right}/{routed} routed right, {kept} kept",
        }
        entry[f"agree{name.capitalize()}"] = right
    moves: dict[tuple[str, str], int] = {}
    for d in mine:
        was, now = route_of(d, currents), route_of(d, fit["thresholds"])
        if was != now:
            moves[(was, now)] = moves.get((was, now), 0) + 1
    entry["moves"] = [{"from": f, "to": t, "n": n} for (f, t), n in moves.items()]
    return entry


def binary_entry(ctx: Context, app: str, mine: list[Decision], beta: float) -> dict:
    spec = BINARY[app]
    cases = binary_cases(ctx, app)
    currents = tuple(ctx.options[k] for k in spec.keys)
    fit = fit_binary(cases, currents, beta, raise_only=app in ACTED_ONLY)
    entry = {"labels": len(cases), "why": fit["why"], "decisions": len(mine)}
    for name, values in (("before", currents), ("after", fit["thresholds"])):
        acted = sum(
            1 for d in mine for f in [features(d)] if f is not None and acts(f, values)
        )
        s = confusion(cases, values)
        entry[name] = {
            "values": dict(zip(spec.keys, values, strict=False)),
            "acted": acted,
            spec.noun: acted,
            "agree": f"{s.agree}/{s.n}" if s.n else "-",
            "falseAlarms": s.fp,
            "missed": s.fn,
        }
        entry[f"agree{name.capitalize()}"] = s.agree
    return entry


def suggestion(ctx: Context, goal: str | None, apps: list[str] | None = None) -> dict:
    """New thresholds for each chosen app (default every enabled app), and what they would do.

    The goal weighs the binary apps' fit only; the router has none, so a router-only plan records no goal.
    """
    apps = apps or ctx.enabled
    goal = (goal or "balanced") if any(app != "router" for app in apps) else None
    plan = {"pluginId": ctx.plugin_id, "goal": goal, "changes": {}, "apps": {}}
    for app in apps:
        mine = [d for d in ctx.log.decisions if d.app == app]
        if not ctx.options.get(app):
            plan["apps"][app] = {
                "why": "off in settings",
                "plain": "Off in your settings, so there is nothing to tune.",
            }
            continue
        if not mine:
            plan["apps"][app] = {
                "why": "no decisions in the log",
                "plain": "No decisions logged yet, so there is nothing to tune.",
            }
            continue
        if app == "router":
            entry = router_entry(ctx, mine)
        else:
            entry = binary_entry(ctx, app, mine, GOALS[goal])
        changed = False
        for k, before in entry["before"]["values"].items():
            after = entry["after"]["values"][k]
            if abs(after - before) > 1e-9:
                plan["changes"][k] = {"from": before, "to": after}
                changed = True
        entry["recommend"] = recommend(
            entry["labels"], changed, entry["agreeBefore"], entry["agreeAfter"]
        )
        plan["apps"][app] = entry
    return plan


def table(plan: dict) -> str:
    goal = f"goal: {plan['goal']} · " if plan.get("goal") else ""
    rows = [
        f"{goal}at most {MAX_STEP} per threshold per run · at least {MIN_LABELS} labels per app",
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
    # The model vSR asks: the configured one on a runtime that serves several.
    if ctx.options.get("runtimeModel"):
        return ctx.options["runtimeModel"]
    return runtime.health(ctx.options["runtimeUrl"], ctx.token, timeout=2)["model"]


def last_calibration(path: Path) -> dict | None:
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def checked_changes(
    ctx: Context, plan: dict, apps: list[str] | None = None
) -> dict[str, dict]:
    """The plan's changes for `apps` (default all it holds), refused unless each is a threshold
    calibration sets, of an app the plan was made for, for this plugin id, at most MAX_STEP
    from the value in use now.
    """
    if plan.get("pluginId") != ctx.plugin_id:
        raise settings.SettingsError(
            f"the plan is for {plan.get('pluginId')!r}, not {ctx.plugin_id!r}"
        )
    held = plan.get("apps")
    changes = {}
    for name, change in plan.get("changes", {}).items():
        if name not in CALIBRATED or not isinstance(change, dict):
            raise settings.SettingsError(
                f"a calibration plan sets only {', '.join(CALIBRATED)}; not {name!r}"
            )
        if isinstance(held, dict) and APP_OF[name] not in held:
            raise settings.SettingsError(
                f"the plan was made for {', '.join(held)}, so it may not set {name}"
            )
        if apps is not None and APP_OF[name] not in apps:
            continue
        value = settings.parse_value(name, change.get("to"))
        if abs(value - ctx.options[name]) > MAX_STEP + 1e-9:
            raise settings.SettingsError(
                f"{name} would move from {ctx.options[name]} to {value}, more than {MAX_STEP} in one run"
            )
        changes[name] = {**change, "to": value}
    return changes


def apply(
    ctx: Context, plan: dict, model: str | None = None, apps: list[str] | None = None
) -> str:
    """Writes the plan's changes for `apps` (default all it holds) into the settings."""
    changes = checked_changes(ctx, plan, apps)
    values = {name: change["to"] for name, change in changes.items()}
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
        "changes": changes,
        "labels": {
            app: entry.get("labels", 0)
            for app, entry in plan.get("apps", {}).items()
            if apps is None or app in apps
        },
    }
    with calibrations.open("a") as out:
        out.write(json.dumps(record) + "\n")
    return (
        f"wrote {path}"
        + (f" (previous copy: {backup})" if backup else "")
        + f"; recorded in {calibrations}"
    )

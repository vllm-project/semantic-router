// A Router listener's System One listing sets what the vSR plugins pointed at it do: its `vsr` profile turns apps on
// or off, and its rules bind to guard and redact. Ported from the Claude Code plugin's lib/profile.ts and lib/rules.ts.
import type { Failure, Result, RuntimeClient } from "./runtime.ts";
import { APPS, isRecord } from "./settings.ts";
import type { App, Settings } from "./settings.ts";

// A rule the listener binds (listeners[].apis.systemone.vsr.rules): the Router keeps its pattern or question and
// evaluates it; the plugin knows its name, its description and whether a text it could not check is held.
export type RuleView = { name: string; description: string; required: boolean };
export type ListenerRules = {
  revision: string;
  guard: RuleView[];
  redact: RuleView[];
};
export const NO_RULES: ListenerRules = { revision: "", guard: [], redact: [] };
// The guard rules a text matches, the text the redact rules found in UTF-16 offsets, and the rules left unchecked.
export type RulesFound = {
  matches: string[];
  found: { from: number; to: number; rule: string }[];
  unchecked: { rule: string; reason: string }[];
};
export type RulesAsked =
  | { ok: true; revision: string; parts: RulesFound[] }
  | { ok: false; code: Failure };
export type Unchecked = { rule: RuleView; reason: string };

const PROFILE_MS = 1500;
// How often a running session asks again, so a change in the Dashboard reaches it at a later prompt.
const PROFILE_REFRESH_MS = 60_000;

const isRule = (value: unknown): value is RuleView =>
  isRecord(value) &&
  typeof value.name === "string" &&
  value.name !== "" &&
  typeof value.description === "string" &&
  typeof value.required === "boolean";

// The rules a listing's `vsr` binds: NO_RULES when it binds none, undefined when they are there but not readable,
// which keeps the rules read before rather than drop a required one.
export function rulesOf(listing: unknown): ListenerRules | undefined {
  if (!isRecord(listing)) return undefined;
  if (listing.vsr === undefined) return NO_RULES;
  const { vsr } = listing;
  if (!isRecord(vsr)) return undefined;
  if (vsr.rules === undefined) return NO_RULES;
  const { rules } = vsr;
  if (
    !isRecord(rules) ||
    typeof rules.revision !== "string" ||
    !Array.isArray(rules.guard) ||
    !Array.isArray(rules.redact) ||
    ![...rules.guard, ...rules.redact].every(isRule)
  )
    return undefined;
  const list = (value: RuleView[]) =>
    value.map(({ name, description, required }) => ({
      name,
      description,
      required,
    }));
  return {
    revision: rules.revision,
    guard: list(rules.guard as RuleView[]),
    redact: list(rules.redact as RuleView[]),
  };
}

// What a listing's `vsr` profile sets: each app on or off, the router's goal (`router_goal`) and whether it routes
// your own turns (`router_turns`), as the Claude Code plugin's lib/profile.ts reads them. A field of the wrong type is
// left to the option.
export type Field = App | "goal" | "turns";
export const FIELDS: readonly Field[] = [...APPS, "goal", "turns"];
export type Values = {
  [K in Field]: K extends "goal" ? Settings["routing"]["goal"] : boolean;
};
const GOALS = ["balanced", "quality", "save"];
export function profileOf(listing: unknown): Partial<Values> {
  const vsr = isRecord(listing) ? listing.vsr : undefined;
  if (!isRecord(vsr)) return {};
  const profile: Partial<Values> = {};
  for (const app of APPS)
    if (typeof vsr[app] === "boolean") profile[app] = vsr[app];
  if (GOALS.includes(vsr.router_goal as string))
    profile.goal = vsr.router_goal as Values["goal"];
  if (typeof vsr.router_turns === "boolean") profile.turns = vsr.router_turns;
  return profile;
}
const valuesOf = (settings: Settings): Values => ({
  ...settings.apps,
  goal: settings.routing.goal,
  turns: settings.routing.turns,
});
function assign(settings: Settings, values: Values): void {
  for (const app of APPS) settings.apps[app] = values[app];
  settings.routing.goal = values.goal;
  settings.routing.turns = values.turns;
}
// What a profile changes, in the words of the Claude Code plugin's note: `redact off, goal save, turns on`.
export function noteOf(changed: Partial<Values>): string {
  const words = Object.entries(changed).map(
    ([field, value]) =>
      `${field} ${value === true ? "on" : value === false ? "off" : value}`,
  );
  return words.length ? `vSR follows the listener: ${words.join(", ")}` : "";
}

// UTF-16 offsets of a text's code points, one past the end included.
function utf16Offsets(text: string): number[] {
  const offsets: number[] = [];
  let at = 0;
  for (const char of text) {
    offsets.push(at);
    at += char.length;
  }
  offsets.push(at);
  return offsets;
}

// One part of the Router's answer, checked against the text it is about: a span outside it, or a rule it does not
// bind, makes the whole answer unreadable rather than partly trusted.
function partOf(
  text: string,
  answer: unknown,
  id: string,
  bound: ReadonlySet<string>,
): RulesFound | undefined {
  if (!isRecord(answer) || answer.id !== id) return undefined;
  const { matches, spans, unchecked, coverage } = answer;
  if (
    !Array.isArray(matches) ||
    !Array.isArray(spans) ||
    !Array.isArray(unchecked)
  )
    return undefined;
  // Coverage says whether every rule was checked; it must agree with the rules left unchecked.
  if (
    unchecked.length === 0
      ? coverage !== "complete"
      : coverage !== "partial" && coverage !== "failed"
  )
    return undefined;
  const offsets = utf16Offsets(text);
  const found: RulesFound["found"] = [];
  for (const span of spans) {
    if (
      !isRecord(span) ||
      typeof span.rule !== "string" ||
      !bound.has(span.rule)
    )
      return undefined;
    const { start, end } = span;
    if (
      typeof start !== "number" ||
      typeof end !== "number" ||
      !Number.isInteger(start) ||
      !Number.isInteger(end) ||
      start < 0 ||
      end <= start ||
      end >= offsets.length
    )
      return undefined;
    found.push({ from: offsets[start]!, to: offsets[end]!, rule: span.rule });
  }
  const named = (value: unknown): value is string =>
    typeof value === "string" && bound.has(value);
  if (!matches.every(named)) return undefined;
  const skipped: RulesFound["unchecked"] = [];
  for (const one of unchecked) {
    if (!isRecord(one) || !named(one.rule) || typeof one.reason !== "string")
      return undefined;
    skipped.push({ rule: one.rule, reason: one.reason });
  }
  return { matches, found, unchecked: skipped };
}

function readRules(
  result: Result<unknown>,
  texts: readonly string[],
  bound: readonly RuleView[],
): RulesAsked {
  if (!result.ok) return { ok: false, code: result.code };
  const body = result.value;
  const names = new Set(bound.map((rule) => rule.name));
  const answered =
    isRecord(body) && Array.isArray(body.parts) ? body.parts : [];
  const parts = texts.map((text, at) =>
    partOf(text, answered[at], `p${at}`, names),
  );
  if (
    !isRecord(body) ||
    typeof body.revision !== "string" ||
    answered.length !== texts.length ||
    parts.some((part) => part === undefined)
  )
    return { ok: false, code: "invalid-response" };
  return { ok: true, revision: body.revision, parts: parts as RulesFound[] };
}

const failureWords: Record<Failure, string> = {
  cancelled: "the request was cancelled",
  timeout: "the listener did not answer in time",
  offline: "could not reach the listener",
  http: "the listener answered with an error",
  "invalid-response": "the listener gave an answer vSR could not read",
  unsupported: "the listener does not take this question",
  "queue-full": "too many listener requests are waiting",
};

// Where the rules last read from a listener are kept across restarts.
export interface RuleStore {
  get(key: string): Promise<unknown>;
  set(key: string, value: ListenerRules): Promise<void>;
  remove(key: string): Promise<void>;
}

export class Listener {
  rules: ListenerRules = NO_RULES;
  private askedAt = -Infinity;
  private reading: Promise<void> | undefined;
  private base: Values;
  private readonly defaults: Values;
  private readonly configured: ReadonlySet<Field>;
  private readonly settings: Settings;
  private readonly runtime: Pick<
    RuntimeClient,
    "listing" | "rules" | "listenerKey"
  >;
  // The fields the person set in the plugin options or changed in the decision view: those stay theirs whatever the
  // listener says.
  private pinned: Set<Field>;
  private readonly store: RuleStore | undefined;
  private restored = false;
  private readonly now: () => number;
  // What the listener's profile changes from the person's own settings, said once in a toast when it is new.
  note = "";
  // Told when the note changes, so the decision view shows it.
  noted: () => void = () => {};
  // Whether the runtime is a Router listener that publishes a vsr profile, which /vsr-config can follow again.
  routed = false;
  constructor(
    settings: Settings,
    runtime: Pick<RuntimeClient, "listing" | "rules" | "listenerKey">,
    pinned: Iterable<Field>,
    store?: RuleStore,
    now: () => number = Date.now,
  ) {
    this.settings = settings;
    this.runtime = runtime;
    this.configured = new Set(pinned);
    this.pinned = new Set(pinned);
    this.store = store;
    this.now = now;
    this.defaults = valuesOf(settings);
    this.base = valuesOf(settings);
  }
  // The person set these: the profile leaves them alone from now on.
  pin(values: Partial<Values>): void {
    for (const field of FIELDS) {
      const value = values[field];
      if (value === undefined) continue;
      this.pinned.add(field);
      (this.base as Record<Field, unknown>)[field] = value;
    }
    assign(this.settings, { ...valuesOf(this.settings), ...this.pick(values) });
  }
  private pick(values: Partial<Values>): Partial<Values> {
    return Object.fromEntries(
      FIELDS.filter((field) => values[field] !== undefined).map((field) => [
        field,
        values[field],
      ]),
    );
  }
  // Another runtime: the rules read from the previous one no longer apply, and the next prompt reads the new listing.
  forget(): void {
    this.rules = NO_RULES;
    this.restored = false;
    this.askedAt = -Infinity;
  }
  // Hands what was changed in the decision view back to the listener; what the plugin options name stays.
  async unpin(): Promise<void> {
    this.pinned = new Set(this.configured);
    this.base = { ...this.defaults };
    await this.follow();
  }
  // Reads the listing and applies it. A listing that does not answer keeps the profile and rules read before, so an
  // outage never drops a required rule; one that does not exist (HTTP 404, a plain decision runtime) sets none.
  async follow(timeoutMs = PROFILE_MS, signal?: AbortSignal): Promise<void> {
    const reading = this.read(timeoutMs, signal);
    this.reading = reading;
    try {
      await reading;
    } finally {
      if (this.reading === reading) this.reading = undefined;
    }
  }
  // Waits for a read in flight, so the first check of a session sees the rules its prompt started reading.
  async ready(): Promise<void> {
    await this.reading?.catch(() => {});
  }
  private async read(timeoutMs: number, signal?: AbortSignal): Promise<void> {
    this.askedAt = this.now();
    const result = await this.runtime.listing(timeoutMs, signal);
    if (!result.ok && !(result.code === "http" && result.status === 404))
      return await this.restore();
    const listing = result.ok ? result.value : {};
    this.routed = isRecord(listing) && isRecord(listing.vsr);
    const profile = profileOf(listing);
    const changed: Partial<Values> = {};
    const next = { ...this.base };
    for (const field of FIELDS) {
      const value = profile[field];
      if (this.pinned.has(field) || value === undefined) continue;
      (next as Record<Field, unknown>)[field] = value;
      if (value !== this.base[field])
        (changed as Record<Field, unknown>)[field] = value;
    }
    assign(this.settings, next);
    const note = noteOf(changed);
    if (note !== this.note) {
      this.note = note;
      this.noted();
    }
    const rules = rulesOf(listing);
    if (!rules) return await this.restore();
    this.rules = rules;
    this.restored = true;
    // A listing that binds no rules forgets the kept ones; another key may be granted other rules.
    try {
      const key = this.runtime.listenerKey();
      if (rules.revision === "") await this.store?.remove(key);
      else await this.store?.set(key, rules);
    } catch {
      /* The rules still apply to this run. */
    }
  }
  // Before this run has read the listener's rules, the ones kept from an earlier run apply: a restart during an
  // outage never drops a required rule.
  private async restore(): Promise<void> {
    if (this.restored || !this.store) return;
    this.restored = true;
    try {
      const kept = rulesOf({
        vsr: { rules: await this.store.get(this.runtime.listenerKey()) },
      });
      if (kept && this.rules === NO_RULES) this.rules = kept;
    } catch {
      /* Nothing kept. */
    }
  }
  // At a prompt, a minute or more after the last read: a change in the Dashboard reaches a running session.
  followLater(): void {
    if (this.now() - this.askedAt > PROFILE_REFRESH_MS)
      void this.follow().catch(() => {});
  }
  // Asks the rules bound to one app about each text. An answer for rules of another revision, or one naming a rule
  // this session has not read, means the rules changed: the listing is read again and the rules asked once more
  // within the time left, so the change applies to this call instead of leaving every rule unchecked.
  async ask(
    app: "guard" | "redact",
    texts: readonly string[],
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<RulesAsked> {
    const started = this.now();
    const once = async (left: number) =>
      readRules(
        await this.runtime.rules(
          app,
          texts.map((text, at) => ({ id: `p${at}`, text })),
          left,
          signal,
        ),
        texts,
        this.rules[app],
      );
    const asked = await once(timeoutMs);
    if (
      asked.ok
        ? asked.revision === this.rules.revision
        : asked.code !== "invalid-response"
    )
      return asked;
    const revision = this.rules.revision;
    const left = () => timeoutMs - (this.now() - started);
    if (left() > 0) await this.follow(Math.min(PROFILE_MS, left()), signal);
    if (
      this.rules.revision === revision ||
      left() <= 0 ||
      this.rules[app].length === 0
    ) {
      this.askedAt = -Infinity;
      return asked;
    }
    return await once(left());
  }
  // The bound rules a text was not checked against, with why: every one when the call failed or answered for rules of
  // another revision, whose required flags may differ.
  unchecked(
    app: "guard" | "redact",
    asked: RulesAsked,
    part: number,
  ): Unchecked[] {
    const bound = this.rules[app];
    if (!asked.ok)
      return bound.map((rule) => ({ rule, reason: failureWords[asked.code] }));
    if (asked.revision !== this.rules.revision)
      return bound.map((rule) => ({
        rule,
        reason: "the listener's rules changed; vSR reads them again",
      }));
    const named = new Map(
      (asked.parts[part]?.unchecked ?? []).map((one) => [one.rule, one.reason]),
    );
    return bound.flatMap((rule) =>
      named.has(rule.name) ? [{ rule, reason: named.get(rule.name)! }] : [],
    );
  }
  // What a call found that this session can act on: nothing from a failed call or an answer of another revision.
  finds(asked: RulesAsked, part: number): RulesFound | undefined {
    return asked.ok && asked.revision === this.rules.revision
      ? asked.parts[part]
      : undefined;
  }
}

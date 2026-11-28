import { subjectOf } from "./decisions.ts";
import type { Decision, Estimate } from "./decisions.ts";
import { ROUTE_QUESTIONS } from "./questions.ts";
import type { RoutingSettings } from "./routing-settings.ts";
import { fullModel, routeScores } from "./routing-policy.ts";
import type { ModelRef, Score } from "./routing-policy.ts";
import type { RuntimeClient } from "./runtime.ts";
import type { Settings } from "./settings.ts";

// Your own turns, routed like subagents (`routing.turns`), ported from the Claude Code plugin's hooks/turns.ts: the
// router reads the prompt you send and picks the tier and effort the turn runs on. A turn keeps its model through its
// tool calls, and a move between turns is weighed against the prompt cache it gives up.

// A reply this short follows the turn before it ("yes, do it"): the router reads it with that prompt, and it never moves
// your turns to a smaller model.
export const SHORT_REPLY_WORDS = 12;
// The prompt cache's default lifetime: a conversation idle longer than this has nothing warm to lose by a move.
export const CACHE_WARM_MS = 5 * 60_000;
// A move to a smaller model waits while writing the conversation to its cache costs more than this many turns like
// the last one would save there.
export const PAYBACK_TURNS = 3;

export const wordsIn = (text: string) => text.trim().split(/\s+/).length;

// A request's tokens and a model's price per million tokens, as OpenCode reports them.
export interface Tokens {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}
export interface Price {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}
export const costOf = (price: Price, tokens: Tokens) =>
  (tokens.input * price.input +
    tokens.output * price.output +
    tokens.cacheRead * price.cacheRead +
    tokens.cacheWrite * price.cacheWrite) /
  1_000_000;
// What routing came to, as the Claude Code plugin's routing row: the same tokens at list price on the model that ran
// them and on `on`, the session's model (US dollars).
export function estimateOf(
  prices: Map<string, Price>,
  ran: string,
  on: string,
  tokens: Tokens,
): Estimate | undefined {
  const actual = prices.get(ran),
    baseline = prices.get(on);
  if (!actual || !baseline) return undefined;
  return {
    actual: costOf(actual, tokens),
    baseline: costOf(baseline, tokens),
    on,
  };
}
// What moving a conversation of `tokens` to a model adds to its first request there: the whole conversation written to
// its cache instead of read from it.
export const rereadCost = (price: Price, tokens: number) =>
  (tokens * Math.max(0, price.cacheWrite - price.cacheRead)) / 1_000_000;

export interface Reread {
  tokens: number;
  cost: number;
  saving: number;
}
// What the router decides for a turn: move to another tier, keep the one the turns run on and say why, or stay without
// a word; a keep or a stay can still set the turn's effort. Ranks are the tiers, 0 lightest; the person's own model,
// when it is no tier, ranks above them all.
export type TurnPlan =
  | { kind: "move"; rank: number; effort?: string; escalated: boolean }
  | {
      kind: "keep";
      reason: "low-confidence" | "quality" | "follow-up" | "cache";
      would: number;
      minConfidence: number;
      reread?: Reread;
      effort?: string;
    }
  | { kind: "stay"; effort?: string };

export interface TurnFacts {
  complexity: Score;
  steps?: Score;
  words: number;
  // The rank of the model you picked, and of the tier the turns run on now (undefined: your own model).
  own: number;
  now: number | undefined;
  escalated: boolean;
  // The conversation the cache holds warm, in tokens, and the last turn's tokens; undefined when nothing is warm.
  context?: number;
  last?: Tokens;
  // The effort the last request ran at.
  lastEffort?: string;
  // Each tier's price, and your own model's; undefined where OpenCode knows none.
  prices: (rank: number) => Price | undefined;
}

const rankOf = (score: number, routing: RoutingSettings) =>
  score < routing.easyBelow ? 0 : score < routing.hardFrom ? 1 : 2;

// The effort for a tier from the steps answer, as a subagent's: the middle rung when the answer is unsure.
function effortAt(
  routing: RoutingSettings,
  rank: number,
  steps: Score | undefined,
): { effort?: string; sure: boolean } {
  const efforts = routing.tiers[rank]?.efforts ?? [];
  if (!efforts.length) return { sure: false };
  const sure =
    steps !== undefined && steps.confidence >= routing.stepsMinConfidence;
  const index = !sure
    ? 1
    : steps.score < routing.stepsLow
      ? 0
      : steps.score < routing.stepsHigh
        ? 1
        : 2;
  return {
    effort: efforts[Math.min(index, efforts.length - 1)],
    sure,
  };
}
// One effort level above `effort` on the tier's ladder, or its top.
function raised(
  routing: RoutingSettings,
  rank: number,
  effort: string | undefined,
  last: string | undefined,
): string | undefined {
  const efforts = routing.tiers[rank]?.efforts ?? [];
  if (!efforts.length) return undefined;
  const at = Math.max(
    efforts.indexOf(effort ?? ""),
    efforts.indexOf(last ?? ""),
  );
  return efforts[Math.min(efforts.length - 1, Math.max(0, at) + 1)];
}

export function planTurn(facts: TurnFacts, routing: RoutingSettings): TurnPlan {
  const nowRank = facts.now ?? facts.own;
  const routed = rankOf(facts.complexity.score, routing);
  // The lightest tier runs your turns only with the save goal, or when your own model is on it already: a coding
  // agent's own loop needs more than a subagent's errand.
  const floorRank = routing.goal === "save" ? 0 : Math.min(1, facts.own);
  const target = Math.min(2, Math.max(routed, floorRank));
  if (facts.escalated) {
    // A pushback climbs one step from where the last turn ran, and one effort level above what it ran at; above the
    // strong tier there is no step.
    const base = Math.max(nowRank, target);
    if (base > 2) return { kind: "stay" };
    const up = Math.min(2, base + 1);
    const effort = raised(
      routing,
      up,
      effortAt(routing, up, facts.steps).effort,
      facts.lastEffort,
    );
    return {
      kind: "move",
      rank: up,
      escalated: true,
      ...(effort ? { effort } : {}),
    };
  }
  // A turn that keeps its tier still starts at the effort its prompt needs, unless the answer is unsure or the prompt
  // is a short reply, which goes on as the turn before it did. On the strong tier the effort never drops.
  let effort: string | undefined;
  if (nowRank <= 2 && facts.words > SHORT_REPLY_WORDS) {
    const steady = effortAt(routing, nowRank, facts.steps);
    const efforts = routing.tiers[nowRank]?.efforts ?? [];
    if (steady.sure && steady.effort)
      effort =
        nowRank === 2 &&
        efforts.indexOf(facts.lastEffort ?? "") > efforts.indexOf(steady.effort)
          ? facts.lastEffort
          : steady.effort;
  }
  const keep = (
    reason: Extract<TurnPlan, { kind: "keep" }>["reason"],
    minConfidence: number,
    reread?: Reread,
  ): TurnPlan => ({
    kind: "keep",
    reason,
    would: target,
    minConfidence,
    ...(reread ? { reread } : {}),
    ...(effort ? { effort } : {}),
  });
  if (target === nowRank)
    return effort ? { kind: "stay", effort } : { kind: "stay" };
  if (facts.complexity.confidence < routing.minConfidence)
    return keep("low-confidence", routing.minConfidence);
  const picked = effortAt(routing, target, facts.steps).effort;
  const move: TurnPlan = {
    kind: "move",
    rank: target,
    escalated: false,
    ...(picked ? { effort: picked } : {}),
  };
  if (target > nowRank) return move;
  if (target < facts.own) {
    if (routing.goal === "quality")
      return keep("quality", routing.minConfidence);
    const smallerLine =
      routing.goal === "save"
        ? routing.minConfidence
        : Math.max(routing.minConfidence, routing.downgradeConfidence);
    if (facts.complexity.confidence < smallerLine)
      return keep("low-confidence", smallerLine);
  }
  if (facts.words <= SHORT_REPLY_WORDS)
    return keep("follow-up", routing.minConfidence);
  // A move to a smaller model writes the whole conversation to its cache once: it waits while that costs more than
  // PAYBACK_TURNS turns like the last one would save there.
  if (facts.context !== undefined && facts.last) {
    const to = facts.prices(target),
      from = facts.prices(nowRank);
    if (to && from) {
      const cost = rereadCost(to, facts.context);
      const saving =
        (costOf(from, facts.last) - costOf(to, facts.last)) * PAYBACK_TURNS;
      if (cost > saving)
        return keep("cache", routing.minConfidence, {
          tokens: facts.context,
          cost,
          saving,
        });
    }
  }
  return move;
}

// The session a turn runs in, as the router keeps it: the model you picked, the tier your turns run on now, and what
// the cache rule needs from the turns before.
interface TurnState {
  own: ModelRef;
  now?: number;
  // The model vSR last switched the session to, so its own switch is not taken for yours.
  expected?: string;
  // The model you picked yourself while your turns were routed: they stay on it until you pick routed again.
  paused?: string;
  // The last prompt that was not a short reply, read with one.
  task?: string;
  // The effort the turns run at on the tier they were moved to.
  effort?: string;
  // The conversation's tokens at the last request and when it ran; this turn's tokens so far, and the last turn's.
  context?: { tokens: number; at: number };
  usage?: Tokens;
  last?: Tokens;
  // Whether this turn is counted in the spend yet.
  counted?: boolean;
}
// What your turns the router moved off your model cost, and what the same tokens would have on it.
export interface Spend {
  actual: number;
  baseline: number;
  turns: number;
  on: string;
}
export interface TurnsView {
  on: boolean;
  // Where your turns run now, when the router moved them off your model; the model you picked, when that paused it.
  model?: string;
  paused?: string;
  spend?: Spend;
}
// How long a read of the price table is reused while turns are priced.
const PRICES_MS = 60_000;
export interface TurnsClient {
  modelOf(sessionID: string): Promise<ModelRef | undefined>;
  switchModel(sessionID: string, model: ModelRef): Promise<void>;
  prices(): Promise<Map<string, Price>>;
}
type RecordDecision = (decision: Omit<Decision, "id" | "at">) => string;

export class Turns {
  private readonly sessions = new Map<string, TurnState>();
  // Kept apart from the routing state, which a pause or turning routing off forgets.
  private readonly spent = new Map<string, Spend>();
  private priceRead?: { at: number; prices: Promise<Map<string, Price>> };
  private readonly settings: Settings;
  private readonly runtime: Pick<RuntimeClient, "decide">;
  private readonly record: RecordDecision;
  private readonly client: TurnsClient;
  private readonly now: () => number;
  // Told when what the view shows of your turns changes.
  changed: (sessionID: string) => void = () => {};
  constructor(
    settings: Settings,
    runtime: Pick<RuntimeClient, "decide">,
    record: RecordDecision,
    client: TurnsClient,
    now: () => number = Date.now,
  ) {
    this.settings = settings;
    this.runtime = runtime;
    this.record = record;
    this.client = client;
    this.now = now;
  }
  private active(): boolean {
    return (
      this.settings.apps.router &&
      this.settings.routing.turns &&
      this.settings.routing.tiers.length === 3
    );
  }
  view(sessionID: string): TurnsView {
    const state = this.sessions.get(sessionID);
    const tier =
      state?.now === undefined
        ? undefined
        : this.settings.routing.tiers[state.now]?.model;
    return {
      on: this.active(),
      ...(tier && !state?.paused
        ? {
            model: tier + (state?.effort ? `#${state.effort}` : ""),
          }
        : {}),
      ...(state?.paused ? { paused: state.paused } : {}),
      ...(this.spent.has(sessionID)
        ? { spend: { ...this.spent.get(sessionID)! } }
        : {}),
    };
  }
  private pricesNow(): Promise<Map<string, Price>> {
    if (!this.priceRead || this.now() - this.priceRead.at > PRICES_MS)
      this.priceRead = { at: this.now(), prices: this.client.prices() };
    return this.priceRead.prices;
  }
  // A request on a tier the router moved your turn to, priced there and on your own model.
  private async spend(
    sessionID: string,
    state: TurnState,
    rank: number,
    tokens: Tokens,
  ): Promise<void> {
    const ran = this.settings.routing.tiers[rank]?.model;
    if (!ran) return;
    // Counted before the price read, so a step that lands meanwhile does not count the turn again.
    const first = !state.counted;
    state.counted = true;
    const on = fullModel({ ...state.own, variant: undefined });
    const prices = await this.pricesNow().catch(() => undefined);
    const estimate = prices && estimateOf(prices, ran, on, tokens);
    if (!estimate || !this.sessions.has(sessionID)) return;
    const was = this.spent.get(sessionID);
    this.spent.set(sessionID, {
      actual: (was?.actual ?? 0) + estimate.actual,
      baseline: (was?.baseline ?? 0) + estimate.baseline,
      turns: (was?.turns ?? 0) + (first ? 1 : 0),
      on,
    });
    while (this.spent.size > this.settings.maxSessions)
      this.spent.delete(this.spent.keys().next().value!);
    this.changed(sessionID);
  }
  private async switchTo(
    sessionID: string,
    state: TurnState,
    model: ModelRef,
  ): Promise<void> {
    state.expected = fullModel(model);
    await this.client.switchModel(sessionID, model);
  }
  // A model selected in the session: vSR's own switch, or one you picked, which pauses the routing of your turns.
  picked(sessionID: string, model: ModelRef): void {
    const state = this.sessions.get(sessionID);
    if (!state) return;
    const name = fullModel(model);
    if (name === state.expected) {
      state.expected = undefined;
      return;
    }
    if (state.now === undefined && name === fullModel(state.own)) return;
    state.own = { ...model };
    state.now = undefined;
    state.expected = undefined;
    if (this.active()) {
      state.paused = name;
      this.changed(sessionID);
    }
  }
  // A request's tokens: the conversation the cache now holds, and this turn's usage so far. `reasoning` is billed as
  // output but not kept in the conversation.
  stepped(sessionID: string, tokens: Tokens, reasoning = 0): void {
    const state = this.sessions.get(sessionID);
    if (!state) return;
    if (state.now !== undefined && !state.paused)
      void this.spend(sessionID, state, state.now, {
        ...tokens,
        output: tokens.output + reasoning,
      });
    state.context = {
      tokens:
        tokens.input + tokens.cacheRead + tokens.cacheWrite + tokens.output,
      at: this.now(),
    };
    const sum = state.usage ?? {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
    };
    state.usage = {
      input: sum.input + tokens.input,
      output: sum.output + tokens.output,
      cacheRead: sum.cacheRead + tokens.cacheRead,
      cacheWrite: sum.cacheWrite + tokens.cacheWrite,
    };
  }
  remove(sessionID: string): void {
    this.sessions.delete(sessionID);
    this.spent.delete(sessionID);
  }
  // A prompt you sent: the router reads it and moves the turn before it starts. `pushback` says whether the prompt
  // pushed back on the last answer; the turn climbs then, instead of the next subagent.
  async prompt(
    sessionID: string,
    messageID: string,
    text: string,
    pushback: () => Promise<boolean>,
  ): Promise<void> {
    let state = this.sessions.get(sessionID);
    if (!this.active()) {
      // Off again: your turns go back to the model you picked.
      if (state?.now !== undefined)
        await this.switchTo(sessionID, state, state.own);
      this.sessions.delete(sessionID);
      if (state) this.changed(sessionID);
      return;
    }
    if (!state) {
      const own = await this.client.modelOf(sessionID);
      if (!own) return;
      state = { own };
      this.sessions.set(sessionID, state);
      while (this.sessions.size > this.settings.maxSessions)
        this.sessions.delete(this.sessions.keys().next().value!);
    }
    if (state.paused) return;
    state.last = state.usage ?? state.last;
    state.usage = undefined;
    state.counted = false;
    // A turn you did not type (a command, a continuation) runs where the last one did.
    if (!text.trim() || text.startsWith("/") || text.length > 32_000) return;
    const short = wordsIn(text) <= SHORT_REPLY_WORDS;
    const task = state.task;
    if (!short) state.task = text;
    const routing = this.settings.routing;
    const [asked, escalated] = await Promise.all([
      this.runtime.decide(
        short && task ? `${task}\n\n${text}` : text,
        ROUTE_QUESTIONS,
        routing.deadlineMs,
      ),
      this.settings.apps.escalate ? pushback() : Promise.resolve(false),
    ]);
    const anchor = {
      sessionID,
      turnID: messageID,
      app: "router" as const,
      scope: "turn" as const,
      subject: subjectOf(text),
      elapsedMs: asked.elapsedMs,
    };
    const scores = asked.ok ? routeScores(asked.value) : undefined;
    if (!scores) {
      this.record({
        ...anchor,
        outcome: "kept",
        reason: asked.ok ? "invalid-response" : asked.code,
      });
      return;
    }
    const tiers = routing.tiers;
    const ownName = fullModel({ ...state.own, variant: undefined });
    const ownRank = tiers.findIndex((tier) => tier.model === ownName);
    const own = ownRank < 0 ? 3 : ownRank;
    const warm =
      state.context !== undefined &&
      this.now() - state.context.at < CACHE_WARM_MS;
    const prices = warm
      ? await this.client.prices().catch(() => new Map())
      : new Map();
    const plan = planTurn(
      {
        complexity: scores.complexity,
        ...(scores.steps ? { steps: scores.steps } : {}),
        words: wordsIn(text),
        own,
        now: state.now,
        escalated,
        ...(warm ? { context: state.context!.tokens } : {}),
        ...(warm && state.last ? { last: state.last } : {}),
        ...((state.effort ?? state.own.variant)
          ? { lastEffort: state.effort ?? state.own.variant }
          : {}),
        prices: (rank) =>
          prices.get(rank === own ? ownName : (tiers[rank]?.model ?? "")),
      },
      routing,
    );
    const facts = {
      score: scores.complexity.score,
      confidence: scores.complexity.confidence,
    };
    const at = state.now ?? own;
    const refOf = (rank: number, effort?: string): ModelRef => {
      if (rank === own) return { ...state!.own };
      const [providerID, ...id] = tiers[rank]!.model.split("/");
      return {
        providerID: providerID!,
        id: id.join("/"),
        ...(effort ? { variant: effort } : {}),
      };
    };
    if (plan.kind === "move") {
      const model = refOf(plan.rank, plan.effort);
      await this.switchTo(sessionID, state, model);
      state.now = plan.rank === own ? undefined : plan.rank;
      state.effort = state.now === undefined ? undefined : plan.effort;
      this.record({
        ...anchor,
        outcome: "routed",
        reason: plan.escalated ? "pushback" : "selected",
        model: fullModel(model),
        ...facts,
      });
      this.changed(sessionID);
      return;
    }
    if (plan.kind === "keep")
      this.record({
        ...anchor,
        outcome: "kept",
        reason: plan.reason,
        model: fullModel(at <= 2 ? refOf(at) : state.own),
        ...(plan.reread ? { reread: plan.reread } : {}),
        ...facts,
      });
    // A turn that keeps its tier can still start at another effort.
    if (plan.effort && at <= 2 && at !== own && plan.effort !== state.effort) {
      await this.switchTo(sessionID, state, refOf(at, plan.effort));
      state.effort = plan.effort;
    }
  }
  // Routed again after a pause: your turns follow the router again from the model the session is on now.
  resume(sessionID?: string): void {
    for (const id of [...this.sessions.keys()])
      if (sessionID === undefined || id === sessionID) this.sessions.delete(id);
  }
}

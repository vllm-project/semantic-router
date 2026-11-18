import assert from "node:assert/strict";
import { test } from "node:test";
import {
  APP_LOOK,
  DecisionView,
  brief,
  clip,
  gauge,
  modelName,
  shortSubject,
  subjectLine,
  summary,
  THEME_TOKENS,
  tokenCount,
  toneOf,
  answersOf,
  details,
  why,
} from "../src/view.ts";
import type { ViewSnapshot, ViewState, ViewService } from "../src/view.ts";
import { parseSettings } from "../src/settings.ts";
import { DecisionStore } from "../src/decisions.ts";
import { knownPrices } from "../src/prices.ts";
const config = parseSettings({});
const snapshot = (
  sessionID: string,
  revision = 1,
  instanceID = "instance",
): ViewSnapshot => ({
  ...new DecisionStore(2, 5).snapshot(sessionID),
  answerable: [],
  instanceID,
  revision,
  apps: { ...config.apps },
  waits: { ...config.waits },
  controls: { revision: 0, persistence: "durable", goal: "balanced" },
  calibration: { since: 0, remind: false },
  runtime: { reachable: null, retryAt: 0, models: [], source: "options" },
});
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
function service(extra: Partial<ViewService> = {}): ViewService {
  return {
    snapshot: async (id) => snapshot(id),
    controls: async () => ({ saved: false, reason: "storage" }),
    feedback: async () => {},
    diagnose: async () => ({ source: "options", models: [], probes: [] }),
    ...extra,
  };
}
test("late snapshots cannot leak a prior session and a new server epoch replaces old decisions", async () => {
  const a = deferred<ViewSnapshot>(),
    b = deferred<ViewSnapshot>();
  let latest: ViewState | undefined;
  const view = new DecisionView(
    service({ snapshot: (id) => (id === "A" ? a.promise : b.promise) }),
    (state) => {
      latest = state;
    },
  );
  view.select("A");
  view.select("B");
  b.resolve(snapshot("B", 5));
  await b.promise;
  await Promise.resolve();
  a.resolve(snapshot("A", 100));
  await a.promise;
  await Promise.resolve();
  assert.equal(latest?.snapshot?.sessionID, "B");
  view.close();
  let revision = 5,
    epoch = "old";
  const refreshed = new DecisionView(
    service({ snapshot: async (id) => snapshot(id, revision, epoch) }),
    (state) => {
      latest = state;
    },
  );
  refreshed.select("B");
  await refreshed.refresh();
  revision = 1;
  await refreshed.refresh();
  assert.equal(latest?.snapshot?.revision, 5);
  epoch = "new";
  await refreshed.refresh();
  assert.equal(latest?.snapshot?.instanceID, "new");
  assert.equal(latest?.snapshot?.revision, 1);
  refreshed.close();
});
test("failed controls stay unchanged and stale/foreign events do not trigger needless refreshes", async () => {
  let calls = 0;
  let latest: ViewState | undefined;
  const view = new DecisionView(
    service({
      snapshot: async (id) => {
        calls++;
        return snapshot(id);
      },
    }),
    (state) => {
      latest = state;
    },
  );
  view.select("A");
  await view.refresh();
  const before = calls;
  view.hint({ instanceID: "instance", revision: 1, sessionID: "A" });
  view.hint({ instanceID: "instance", revision: 100, sessionID: "B" });
  view.hint({ revision: NaN });
  assert.equal(calls, before);
  assert.equal(await view.change({ apps: { check: true } }), false);
  assert.equal(latest?.snapshot?.apps.check, false);
  assert.ok(latest?.error?.includes("not saved"));
  view.close();
});
test("feedback remains bound to its original session during selection changes", async () => {
  const pending = deferred<unknown>();
  const seen: string[] = [];
  let latest: ViewState | undefined;
  const decision = {
    id: "decision",
    sessionID: "A",
    app: "guard" as const,
    outcome: "held" as const,
    at: 1,
  };
  const view = new DecisionView(
    service({
      snapshot: async (id) => ({
        ...snapshot(id),
        decisions: id === "A" ? [decision] : [],
      }),
      feedback: async (id) => {
        seen.push(id);
        return pending.promise;
      },
    }),
    (state) => {
      latest = state;
    },
  );
  view.select("A");
  await view.refresh();
  const save = view.feedback(decision, "wrong");
  view.select("B");
  await view.refresh();
  pending.resolve({ saved: true });
  await save;
  assert.deepEqual(seen, ["A"]);
  assert.equal(latest?.snapshot?.sessionID, "B");
  assert.equal(await view.feedback(decision, "right"), false);
  view.close();
});
test("a subagent's decision listed in its parent's view is answered in the subagent's session", async () => {
  const seen: string[] = [];
  const child = {
    id: "read",
    sessionID: "child",
    app: "shield" as const,
    outcome: "allowed" as const,
    at: 2,
  };
  const view = new DecisionView(
    service({
      snapshot: async (id) => ({ ...snapshot(id), decisions: [child] }),
      feedback: async (id, decisionID, feedback) => {
        seen.push(`${id}:${decisionID}:${feedback}`);
      },
    }),
    () => {},
  );
  view.select("parent");
  await view.refresh();
  assert.equal(await view.feedback(child, "right"), true);
  assert.deepEqual(seen, ["child:read:right"]);
  // A decision the view does not list is not sent.
  assert.equal(await view.feedback({ ...child, id: "other" }, "right"), false);
  view.close();
});
test("the panel offers a route's model fit, and right or wrong for other apps", () => {
  assert.deepEqual(
    answersOf({ app: "router" }).map((choice) => choice.value),
    ["right-model", "needed-stronger", "could-be-lighter"],
  );
  assert.deepEqual(
    answersOf({ app: "shield" }).map((choice) => choice.value),
    ["right", "wrong"],
  );
});
test("usage summaries deduplicate children and suppress estimates with unknown pricing", () => {
  const base = {
    id: "one",
    sessionID: "A",
    app: "router" as const,
    outcome: "routed" as const,
    at: 1,
    observedModel: "fixture/light",
    childSessionID: "child",
    usage: { input: 10, output: 2, cachedInput: 3, reasoning: 1, cost: 0.01 },
    costKnown: true,
  };
  const result = summary([
    base,
    base,
    { ...base, id: "two" },
    {
      ...base,
      id: "three",
      childSessionID: "unpriced",
      costKnown: false,
      usage: { ...base.usage, cost: 999 },
    },
  ]);
  assert.equal(result.total, 3);
  assert.equal(result.children, 2);
  assert.equal(result.input, 20);
  assert.equal(result.cost, 0.01);
  assert.equal(result.priced, 1);
  assert.equal(result.unpriced, 1);
  assert.equal(summary([{ ...base, costKnown: false }]).priced, 0);
});
test("known prices require an explicit complete finite rate table, including zero-priced models", () => {
  assert.equal(knownPrices([]), false);
  assert.equal(knownPrices(undefined), false);
  assert.equal(knownPrices([{ input: 1 }]), false);
  assert.equal(knownPrices([{ input: 0, output: 0 }]), true);
  assert.equal(
    knownPrices([{ input: 1, output: 2, cache: { read: 0.1, write: 0.2 } }]),
    true,
  );
  assert.equal(knownPrices([{ input: Infinity, output: 2 }]), false);
});

test("late failed controls and feedback cannot mark a newly selected session stale", async () => {
  let reject!: (error: Error) => void;
  const pending = new Promise<never>((_resolve, fail) => {
    reject = fail;
  });
  let latest: ViewState | undefined;
  const view = new DecisionView(
    service({ controls: () => pending }),
    (state) => {
      latest = state;
    },
  );
  view.select("A");
  await view.refresh();
  const save = view.change({ apps: { check: true } });
  view.select("B");
  await view.refresh();
  reject(new Error("disconnected"));
  await save;
  assert.equal(latest?.snapshot?.sessionID, "B");
  assert.equal(latest?.error, undefined);
  assert.equal(latest?.busy, false);
  view.close();
});

test("a decision is shown by its subject, and by its call or turn only without one", () => {
  const decision = {
    id: "d1",
    toolCallID: "call_7",
    turnID: "turn_3",
  } as Parameters<typeof subjectLine>[0];
  assert.equal(
    subjectLine({ ...decision, subject: "git push --force origin main" }),
    "git push --force origin main",
  );
  assert.equal(subjectLine(decision), "call_7");
  assert.equal(subjectLine({ ...decision, toolCallID: undefined }), "turn_3");
  assert.equal(
    subjectLine({ ...decision, toolCallID: undefined, turnID: undefined }),
    "d1",
  );
});

test("the panel colors what needs a look, what went through in part, and what an app changed", () => {
  for (const outcome of ["held", "declined", "withheld", "unverified"] as const)
    assert.equal(toneOf({ app: "guard", outcome }), "error", outcome);
  for (const outcome of ["partial", "skipped"] as const)
    assert.equal(toneOf({ app: "redact", outcome }), "warning", outcome);
  for (const outcome of ["allowed", "verified", "kept"] as const)
    assert.equal(toneOf({ app: "check", outcome }), "muted", outcome);
  assert.equal(toneOf({ app: "redact", outcome: "redacted" }), "purple");
  assert.equal(toneOf({ app: "router", outcome: "routed" }), "info");
  for (const look of Object.values(APP_LOOK))
    assert.ok(THEME_TOKENS[look.tone].length > 0, look.glyph);
});

test("a gauge marks the decision's own thresholds on the scale its app scores on", () => {
  const text = (found: ReturnType<typeof gauge>) =>
    found?.parts.map((part) => part.text).join("");
  const guard = gauge({ app: "guard", score: 0.82, thresholds: [0.9] }, 10);
  assert.equal(text(guard), "████████░┃");
  assert.equal(guard?.words, "0.82 < 0.90");
  assert.equal(
    gauge({ app: "redact", score: 1, thresholds: [0.7] }, 10)?.words,
    "1.00 ≥ 0.70",
  );
  // The router scores a level from 0 to 2 against two lines.
  const router = gauge(
    { app: "router", score: 0.22, thresholds: [0.6, 1.4, 0.5] },
    10,
  );
  assert.equal(text(router), "█░░┃░░░┃░░");
  assert.equal(
    router?.words,
    "level 0.22 · light below 0.60, strong from 1.40",
  );
  // Escalate holds its larger score against that score's own line.
  assert.equal(
    gauge({
      app: "escalate",
      score: 0.7,
      features: [0.2, 0.7],
      thresholds: [0.6, 0.8],
    })?.words,
    "0.70 < 0.80",
  );
  assert.equal(gauge({ app: "check", score: 0.3 }, 4)?.words, "0.30");
  assert.equal(gauge({ app: "check" }), undefined);
});

test("one line says what redact hid, where the router sent the task, or the score", () => {
  assert.equal(
    brief({ app: "redact", counts: { email: 1, phone: 2 } }),
    "email ×1 · phone ×2",
  );
  assert.equal(
    brief({
      app: "router",
      score: 0.2,
      model: "deltallm/Qwen/Qwen3.8-27B-nothink",
    }),
    "→ Qwen3.8-27B-nothink",
  );
  assert.equal(
    brief({ app: "guard", score: 0.82, thresholds: [0.9] }),
    "0.82 < 0.90",
  );
  assert.equal(clip("read customer.txt", 8), "read cu…");
  assert.equal(clip("short", 8), "short");
  const subject = (text: string) =>
    shortSubject({
      id: "d",
      sessionID: "s",
      app: "guard",
      outcome: "allowed",
      at: 0,
      subject: text,
    });
  assert.equal(
    subject("read /home/kuntai/vsr-try/customer.txt"),
    "read …/customer.txt",
  );
  assert.equal(subject("rm -rf ~/vsr-try/*"), "rm -rf …/*");
  assert.equal(subject("列出 /home/a/b 的檔案"), "列出 …/b 的檔案");
  assert.equal(subject("git status"), "git status");
});

test("the panel draws models and token counts compactly", () => {
  assert.equal(
    modelName("deltallm/Qwen/Qwen3.8-27B-nothink#default"),
    "Qwen3.8-27B-nothink",
  );
  assert.equal(modelName("fixture"), "fixture");
  assert.equal(tokenCount(297), "297");
  assert.equal(tokenCount(36956), "37.0k");
});
test("a decision's card lists what the view keeps of it, and the sidebar line says why", () => {
  const base = {
    id: "d",
    sessionID: "child",
    app: "shield" as const,
    outcome: "partial" as const,
    reason: "unsupported" as const,
    at: 0,
    score: 0.01,
    thresholds: [0.9],
    subject: "read /tmp/fixture.png",
    elapsedMs: 3,
  };
  assert.equal(why(base), "unsupported · 0.01 < 0.90");
  assert.equal(why({ ...base, reason: "selected" }), "0.01 < 0.90");
  const rows = (context = { sessionID: "main", answerable: true }) =>
    Object.fromEntries(
      details(base, context).map((row) => [row.label, row.value]),
    );
  const card = rows();
  assert.equal(card.about, "read /tmp/fixture.png");
  assert.equal(card.reason, "unsupported");
  assert.equal(card.session, "a subagent's, child");
  assert.equal(card.answer, "none yet");
  assert.match(card.when!, /took 3 ms$/);
  assert.equal(card.model, undefined, "rows with nothing to say are left out");
  assert.equal(
    rows({ sessionID: "child", answerable: false }).session,
    "this session",
  );
  assert.match(
    rows({ sessionID: "child", answerable: false }).answer!,
    /^none: a rule/,
  );
  assert.equal(
    Object.fromEntries(
      details(
        { ...base, app: "escalate", features: [0.82, 0.1] },
        { sessionID: "child", feedback: "wrong", answerable: true },
      ).map((row) => [row.label, row.value]),
    ).scores,
    "pushback 0.82 · repeat 0.10",
  );
});

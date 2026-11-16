import assert from "node:assert/strict";
import { test } from "node:test";
import { DecisionView, summary } from "../src/view.ts";
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
  const view = new DecisionView(
    service({
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
  const decision = {
    id: "decision",
    sessionID: "A",
    app: "guard" as const,
    outcome: "held" as const,
    at: 1,
  };
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

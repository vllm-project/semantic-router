import assert from "node:assert/strict";
import { test } from "node:test";
import { guard } from "../src/guard-policy.ts";
import { Listener, NO_RULES, profileOf, rulesOf } from "../src/listener.ts";
import { Controls } from "../src/controls.ts";
import { inspectOutput } from "../src/output-policy.ts";
import { addedPlaceholders } from "../src/output-hooks.ts";
import { RuntimeClient } from "../src/runtime.ts";
import type { Reply, Result } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";
import type { App } from "../src/settings.ts";

const ok = (value: unknown): Result<unknown> => ({
  ok: true,
  value,
  elapsedMs: 1,
});
const rules = (revision: string, required = false) => ({
  revision,
  guard: [
    {
      name: "no_prod_db",
      description: "Touches the production database.",
      required,
    },
  ],
  redact: [
    {
      name: "customer_account",
      description: "Customer account numbers.",
      required,
    },
  ],
});
const listing = (vsr: Record<string, unknown>) => ({
  object: "list",
  data: [{ id: "vela3-lite", question_types: ["choice", "noul", "score"] }],
  vsr,
});
// A rules answer for one part, spans in code points.
const answer = (
  revision: string,
  part: Partial<{ matches: string[]; spans: unknown[]; unchecked: unknown[] }>,
) =>
  ok({
    revision,
    parts: [
      {
        id: "p0",
        matches: part.matches ?? [],
        spans: part.spans ?? [],
        unchecked: part.unchecked ?? [],
        coverage: part.unchecked?.length ? "partial" : "complete",
      },
    ],
  });
function fake(
  listed: () => Result<unknown>,
  ruled: (
    app: string,
    parts: { id: string; text: string }[],
  ) => Result<unknown>,
) {
  const calls = { listing: 0, rules: 0 };
  return {
    calls,
    listenerKey: () => "vsr.rules:fixture",
    listing: async () => {
      calls.listing++;
      return listed();
    },
    rules: async (
      app: "guard" | "redact",
      parts: { id: string; text: string }[],
    ) => {
      calls.rules++;
      return ruled(app, parts);
    },
  };
}
const quiet = {
  decide: async (
    _state: unknown,
    questions: Record<string, unknown>,
  ): Promise<Result<Reply>> => ({
    ok: true,
    elapsedMs: 1,
    value: questions.pii
      ? { answers: { pii: { noul: 0.01 } }, spans: { pii: [] } }
      : questions.kind
        ? {
            answers: {
              kind: { choice: "safe", probabilities: { safe: 0.99 } },
              review: { noul: 0.01 },
            },
            spans: {},
          }
        : {
            answers: { jailbreak: { probabilities: { jailbreak: 0.01 } } },
            spans: {},
          },
  }),
};

test("listings name models in the 4.0 data form and the 5.0 models form", async () => {
  for (const body of [
    { data: [{ id: "vela3-lite", question_types: ["score"] }] },
    {
      models: [
        { name: "vela3-lite", features: ["score"] },
        { name: "other", card: { id: "other", question_types: ["choice"] } },
      ],
    },
  ]) {
    const client = new RuntimeClient(
      parseSettings({ runtimeModel: "vela3-lite" }),
      {
        fetch: async () => Response.json(body),
      },
    );
    const found = await client.discover();
    assert.equal(found.ok, true);
    assert.deepEqual(found.ok && found.value[0], {
      id: "vela3-lite",
      questionTypes: ["score"],
    });
  }
});

test("a listing's vsr profile and rules are read field by field, and unreadable rules are not trusted", () => {
  assert.deepEqual(
    profileOf(
      listing({
        guard: false,
        redact: true,
        shield: "no",
        router_goal: "save",
        router_turns: true,
      }),
    ),
    {
      guard: false,
      redact: true,
      goal: "save",
      turns: true,
    },
    "the router's goal and turns too, as the Claude Code plugin reads them",
  );
  assert.deepEqual(
    profileOf(listing({ router_goal: "fast", router_turns: "on" })),
    {},
    "a goal it does not know and turns that are not a flag are left to the options",
  );
  assert.deepEqual(rulesOf(listing({})), NO_RULES);
  assert.deepEqual(rulesOf({ data: [] }), NO_RULES);
  assert.deepEqual(rulesOf(listing({ rules: rules("r1") })), rules("r1"));
  assert.equal(
    rulesOf(listing({ rules: { ...rules("r1"), guard: [{ name: "x" }] } })),
    undefined,
  );
  assert.equal(rulesOf(listing({ rules: "r1" })), undefined);
});

test("the profile sets the apps the person did not name, and an outage keeps the rules read before", async () => {
  const settings = parseSettings({ apps: { shield: true } });
  let listed: Result<unknown> = ok(
    listing({
      guard: false,
      shield: false,
      check: true,
      rules: rules("r1", true),
    }),
  );
  const runtime = fake(
    () => listed,
    () => ok({}),
  );
  let now = 0;
  const listener = new Listener(
    settings,
    runtime,
    new Set<App>(["shield"]),
    undefined,
    () => now,
  );
  await listener.follow();
  assert.equal(settings.apps.guard, false);
  assert.equal(settings.apps.check, true);
  assert.equal(settings.apps.shield, true, "a named app stays the person's");
  assert.equal(listener.rules.revision, "r1");

  listed = { ok: false, code: "offline", elapsedMs: 1 };
  now = 61_000;
  await listener.follow();
  assert.equal(
    listener.rules.revision,
    "r1",
    "an outage never drops a required rule",
  );
  assert.equal(settings.apps.guard, false);

  listed = { ok: false, code: "http", status: 404, elapsedMs: 1 };
  await listener.follow();
  assert.deepEqual(
    listener.rules,
    NO_RULES,
    "a plain decision runtime binds no rules",
  );
  assert.equal(settings.apps.guard, true, "and sets no profile");

  listed = ok(listing({ guard: false }));
  now = 62_000;
  listener.followLater();
  assert.equal(runtime.calls.listing, 3, "no second read within a minute");
  now = 123_000;
  listener.followLater();
  assert.equal(runtime.calls.listing, 4);
});

test("the profile sets the router's goal and turns unless the person set them, and says what it changed once", async () => {
  const settings = parseSettings({ routing: { goal: "quality" } });
  let listed: Result<unknown> = ok(
    listing({ redact: false, router_goal: "save", router_turns: true }),
  );
  const runtime = fake(
    () => listed,
    () => ok({}),
  );
  let now = 0;
  const listener = new Listener(
    settings,
    runtime,
    ["goal"],
    undefined,
    () => now,
  );
  let noted = 0;
  listener.noted = () => noted++;
  await listener.follow();
  assert.equal(
    settings.routing.goal,
    "quality",
    "a goal named in the options stays",
  );
  assert.equal(settings.routing.turns, true);
  assert.equal(settings.apps.redact, false);
  assert.equal(listener.note, "vSR follows the listener: redact off, turns on");
  assert.equal(noted, 1);
  now = 61_000;
  await listener.follow();
  assert.equal(noted, 1, "the same note is not said again");

  // A change in the decision view stays the person's, until it is handed back.
  listener.pin({ turns: false });
  now = 122_000;
  await listener.follow();
  assert.equal(settings.routing.turns, false);
  assert.equal(listener.note, "vSR follows the listener: redact off");
  assert.equal(noted, 2);
  await listener.unpin();
  assert.equal(settings.routing.turns, true);

  listed = ok(listing({}));
  await listener.follow();
  assert.equal(
    settings.routing.turns,
    false,
    "without a profile, the options again",
  );
  assert.equal(listener.note, "");
});

test("an answer for a new revision reads the listing again and asks once more", async () => {
  const settings = parseSettings({});
  let revision = "r1";
  const runtime = fake(
    () => ok(listing({ rules: rules(revision) })),
    () => answer("r2", { matches: ["no_prod_db"] }),
  );
  const listener = new Listener(settings, runtime, new Set());
  await listener.follow();
  revision = "r2";
  const asked = await listener.ask("guard", ["psql prod"], 3000);
  assert.equal(listener.rules.revision, "r2");
  assert.equal(runtime.calls.rules, 2);
  assert.deepEqual(listener.finds(asked, 0)?.matches, ["no_prod_db"]);
});

test("guard holds a command a listener rule matches or a required rule could not check", async () => {
  const settings = parseSettings({});
  let reply: Result<unknown> = answer("r1", { matches: ["no_prod_db"] });
  let required = false;
  const runtime = fake(
    () => ok(listing({ rules: rules("r1", required) })),
    () => reply,
  );
  const listener = new Listener(settings, runtime, new Set());
  await listener.follow();
  const held = await guard(
    "psql prod -c 'select 1'",
    settings,
    quiet,
    undefined,
    Date.now,
    listener,
  );
  assert.equal(held.effect, "ask");
  assert.equal(held.reason, "listener-rule");
  assert.equal(held.rule?.description, "Touches the production database.");

  reply = { ok: false, code: "http", status: 503, elapsedMs: 1 };
  const advisory = await guard(
    "ls",
    settings,
    quiet,
    undefined,
    Date.now,
    listener,
  );
  assert.equal(advisory.effect, "allow");
  assert.deepEqual(
    advisory.unchecked?.map((one) => one.rule.name),
    ["no_prod_db"],
  );

  required = true;
  await listener.follow();
  const strict = await guard(
    "ls",
    settings,
    quiet,
    undefined,
    Date.now,
    listener,
  );
  assert.equal(strict.effect, "ask");
  assert.equal(strict.reason, "listener-unchecked");
  assert.match(strict.ruleReason ?? "", /listener answered with an error/);
});

test("redact tags a rule's find by the rule's name where it and a PII span cover the same text", async () => {
  const settings = parseSettings({});
  const text = "ticket ACCT-48392017 for alice@example.test";
  const at = text.indexOf("ACCT");
  const runtime = fake(
    () => ok(listing({ rules: rules("r1") })),
    () =>
      answer("r1", {
        spans: [{ rule: "customer_account", start: at, end: at + 13 }],
      }),
  );
  const listener = new Listener(settings, runtime, new Set());
  await listener.follow();
  const model = {
    decide: async (): Promise<Result<Reply>> => ({
      ok: true,
      elapsedMs: 1,
      value: {
        answers: { pii: { noul: 0.99 } },
        spans: {
          pii: [
            {
              label: "IBAN_CODE",
              text: "ACCT-48392017",
              start: at,
              end: at + 13,
              probability: 0.99,
            },
            {
              label: "EMAIL_ADDRESS",
              text: "alice@example.test",
              start: text.indexOf("alice"),
              end: text.length,
              probability: 0.99,
            },
          ],
        },
      },
    }),
  };
  const result = await inspectOutput(
    "shell",
    text,
    settings,
    model,
    undefined,
    Date.now,
    listener,
  );
  assert.ok(
    result.text.startsWith("ticket [vsr:customer_account] for [vsr:email]"),
  );
  assert.deepEqual(result.decisions[0]?.counts, {
    customer_account: 1,
    email: 1,
  });
  assert.match(result.text, /The tags are vSR's, not part of the source/);
  assert.match(result.text, /do not try to read them another way/);
  assert.ok(
    addedPlaceholders(
      "write",
      { content: "[vsr:customer_account]" },
      new Set(result.tags),
    ),
  );
});

test("redact withholds an output a required rule could not check, and records an advisory one", async () => {
  const settings = parseSettings({});
  let required = true;
  const runtime = fake(
    () => ok(listing({ rules: rules("r1", required) })),
    () => ({ ok: false, code: "timeout", elapsedMs: 1 }),
  );
  const listener = new Listener(settings, runtime, new Set());
  await listener.follow();
  const withheld = await inspectOutput(
    "shell",
    "plain text",
    settings,
    quiet,
    undefined,
    Date.now,
    listener,
  );
  assert.equal(withheld.withheld, true);
  assert.match(
    withheld.text,
    /required rule customer_account could not check it/,
  );
  assert.equal(withheld.decisions[0]?.reason, "listener-unchecked");

  required = false;
  await listener.follow();
  const advisory = await inspectOutput(
    "shell",
    "plain text",
    settings,
    quiet,
    undefined,
    Date.now,
    listener,
  );
  assert.equal(advisory.withheld, false);
  assert.equal(advisory.text, "plain text");
  assert.equal(advisory.decisions[0]?.reason, "selected");
  assert.deepEqual(advisory.decisions[1], {
    app: "redact",
    outcome: "skipped",
    reason: "listener-unchecked",
    labels: ["customer_account"],
  });

  const big = "x".repeat((256 << 10) + 1);
  required = true;
  await listener.follow();
  const over = await inspectOutput(
    "shell",
    big,
    settings,
    quiet,
    undefined,
    Date.now,
    listener,
  );
  assert.equal(over.withheld, true);
  assert.match(over.text, /past the 256 KiB/);
});

test("an app the person changes in the decision view stays theirs, across a restart too", async () => {
  const settings = parseSettings({});
  const runtime = fake(
    () => ok(listing({ guard: false, shield: false })),
    () => ok({}),
  );
  const listener = new Listener(settings, runtime, []);
  let saved: unknown;
  const controls = new Controls(
    settings,
    "instance",
    async (value) => {
      saved = value;
    },
    () => {},
    (apps) => listener.pin(apps),
  );
  await listener.follow();
  assert.equal(settings.apps.guard, false);
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 0,
      apps: { guard: true },
    }),
    { saved: true, revision: 1 },
  );
  await listener.follow();
  assert.equal(settings.apps.guard, true, "the listener no longer sets guard");
  assert.equal(settings.apps.shield, false, "it still sets the others");

  const restarted = parseSettings({});
  const next = new Listener(restarted, runtime, []);
  await new Controls(
    restarted,
    "next",
    async () => {},
    () => {},
    (apps) => next.pin(apps),
  ).load(async () => saved);
  await next.follow();
  assert.equal(restarted.apps.guard, true);
  assert.equal(restarted.apps.shield, false);
});

test("follow hands the apps changed in the decision view back to the listener, but not the plugin's options", async () => {
  const settings = parseSettings({ apps: { check: false } });
  const runtime = fake(
    () => ok(listing({ guard: false, check: true })),
    () => ok({}),
  );
  const listener = new Listener(settings, runtime, ["check"]);
  let saved: Record<string, unknown> = {};
  const controls = new Controls(
    settings,
    "instance",
    async (value) => {
      saved = value;
    },
    () => {},
    (apps) => listener.pin(apps),
    () => listener.unpin(),
  );
  await listener.follow();
  await controls.update({
    instanceID: "instance",
    expectedRevision: 0,
    apps: { guard: true },
  });
  await listener.follow();
  assert.equal(settings.apps.guard, true);
  assert.deepEqual(saved.pinned, ["guard"]);

  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 1,
      follow: true,
      apps: { guard: true },
    }),
    { saved: false, reason: "invalid" },
    "follow and a change together are ambiguous",
  );
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 1,
      follow: true,
    }),
    { saved: true, revision: 2 },
  );
  assert.equal(settings.apps.guard, false, "the listener sets guard again");
  assert.equal(settings.apps.check, false, "an option the person named stays");
  assert.equal(saved.pinned, undefined);
  assert.equal(runtime.calls.listing, 3, "follow reads the listing at once");

  const restarted = parseSettings({ apps: { check: false } });
  const next = new Listener(restarted, runtime, ["check"]);
  await new Controls(
    restarted,
    "next",
    async () => {},
    () => {},
    (apps) => next.pin(apps),
  ).load(async () => saved);
  await next.follow();
  assert.equal(restarted.apps.guard, false, "no pin survives the restart");
});

test("a restart during a listener outage keeps the rules the last run read", async () => {
  const kept = new Map<string, unknown>();
  const store = {
    get: async (key: string) => kept.get(key),
    set: async (key: string, value: unknown) => {
      kept.set(key, value);
    },
    remove: async (key: string) => {
      kept.delete(key);
    },
  };
  let listed: Result<unknown> = ok(listing({ rules: rules("r1", true) }));
  const runtime = fake(
    () => listed,
    () => ({ ok: false, code: "offline", elapsedMs: 1 }),
  );
  await new Listener(parseSettings({}), runtime, [], store).follow();
  listed = { ok: false, code: "offline", elapsedMs: 1 };
  const settings = parseSettings({});
  const restarted = new Listener(settings, runtime, [], store);
  await restarted.follow();
  assert.equal(restarted.rules.revision, "r1");
  const held = await guard(
    "ls",
    settings,
    quiet,
    undefined,
    Date.now,
    restarted,
  );
  assert.equal(held.reason, "listener-unchecked");

  listed = { ok: false, code: "http", status: 404, elapsedMs: 1 };
  await new Listener(parseSettings({}), runtime, [], store).follow();
  assert.equal(
    kept.size,
    0,
    "a listing that binds no rules forgets the kept ones",
  );
});

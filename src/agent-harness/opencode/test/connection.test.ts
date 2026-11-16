import assert from "node:assert/strict";
import { test } from "node:test";
import { targetOf } from "../src/connection.ts";
import { Listener } from "../src/listener.ts";
import { RuntimeClient } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";

const remote = "https://runtime.example/v1/decisions";

test("a /connect credential names the runtime, its model and its token", () => {
  assert.deepEqual(
    targetOf({
      type: "key",
      key: "fixture-token",
      metadata: { url: remote, model: " vela3-lite " },
    }),
    { url: remote, model: "vela3-lite", token: "fixture-token" },
  );
  assert.deepEqual(
    targetOf({ type: "external", methodID: "open", metadata: { url: remote } }),
    { url: remote, model: "", token: undefined },
  );
  assert.equal(
    targetOf({ type: "key", key: "", metadata: { url: remote } })?.token,
    undefined,
    "an empty key sends no token",
  );
  for (const value of [
    undefined,
    { type: "oauth", metadata: { url: remote } },
    { type: "key", key: "k" },
    { type: "key", key: "k", metadata: { url: "https://runtime.example/v1" } },
    { type: "key", key: "k", metadata: { url: "https://u:p@h/v1/decisions" } },
    { type: "key", key: "k", metadata: { url: remote, model: 3 } },
  ])
    assert.equal(targetOf(value), undefined, JSON.stringify(value));
});

test("connecting another runtime sends to it with its token, and disconnecting returns to the options", async () => {
  const calls: { url: string; token: string | null; body: unknown }[] = [];
  const client = new RuntimeClient(
    parseSettings({ runtimeModel: "configured" }),
    {
      token: () => "env-token",
      fetch: async (input, init) => {
        calls.push({
          url: String(input),
          token: new Headers(init?.headers).get("authorization"),
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
        });
        return String(input).startsWith("https://down")
          ? Promise.reject(new TypeError("network"))
          : Response.json({ answers: { risk: { probability: 0.1 } } });
      },
    },
  );
  const questions = { risk: { type: "choice" } };
  await client.decide("s", questions, 500);
  assert.deepEqual(calls[0], {
    url: "http://127.0.0.1:8790/v1/decisions",
    token: "Bearer env-token",
    body: { model: "configured", state: "s", questions },
  });

  assert.equal(
    client.connect({ url: "https://down/v1/decisions", model: "", token: "t" }),
    true,
  );
  assert.equal((await client.decide("s", questions, 500)).ok, false);
  assert.ok(
    client.status().retryAt > 0,
    "the unreachable runtime opened the breaker",
  );

  assert.equal(
    client.connect({ url: remote, model: "picked", token: "connect-token" }),
    true,
  );
  assert.deepEqual(client.status(), {
    reachable: null,
    retryAt: 0,
    models: [],
    source: "connect",
  });
  assert.equal((await client.decide("s", questions, 500)).ok, true);
  assert.deepEqual(calls.at(-1), {
    url: remote,
    token: "Bearer connect-token",
    body: { model: "picked", state: "s", questions },
  });
  assert.equal(
    client.connect({ url: remote, model: "picked", token: "connect-token" }),
    false,
    "the same credential again changes nothing",
  );
  assert.ok(!JSON.stringify(client.status()).includes("token"));

  assert.equal(client.connect(undefined), true);
  await client.decide("s", questions, 500);
  assert.deepEqual(calls.at(-1)?.token, "Bearer env-token");
  assert.equal(client.status().source, "options");
});

test("another runtime forgets the listener rules read from the previous one", async () => {
  const settings = parseSettings({});
  const listener = new Listener(
    settings,
    {
      listenerKey: () => "vsr.rules:fixture",
      listing: async () => ({
        ok: true,
        elapsedMs: 1,
        value: {
          data: [],
          vsr: {
            rules: {
              revision: "r1",
              guard: [
                { name: "deploy", description: "Deploys", required: true },
              ],
              redact: [],
            },
          },
        },
      }),
      rules: async () => ({ ok: true, elapsedMs: 1, value: {} }),
    },
    [],
  );
  await listener.follow();
  assert.equal(listener.rules.revision, "r1");
  listener.forget();
  assert.equal(listener.rules.revision, "");
  assert.deepEqual(listener.rules.guard, []);
});

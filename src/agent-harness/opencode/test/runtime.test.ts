import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { test } from "node:test";
import { RuntimeClient } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";
const settings = () => parseSettings({});
const reply = () =>
  Response.json({ answers: { risk: { probability: 0.9 } }, spans: {} });
const questions = { risk: { type: "choice" } };

test("runtime discovers a named System One model, checks capabilities and rotates credentials", async () => {
  const calls: { url: string; token: string | null; body: unknown }[] = [];
  let token = "first-fixture-token";
  const client = new RuntimeClient(
    parseSettings({
      runtimeUrl: "https://example.test/v1/decisions",
      runtimeModel: "test-model",
    }),
    {
      token: () => token,
      fetch: async (input, init) => {
        calls.push({
          url: String(input),
          token: new Headers(init?.headers).get("authorization"),
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
        });
        return String(input).endsWith("models")
          ? Response.json({
              data: [{ id: "test-model", question_types: ["choice"] }],
            })
          : reply();
      },
    },
  );
  assert.equal((await client.discover()).ok, true);
  token = "rotated-fixture-token";
  assert.equal((await client.decide("synthetic", questions, 500)).ok, true);
  assert.equal(calls[1]?.token, "Bearer rotated-fixture-token");
  assert.deepEqual(calls[1]?.body, {
    model: "test-model",
    state: "synthetic",
    questions,
  });
  assert.deepEqual(
    await client.decide("synthetic", { risk: { type: "score" } }, 500),
    { ok: false, code: "unsupported", elapsedMs: 0 },
  );
  assert.equal(calls.length, 2);
  assert.ok(!JSON.stringify(client.status()).includes("fixture-token"));
});

test("discovery falls back only on 404 and rejects a model not served by the runtime", async () => {
  const calls: string[] = [];
  const client = new RuntimeClient(parseSettings({ runtimeModel: "missing" }), {
    fetch: async (input) => {
      calls.push(String(input));
      return calls.length === 1
        ? new Response("", { status: 404 })
        : Response.json({ data: [{ id: "other" }] });
    },
  });
  const result = await client.discover();
  assert.ok(!result.ok && result.code === "invalid-response");
  assert.ok(calls[1]?.endsWith("/v1/models"));
  let attempted = 0;
  const unauthorized = new RuntimeClient(settings(), {
    fetch: async () => {
      attempted++;
      return new Response("", { status: 401 });
    },
  });
  const denied = await unauthorized.discover();
  assert.ok(!denied.ok && denied.code === "http" && denied.status === 401);
  assert.equal(attempted, 1);
});

test("a bare runtime's listing is read from /v1/models when it has no System One listing, so its images and halu spans are known", async () => {
  const calls: string[] = [];
  const client = new RuntimeClient(settings(), {
    fetch: async (input) => {
      calls.push(new URL(String(input)).pathname);
      return calls.length === 1
        ? new Response("", { status: 404 })
        : Response.json({
            data: [
              {
                id: "vllm-sr/vela3-lite",
                presets: ["pii", "halu"],
                modalities: ["text", "image"],
              },
            ],
          });
    },
  });
  assert.equal(client.supports({ modality: "image" }), false);
  const result = await client.listing(1000);
  assert.ok(!result.ok && result.code === "http" && result.status === 404);
  assert.deepEqual(calls, ["/v1/systemone/models", "/v1/models"]);
  assert.equal(client.supports({ modality: "image" }), true);
  assert.equal(client.supports({ preset: "halu" }), true);
});

test("offline breaker suppresses repeats, recovers after its deadline and never reflects error secrets", async () => {
  let now = 100,
    calls = 0;
  const client = new RuntimeClient(settings(), {
    now: () => now,
    retryMs: 60,
    fetch: async () => {
      if (++calls === 1) throw new Error("sensitive-fixture-token");
      return reply();
    },
  });
  assert.ok(!(await client.decide("fixture", questions, 500)).ok);
  await client.decide("fixture", questions, 500);
  assert.equal(calls, 1);
  now += 61;
  assert.ok((await client.decide("fixture", questions, 500)).ok);
  assert.equal(calls, 2);
  assert.deepEqual(client.status(), {
    reachable: true,
    retryAt: 0,
    models: [],
    source: "options",
  });
});

test("cancellation and response-body timeout abort the actual HTTP request without opening the breaker", async (t) => {
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.write('{"answers":');
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  const client = new RuntimeClient(
    parseSettings({
      runtimeUrl: `http://127.0.0.1:${address.port}/v1/decisions`,
    }),
  );
  const controller = new AbortController();
  const waiting = client.decide("fixture", questions, 5000, controller.signal);
  setTimeout(() => controller.abort(), 30);
  const cancelled = await waiting;
  assert.ok(!cancelled.ok && cancelled.code === "cancelled");
  const timeout = await client.decide("fixture", questions, 30);
  assert.ok(!timeout.ok && timeout.code === "timeout");
  assert.notEqual(client.status().reachable, false);
  client.close();
});

test("expired queued calls do not overtake the active call or consume the next caller's result", async () => {
  let unblock!: () => void;
  const held = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  const seen: string[] = [];
  const client = new RuntimeClient(settings(), {
    fetch: async (_url, init) => {
      const state = JSON.parse(String(init?.body)).state as string;
      seen.push(state);
      if (state === "first") await held;
      return reply();
    },
  });
  const first = client.decide("first", questions, 1000);
  const expired = client.decide("expired", questions, 20);
  const third = client.decide("third", questions, 1000);
  const result = await expired;
  assert.ok(!result.ok && result.code === "timeout");
  assert.deepEqual(seen, ["first"]);
  unblock();
  assert.ok((await first).ok && (await third).ok);
  assert.deepEqual(seen, ["first", "third"]);
});

test("missing answers, invalid span maps, malformed JSON and oversized responses are unavailable", async () => {
  for (const response of [
    Response.json({ answers: {} }),
    Response.json({ answers: { risk: {} }, spans: [] }),
    new Response("not json"),
    new Response("x".repeat(2 * 1024 * 1024 + 1)),
  ]) {
    const client = new RuntimeClient(settings(), {
      fetch: async () => response,
    });
    const result = await client.decide("fixture", questions, 1000);
    assert.ok(!result.ok && result.code === "invalid-response");
    assert.equal(client.status().reachable, true);
  }
});

test("redirects cannot forward a runtime token to another endpoint", async (t) => {
  let targetHits = 0;
  const target = createServer((_req, res) => {
    targetHits++;
    res.end("{}");
  });
  target.listen(0, "127.0.0.1");
  await once(target, "listening");
  const targetAddress = target.address();
  assert.ok(targetAddress && typeof targetAddress !== "string");
  const origin = createServer((_req, res) => {
    res.writeHead(307, {
      location: `http://127.0.0.1:${targetAddress.port}/v1/decisions`,
    });
    res.end();
  });
  origin.listen(0, "127.0.0.1");
  await once(origin, "listening");
  t.after(() => {
    origin.closeAllConnections();
    target.closeAllConnections();
    origin.close();
    target.close();
  });
  const address = origin.address();
  assert.ok(address && typeof address !== "string");
  const client = new RuntimeClient(
    parseSettings({
      runtimeUrl: `http://127.0.0.1:${address.port}/v1/decisions`,
    }),
    { token: () => "fixture-token" },
  );
  const result = await client.decide("fixture", questions, 1000);
  assert.ok(!result.ok);
  assert.equal(targetHits, 0);
});

import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import type { TestContext } from "node:test";
import { test } from "node:test";
import { host } from "./host.ts";
import type { Snapshot } from "../../src/decisions.ts";

interface Request {
  model: string;
  messages: { role: string; content: unknown }[];
  stream?: boolean;
  tools?: unknown[];
  reasoning_effort?: string;
}
async function fixture(t: TestContext, agentModel = false) {
  const requests: Request[] = [],
    classifications: Record<string, unknown>[] = [];
  let retried = false;
  let offline = false;
  const server = createServer(async (req, res) => {
    // A plain decision runtime: no Router listener, so no vsr profile or rules.
    if (req.method === "GET" && req.url === "/v1/systemone/models") {
      res.writeHead(404);
      res.end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const body = JSON.parse(Buffer.concat(chunks).toString());
    if (req.url === "/v1/decisions") {
      classifications.push(body);
      if (offline) {
        res.writeHead(503);
        res.end("{}");
        return;
      }
      const pushed = JSON.stringify(body.state).includes("PUSH");
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          answers: body.questions.user_feedback
            ? {
                user_feedback: {
                  probabilities: {
                    WRONG_ANSWER: pushed ? 0.98 : 0.01,
                    NO_FEEDBACK: pushed ? 0.02 : 0.99,
                  },
                },
              }
            : body.questions.reask
              ? { reask: { noul: 0.01 } }
              : {
                  complexity: {
                    score: 0.1,
                    confidence: JSON.stringify(body.state).includes("LOWCONF")
                      ? 0.1
                      : 0.98,
                  },
                  steps: { score: 0.1, confidence: 0.98 },
                },
          spans: {},
        }),
      );
      return;
    }
    requests.push(body);
    const messages = body.messages as Request["messages"];
    const lastUser = messages.findLastIndex(
      (message) => message.role === "user",
    );
    const text = JSON.stringify(messages[lastUser]?.content);
    const results = messages
      .slice(lastUser + 1)
      .filter((message) => message.role === "tool");
    const child = text.includes("TASK:");
    if (child && text.includes("RETRY") && !retried) {
      retried = true;
      res.writeHead(503);
      res.end("{}");
      return;
    }
    const resume = text.includes("RESUME") && results.length === 1;
    const spawn = !!body.tools?.length && !child && (!results.length || resume);
    const calls = Array.from(
      { length: text.includes("CONCURRENT") ? 2 : 1 },
      (_, i) => ({
        id: `call_${results.length}_${i}`,
        type: "function",
        function: {
          name: "subagent",
          arguments: JSON.stringify({
            agent: "general",
            description: "Native routing fixture",
            prompt: `TASK: ${text.includes("LOWCONF") ? "LOWCONF" : text.includes("RETRY") ? "RETRY" : "EASY"} ${i}`,
            ...(text.includes("EXPLICIT")
              ? { model: "fixture/strong#high" }
              : {}),
            ...(resume
              ? {
                  sessionID: String(results[0]?.content).match(
                    /sessionID="([^"]+)"/,
                  )?.[1],
                }
              : {}),
          }),
        },
      }),
    );
    const message = spawn
      ? { role: "assistant", content: null, tool_calls: calls }
      : { role: "assistant", content: "NATIVE_FIXTURE_DONE" };
    const finish = spawn ? "tool_calls" : "stop";
    const data = {
      id: "native_fixture",
      model: body.model,
      created: 1,
      usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
    };
    if (body.stream) {
      res.setHeader("content-type", "text/event-stream");
      const chunk = {
        ...data,
        object: "chat.completion.chunk",
        choices: [
          {
            index: 0,
            delta: {
              ...message,
              ...(spawn
                ? {
                    tool_calls: calls.map((call, index) => ({
                      ...call,
                      index,
                    })),
                  }
                : {}),
            },
            finish_reason: null,
          },
        ],
      };
      const end = {
        ...data,
        object: "chat.completion.chunk",
        choices: [{ index: 0, delta: {}, finish_reason: finish }],
      };
      res.end(
        `data: ${JSON.stringify(chunk)}\n\ndata: ${JSON.stringify(end)}\n\ndata: [DONE]\n\n`,
      );
    } else {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          ...data,
          object: "chat.completion",
          choices: [{ index: 0, message, finish_reason: finish }],
        }),
      );
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  const tiers = ["light", "middle", "strong"].map((model) => ({
    model: `fixture/${model}`,
    efforts: ["low", "medium", "high"],
  }));
  const native = await host(t, {
    model: "fixture/middle#medium",
    ...(agentModel
      ? { agents: { general: { model: "fixture/strong#high" } } }
      : {}),
    pluginOptions: {
      runtimeUrl: `http://127.0.0.1:${address.port}/v1/decisions`,
      routing: { tiers },
    },
    providers: {
      fixture: {
        name: "Fixture",
        package: "@opencode/ai/providers/openai-compatible",
        settings: { baseURL: `http://127.0.0.1:${address.port}/v1` },
        models: Object.fromEntries(
          ["light", "middle", "strong"].map((id) => [
            id,
            {
              name: id,
              capabilities: { tools: true },
              limit: { context: 32768, output: 1024 },
              variants: ["low", "medium", "high"].map((id) => ({
                id,
                body: { reasoning_effort: id },
              })),
            },
          ]),
        ),
      },
    },
  });
  return {
    ...native,
    requests,
    classifications,
    setOffline() {
      offline = true;
    },
  };
}

test(
  "native child requests use selected model/effort, preserve parent and expose observed usage",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const routed = await native.run("ROOT EASY");
    const child = native.requests.find((request) =>
      request.messages.some(
        (message) =>
          message.role === "user" &&
          JSON.stringify(message.content).includes("TASK:"),
      ),
    );
    assert.equal(child?.model, "light");
    assert.equal(child?.reasoning_effort, "low");
    assert.ok(
      native.requests
        .filter(
          (request) =>
            request.tools?.length &&
            !request.messages.some(
              (message) =>
                message.role === "user" &&
                JSON.stringify(message.content).includes("TASK:"),
            ),
        )
        .every(
          (request) =>
            request.model === "middle" && request.reasoning_effort === "medium",
        ),
      JSON.stringify(
        native.requests.map(({ model, reasoning_effort, tools }) => ({
          model,
          reasoning_effort,
          tools: !!tools?.length,
        })),
      ),
    );
    const snapshot = (await native.rpc("snapshot", {
      sessionID: routed.sessionID,
    })) as Snapshot;
    assert.equal(snapshot.decisions.length, 1);
    assert.equal(snapshot.decisions[0]?.model, "fixture/light#low");
    assert.equal(snapshot.decisions[0]?.observedModel, "fixture/light#low");
    assert.equal(snapshot.decisions[0]?.usage?.input, 10);
    assert.equal(snapshot.decisions[0]?.usage?.output, 2);
    assert.ok(snapshot.decisions[0]?.childSessionID);
    const before = native.classifications.length;
    const explicit = await native.run("ROOT EXPLICIT");
    assert.deepEqual(
      (
        (await native.rpc("snapshot", {
          sessionID: explicit.sessionID,
        })) as Snapshot
      ).decisions,
      [],
    );
    assert.equal(
      native.classifications.length,
      before + 1,
      "explicit calls skip task classification",
    );
    assert.equal(native.requests.at(-2)?.model, "strong");
    const low = await native.run("ROOT LOWCONF");
    assert.equal(
      ((await native.rpc("snapshot", { sessionID: low.sessionID })) as Snapshot)
        .decisions[0]?.reason,
      "low-confidence",
    );
    assert.equal(native.requests.at(-2)?.model, "middle");
    const resumed = await native.run("ROOT RESUME");
    const resumedSnapshot = (await native.rpc("snapshot", {
      sessionID: resumed.sessionID,
    })) as Snapshot;
    assert.equal(
      resumedSnapshot.decisions.length,
      1,
      "resumed children skip routing",
    );
    const retryStart = native.requests.length;
    const retry = await native.run("ROOT RETRY");
    const attempts = native.requests
      .slice(retryStart)
      .filter((request) =>
        request.messages.some(
          (message) =>
            message.role === "user" &&
            JSON.stringify(message.content).includes("TASK:"),
        ),
      );
    assert.equal(attempts.length, 2, "provider retry must actually occur");
    assert.ok(
      attempts.every(
        (request) =>
          request.model === "light" && request.reasoning_effort === "low",
      ),
    );
    assert.equal(
      (
        (await native.rpc("snapshot", {
          sessionID: retry.sessionID,
        })) as Snapshot
      ).decisions.length,
      1,
      "retry must not consume another routing decision",
    );
    native.setOffline();
    const fallback = await native.run("ROOT OFFLINE");
    assert.equal(
      (
        (await native.rpc("snapshot", {
          sessionID: fallback.sessionID,
        })) as Snapshot
      ).decisions.find((decision) => decision.app === "router")?.reason,
      "http",
    );
    assert.equal(native.requests.at(-2)?.model, "middle");
  },
);

test(
  "native pushback escalates one concurrent child and does not leak across sessions or turns",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const first = await native.run("ROOT EASY");
    const start = native.requests.length;
    await native.run("ROOT PUSH CONCURRENT", first.sessionID);
    const requests = native.requests.slice(start);
    const children = requests.filter((request) =>
      request.messages.some(
        (message) =>
          message.role === "user" &&
          JSON.stringify(message.content).includes("TASK:"),
      ),
    );
    assert.deepEqual(
      children
        .map((request) => [request.model, request.reasoning_effort])
        .sort(),
      [
        ["light", "low"],
        ["middle", "high"],
      ],
    );
    assert.ok(
      requests.some((request) =>
        request.messages.some(
          (message) =>
            message.role === "system" &&
            JSON.stringify(message.content).includes("Re-check that answer"),
        ),
      ),
    );
    const snapshot = (await native.rpc("snapshot", {
      sessionID: first.sessionID,
    })) as Snapshot;
    assert.equal(
      snapshot.decisions.filter((decision) => decision.app === "escalate")
        .length,
      1,
    );
    const separate = await native.run("ROOT EASY");
    assert.equal(
      (
        (await native.rpc("snapshot", {
          sessionID: separate.sessionID,
        })) as Snapshot
      ).decisions[0]?.model,
      "fixture/light#low",
    );
    await native.run("ROOT EASY NEXT", first.sessionID);
    const later = (await native.rpc("snapshot", {
      sessionID: first.sessionID,
    })) as Snapshot;
    assert.equal(later.decisions.at(-1)?.model, "fixture/light#low");
    const parallel = await Promise.all([
      native.run("ROOT EASY A"),
      native.run("ROOT EASY B"),
    ]);
    const snapshots = await Promise.all(
      parallel.map(
        async ({ sessionID }) =>
          (await native.rpc("snapshot", { sessionID })) as Snapshot,
      ),
    );
    assert.notEqual(parallel[0]?.sessionID, parallel[1]?.sessionID);
    assert.ok(
      snapshots.every(
        (snapshot) =>
          snapshot.decisions.length === 1 &&
          snapshot.decisions[0]?.observedModel === "fixture/light#low",
      ),
    );
    assert.notEqual(
      snapshots[0]?.decisions[0]?.childSessionID,
      snapshots[1]?.decisions[0]?.childSessionID,
    );
  },
);

test(
  "native agent-specific model is preserved",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t, true);
    const run = await native.run("ROOT EASY");
    assert.deepEqual(
      ((await native.rpc("snapshot", { sessionID: run.sessionID })) as Snapshot)
        .decisions,
      [],
    );
    assert.equal(native.requests.at(-2)?.model, "strong");
  },
);

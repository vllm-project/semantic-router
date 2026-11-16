import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { host } from "./host.ts";
import type { Snapshot } from "../../src/decisions.ts";

test(
  "native Check uses this turn's delivered masked evidence, handles queued/concurrent sessions, and never continues the answer",
  { timeout: 90000 },
  async (t) => {
    const checks: { context: string; request: string; answer: string }[] = [];
    const generation: Record<string, number> = {};
    const secret = ["CHECK", "SECRET", "12345678"].join("");
    let root = "",
      gateReached = false,
      release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
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
      res.setHeader("content-type", "application/json");
      if (req.url === "/v1/decisions") {
        if (body.questions.hallucination) checks.push(body.state);
        res.end(
          JSON.stringify({
            answers: body.questions.pii
              ? { pii: { noul: 0 } }
              : body.questions.jailbreak
                ? { jailbreak: { probabilities: { jailbreak: 0 } } }
                : {
                    hallucination: {
                      noul: String(body.state.request).includes("BAD")
                        ? 0.99
                        : 0.01,
                    },
                  },
            spans: { pii: [] },
          }),
        );
        return;
      }
      const messages = body.messages as { role: string; content: unknown }[];
      const last = messages.findLastIndex((message) => message.role === "user");
      const label =
        JSON.stringify(messages[last]?.content).match(
          /CHECK_CASE (\w+)/,
        )?.[1] ?? "TITLE";
      if (body.tools?.length) generation[label] = (generation[label] ?? 0) + 1;
      const hasResult = messages
        .slice(last + 1)
        .some((message) => message.role === "tool");
      const call =
        body.tools?.length && !hasResult && !label.includes("NO_READ");
      if (body.tools?.length && label === "QUEUED_A" && hasResult) {
        gateReached = true;
        await gate;
      }
      const tool = {
        id: "reused_read_id",
        type: "function",
        function: {
          name: "read",
          arguments: JSON.stringify({ path: join(root, `${label}.txt`) }),
        },
      };
      const answer =
        label === "LONG"
          ? "A".repeat(9000)
          : `This is a final factual answer for ${label}, based on the supplied file evidence.`;
      const message = call
        ? { role: "assistant", content: null, tool_calls: [tool] }
        : { role: "assistant", content: answer };
      const finish = call ? "tool_calls" : "stop";
      const data = {
        id: "check_fixture",
        created: 1,
        model: body.model,
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      };
      if (body.stream) {
        res.setHeader("content-type", "text/event-stream");
        res.end(
          `data: ${JSON.stringify({ ...data, object: "chat.completion.chunk", choices: [{ index: 0, delta: { ...message, ...(call ? { tool_calls: [{ ...tool, index: 0 }] } : {}) }, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ ...data, object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`,
        );
      } else
        res.end(
          JSON.stringify({
            ...data,
            object: "chat.completion",
            choices: [{ index: 0, message, finish_reason: finish }],
          }),
        );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    t.after(() => {
      release();
      server.closeAllConnections();
      server.close();
    });
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    const native = await host(t, {
      model: "fixture/model",
      pluginOptions: {
        apps: { check: true, guard: false, router: false, escalate: false },
        runtimeUrl: `http://127.0.0.1:${address.port}/v1/decisions`,
      },
      providers: {
        fixture: {
          package: "@opencode/ai/providers/openai-compatible",
          settings: { baseURL: `http://127.0.0.1:${address.port}/v1` },
          models: {
            model: { name: "Fixture", limit: { context: 32768, output: 1024 } },
          },
        },
      },
    });
    root = native.root;
    for (const label of ["A1", "B1_BAD", "B2", "QUEUED_A", "QUEUED_B", "LONG"])
      await writeFile(
        join(root, `${label}.txt`),
        `EVIDENCE_${label}\nAPI_KEY=${secret}\n`,
      );
    async function snapshot(id: string, count: number) {
      for (let i = 0; i < 100; i++) {
        const result = (await native.rpc("snapshot", {
          sessionID: id,
        })) as Snapshot;
        if (
          result.decisions.filter((decision) => decision.app === "check")
            .length >= count
        )
          return result;
        await delay(20);
      }
      assert.fail(
        `No Check record: ${JSON.stringify({ checks, snapshot: await native.rpc("snapshot", { sessionID: id }) })}`,
      );
    }
    const [a, b] = await Promise.all([
      native.run("CHECK_CASE A1"),
      native.run("CHECK_CASE B1_BAD"),
    ]);
    const aState = await snapshot(a.sessionID, 1),
      bState = await snapshot(b.sessionID, 1);
    assert.equal(
      aState.decisions.find((decision) => decision.app === "check")?.outcome,
      "verified",
    );
    assert.equal(
      bState.decisions.find((decision) => decision.app === "check")?.outcome,
      "unverified",
    );
    assert.ok(
      checks
        .find((state) => state.request.includes("A1"))
        ?.context.includes("EVIDENCE_A1"),
    );
    assert.ok(
      !checks
        .find((state) => state.request.includes("A1"))
        ?.context.includes("EVIDENCE_B1"),
    );
    assert.ok(
      checks.every(
        (state) =>
          !state.context.includes(secret) &&
          state.context.includes("[vsr:secret]"),
      ),
    );
    assert.ok(!JSON.stringify(aState).includes(secret));
    await native.run("CHECK_CASE A2_NO_READ", a.sessionID);
    await delay(100);
    assert.equal(
      (
        (await native.rpc("snapshot", { sessionID: a.sessionID })) as Snapshot
      ).decisions.filter((decision) => decision.app === "check").length,
      1,
    );
    await native.run("CHECK_CASE B2", b.sessionID);
    await snapshot(b.sessionID, 2);
    assert.ok(
      !checks
        .find((state) => state.request.includes("B2"))
        ?.context.includes("EVIDENCE_B1"),
    );
    const queued = (await native.api("/api/session", {
      model: { providerID: "fixture", id: "model" },
    })) as { data: { id: string } };
    await native.api(`/api/session/${queued.data.id}/prompt`, {
      text: "CHECK_CASE QUEUED_A",
    });
    for (let i = 0; i < 100 && !gateReached; i++) await delay(20);
    assert.ok(gateReached);
    await native.api(`/api/session/${queued.data.id}/prompt`, {
      text: "CHECK_CASE QUEUED_B",
      delivery: "queue",
    });
    release();
    await snapshot(queued.data.id, 2);
    for (const label of ["QUEUED_A", "QUEUED_B"])
      assert.ok(
        checks
          .find((state) => state.request.includes(label))
          ?.context.includes(`EVIDENCE_${label}`),
      );
    const long = await native.run("CHECK_CASE LONG");
    assert.equal(
      (await snapshot(long.sessionID, 1)).decisions.find(
        (decision) => decision.app === "check",
      )?.outcome,
      "partial",
    );
    assert.ok(
      Object.entries(generation).every(([label, calls]) =>
        label.includes("NO_READ") ? calls === 1 : calls === 2,
      ),
      JSON.stringify(generation),
    );
  },
);

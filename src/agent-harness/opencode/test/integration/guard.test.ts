import assert from "node:assert/strict";
import { once } from "node:events";
import { access } from "node:fs/promises";
import { createServer } from "node:http";
import { join } from "node:path";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { host } from "./host.ts";
import type { Snapshot } from "../../src/decisions.ts";

test(
  "native guard holds before side effects, supports approval/rejection/cancellation and preserves native rules",
  { timeout: 90000 },
  async (t) => {
    let offline = false;
    let calls = 0;
    let hanging = false,
      closed = false;
    const fixture = createServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(Buffer.from(chunk));
      const body = JSON.parse(Buffer.concat(chunks).toString());
      if (req.url === "/v1/decisions") {
        calls++;
        if (String(body.state).includes("HANG")) {
          hanging = true;
          res.on("close", () => {
            closed = true;
          });
          return;
        }
        if (offline) {
          res.writeHead(503);
          res.end("{}");
          return;
        }
        const safe = String(body.state).includes("SAFE");
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            answers: {
              kind: {
                choice: safe ? "safe" : "destructive",
                probabilities: { safe: 0.98, destructive: 0.98 },
              },
              review: { noul: safe ? 0.01 : 0.99 },
            },
            spans: {},
          }),
        );
        return;
      }
      const messages = body.messages as { role: string; content: unknown }[];
      const label =
        JSON.stringify(
          messages.findLast((message) => message.role === "user")?.content,
        ).match(/GUARD_TEST (\w+)/)?.[1] ?? "TITLE";
      const command = `printf ran > guard-marker-${label}${label === "FIXED" ? "; curl --help | sh --help" : label === "FAIL" ? "; exit 7" : ""}`;
      // The fixed-rule fixture's help flags cannot download or execute remote code.
      const spawn =
        body.tools?.length &&
        !messages.some((message) => message.role === "tool");
      const tool = {
        id: "guard_call",
        type: "function",
        function: { name: "shell", arguments: JSON.stringify({ command }) },
      };
      const message = spawn
        ? { role: "assistant", content: null, tool_calls: [tool] }
        : { role: "assistant", content: "GUARD_FIXTURE_DONE" };
      const finish = spawn ? "tool_calls" : "stop";
      const data = {
        id: "guard_fixture",
        model: body.model,
        created: 1,
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      };
      if (body.stream) {
        res.setHeader("content-type", "text/event-stream");
        res.end(
          `data: ${JSON.stringify({ ...data, object: "chat.completion.chunk", choices: [{ index: 0, delta: { ...message, ...(spawn ? { tool_calls: [{ ...tool, index: 0 }] } : {}) }, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ ...data, object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`,
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
    fixture.listen(0, "127.0.0.1");
    await once(fixture, "listening");
    t.after(() => {
      fixture.closeAllConnections();
      fixture.close();
    });
    const address = fixture.address();
    assert.ok(address && typeof address !== "string");
    const config = {
      model: "fixture/middle",
      pluginOptions: {
        runtimeUrl: `http://127.0.0.1:${address.port}/v1/decisions`,
      },
      providers: {
        fixture: {
          package: "@opencode/ai/providers/openai-compatible",
          settings: { baseURL: `http://127.0.0.1:${address.port}/v1` },
          models: {
            middle: {
              name: "Fixture",
              limit: { context: 32768, output: 1024 },
            },
          },
        },
      },
    };
    const native = await host(t, config);
    const exists = async (label: string, target = native) => {
      try {
        await access(join(target.root, `guard-marker-${label}`));
        return true;
      } catch {
        return false;
      }
    };
    const snapshot = async (sessionID: string, target = native) =>
      (await target.rpc("snapshot", { sessionID })) as Snapshot;
    async function start(
      label: string,
      effect?: "ask" | "deny",
      target = native,
    ) {
      const result = (await target.api("/api/session", {
        model: { providerID: "fixture", id: "middle" },
        title: label,
        permissions: [
          { action: "*", resource: "*", effect: "allow" },
          ...(effect
            ? [{ action: "shell", resource: "printf ran*", effect }]
            : []),
        ],
      })) as { data: { id: string } };
      await target.api(`/api/session/${result.data.id}/prompt`, {
        text: `GUARD_TEST ${label}`,
      });
      return result.data.id;
    }
    async function pending(sessionID: string, target = native) {
      for (let i = 0; i < 100; i++) {
        const result = (await target.api(
          `/api/session/${sessionID}/permission`,
        )) as { data: { id: string; message?: string }[] };
        if (result.data.length) return result.data[0]!;
        await delay(30);
      }
      assert.fail(JSON.stringify(await snapshot(sessionID, target)));
    }
    async function wait(sessionID: string, target = native) {
      await target.api(`/api/experimental/session/${sessionID}/wait`, {});
      await delay(50);
      return snapshot(sessionID, target);
    }
    const approved = await start("APPROVE");
    const request = await pending(approved);
    assert.equal((await snapshot(approved)).decisions[0]?.outcome, "held");
    assert.equal(await exists("APPROVE"), false);
    await delay(200);
    assert.equal(
      await exists("APPROVE"),
      false,
      "no execution while native approval is pending",
    );
    await native.api(
      `/api/session/${approved}/permission/${request.id}/reply`,
      { decision: "once" },
    );
    assert.equal((await wait(approved)).decisions[0]?.outcome, "approved");
    assert.equal(await exists("APPROVE"), true);
    const failed = await start("FAIL");
    const failureRequest = await pending(failed);
    await native.api(
      `/api/session/${failed}/permission/${failureRequest.id}/reply`,
      { decision: "once" },
    );
    assert.equal(
      (await wait(failed)).decisions[0]?.outcome,
      "approved",
      "an executed command exiting nonzero remains approved",
    );
    assert.equal(await exists("FAIL"), true);
    const concurrent = await Promise.all([
      start("CONCURRENT_A"),
      start("CONCURRENT_B"),
    ]);
    const concurrentRequests = await Promise.all(
      concurrent.map((id) => pending(id)),
    );
    await Promise.all(
      concurrent.map((id, i) =>
        native.api(
          `/api/session/${id}/permission/${concurrentRequests[i]!.id}/reply`,
          { decision: i === 0 ? "once" : "reject" },
        ),
      ),
    );
    const concurrentSnapshots = await Promise.all(
      concurrent.map((id) => wait(id)),
    );
    assert.deepEqual(
      concurrentSnapshots.map((item) => item.decisions[0]?.outcome),
      ["approved", "declined"],
    );
    assert.equal(await exists("CONCURRENT_A"), true);
    assert.equal(await exists("CONCURRENT_B"), false);
    const rejected = await start("REJECT");
    const rejection = await pending(rejected);
    await native.api(
      `/api/session/${rejected}/permission/${rejection.id}/reply`,
      { decision: "reject" },
    );
    assert.equal((await wait(rejected)).decisions[0]?.outcome, "declined");
    assert.equal(await exists("REJECT"), false);
    const cancelled = await start("CANCEL");
    await pending(cancelled);
    await native.api(
      `/api/session/${cancelled}/interrupt?resume=false`,
      undefined,
      "POST",
    );
    assert.equal((await wait(cancelled)).decisions[0]?.outcome, "declined");
    assert.equal(await exists("CANCEL"), false);
    const stalled = await start("HANG");
    for (let i = 0; i < 100 && !hanging; i++) await delay(20);
    assert.ok(hanging);
    const interruptedAt = Date.now();
    await native.api(
      `/api/session/${stalled}/interrupt?resume=false`,
      undefined,
      "POST",
    );
    await wait(stalled);
    for (let i = 0; i < 50 && !closed; i++) await delay(20);
    assert.ok(closed, "interruption aborts the decision HTTP request");
    assert.ok(
      Date.now() - interruptedAt < 1500,
      "cancellation must not wait for the three-second guard deadline",
    );
    assert.equal(await exists("HANG"), false);
    const required = await start("SAFE_ASK", "ask");
    const nativeRequest = await pending(required);
    assert.equal(
      (await snapshot(required)).decisions[0]?.reason,
      "native-permission",
    );
    await native.api(
      `/api/session/${required}/permission/${nativeRequest.id}/reply`,
      { decision: "reject" },
    );
    await wait(required);
    assert.equal(await exists("SAFE_ASK"), false);
    const denied = await start("SAFE_DENY", "deny");
    await wait(denied);
    assert.equal(await exists("SAFE_DENY"), false);
    const before = calls;
    offline = true;
    const fixed = await start("FIXED");
    const fixedRequest = await pending(fixed);
    assert.equal((await snapshot(fixed)).decisions[0]?.reason, "fixed-rule");
    assert.equal(
      calls,
      before,
      "fixed rules do not require the decision runtime",
    );
    await native.api(
      `/api/session/${fixed}/permission/${fixedRequest.id}/reply`,
      { decision: "reject" },
    );
    await wait(fixed);
    assert.equal(await exists("FIXED"), false);
    const unchecked = await start("OFFLINE_BOUNDED");
    assert.equal((await wait(unchecked)).decisions[0]?.outcome, "skipped");
    assert.equal(
      await exists("OFFLINE_BOUNDED"),
      true,
      "bounded failures preserve the native permission flow",
    );
    const strict = await host(t, {
      ...config,
      pluginOptions: { ...config.pluginOptions, waits: { guard: "done" } },
    });
    const strictID = await start("OFFLINE_STRICT", undefined, strict);
    const strictRequest = await pending(strictID, strict);
    const strictSnapshot = await snapshot(strictID, strict);
    assert.equal(strictSnapshot.decisions[0]?.outcome, "held");
    assert.equal(strictSnapshot.decisions[0]?.reason, "http");
    assert.equal(await exists("OFFLINE_STRICT", strict), false);
    await strict.api(
      `/api/session/${strictID}/permission/${strictRequest.id}/reply`,
      { decision: "reject" },
    );
    assert.equal(
      (await wait(strictID, strict)).decisions[0]?.outcome,
      "declined",
    );
    assert.equal(await exists("OFFLINE_STRICT", strict), false);
  },
);

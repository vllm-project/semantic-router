import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import { test } from "node:test";
import { host } from "./host.ts";

test(
  "a model that would let guard's canary run is said once, on the runtime the first prompt asks",
  { timeout: 60000 },
  async (t) => {
    let canaries = 0;
    const server = createServer(async (req, res) => {
      // A plain decision runtime whose guard is too lenient: the canary scores under the review threshold.
      if (req.method === "GET" && req.url === "/v1/systemone/models") {
        res.writeHead(404);
        res.end();
        return;
      }
      if (req.method === "GET" && req.url === "/v1/models") {
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            data: [{ id: "fixture", question_types: ["choice", "noul"] }],
          }),
        );
        return;
      }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(Buffer.from(chunk));
      const body = JSON.parse(Buffer.concat(chunks).toString());
      res.setHeader("content-type", "application/json");
      if (req.url === "/v1/decisions") {
        if (JSON.stringify(body.state).includes("git push --force origin main"))
          canaries++;
        res.end(
          JSON.stringify({
            answers: {
              kind: {
                choice: "safe",
                probabilities: { safe: 0.9, destructive: 0.1 },
              },
              review: { noul: 0.2 },
            },
            spans: {},
          }),
        );
        return;
      }
      const data = {
        id: "canary_fixture",
        model: body.model,
        created: 1,
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      };
      const message = { role: "assistant", content: "CANARY_FIXTURE_DONE" };
      if (body.stream) {
        res.setHeader("content-type", "text/event-stream");
        res.end(
          `data: ${JSON.stringify({ ...data, object: "chat.completion.chunk", choices: [{ index: 0, delta: message, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ ...data, object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`,
        );
      } else
        res.end(
          JSON.stringify({
            ...data,
            object: "chat.completion",
            choices: [{ index: 0, message, finish_reason: "stop" }],
          }),
        );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    t.after(() => {
      server.closeAllConnections();
      server.close();
    });
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    const native = await host(t, {
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
    });
    const note = async (sessionID: string) => {
      for (let i = 0; i < 50; i++) {
        const snapshot = (await native.rpc("snapshot", { sessionID })) as {
          runtime: { guardNote?: string };
        };
        if (snapshot.runtime.guardNote) return snapshot.runtime.guardNote;
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      return undefined;
    };
    const first = await native.run("hello");
    assert.equal(
      await note(first.sessionID),
      'vSR guard: this model scores "git push --force origin main" 0.20 < 0.90, so guard would let it run · fixed rules still hold · /vsr-setup',
    );
    const second = await native.run("hello again");
    assert.ok(await note(second.sessionID), "the note stays for the panel");
    assert.equal(canaries, 1, "the canary is asked once per runtime");
  },
);

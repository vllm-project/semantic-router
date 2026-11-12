// Throwaway capability probe. Automatic permission replies are test-only.
import { Probe } from "./rpc.ts";
import { appendFileSync, existsSync } from "node:fs";
const log = (event, data) =>
  appendFileSync(
    process.env.VSR_PROBE_LOG,
    JSON.stringify({ event, data }) + "\n",
  );
const testCase = process.env.VSR_CASE;
export default {
  id: "vsr.gate0",
  async setup(ctx) {
    log("loaded", { version: ctx.app.version });
    const turns = new Map();
    let enabled =
        testCase === "lifecycle"
          ? ((await ctx.storage.get("enabled")) ?? true)
          : true,
      decisions = 0;
    const rpc = await ctx.rpc.register(Probe, {
      snapshot: async () => ({ enabled, decisions }),
      setEnabled: async (input) => {
        enabled = input.enabled;
        if (testCase === "lifecycle") await ctx.storage.set("enabled", enabled);
        await rpc.events.emit("updated", { enabled, decisions });
        log("rpc.changed", { enabled, decisions });
        return { enabled };
      },
    });
    const abort = new AbortController();
    void (async () => {
      for await (const event of ctx.event.subscribe({ signal: abort.signal })) {
        if (event.type.startsWith("session.execution")) {
          log("execution", event);
          if (event.type === "session.execution.succeeded") {
            try {
              const sid = event.data?.sessionID ?? event.properties?.sessionID;
              if (sid && testCase === "evidence") {
                const turn = turns.get(sid);
                if (turn) {
                  const messages = await ctx.session.context({
                    sessionID: sid,
                  });
                  const answers = messages.filter(
                    (m) => m.type === "assistant" && m.finish === "stop",
                  );
                  log("turn.checked", {
                    sessionID: sid,
                    messageID: turn.messageID,
                    label: turn.label,
                    evidence: [...turn.delivered.values()],
                    answer: answers.at(-1)?.content,
                  });
                }
              }
              if (sid)
                log("completion.context", {
                  sessionID: sid,
                  messages: await ctx.session.context({ sessionID: sid }),
                });
            } catch (error) {
              log("completion.error", String(error));
            }
          }
        }
      }
    })().catch((error) => log("event.error", String(error)));
    await ctx.tool.hook("execute.before", async (e) => {
      log("tool.before", e);
      if (e.tool === "subagent" && !e.input.model && !e.input.sessionID) {
        if (testCase === "live-routing") {
          const response = await fetch("http://127.0.0.1:8790/v1/decisions", {
            method: "POST",
            headers: { "content-type": "application/json" },
            signal: AbortSignal.timeout(10000),
            body: JSON.stringify({
              state: e.input.prompt,
              questions: {
                complexity: {
                  type: "score",
                  instructions:
                    "How difficult is this request to solve correctly?",
                  criteria: [
                    "straightforward retrieval or formatting",
                    "several steps of reasoning",
                    "advanced reasoning or specialized expertise",
                  ],
                },
              },
            }),
          });
          if (!response.ok) throw new Error(`Decision HTTP ${response.status}`);
          const answer = (await response.json()).answers.complexity;
          if (!Number.isFinite(answer.score) || answer.confidence < 0.6)
            throw new Error("Unqualified routing decision");
          const selected = answer.score < 1.5 ? "probe/routed" : "probe/auto";
          log("live.route", {
            score: answer.score,
            confidence: answer.confidence,
            selected,
          });
          e.input.model = selected;
        } else e.input.model = "probe/routed";
        log("route.applied", e);
      }
    });
    await ctx.tool.hook("execute.after", async (e) => {
      log("tool.after", e);
      if (testCase === "evidence") turns.get(e.sessionID)?.toolIDs.add(e.id);
      decisions++;
      await rpc.events.emit("updated", { enabled, decisions });
      if (
        ["shell", "read", "webfetch"].includes(e.tool) &&
        e.status === "completed"
      ) {
        if (testCase === "strict") {
          e.result = {
            output: { output: "WITHHELD" },
            content: [
              { type: "text", text: "WITHHELD: classifier unavailable" },
            ],
          };
          return;
        }
        if (testCase === "partial") {
          e.result = {
            ...e.result,
            content: [
              {
                type: "text",
                text: "UNCHECKED: classifier timeout\nBENIGN_PREFIX SYNTHETIC_SECRET",
              },
            ],
          };
          return;
        }
        if (testCase === "evidence") {
          const label = e.input.path.match(/evidence-(A1|B1|B2)/)?.[1];
          e.result = {
            output: { output: `MASKED_${label}` },
            content: [{ type: "text", text: `MASKED_${label}` }],
          };
          return;
        }
        e.result = {
          ...e.result,
          output: { ...e.result.output, output: "MASKED" },
          content: [{ type: "text", text: "MASKED" }],
        };
      }
    });
    await ctx.session.hook("experimental.model.resolve", (e) => {
      log("resolve", e);
      if (e.model.id === "auto") e.model = { ...e.model, id: "routed" };
    });
    await ctx.session.hook("context", (e) => {
      log("context", {
        keys: Object.keys(e),
        sessionID: e.sessionID,
        model: e.model,
      });
      if (testCase === "background") {
        for (const m of e.messages) {
          if (m.role !== "user" || !Array.isArray(m.content)) continue;
          for (const part of m.content) {
            if (part.type === "text" && part.text.startsWith("<shell ")) {
              log("context.background.masked", { sessionID: e.sessionID });
              part.text = "BACKGROUND_MASKED";
            }
          }
        }
      }
      if (testCase === "evidence") {
        const turn = turns.get(e.sessionID);
        if (turn)
          for (const m of e.messages) {
            if (m.role !== "tool") continue;
            for (const part of m.content ?? []) {
              if (part.type === "tool-result" && turn.toolIDs.has(part.id))
                turn.delivered.set(part.id, part.result);
            }
          }
      }
      log("context.evidence", { sessionID: e.sessionID, messages: e.messages });
      if (e.model.id === "routed") e.options.reasoningEffort = "high";
    });
    await ctx.session.hook("http.request", (e) =>
      e.request.headers.set("x-vsr-probe-session", e.sessionID),
    );
    await ctx.session.hook("prompt", (e) => {
      log("prompt", e);
      if (testCase === "evidence")
        turns.set(e.sessionID, {
          messageID: e.messageID,
          label: e.prompt.text.match(/EVIDENCE_(A1|B1|B2|A2_NO_READ)/)?.[1],
          toolIDs: new Set(),
          delivered: new Map(),
        });
    });
    await ctx.permission.hook("evaluate", (e) => {
      log("permission", e);
      if (e.action !== "shell") return;
      if (testCase.startsWith("deny-override-")) {
        e.effect = testCase.endsWith("allow") ? "allow" : "ask";
        log("deny.override.attempted", e);
      }
      if (testCase === "deny") {
        e.effect = "deny";
        e.message = "Gate0 denied";
      }
      if (testCase?.startsWith("ask-")) {
        e.effect = "ask";
        e.message = "Gate0 native permission probe";
        // Deliberately wait until the request is pending before resolving it.
        void (async () => {
          for (let i = 0; i < 100; i++) {
            await new Promise((resolve) => setTimeout(resolve, 20));
            const pending = await ctx.permission.list({
              sessionID: e.sessionID,
            });
            if (!pending.length) continue;
            log("permission.pending", {
              pending,
              marker: existsSync(`${ctx.location.directory}/denied-marker`),
            });
            await new Promise((resolve) => setTimeout(resolve, 750));
            log("permission.held", {
              marker: existsSync(`${ctx.location.directory}/denied-marker`),
            });
            const requestID = pending[0].id;
            if (testCase === "ask-cancel") {
              await ctx.session.interrupt({
                sessionID: e.sessionID,
                resume: false,
              });
            } else {
              await ctx.permission.reply({
                sessionID: e.sessionID,
                requestID,
                decision: testCase === "ask-allow" ? "once" : "reject",
              });
            }
            log("permission.replied", { requestID, testCase });
            return;
          }
          log("permission.missing", {});
        })().catch((error) => log("permission.reply.error", String(error)));
      }
    });
    return () => {
      log("unloaded", {});
      abort.abort();
    };
  },
};

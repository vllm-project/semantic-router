import assert from "node:assert/strict";
import { test } from "node:test";
import type { Reply, Result } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";
import { verify } from "../src/verify.ts";
import { report } from "../src/view.ts";

// Answers the way vela3-lite answers the Claude plugin's smoke cases.
function measured(failing = new Set<string>()) {
  const asked: string[] = [];
  return {
    asked,
    decide: async (
      state: unknown,
      questions: Record<string, unknown>,
    ): Promise<Result<Reply>> => {
      const key = Object.keys(questions)[0]!;
      asked.push(key);
      if (failing.has(key)) return { ok: false, code: "offline", elapsedMs: 1 };
      const text = typeof state === "string" ? state : "";
      const email = "maria.gonzalez@example.com";
      const at = text.indexOf(email);
      const answers: Record<string, Record<string, unknown>> = {
        kind: {
          kind: {
            choice: "destructive",
            probabilities: { destructive: 0.9 },
          },
          review: { noul: 0.98 },
        },
        pii: { pii: { noul: 0.99 } },
        jailbreak: { jailbreak: { probabilities: { jailbreak: 0.99 } } },
        complexity: {
          complexity: { score: 0.03, confidence: 0.9 },
          steps: { score: 0.17, confidence: 0.9 },
        },
        user_feedback: {
          user_feedback: { probabilities: { WRONG_ANSWER: 0.92 } },
        },
        hallucination: { hallucination: { noul: 0.65 } },
      };
      return {
        ok: true,
        elapsedMs: 10,
        value: {
          answers: answers[key]!,
          spans:
            key === "pii" && at >= 0
              ? {
                  pii: [
                    {
                      label: "EMAIL_ADDRESS",
                      text: email,
                      start: Array.from(text.slice(0, at)).length,
                      end: Array.from(text.slice(0, at)).length + email.length,
                      probability: 0.99,
                    },
                  ],
                }
              : {},
        },
      };
    },
  };
}
const all = () =>
  parseSettings({
    apps: {
      router: true,
      escalate: true,
      redact: true,
      shield: true,
      guard: true,
      check: true,
    },
    routing: {
      tiers: [
        { model: "p/light" },
        { model: "p/middle" },
        { model: "p/strong" },
      ],
    },
  });

test("each app's smoke case decides as measured on vela3-lite", async () => {
  const runtime = measured();
  let clock = 0;
  const probes = await verify(
    all(),
    runtime,
    new AbortController().signal,
    () => (clock += 5),
  );
  assert.deepEqual(
    probes.map((probe) => [probe.app, probe.verdict]),
    [
      ["router", "acts"],
      ["escalate", "acts"],
      ["redact", "acts"],
      ["shield", "acts"],
      ["guard", "acts"],
      ["check", "acts"],
    ],
  );
  assert.match(probes.find((p) => p.app === "redact")!.detail, /email/);
  assert.match(probes.find((p) => p.app === "guard")!.detail, /review 0\.98/);
  const text = report({ source: "connect", models: ["vela3-lite"], probes });
  assert.match(text, /^Runtime: from \/connect · models vela3-lite/);
  assert.equal(text.match(/✓/g)?.length, 6);
  assert.match(text, /Median \d+ ms/);
});

test("an app that is off is listed and not asked, and one the runtime does not answer fails", async () => {
  const settings = all();
  settings.apps.router = false;
  const runtime = measured(new Set(["hallucination", "kind"]));
  const probes = await verify(settings, runtime, new AbortController().signal);
  assert.ok(!runtime.asked.includes("complexity"));
  assert.deepEqual(
    Object.fromEntries(probes.map((probe) => [probe.app, probe.verdict])),
    {
      router: "off",
      escalate: "acts",
      redact: "acts",
      shield: "acts",
      guard: "no answer",
      check: "no answer",
    },
  );
  const text = report({ source: "options", models: [], probes });
  assert.match(text, /– Router off/);
  assert.match(
    text,
    /✗ Guard no answer · git push --force origin main · offline/,
  );
  assert.match(text, /slower than the app's wait/);
  assert.doesNotMatch(
    text,
    /Escalate acts.*offline/,
    "an app that answered names no failure",
  );
});

test("redact counts only what the model found: secret patterns alone are not an answer", async () => {
  const settings = all();
  const runtime = measured(new Set(["pii"]));
  const probes = await verify(settings, runtime, new AbortController().signal);
  assert.equal(probes.find((p) => p.app === "redact")?.verdict, "no answer");
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { MEASURED_AT, PROBES, probe } from "../src/probes.ts";
import type { ProbeRow } from "../src/probes.ts";
import type { Reply, Result } from "../src/runtime.ts";
import { modelChanged, probeText } from "../src/view.ts";

const CLAUDE_CODE = new URL(
  "../../claude-code/skills/setup/scripts/probes.jsonl",
  import.meta.url,
);

test("the known cases are the Claude Code plugin's, so both probes measure the same thing", () => {
  const theirs = readFileSync(CLAUDE_CODE, "utf8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => {
      const { smoke: _, ...found } = JSON.parse(line);
      return found;
    });
  assert.deepEqual(PROBES, theirs);
  assert.equal(PROBES.length, 30);
});

// A runtime that answers each case with the score `scores` gives its name, and records what it was asked.
function runtime(scores: Record<string, number | "fail">) {
  const asked: unknown[] = [];
  return {
    asked,
    decide: async (
      state: string | Record<string, string>,
      questions: Record<string, unknown>,
    ): Promise<Result<Reply>> => {
      asked.push(state);
      const text = typeof state === "string" ? state : JSON.stringify(state);
      const found = PROBES.find((one) =>
        text.includes(
          typeof one.input === "string"
            ? one.input
            : (one.input.current ?? one.input.text ?? one.input.answer)!,
        ),
      )!;
      const given = scores[found.name] ?? found.measured ?? 0;
      if (given === "fail") return { ok: false, code: "timeout", elapsedMs: 5 };
      const key = Object.keys(questions)[0]!;
      const answers: Record<string, unknown> =
        key === "kind"
          ? { kind: { choice: "safe" }, review: { noul: given } }
          : key === "jailbreak"
            ? { jailbreak: { probabilities: { jailbreak: given } } }
            : key === "user_feedback"
              ? { user_feedback: { probabilities: { WRONG_ANSWER: given } } }
              : key === "complexity"
                ? { complexity: { score: given, confidence: 0.9 }, steps: {} }
                : key === "pii"
                  ? { pii: { noul: 0.9 } }
                  : { [key]: { noul: given } };
      const spans =
        key === "pii" && found.expect === "acts"
          ? {
              pii: [{ label: "EMAIL_ADDRESS", probability: 0.8 }],
            }
          : { pii: [] };
      return { ok: true, value: { answers, spans }, elapsedMs: 5 };
    },
  };
}
const thresholds = () => structuredClone(MEASURED_AT);

test("a probe judges each case at the thresholds it was measured at: ok, flip, drift or no answer", async () => {
  const fake = runtime({
    "git push origin main": 0.93, // over 0.9: flipped from passes
    "rm -rf ~": 0.55, // still passes, but moved 0.23
    "git status": "fail",
  });
  const mine = { ...thresholds(), guard: [0.95] };
  const rows = await probe(
    ["guard", "escalate", "router", "redact", "shield", "check"],
    mine,
    fake,
    new AbortController().signal,
  );
  const of = (name: string) => rows.find((row) => row.name === name)!;
  assert.equal(rows.length, 30);
  assert.equal(of("git push origin main").status, "flip");
  assert.equal(of("git push origin main").now, "acts");
  assert.equal(
    of("git push origin main").yours,
    "passes",
    "yours: at the guard line in use, 0.95",
  );
  assert.equal(of("rm -rf ~").status, "drift");
  assert.equal(of("rm -rf ~").drift, -0.23);
  assert.equal(of("git status").status, "error");
  assert.equal(of("git status").error, "timeout");
  assert.equal(of("git reset --hard HEAD~1").status, "ok");
  assert.equal(of("List the files in this repo.").now, "light");
  assert.equal(of("support ticket with personal data").now, "acts");
  assert.equal(of("code with no personal data").now, "passes");
  const repeat = of("Unrelated follow-up");
  assert.equal(repeat.app, "escalate");
  assert.equal(repeat.repeat, true);
  assert.equal(
    rows.filter((row) => row.status === "ok").length,
    27,
    JSON.stringify(rows.filter((row) => row.status !== "ok")),
  );
  // Framed as the hooks frame them.
  assert.ok(fake.asked.includes("Tool: Bash\nCommand: git push origin main"));
  assert.ok(
    fake.asked.some(
      (state) =>
        typeof state === "string" &&
        state.startsWith("Contents of vendor/INSTALL.md:\n# Acme"),
    ),
  );

  const guardOnly = await probe(
    ["guard"],
    mine,
    runtime({}),
    new AbortController().signal,
  );
  assert.equal(guardOnly.length, 13);
});

test("the probe says in plain words what flipped or drifted, and warns when the model changed since calibrating", () => {
  const row = (extra: Partial<ProbeRow>): ProbeRow => ({
    app: "guard",
    name: "",
    expect: "passes",
    status: "ok",
    elapsedMs: 1,
    ...extra,
  });
  const rows = [
    row({ name: "git status", measured: 0, score: 0.01, now: "passes" }),
    row({
      name: "git push origin main",
      measured: 0.76,
      score: 0.93,
      now: "acts",
      status: "flip",
    }),
    row({
      name: "rm -rf ~",
      measured: 0.78,
      score: 0.55,
      now: "passes",
      status: "drift",
    }),
  ];
  const changed = { model: "vela3-fast", calibratedModel: "vela3-lite" };
  assert.equal(modelChanged(changed), true);
  assert.equal(modelChanged({ model: "vela3-lite" }), false);
  const text = probeText({ ...changed, rows });
  assert.match(
    text,
    /^Decision model: vela3-fast; you last calibrated with vela3-lite\./,
  );
  assert.match(
    text,
    /Guard, 3 known cases: 1 as measured, 1 flipped, 1 drifted\./,
  );
  assert.match(
    text,
    /Flipped: git push origin main: vSR holds it now \(0\.93\), measured it runs \(0\.76\)\./,
  );
  assert.match(text, /Drifted: rm -rf ~: 0\.78 → 0\.55\./);
  assert.match(text, /answer some newer decisions before you apply/);
  assert.doesNotMatch(
    probeText({ model: "vela3-lite", rows }),
    /newer decisions/,
    "no warning when the model is the one calibrated with",
  );
});

import assert from "node:assert/strict";
import { test } from "node:test";
import { inspectImages } from "../src/output-policy.ts";
import type { FilePart } from "../src/output-policy.ts";
import type { Reply, Result } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";

const png = (tag: string): FilePart => ({
  type: "file",
  uri: `data:image/png;base64,${Buffer.from(tag).toString("base64")}`,
  mime: "image/png",
  name: `${tag}.png`,
});
// A runtime that reads images and answers by the image's tag: PII, INJECT, FAIL, or anything else as benign.
function reader(images = true) {
  const asked: {
    state: unknown;
    questions: string[];
    images?: readonly string[];
  }[] = [];
  return {
    asked,
    supports: (feature: { preset: string } | { modality: string }) =>
      images && "modality" in feature && feature.modality === "image",
    decide: async (
      state: unknown,
      questions: Record<string, unknown>,
      _timeout: number,
      _signal?: AbortSignal,
      images?: readonly string[],
    ): Promise<Result<Reply>> => {
      asked.push({ state, questions: Object.keys(questions), images });
      const tag = Buffer.from(images![0]!.split(",")[1]!, "base64").toString();
      if (tag === "FAIL") return { ok: false, code: "timeout", elapsedMs: 1 };
      return {
        ok: true,
        elapsedMs: 1,
        value: {
          answers: {
            image_pii: { noul: tag === "PII" ? 0.99 : 0.01 },
            jailbreak: {
              probabilities: { jailbreak: tag === "INJECT" ? 0.99 : 0.02 },
            },
          },
          spans: {},
        },
      };
    },
  };
}

test("an image showing personal data is withheld, one carrying instructions is passed with shield's note", async () => {
  const settings = parseSettings({});
  const runtime = reader();
  const result = await inspectImages(
    "read",
    [png("CODE"), png("PII"), png("INJECT")],
    settings,
    runtime,
  );
  assert.deepEqual(
    result?.files.map((file) => file.name),
    ["CODE.png", "INJECT.png"],
  );
  assert.equal(result?.unchecked, 0);
  assert.deepEqual(
    result?.decisions.map((d) => [d.app, d.outcome]),
    [
      ["redact", "redacted"],
      ["shield", "shielded"],
    ],
  );
  assert.match(result!.notes[0]!, /^vSR redact withheld an image/);
  assert.match(
    result!.notes[1]!,
    /^vSR shield: an image .* may carry prompt injection/,
  );
  // Each image is asked alone, both apps' questions in one request, with no words beside it: not even its name.
  assert.deepEqual(
    runtime.asked.map((one) => [one.state, one.questions, one.images?.length]),
    Array(3).fill(["", ["image_pii", "jailbreak"], 1]),
  );
});

test("an image the model cannot read, or did not answer, stays unchecked and the decision partial", async () => {
  const settings = parseSettings({});
  const pdf: FilePart = {
    type: "file",
    uri: "data:application/pdf;base64,JVBERi0=",
    mime: "application/pdf",
  };
  const result = await inspectImages(
    "read",
    [png("FAIL"), pdf, png("CODE")],
    settings,
    reader(),
  );
  assert.equal(result?.files.length, 3, "nothing unchecked is dropped");
  assert.equal(result?.unchecked, 2);
  assert.deepEqual(
    result?.decisions.map((d) => [d.app, d.outcome, d.reason]),
    [
      ["redact", "partial", "timeout"],
      ["shield", "partial", "timeout"],
    ],
  );
});

test("no image question goes to a model that reads no images, or for an app off on that tool", async () => {
  const settings = parseSettings({});
  assert.equal(
    await inspectImages("read", [png("PII")], settings, reader(false)),
    undefined,
  );
  // webfetch output is shield's alone, as for its text; shell output is redact's alone.
  const runtime = reader();
  const fetched = await inspectImages(
    "webfetch",
    [png("PII")],
    settings,
    runtime,
  );
  assert.deepEqual(runtime.asked[0]?.questions, ["jailbreak"]);
  assert.equal(fetched?.files.length, 1, "shield alone never withholds");
  settings.apps.redact = false;
  assert.equal(
    await inspectImages("shell", [png("PII")], settings, reader()),
    undefined,
  );
});

test("at most eight images are asked; the rest stay unchecked", async () => {
  const runtime = reader();
  const result = await inspectImages(
    "read",
    Array.from({ length: 10 }, (_, index) => png(`CODE${index}`)),
    parseSettings({}),
    runtime,
  );
  assert.equal(runtime.asked.length, 8);
  assert.equal(result?.unchecked, 2);
  assert.equal(result?.files.length, 10);
});

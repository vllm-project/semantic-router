import assert from "node:assert/strict";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { fixture } from "./output-fixture.ts";
import type { ViewSnapshot } from "../../src/view.ts";

test(
  "native /connect names the decision runtime and keeps its token across a restart, and vsr-setup is offered",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t);
    const first = await native.run("OUTPUT_CASE READ");
    // The way a person finds it: /connect, search, open its methods.
    const screen = await native.terminal(
      first.sessionID,
      [
        { at: 2000, keys: "/connect" },
        { at: 2800, keys: "\r" },
        { at: 4000, keys: "vSR decision" },
        { at: 5000, keys: "\r" },
      ],
      7,
    );
    assert.ok(
      screen.text.includes("vSR decision runtime") &&
        screen.text.includes("Runtime without a token") &&
        screen.text.includes("Runtime with a token"),
      screen.text.slice(-6000),
    );
    const integration = (await native.api("/api/integration/vsr")) as {
      data: { name: string; methods: { type: string }[] };
    };
    assert.equal(integration.data.name, "vSR decision runtime");
    assert.deepEqual(
      integration.data.methods.map((method) => method.type),
      ["key", "external"],
    );
    const skills = (await native.api("/api/skill")) as {
      data: { name: string; path: string; content: string }[];
    };
    const setup = skills.data.find((skill) => skill.name === "vsr-setup");
    assert.ok(setup, JSON.stringify(skills.data.map((skill) => skill.name)));
    assert.match(setup.path, /skills\/vsr-setup\/SKILL\.md$/);
    assert.match(setup.content, /\/connect/);
    assert.ok(native.asked.length > 0);
    assert.ok(native.asked.every((call) => !call.authorization));
    assert.ok(native.asked.every((call) => call.host?.startsWith("127.0.0.1")));

    // localhost reaches the same fixture under another URL, so the requests show which runtime vSR asked.
    await native.api("/api/integration/vsr/connect/key", {
      key: "connect-fixture-token",
      answer: { url: `http://localhost:${native.port}/v1/decisions` },
    });
    const source = async () =>
      (
        (await native.rpc("snapshot", {
          sessionID: first.sessionID,
        })) as ViewSnapshot
      ).runtime.source;
    for (let i = 0; i < 50 && (await source()) !== "connect"; i++)
      await delay(100);
    assert.equal(await source(), "connect");

    const before = native.asked.length;
    await native.run("OUTPUT_CASE READ");
    const after = native.asked.slice(before);
    assert.ok(after.length > 0);
    assert.ok(
      after.every(
        (call) =>
          call.authorization === "Bearer connect-fixture-token" &&
          call.host === `localhost:${native.port}`,
      ),
      JSON.stringify(after),
    );
    const snapshot = JSON.stringify(
      await native.rpc("snapshot", { sessionID: first.sessionID }),
    );
    assert.ok(!snapshot.includes("connect-fixture-token"));

    await native.restart();
    assert.equal(
      await source(),
      "connect",
      "the credential outlives the server",
    );
  },
);

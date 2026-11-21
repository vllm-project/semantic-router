import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn, execFileSync } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";

const packageRoot = resolve(import.meta.dirname, "../..");
test(
  "packed plugin loads in OpenCode 2.2.0 and exposes authenticated bounded diagnostics",
  { timeout: 60000 },
  async (t) => {
    assert.equal(
      execFileSync("opencode", ["--version"], { encoding: "utf8" }).trim(),
      "opencode v2.2.0",
    );
    const root = await mkdtemp(join(tmpdir(), "vsr-opencode-package-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const [packed] = JSON.parse(
      execFileSync("npm", ["pack", "--json", "--pack-destination", root], {
        cwd: packageRoot,
        encoding: "utf8",
      }),
    ) as { filename: string; files: { path: string }[] }[];
    assert.ok(packed);
    assert.ok(packed.files.some((file) => file.path === "src/index.ts"));
    assert.ok(
      packed.files.every(
        (file) =>
          !file.path.startsWith("test/") && !file.path.includes("node_modules"),
      ),
    );
    execFileSync("tar", ["-xzf", join(root, packed.filename), "-C", root]);
    let authorization: string | undefined;
    const fixture = createServer((req, res) => {
      authorization = req.headers.authorization;
      res.setHeader("content-type", "application/json");
      if (req.url === "/v1/systemone/models") {
        res.writeHead(404);
        res.end("{}");
      } else
        res.end(
          JSON.stringify({
            data: [
              { id: "fixture", question_types: ["score", "choice", "noul"] },
            ],
          }),
        );
    });
    fixture.listen(0, "127.0.0.1");
    await once(fixture, "listening");
    t.after(() => {
      fixture.closeAllConnections();
      fixture.close();
    });
    const fixtureAddress = fixture.address();
    assert.ok(fixtureAddress && typeof fixtureAddress !== "string");
    const probe = createServer();
    probe.listen(0, "127.0.0.1");
    await once(probe, "listening");
    const address = probe.address();
    assert.ok(address && typeof address !== "string");
    await new Promise<void>((resolve) => probe.close(() => resolve()));
    const url = `http://127.0.0.1:${address.port}`;
    await writeFile(
      join(root, "opencode.json"),
      JSON.stringify({
        plugins: [
          {
            package: join(root, "package"),
            options: {
              runtimeUrl: `http://127.0.0.1:${fixtureAddress.port}/v1/decisions`,
              runtimeTokenEnv: "VSR_PACKED_TEST_TOKEN",
            },
          },
        ],
      }),
    );
    const env: NodeJS.ProcessEnv = Object.fromEntries(
      ["PATH", "USER", "SHELL", "LANG", "TERM"].flatMap((key) =>
        process.env[key] ? [[key, process.env[key]]] : [],
      ),
    );
    for (const key of [
      "XDG_CONFIG_HOME",
      "XDG_DATA_HOME",
      "XDG_CACHE_HOME",
      "XDG_STATE_HOME",
    ]) {
      env[key] = join(root, key);
      await mkdir(env[key]!);
    }
    env.OPENCODE_PASSWORD = "packed-fixture-password";
    env.VSR_PACKED_TEST_TOKEN = "synthetic-private-token";
    const server = spawn(
      "opencode",
      ["serve", "--hostname", "127.0.0.1", "--port", String(address.port)],
      { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] },
    );
    let logs = "";
    server.stdout.on("data", (data) => {
      logs += String(data);
    });
    server.stderr.on("data", (data) => {
      logs += String(data);
    });
    t.after(async () => {
      if (server.exitCode === null) {
        server.kill("SIGTERM");
        await once(server, "exit");
      }
    });
    const headers = {
      authorization: `Basic ${Buffer.from("opencode:packed-fixture-password").toString("base64")}`,
      "content-type": "application/json",
    };
    let ready = false;
    for (let i = 0; i < 150; i++) {
      try {
        const result = await fetch(url + "/global/health", { headers });
        if (result.ok) {
          ready = true;
          break;
        }
      } catch {
        /* server starting */
      }
      assert.equal(server.exitCode, null, logs);
      await delay(100);
    }
    assert.ok(ready, logs);
    async function rpc(method: string, input: unknown): Promise<Response> {
      return fetch(`${url}/api/rpc/vsr.opencode.v1/${method}`, {
        method: "POST",
        headers,
        body: JSON.stringify({ input }),
      });
    }
    const response = await rpc("snapshot", { sessionID: "fixture-session" });
    assert.equal(response.status, 200, await response.clone().text());
    const snapshot = (await response.json()) as {
      output: {
        sessionID: string;
        decisions: unknown[];
        apps: Record<string, boolean>;
        runtime: { reachable: boolean | null };
      };
    };
    assert.equal(snapshot.output.sessionID, "fixture-session");
    assert.deepEqual(snapshot.output.decisions, []);
    assert.equal(snapshot.output.apps.check, false);
    assert.equal(snapshot.output.runtime.reachable, null);
    assert.equal(
      authorization,
      undefined,
      "loading the package must not call a runtime",
    );
    const diagnostics = await rpc("diagnose", {});
    assert.equal(diagnostics.status, 200, await diagnostics.clone().text());
    const diagnosis = (await diagnostics.json()) as {
      output: { ok: boolean; value: { id: string }[] };
    };
    assert.equal(diagnosis.output.ok, true);
    assert.equal(diagnosis.output.value[0]?.id, "fixture");
    assert.equal(authorization, "Bearer synthetic-private-token");
    assert.ok(!JSON.stringify(snapshot).includes("synthetic-private-token"));
    assert.ok(!JSON.stringify(diagnosis).includes("synthetic-private-token"));
    assert.equal((await rpc("snapshot", { sessionID: "" })).status, 400);
    assert.equal(
      (
        await rpc("feedback", {
          sessionID: "fixture-session",
          decisionID: "missing",
          feedback: "invalid",
        })
      ).status,
      400,
    );
    assert.ok(
      !(await readFile(join(root, "package/src/index.ts"), "utf8")).includes(
        "claude-code/",
      ),
    );
  },
);

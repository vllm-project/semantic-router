import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { TestContext } from "node:test";
import { setTimeout as delay } from "node:timers/promises";

export async function host(t: TestContext, config: Record<string, unknown>) {
  assert.equal(
    execFileSync("opencode", ["--version"], { encoding: "utf8" }).trim(),
    "opencode v2.2.0",
  );
  const root = await mkdtemp(join(tmpdir(), "vsr-opencode-native-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const [packed] = JSON.parse(
    execFileSync("npm", ["pack", "--json", "--pack-destination", root], {
      cwd: resolve(import.meta.dirname, "../.."),
      encoding: "utf8",
    }),
  ) as { filename: string }[];
  assert.ok(packed);
  execFileSync("tar", ["-xzf", join(root, packed.filename), "-C", root]);
  await writeFile(
    join(root, "opencode.json"),
    JSON.stringify({
      ...config,
      plugins: [
        { package: join(root, "package"), options: config.pluginOptions },
      ],
      pluginOptions: undefined,
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
  env.OPENCODE_PASSWORD = "native-fixture-password";
  const port = createServer();
  port.listen(0, "127.0.0.1");
  await once(port, "listening");
  const address = port.address();
  assert.ok(address && typeof address !== "string");
  await new Promise<void>((resolve) => port.close(() => resolve()));
  const url = `http://127.0.0.1:${address.port}`;
  let logs = "";
  const start = () => {
    const child = spawn(
      "opencode",
      ["serve", "--hostname", "127.0.0.1", "--port", String(address.port)],
      { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] },
    );
    child.stdout.on("data", (data) => {
      logs += String(data);
    });
    child.stderr.on("data", (data) => {
      logs += String(data);
    });
    return child;
  };
  let server = start();
  t.after(async () => {
    if (server.exitCode === null) {
      server.kill("SIGTERM");
      await once(server, "exit");
    }
  });
  const headers = {
    authorization: `Basic ${Buffer.from("opencode:native-fixture-password").toString("base64")}`,
    "content-type": "application/json",
  };
  async function waitReady() {
    let ready = false;
    for (let i = 0; i < 150; i++) {
      try {
        if ((await fetch(url + "/global/health", { headers })).ok) {
          ready = true;
          break;
        }
      } catch {
        /* starting */
      }
      assert.equal(server.exitCode, null, logs);
      await delay(100);
    }
    assert.ok(ready, logs);
  }
  await waitReady();
  return {
    root,
    async restart() {
      server.kill("SIGTERM");
      await once(server, "exit");
      server = start();
      await waitReady();
    },
    async terminal(
      sessionID: string,
      steps: { at: number; keys?: string; resize?: [number, number] }[],
      duration = 15,
    ): Promise<{ text: string; exit: number }> {
      const driver = spawn(
        "python3",
        [resolve(import.meta.dirname, "terminal.py")],
        { cwd: root, env, stdio: ["pipe", "pipe", "pipe"] },
      );
      let output = "",
        error = "";
      driver.stdout.on("data", (data) => {
        output += String(data);
      });
      driver.stderr.on("data", (data) => {
        error += String(data);
      });
      driver.stdin.end(JSON.stringify({ url, sessionID, steps, duration }));
      const [code] = await once(driver, "exit");
      assert.equal(code, 0, error);
      return JSON.parse(output) as { text: string; exit: number };
    },
    async api(path: string, body?: unknown, method?: string): Promise<unknown> {
      const response = await fetch(url + path, {
        headers,
        method: method ?? (body === undefined ? "GET" : "POST"),
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
      assert.ok(response.ok, await response.clone().text());
      return response.status === 204 ? undefined : await response.json();
    },
    async rpc(method: string, input: unknown) {
      const response = await fetch(`${url}/api/rpc/vsr.opencode.v1/${method}`, {
        method: "POST",
        headers,
        body: JSON.stringify({ input }),
      });
      assert.equal(response.status, 200, await response.clone().text());
      return ((await response.json()) as { output: unknown }).output;
    },
    async run(prompt: string, sessionID?: string) {
      const args = [
        "run",
        "--server",
        url,
        "--format",
        "json",
        ...(typeof config.model === "string" ? ["--model", config.model] : []),
        ...(sessionID ? ["--session", sessionID] : []),
        prompt,
      ];
      const child = spawn("opencode", args, {
        cwd: root,
        env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      let output = "",
        error = "";
      child.stdout.on("data", (data) => {
        output += String(data);
      });
      child.stderr.on("data", (data) => {
        error += String(data);
      });
      const timer = setTimeout(() => child.kill("SIGKILL"), 25000);
      const [code] = await once(child, "exit");
      clearTimeout(timer);
      assert.equal(code, 0, error + output + logs);
      const events = output
        .split("\n")
        .filter((line) => line.startsWith("{"))
        .map((line) => JSON.parse(line));
      assert.ok(!events.some((event) => event.type === "error"), output);
      const id = events.find((event) => event.sessionID)?.sessionID as
        | string
        | undefined;
      assert.ok(id, error + output + logs);
      return { sessionID: id, events, output };
    },
  };
}

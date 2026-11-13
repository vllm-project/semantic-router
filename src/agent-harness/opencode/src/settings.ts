import { parseRouting } from "./routing-settings.ts";
import type { RoutingSettings } from "./routing-settings.ts";
export const APPS = [
  "router",
  "escalate",
  "redact",
  "shield",
  "guard",
  "check",
] as const;
export type App = (typeof APPS)[number];
export type Wait = 1000 | 2000 | 3000 | "done";
export interface Settings {
  guard: { reviewThreshold: number };
  redact: { piiThreshold: number };
  shield: { injectionThreshold: number };
  routing: RoutingSettings;
  runtimeUrl: string;
  runtimeModel: string;
  runtimeTokenEnv: string;
  apps: Record<App, boolean>;
  waits: { redact: Wait; shield: Wait; guard: Wait };
  maxDecisions: number;
  maxSessions: number;
}
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function string(value: unknown, fallback: string, key: string): string {
  if (value === undefined) return fallback;
  if (typeof value !== "string") throw new Error(`Invalid ${key}`);
  return value;
}
function limit(
  value: unknown,
  fallback: number,
  maximum: number,
  key: string,
): number {
  if (value === undefined) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < 1 ||
    value > maximum
  )
    throw new Error(`Invalid ${key}`);
  return value;
}
function wait(value: unknown, fallback: Wait): Wait {
  if (value === undefined) return fallback;
  if (value === "done" || value === 1000 || value === 2000 || value === 3000)
    return value;
  throw new Error("Invalid wait; use 1000, 2000, 3000 or done");
}
export function parseSettings(input: unknown): Settings {
  if (input !== undefined && !isRecord(input))
    throw new Error("Invalid plugin options");
  const options = input ?? {};
  const known = new Set([
    "runtimeUrl",
    "runtimeModel",
    "runtimeTokenEnv",
    "apps",
    "waits",
    "maxDecisions",
    "maxSessions",
    "routing",
    "guard",
    "redact",
    "shield",
  ]);
  for (const key of Object.keys(options))
    if (!known.has(key)) throw new Error(`Unknown option: ${key}`);
  const urlText = string(
    options.runtimeUrl,
    "http://127.0.0.1:8790/v1/decisions",
    "runtimeUrl",
  );
  let url: URL;
  try {
    url = new URL(urlText);
  } catch {
    throw new Error("Invalid runtimeUrl");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    !url.pathname.endsWith("/v1/decisions")
  )
    throw new Error(
      "runtimeUrl must end in /v1/decisions without embedded credentials, query or fragment",
    );
  const tokenEnv = string(
    options.runtimeTokenEnv,
    "VSR_RUNTIME_TOKEN",
    "runtimeTokenEnv",
  );
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(tokenEnv))
    throw new Error("Invalid runtimeTokenEnv");
  if (options.apps !== undefined && !isRecord(options.apps))
    throw new Error("Invalid apps");
  const flags = options.apps ?? {};
  for (const key of Object.keys(flags))
    if (!APPS.includes(key as App)) throw new Error(`Unknown app: ${key}`);
  const apps = Object.fromEntries(
    APPS.map((app) => {
      const value = flags[app];
      if (value !== undefined && typeof value !== "boolean")
        throw new Error(`Invalid app: ${app}`);
      return [app, value ?? app !== "check"];
    }),
  ) as Record<App, boolean>;
  if (options.waits !== undefined && !isRecord(options.waits))
    throw new Error("Invalid waits");
  const waits = options.waits ?? {};
  for (const key of Object.keys(waits))
    if (!["redact", "shield", "guard"].includes(key))
      throw new Error(`Unknown wait: ${key}`);
  const guard = options.guard === undefined ? {} : options.guard;
  if (
    !isRecord(guard) ||
    Object.keys(guard).some((key) => key !== "reviewThreshold")
  )
    throw new Error("Invalid guard options");
  const reviewThreshold =
    guard.reviewThreshold === undefined ? 0.9 : guard.reviewThreshold;
  if (
    typeof reviewThreshold !== "number" ||
    !Number.isFinite(reviewThreshold) ||
    reviewThreshold < 0 ||
    reviewThreshold > 1
  )
    throw new Error("Invalid guard reviewThreshold");
  const threshold = (app: string, key: string, fallback: number): number => {
    const input = options[app] === undefined ? {} : options[app];
    if (!isRecord(input) || Object.keys(input).some((name) => name !== key))
      throw new Error(`Invalid ${app} options`);
    const value = input[key] === undefined ? fallback : input[key];
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value < 0 ||
      value > 1
    )
      throw new Error(`Invalid ${app} threshold`);
    return value;
  };
  return {
    guard: { reviewThreshold },
    redact: { piiThreshold: threshold("redact", "piiThreshold", 0.7) },
    shield: {
      injectionThreshold: threshold("shield", "injectionThreshold", 0.9),
    },
    routing: parseRouting(options.routing),
    runtimeUrl: url.href,
    runtimeModel: string(options.runtimeModel, "", "runtimeModel"),
    runtimeTokenEnv: tokenEnv,
    apps,
    waits: {
      redact: wait(waits.redact, 3000),
      shield: wait(waits.shield, 2000),
      guard: wait(waits.guard, 3000),
    },
    maxDecisions: limit(options.maxDecisions, 500, 5000, "maxDecisions"),
    maxSessions: limit(options.maxSessions, 64, 512, "maxSessions"),
  };
}

import { APPS, isRecord } from "./settings.ts";
import type { App, Settings } from "./settings.ts";

export type Thresholds = Record<App, number[]>;
// After this many decisions since the last calibration, the decision view suggests another.
export const REMIND_EVERY = 300;

// The thresholds each app holds its scores against, in the order its log lines keep them.
export function thresholdsOf(settings: Settings): Thresholds {
  return {
    router: [
      settings.routing.easyBelow,
      settings.routing.hardFrom,
      settings.routing.minConfidence,
    ],
    escalate: [
      settings.routing.feedbackThreshold,
      settings.routing.repeatThreshold,
    ],
    redact: [settings.redact.piiThreshold],
    shield: [settings.shield.injectionThreshold],
    guard: [settings.guard.reviewThreshold],
    check: [settings.check.unsupportedThreshold],
  };
}
function assign(settings: Settings, app: App, values: readonly number[]): void {
  if (app === "router")
    [
      settings.routing.easyBelow,
      settings.routing.hardFrom,
      settings.routing.minConfidence,
    ] = values as [number, number, number];
  else if (app === "escalate")
    [settings.routing.feedbackThreshold, settings.routing.repeatThreshold] =
      values as [number, number];
  else if (app === "redact") settings.redact.piiThreshold = values[0]!;
  else if (app === "shield") settings.shield.injectionThreshold = values[0]!;
  else if (app === "guard") settings.guard.reviewThreshold = values[0]!;
  else settings.check.unsupportedThreshold = values[0]!;
}
function valid(app: App, values: unknown): values is number[] {
  const size = app === "router" ? 3 : app === "escalate" ? 2 : 1;
  return (
    Array.isArray(values) &&
    values.length === size &&
    values.every(
      (value, i) =>
        typeof value === "number" &&
        Number.isFinite(value) &&
        value >= 0 &&
        value <= (app === "router" && i < 2 ? 2 : 1),
    ) &&
    (app !== "router" || values[0] < values[1])
  );
}

interface Saved {
  version: 1;
  // The options' thresholds when it was calibrated: changing them in opencode.json drops the calibration.
  base: string;
  at: number;
  thresholds: Partial<Thresholds>;
}
export interface CalibrationState {
  at?: number;
  apps: App[];
}

// Calibrated thresholds, kept in plugin storage and applied over the options' own.
export class Calibrated {
  private readonly settings: Settings;
  private readonly base: string;
  private readonly store: {
    get(key: string): Promise<unknown>;
    set(key: string, value: Saved): Promise<void>;
  };
  private saved: Saved | undefined;
  constructor(
    settings: Settings,
    store: {
      get(key: string): Promise<unknown>;
      set(key: string, value: Saved): Promise<void>;
    },
  ) {
    this.settings = settings;
    this.store = store;
    this.base = JSON.stringify(thresholdsOf(settings));
  }
  async load(): Promise<void> {
    let value: unknown;
    try {
      value = await this.store.get("calibration.v1");
    } catch {
      return;
    }
    if (
      !isRecord(value) ||
      value.version !== 1 ||
      value.base !== this.base ||
      typeof value.at !== "number" ||
      !isRecord(value.thresholds)
    )
      return;
    const thresholds: Partial<Thresholds> = {};
    for (const app of APPS) {
      const values = value.thresholds[app];
      if (values === undefined) continue;
      if (!valid(app, values)) return;
      thresholds[app] = values;
    }
    this.saved = { version: 1, base: this.base, at: value.at, thresholds };
    for (const [app, values] of Object.entries(thresholds))
      assign(this.settings, app as App, values);
  }
  // Applies and keeps one app's new thresholds; the settings change only once they are stored.
  async apply(
    app: App,
    values: readonly number[],
    at = Date.now(),
  ): Promise<void> {
    if (!valid(app, [...values])) throw new Error("Invalid thresholds");
    const next: Saved = {
      version: 1,
      base: this.base,
      at,
      thresholds: { ...this.saved?.thresholds, [app]: [...values] },
    };
    await this.store.set("calibration.v1", next);
    this.saved = next;
    assign(this.settings, app, values);
  }
  state(): CalibrationState {
    return {
      ...(this.saved ? { at: this.saved.at } : {}),
      apps: Object.keys(this.saved?.thresholds ?? {}) as App[],
    };
  }
}

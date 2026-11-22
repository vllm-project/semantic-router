import { APPS, isRecord } from "./settings.ts";
import type { App, Settings } from "./settings.ts";

export type Thresholds = Record<App, number[]>;
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
  // When the last calibration was applied; none when only /vsr-config set thresholds.
  at?: number;
  // The decision model the last calibration was made with, when it was known.
  model?: string;
  thresholds: Partial<Thresholds>;
  // The apps whose thresholds the person set in /vsr-config rather than a calibration fitted.
  manual?: App[];
}
export interface CalibrationState {
  at?: number;
  model?: string;
  apps: App[];
  manual: App[];
}

// Thresholds kept in plugin storage and applied over the options' own: the ones a calibration fitted and the ones the
// person set in /vsr-config, as the Claude Code plugin's calibration and /config write the same settings.
export class Calibrated {
  private readonly settings: Settings;
  private readonly base: string;
  private readonly store: {
    get(key: string): Promise<unknown>;
    set(key: string, value: Saved): Promise<void>;
  };
  private saved: Saved | undefined;
  // The options' own thresholds, which a reset in /vsr-config brings back.
  private readonly options: Thresholds;
  constructor(
    settings: Settings,
    store: {
      get(key: string): Promise<unknown>;
      set(key: string, value: Saved): Promise<void>;
    },
  ) {
    this.settings = settings;
    this.store = store;
    this.options = thresholdsOf(settings);
    this.base = JSON.stringify(this.options);
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
      (value.at !== undefined && typeof value.at !== "number") ||
      (value.model !== undefined && typeof value.model !== "string") ||
      !isRecord(value.thresholds) ||
      (value.manual !== undefined &&
        (!Array.isArray(value.manual) ||
          value.manual.some((app) => !APPS.includes(app as App))))
    )
      return;
    const thresholds: Partial<Thresholds> = {};
    for (const app of APPS) {
      const values = value.thresholds[app];
      if (values === undefined) continue;
      if (!valid(app, values)) return;
      thresholds[app] = values;
    }
    this.saved = {
      version: 1,
      base: this.base,
      ...(typeof value.at === "number" ? { at: value.at } : {}),
      ...(typeof value.model === "string" ? { model: value.model } : {}),
      thresholds,
      ...(Array.isArray(value.manual) && value.manual.length
        ? { manual: value.manual as App[] }
        : {}),
    };
    for (const [app, values] of Object.entries(thresholds))
      assign(this.settings, app as App, values);
  }
  // Applies and keeps one app's new thresholds; the settings change only once they are stored.
  async apply(
    app: App,
    values: readonly number[],
    model?: string,
    at = Date.now(),
  ): Promise<void> {
    if (!valid(app, [...values])) throw new Error("Invalid thresholds");
    await this.save(app, [...values], {
      at,
      ...(model ? { model } : {}),
      manual: (this.saved?.manual ?? []).filter((one) => one !== app),
    });
  }
  // Thresholds the person set in /vsr-config: kept and applied like a calibration's, and named as theirs.
  async set(app: App, values: readonly number[]): Promise<void> {
    if (!valid(app, [...values])) throw new Error("Invalid thresholds");
    await this.save(app, [...values], {
      manual: [...new Set([...(this.saved?.manual ?? []), app])],
    });
  }
  // Back to the options' own thresholds for an app.
  async reset(app: App): Promise<void> {
    await this.save(app, undefined, {
      manual: (this.saved?.manual ?? []).filter((one) => one !== app),
    });
  }
  private async save(
    app: App,
    values: number[] | undefined,
    change: Partial<Saved>,
  ): Promise<void> {
    const thresholds = { ...this.saved?.thresholds };
    if (values) thresholds[app] = values;
    else delete thresholds[app];
    const { manual, ...rest } = { ...this.saved, ...change };
    const next: Saved = {
      ...rest,
      version: 1,
      base: this.base,
      thresholds,
      ...(manual?.length ? { manual } : {}),
    };
    await this.store.set("calibration.v1", next);
    this.saved = next;
    assign(this.settings, app, values ?? this.options[app]);
  }
  // The options' own thresholds for an app.
  optionsOf(app: App): number[] {
    return [...this.options[app]];
  }
  state(): CalibrationState {
    return {
      ...(this.saved?.at === undefined ? {} : { at: this.saved.at }),
      ...(this.saved?.model ? { model: this.saved.model } : {}),
      apps: (Object.keys(this.saved?.thresholds ?? {}) as App[]).filter(
        (app) => !this.saved?.manual?.includes(app),
      ),
      manual: [...(this.saved?.manual ?? [])],
    };
  }
}

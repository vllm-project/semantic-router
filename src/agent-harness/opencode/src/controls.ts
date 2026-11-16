import { APPS, isRecord, parseSettings } from "./settings.ts";
import type { App, Settings } from "./settings.ts";

type Values = Pick<Settings, "apps" | "waits"> & {
  goal: Settings["routing"]["goal"];
};
// `pinned` names the apps the person changed here: a Router listener's profile leaves those alone.
type Saved = Values & {
  version: 1;
  base: string;
  revision: number;
  pinned?: App[];
};
export type ControlResult =
  | { saved: true; revision: number }
  | { saved: false; reason: "stale" | "invalid" | "storage" };
export class Controls {
  private readonly settings: Settings;
  private readonly instanceID: string;
  private readonly base: string;
  private readonly persist: (value: Saved) => Promise<void>;
  private readonly changed: () => void;
  private tail: Promise<unknown> = Promise.resolve();
  private closed = false;
  private revision = 0;
  private persistence: "durable" | "unavailable" = "durable";
  private pinned: App[] = [];
  private readonly pin: (apps: Partial<Record<App, boolean>>) => void;
  private readonly unpin: () => Promise<void>;
  constructor(
    settings: Settings,
    instanceID: string,
    persist: (value: Saved) => Promise<void>,
    changed: () => void,
    pin: (apps: Partial<Record<App, boolean>>) => void = () => {},
    unpin: () => Promise<void> = async () => {},
  ) {
    this.settings = settings;
    this.instanceID = instanceID;
    this.persist = persist;
    this.changed = changed;
    this.pin = pin;
    this.unpin = unpin;
    this.base = JSON.stringify(this.values());
  }
  private values(): Values {
    return {
      apps: { ...this.settings.apps },
      waits: { ...this.settings.waits },
      goal: this.settings.routing.goal,
    };
  }
  state() {
    return {
      revision: this.revision,
      persistence: this.persistence,
      goal: this.settings.routing.goal,
    };
  }
  private parse(input: Record<string, unknown>): Values {
    if (
      input.apps !== undefined &&
      (!isRecord(input.apps) ||
        Object.keys(input.apps).some((key) => !APPS.some((app) => app === key)))
    )
      throw new Error("Invalid app controls");
    if (
      input.waits !== undefined &&
      (!isRecord(input.waits) ||
        Object.keys(input.waits).some(
          (key) => !["redact", "shield", "guard"].includes(key),
        ))
    )
      throw new Error("Invalid wait controls");
    const next = parseSettings({
      ...this.settings,
      apps: { ...this.settings.apps, ...(input.apps as object) },
      waits: { ...this.settings.waits, ...(input.waits as object) },
      routing: {
        ...this.settings.routing,
        goal:
          input.goal === undefined ? this.settings.routing.goal : input.goal,
      },
    });
    return { apps: next.apps, waits: next.waits, goal: next.routing.goal };
  }
  private assign(value: Values): void {
    Object.assign(this.settings.apps, value.apps);
    Object.assign(this.settings.waits, value.waits);
    this.settings.routing.goal = value.goal;
    this.pin(
      Object.fromEntries(this.pinned.map((app) => [app, value.apps[app]])),
    );
  }
  async load(read: () => Promise<unknown>): Promise<void> {
    try {
      const saved = await read();
      if (saved === undefined) return;
      if (
        !isRecord(saved) ||
        saved.version !== 1 ||
        typeof saved.base !== "string" ||
        !isRecord(saved.apps) ||
        APPS.some(
          (app) =>
            typeof (saved.apps as Record<string, unknown>)[app] !== "boolean",
        ) ||
        !isRecord(saved.waits) ||
        ["redact", "shield", "guard"].some(
          (app) => !Object.hasOwn(saved.waits as object, app),
        ) ||
        saved.goal === undefined ||
        !Number.isSafeInteger(saved.revision) ||
        (saved.revision as number) < 0 ||
        (saved.pinned !== undefined &&
          (!Array.isArray(saved.pinned) ||
            saved.pinned.some((app) => !APPS.includes(app as App))))
      )
        throw new Error("Invalid saved controls");
      if (saved.base !== this.base) return; // Explicit changes to configured defaults win.
      const value = this.parse(saved);
      this.pinned = (saved.pinned as App[] | undefined) ?? [];
      this.assign(value);
      this.revision = saved.revision as number;
    } catch {
      this.persistence = "unavailable";
    }
  }
  update(input: unknown): Promise<ControlResult> {
    const transaction = async (): Promise<ControlResult> => {
      if (this.closed) return { saved: false, reason: "stale" };
      if (
        !isRecord(input) ||
        Object.keys(input).some(
          (key) =>
            ![
              "instanceID",
              "expectedRevision",
              "apps",
              "waits",
              "goal",
              "follow",
            ].includes(key),
        ) ||
        !["apps", "waits", "goal", "follow"].some((key) =>
          Object.hasOwn(input, key),
        ) ||
        (input.follow !== undefined &&
          (input.follow !== true || input.apps !== undefined))
      )
        return { saved: false, reason: "invalid" };
      if (
        input.instanceID !== this.instanceID ||
        input.expectedRevision !== this.revision
      )
        return { saved: false, reason: "stale" };
      let next: Values;
      try {
        next = this.parse(input);
      } catch {
        return { saved: false, reason: "invalid" };
      }
      const revision = this.revision + 1;
      // `follow` hands every app changed here back to a Router listener's profile.
      const pinned = input.follow
        ? []
        : APPS.filter(
            (app) =>
              this.pinned.includes(app) ||
              (isRecord(input.apps) && Object.hasOwn(input.apps, app)),
          );
      try {
        await this.persist({
          ...next,
          version: 1,
          base: this.base,
          revision,
          ...(pinned.length ? { pinned } : {}),
        });
      } catch {
        this.persistence = "unavailable";
        return { saved: false, reason: "storage" };
      }
      this.pinned = pinned;
      this.assign(next);
      this.revision = revision;
      this.persistence = "durable";
      if (input.follow) await this.unpin().catch(() => {});
      this.changed();
      return { saved: true, revision };
    };
    const result = this.tail.then(transaction);
    this.tail = result.catch(() => {});
    return result;
  }
  async close(): Promise<void> {
    this.closed = true;
    await this.tail;
  }
}

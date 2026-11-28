import { FIELDS } from "./listener.ts";
import type { Field, Values as Fields } from "./listener.ts";
import { APPS, isRecord, parseSettings } from "./settings.ts";
import type { Settings } from "./settings.ts";

type Values = Pick<Settings, "apps" | "waits"> & {
  goal: Settings["routing"]["goal"];
};
// `pinned` names what the person changed here: a Router listener's profile leaves those alone. `turns` is kept beside
// the rest and out of `base`, so the controls saved before it existed still load; an option that names it wins.
type Saved = Values & {
  version: 1;
  base: string;
  revision: number;
  pinned?: Field[];
  turns?: boolean;
  calibrateEvery?: number;
} & Partial<Record<Tune, number>>;
// The router's lines that calibration does not fit, set in /vsr-config and kept beside the rest, each with its largest
// value: the smaller-model line and the steps answer's three.
export const TUNES = {
  downgradeConfidence: 1,
  stepsLow: 2,
  stepsHigh: 2,
  stepsMinConfidence: 1,
} as const;
export type Tune = keyof typeof TUNES;
const TUNE_KEYS = Object.keys(TUNES) as Tune[];
const tuned = (key: Tune, value: unknown) =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= TUNES[key];
type Named = Partial<Record<"turns" | "calibrateEvery" | Tune, boolean>>;
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
  private pinned: Field[] = [];
  private readonly pin: (values: Partial<Fields>) => void;
  // What the plugin options name of these: then the saved choice does not override them.
  private readonly named: Named;
  // Told when the person turns the routing of their turns on or off, a repeat of the same choice included.
  turned: (on: boolean) => void = () => {};
  private readonly unpin: () => Promise<void>;
  constructor(
    settings: Settings,
    instanceID: string,
    persist: (value: Saved) => Promise<void>,
    changed: () => void,
    pin: (values: Partial<Fields>) => void = () => {},
    unpin: () => Promise<void> = async () => {},
    named: Named = {},
  ) {
    this.named = named;
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
      turns: this.settings.routing.turns,
      calibrateEvery: this.settings.calibrateEvery,
      ...this.tunes(),
    };
  }
  private tunes(): Record<Tune, number> {
    return Object.fromEntries(
      TUNE_KEYS.map((key) => [key, this.settings.routing[key]]),
    ) as Record<Tune, number>;
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
  private assign(value: Values, turns = this.settings.routing.turns): void {
    Object.assign(this.settings.apps, value.apps);
    Object.assign(this.settings.waits, value.waits);
    this.settings.routing.goal = value.goal;
    this.settings.routing.turns = turns;
    const fields: Fields = { ...value.apps, goal: value.goal, turns };
    this.pin(
      Object.fromEntries(this.pinned.map((field) => [field, fields[field]])),
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
            saved.pinned.some((field) => !FIELDS.includes(field as Field)))) ||
        (saved.turns !== undefined && typeof saved.turns !== "boolean") ||
        (saved.calibrateEvery !== undefined &&
          (!Number.isInteger(saved.calibrateEvery) ||
            (saved.calibrateEvery as number) < 0 ||
            (saved.calibrateEvery as number) > 100_000)) ||
        TUNE_KEYS.some(
          (key) => saved[key] !== undefined && !tuned(key, saved[key]),
        )
      )
        throw new Error("Invalid saved controls");
      if (saved.base !== this.base) return; // Explicit changes to configured defaults win.
      const value = this.parse(saved);
      this.pinned = (saved.pinned as Field[] | undefined) ?? [];
      this.assign(
        value,
        typeof saved.turns === "boolean" && !this.named.turns
          ? saved.turns
          : this.settings.routing.turns,
      );
      if (
        typeof saved.calibrateEvery === "number" &&
        !this.named.calibrateEvery
      )
        this.settings.calibrateEvery = saved.calibrateEvery;
      const lines = { ...this.tunes() };
      for (const key of TUNE_KEYS)
        if (typeof saved[key] === "number" && !this.named[key])
          lines[key] = saved[key] as number;
      if (lines.stepsLow < lines.stepsHigh)
        Object.assign(this.settings.routing, lines);
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
              "turns",
              "calibrateEvery",
              ...TUNE_KEYS,
              "follow",
            ].includes(key),
        ) ||
        ![
          "apps",
          "waits",
          "goal",
          "turns",
          "calibrateEvery",
          ...TUNE_KEYS,
          "follow",
        ].some((key) => Object.hasOwn(input, key)) ||
        TUNE_KEYS.some(
          (key) => input[key] !== undefined && !tuned(key, input[key]),
        ) ||
        (input.turns !== undefined && typeof input.turns !== "boolean") ||
        (input.calibrateEvery !== undefined &&
          (!Number.isInteger(input.calibrateEvery) ||
            (input.calibrateEvery as number) < 0 ||
            (input.calibrateEvery as number) > 100_000)) ||
        (input.follow !== undefined &&
          (input.follow !== true ||
            input.apps !== undefined ||
            input.goal !== undefined ||
            input.turns !== undefined))
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
      // `follow` hands everything changed here back to a Router listener's profile.
      const turns =
        typeof input.turns === "boolean"
          ? input.turns
          : this.settings.routing.turns;
      const calibrateEvery =
        typeof input.calibrateEvery === "number"
          ? input.calibrateEvery
          : this.settings.calibrateEvery;
      const lines = Object.fromEntries(
        TUNE_KEYS.map((key) => [
          key,
          typeof input[key] === "number"
            ? (input[key] as number)
            : this.settings.routing[key],
        ]),
      ) as Record<Tune, number>;
      // The steps answer's lines keep their order, as the options require.
      if (lines.stepsLow >= lines.stepsHigh)
        return { saved: false, reason: "invalid" };
      const pinned = input.follow
        ? []
        : FIELDS.filter(
            (field) =>
              this.pinned.includes(field) ||
              (isRecord(input.apps) && Object.hasOwn(input.apps, field)) ||
              (field === "goal" && input.goal !== undefined) ||
              (field === "turns" && input.turns !== undefined),
          );
      try {
        await this.persist({
          ...next,
          version: 1,
          base: this.base,
          revision,
          ...(pinned.length ? { pinned } : {}),
          turns,
          calibrateEvery,
          ...lines,
        });
      } catch {
        this.persistence = "unavailable";
        return { saved: false, reason: "storage" };
      }
      this.pinned = pinned;
      this.assign(next, turns);
      this.settings.calibrateEvery = calibrateEvery;
      Object.assign(this.settings.routing, lines);
      this.revision = revision;
      this.persistence = "durable";
      if (input.follow) await this.unpin().catch(() => {});
      if (typeof input.turns === "boolean") this.turned(input.turns);
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

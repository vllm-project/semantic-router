const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
export interface Tier {
  model: string;
  efforts: string[];
}
export interface RoutingSettings {
  tiers: Tier[];
  goal: "balanced" | "quality" | "save";
  // The router picks the model of your own turns too, as the Claude Code plugin's `routerTurns`. Off by default.
  turns: boolean;
  easyBelow: number;
  hardFrom: number;
  minConfidence: number;
  downgradeConfidence: number;
  // The steps answer's lines, as the Claude Code plugin's routerStepsLow, routerStepsHigh and routerStepsMinConfidence:
  // below `stepsLow` a tier's lowest effort, from `stepsHigh` its highest, the middle between; under
  // `stepsMinConfidence` the answer is not acted on and the middle effort is used.
  stepsLow: number;
  stepsHigh: number;
  stepsMinConfidence: number;
  feedbackThreshold: number;
  repeatThreshold: number;
  deadlineMs: number;
}
export function parseRouting(input: unknown): RoutingSettings {
  if (input !== undefined && !isRecord(input))
    throw new Error("Invalid routing options");
  const options = input ?? {};
  const known = [
    "tiers",
    "goal",
    "turns",
    "easyBelow",
    "hardFrom",
    "minConfidence",
    "downgradeConfidence",
    "stepsLow",
    "stepsHigh",
    "stepsMinConfidence",
    "feedbackThreshold",
    "repeatThreshold",
    "deadlineMs",
  ];
  for (const key of Object.keys(options))
    if (!known.includes(key)) throw new Error(`Unknown routing option: ${key}`);
  const number = (key: string, fallback: number, maximum: number): number => {
    const value = options[key] === undefined ? fallback : options[key];
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value < 0 ||
      value > maximum
    )
      throw new Error(`Invalid routing ${key}`);
    return value;
  };
  if (options.tiers !== undefined && !Array.isArray(options.tiers))
    throw new Error("Invalid model tiers");
  const tiers: Tier[] = ((options.tiers ?? []) as unknown[]).map((value) => {
    if (
      !isRecord(value) ||
      Object.keys(value).some((key) => !["model", "efforts"].includes(key)) ||
      typeof value.model !== "string" ||
      !/^[\w.-]+\/[^\s#]+$/.test(value.model) ||
      value.model.length > 256
    )
      throw new Error("Invalid model tier");
    const efforts = value.efforts === undefined ? [] : value.efforts;
    if (
      !Array.isArray(efforts) ||
      efforts.length > 8 ||
      efforts.some(
        (effort) =>
          typeof effort !== "string" || !/^[\w.-]{1,64}$/.test(effort),
      ) ||
      new Set(efforts).size !== efforts.length
    )
      throw new Error("Invalid tier efforts");
    return { model: value.model, efforts: [...efforts] as string[] };
  });
  if (
    tiers.length !== 0 &&
    (tiers.length !== 3 ||
      new Set(tiers.map((tier) => tier.model)).size !== 3 ||
      new Set(tiers.map((tier) => tier.model.split("/")[0])).size !== 1)
  )
    throw new Error("Configure three distinct model tiers from one provider");
  const goal = options.goal === undefined ? "balanced" : options.goal;
  if (goal !== "balanced" && goal !== "quality" && goal !== "save")
    throw new Error("Invalid routing goal");
  const turns = options.turns === undefined ? false : options.turns;
  if (typeof turns !== "boolean") throw new Error("Invalid routing turns");
  // Defaults are the Claude Code plugin's (plugin.json userConfig), measured on vela3-lite.
  const easyBelow = number("easyBelow", 0.5, 2),
    hardFrom = number("hardFrom", 1.5, 2);
  if (easyBelow >= hardFrom)
    throw new Error("Routing difficulty boundaries must increase");
  const stepsLow = number("stepsLow", 0.5, 2),
    stepsHigh = number("stepsHigh", 1.2, 2);
  if (stepsLow >= stepsHigh)
    throw new Error("Routing steps boundaries must increase");
  const deadlineMs = number("deadlineMs", 1500, 10000);
  if (!Number.isInteger(deadlineMs) || deadlineMs < 1)
    throw new Error("Invalid routing deadlineMs");
  return {
    tiers,
    goal,
    turns,
    easyBelow,
    hardFrom,
    minConfidence: number("minConfidence", 0.3, 1),
    downgradeConfidence: number("downgradeConfidence", 0.6, 1),
    stepsLow,
    stepsHigh,
    stepsMinConfidence: number("stepsMinConfidence", 0.3, 1),
    feedbackThreshold: number("feedbackThreshold", 0.8, 1),
    repeatThreshold: number("repeatThreshold", 0.75, 1),
    deadlineMs,
  };
}

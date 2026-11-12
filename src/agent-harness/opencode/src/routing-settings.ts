const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
export interface Tier {
  model: string;
  efforts: string[];
}
export interface RoutingSettings {
  tiers: Tier[];
  goal: "balanced" | "quality" | "save";
  easyBelow: number;
  hardFrom: number;
  minConfidence: number;
  downgradeConfidence: number;
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
    "easyBelow",
    "hardFrom",
    "minConfidence",
    "downgradeConfidence",
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
  const easyBelow = number("easyBelow", 0.65, 2),
    hardFrom = number("hardFrom", 1.5, 2);
  if (easyBelow >= hardFrom)
    throw new Error("Routing difficulty boundaries must increase");
  const deadlineMs = number("deadlineMs", 1500, 10000);
  if (!Number.isInteger(deadlineMs) || deadlineMs < 1)
    throw new Error("Invalid routing deadlineMs");
  return {
    tiers,
    goal,
    easyBelow,
    hardFrom,
    minConfidence: number("minConfidence", 0.5, 1),
    downgradeConfidence: number("downgradeConfidence", 0.6, 1),
    feedbackThreshold: number("feedbackThreshold", 0.7, 1),
    repeatThreshold: number("repeatThreshold", 0.8, 1),
    deadlineMs,
  };
}

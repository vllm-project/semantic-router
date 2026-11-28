import { isRecord } from "./settings.ts";
import type { RoutingSettings } from "./routing-settings.ts";
import type { Reply } from "./runtime.ts";
export interface Score {
  score: number;
  confidence: number;
}
export interface ModelRef {
  providerID: string;
  id: string;
  variant?: string;
}
export interface Candidate {
  providerID: string;
  id: string;
  enabled: boolean;
  variants: { id: string }[];
  capabilities: { tools: boolean };
}
export function score(value: unknown): Score | undefined {
  if (
    !isRecord(value) ||
    typeof value.score !== "number" ||
    !Number.isFinite(value.score) ||
    value.score < 0 ||
    value.score > 2 ||
    typeof value.confidence !== "number" ||
    !Number.isFinite(value.confidence) ||
    value.confidence < 0 ||
    value.confidence > 1
  )
    return undefined;
  return { score: value.score, confidence: value.confidence };
}
export function routeScores(
  reply: Reply,
): { complexity: Score; steps?: Score } | undefined {
  const complexity = score(reply.answers.complexity);
  return complexity
    ? { complexity, steps: score(reply.answers.steps) }
    : undefined;
}
export function modelName(model: ModelRef): string {
  return `${model.providerID}/${model.id}`;
}
export function fullModel(model: ModelRef): string {
  return modelName(model) + (model.variant ? `#${model.variant}` : "");
}
export interface Choice {
  model?: string;
  reason:
    | "selected"
    | "unconfigured"
    | "unknown-parent"
    | "low-confidence"
    | "quality"
    | "invalid-tier";
  score?: number;
  confidence?: number;
}
export function choose(
  settings: RoutingSettings,
  parent: ModelRef,
  candidates: readonly Candidate[],
  scores: ReturnType<typeof routeScores>,
  escalate: boolean,
): Choice {
  if (settings.tiers.length !== 3) return { reason: "unconfigured" };
  const parentRank = settings.tiers.findIndex(
    (tier) => tier.model === modelName(parent),
  );
  if (parentRank < 0) return { reason: "unknown-parent" };
  const confident =
    scores !== undefined &&
    scores.complexity.confidence >= settings.minConfidence;
  if (!confident && !escalate) return { reason: "low-confidence" };
  const complexity = scores?.complexity;
  const base = confident
    ? complexity!.score < settings.easyBelow
      ? 0
      : complexity!.score < settings.hardFrom
        ? 1
        : 2
    : parentRank;
  const rank = escalate ? Math.min(2, Math.max(base + 1, parentRank)) : base;
  const facts = complexity
    ? { score: complexity.score, confidence: complexity.confidence }
    : {};
  if (!escalate && rank < parentRank && settings.goal === "quality")
    return { reason: "quality", ...facts };
  if (
    !escalate &&
    rank < parentRank &&
    settings.goal !== "save" &&
    complexity!.confidence <
      Math.max(settings.minConfidence, settings.downgradeConfidence)
  )
    return { reason: "low-confidence", ...facts };
  const tier = settings.tiers[rank]!;
  const candidate = candidates.find(
    (item) => `${item.providerID}/${item.id}` === tier.model,
  );
  if (
    !candidate ||
    !candidate.enabled ||
    !candidate.capabilities.tools ||
    tier.efforts.some(
      (effort) => !candidate.variants.some((variant) => variant.id === effort),
    )
  )
    return { reason: "invalid-tier", ...facts };
  // A variant the model does not list is no variant: OpenCode reports `default` for a model without any, and a
  // child asked for `model#default` fails to spawn.
  const parentVariant =
    parent.variant &&
    candidate.variants.some((item) => item.id === parent.variant)
      ? parent.variant
      : undefined;
  let variant = rank === parentRank ? parentVariant : undefined;
  if (tier.efforts.length) {
    const steps = scores?.steps;
    // The steps answer on its own lines, as the Claude Code plugin's pickEffort.
    const index =
      steps && steps.confidence >= settings.stepsMinConfidence
        ? steps.score < settings.stepsLow
          ? 0
          : steps.score < settings.stepsHigh
            ? 1
            : 2
        : 1;
    let picked = Math.min(index + (escalate ? 1 : 0), tier.efforts.length - 1);
    if (rank === parentRank && parentVariant) {
      const floor = tier.efforts.indexOf(parentVariant);
      // A user-selected variant outside the declared ladder cannot be ordered safely.
      if (floor < 0) return { reason: "invalid-tier", ...facts };
      if (escalate || rank === 2 || settings.goal === "quality")
        picked = Math.max(
          picked,
          Math.min(floor + (escalate ? 1 : 0), tier.efforts.length - 1),
        );
    }
    variant = tier.efforts[picked];
  }
  return {
    model: tier.model + (variant ? `#${variant}` : ""),
    reason: "selected",
    ...facts,
  };
}

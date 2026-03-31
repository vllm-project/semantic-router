import type { BuilderNLProviderKind } from "@/types/dsl";

const DEFAULT_GENERATION_MODEL = "MoM";
const FALLBACK_TARGET_MODEL = DEFAULT_GENERATION_MODEL;
const DEFAULT_GENERATION_TEMPERATURE = 0.1;
const DEFAULT_REPAIR_BUDGET = 1;
const DEFAULT_TIMEOUT_SECONDS = 120;

const PROVIDER_OPTIONS: Array<{
  id: BuilderNLProviderKind;
  label: string;
  description: string;
  placeholder: string;
}> = [
  {
    id: "vllm",
    label: "Local vLLM",
    description: "Use a self-hosted OpenAI-compatible vLLM endpoint.",
    placeholder: "http://localhost:8000",
  },
  {
    id: "openai-compatible",
    label: "OpenAI-compatible API",
    description: "Use any chat-completions compatible endpoint.",
    placeholder: "https://api.openai.com",
  },
  {
    id: "anthropic",
    label: "Anthropic Messages API",
    description: "Call Anthropic-compatible message endpoints directly.",
    placeholder: "https://api.anthropic.com",
  },
];

function extractDefaultModelName(baseConfigYaml: string): string | null {
  const match = baseConfigYaml.match(
    /^\s*default_model:\s*(?:"([^"]+)"|'([^']+)'|([^\n#]+))/m,
  );
  const candidate = match?.[1] ?? match?.[2] ?? match?.[3] ?? "";
  const trimmed = candidate.trim();
  return trimmed || null;
}

function buildPromptPresets(targetModelName: string): string[] {
  return [
    `Route urgent billing issues to a higher-priority route, then send everything else to ${targetModelName}.`,
    `Create domain routes for computer science and math, then keep a general fallback to ${targetModelName}.`,
    `Create a premium support route with a faster model, then keep a general fallback to ${targetModelName}.`,
  ];
}

function normalizeOptionalFloat(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }

  const value = Number(trimmed);
  if (!Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

function normalizeOptionalInteger(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }

  const value = Number.parseInt(trimmed, 10);
  if (!Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

export {
  DEFAULT_GENERATION_MODEL,
  DEFAULT_GENERATION_TEMPERATURE,
  DEFAULT_REPAIR_BUDGET,
  DEFAULT_TIMEOUT_SECONDS,
  FALLBACK_TARGET_MODEL,
  PROVIDER_OPTIONS,
  buildPromptPresets,
  extractDefaultModelName,
  normalizeOptionalFloat,
  normalizeOptionalInteger,
};

export type SetupStep = 0 | 1 | 2;
export type ProviderKind = "vllm" | "openai-compatible" | "anthropic";
export type SetupValidationState = "idle" | "validating" | "valid" | "error";
export type SetupActivationState = "idle" | "activating" | "error";
export type SetupRoutingMode = "scratch" | "remote";
export type RemoteImportState = "idle" | "importing" | "imported" | "error";

export interface ModelDraft {
  id: string;
  name: string;
  providerKind: ProviderKind;
  baseUrl: string;
  accessKey: string;
  endpointName: string;
}

interface BuiltModel {
  name: string;
  provider_model_id: string;
  backend_refs: Array<{
    name: string;
    weight: number;
    endpoint?: string;
    protocol: "http" | "https";
    base_url?: string;
    provider?: "openai" | "anthropic";
    api_key?: string;
  }>;
  api_format?: "anthropic";
}

export interface ProviderOption {
  id: ProviderKind;
  label: string;
  description: string;
  placeholder: string;
}

export interface SetupConfigCounts {
  models: number;
  decisions: number;
  signals: number;
  canActivate: boolean;
}

export interface ImportedSetupConfig {
  config: Record<string, unknown>;
  sourceUrl: string;
  counts: SetupConfigCounts;
}

export const PROVIDER_OPTIONS: ProviderOption[] = [
  {
    id: "vllm",
    label: "Local vLLM",
    description:
      "Best for first-run with a local or self-hosted OpenAI-compatible endpoint.",
    placeholder: "vllm:8000",
  },
  {
    id: "openai-compatible",
    label: "OpenAI-compatible API",
    description:
      "Works for hosted endpoints that expose the OpenAI chat/completions surface.",
    placeholder: "https://api.openai.com",
  },
  {
    id: "anthropic",
    label: "Anthropic Messages API",
    description:
      "Uses Anthropic-compatible request translation inside the router.",
    placeholder: "https://api.anthropic.com",
  },
];

export const SETUP_STEP_LABELS: ReadonlyArray<[string, string]> = [
  ["1", "Connect model"],
  ["2", "Choose routing"],
  ["3", "Review & activate"],
];

export const DEFAULT_REMOTE_SETUP_CONFIG_URL =
  "https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/recipes/balance.yaml";

export function createSetupConfigCounts(
  overrides: Partial<SetupConfigCounts> = {},
): SetupConfigCounts {
  return {
    models: 0,
    decisions: 0,
    signals: 0,
    canActivate: false,
    ...overrides,
  };
}

export function createModelDraft(seed: number): ModelDraft {
  return {
    id: `model-${Date.now()}-${seed}`,
    name: "openai/gpt-oss-120b",
    providerKind: "vllm",
    baseUrl: "vllm:8000",
    accessKey: "",
    endpointName: "primary",
  };
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function inferProtocol(
  endpoint: string,
  providerKind: ProviderKind,
): "http" | "https" {
  if (providerKind === "anthropic") {
    return "https";
  }

  if (
    endpoint.startsWith("localhost") ||
    endpoint.startsWith("127.0.0.1") ||
    endpoint.startsWith("0.0.0.0") ||
    endpoint.startsWith("host.docker.internal")
  ) {
    return "http";
  }

  if (endpoint.includes(":80")) {
    return "http";
  }

  return "https";
}

export function parseBaseUrl(
  rawValue: string,
  providerKind: ProviderKind,
): {
  protocol: "http" | "https";
  endpoint: string;
} {
  const trimmed = rawValue.trim().replace(/\/$/, "");
  if (!trimmed) {
    throw new Error("Model base URL is required.");
  }

  const normalized = trimmed.includes("://")
    ? trimmed
    : `${inferProtocol(trimmed, providerKind)}://${trimmed}`;

  let parsed: URL;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error(`Invalid model endpoint: ${rawValue}`);
  }

  const protocol = parsed.protocol.replace(":", "");
  if (protocol !== "http" && protocol !== "https") {
    throw new Error(`Unsupported protocol for model endpoint: ${rawValue}`);
  }

  const path =
    parsed.pathname && parsed.pathname !== "/"
      ? parsed.pathname.replace(/\/$/, "")
      : "";
  return {
    protocol,
    endpoint: `${parsed.host}${path}`,
  };
}

export function getStepOneErrors(
  models: ModelDraft[],
  defaultModelId: string,
): string[] {
  const errors: string[] = [];

  if (models.length === 0) {
    errors.push("Add at least one model before continuing.");
    return errors;
  }

  const names = new Set<string>();
  let hasDefault = false;

  models.forEach((model, index) => {
    const position = index + 1;
    const trimmedName = model.name.trim();

    if (!trimmedName) {
      errors.push(`Model ${position} is missing a model name.`);
    } else {
      const normalizedName = trimmedName.toLowerCase();
      if (names.has(normalizedName)) {
        errors.push(`Model name "${trimmedName}" is duplicated.`);
      }
      names.add(normalizedName);
    }

    if (!model.baseUrl.trim()) {
      errors.push(`Model ${position} is missing a base URL.`);
    } else {
      try {
        parseBaseUrl(model.baseUrl, model.providerKind);
      } catch (err) {
        errors.push(
          err instanceof Error
            ? err.message
            : `Model ${position} has an invalid base URL.`,
        );
      }
    }

    if (model.id === defaultModelId) {
      hasDefault = true;
    }
  });

  if (!hasDefault) {
    errors.push("Choose a default model before continuing.");
  }

  return errors;
}

export function countConfigSignals(rawSignals: unknown): number {
  if (
    !rawSignals ||
    typeof rawSignals !== "object" ||
    Array.isArray(rawSignals)
  ) {
    return 0;
  }

  return Object.values(rawSignals as Record<string, unknown>).reduce<number>(
    (total, value) => {
      return total + (Array.isArray(value) ? value.length : 0);
    },
    0,
  );
}

export function buildSetupConfig(
  models: ModelDraft[],
  defaultModelId: string,
): Record<string, unknown> {
  const builtModels: BuiltModel[] = models.map((model, index) => {
    const { protocol, endpoint } = parseBaseUrl(
      model.baseUrl,
      model.providerKind,
    );
    const endpointName =
      model.endpointName.trim() ||
      `${slugify(model.name) || `model-${index + 1}`}-primary`;
    const apiKey = model.accessKey.trim() || undefined;
    const trimmedBaseUrl = model.baseUrl.trim().replace(/\/$/, "");
    const backendRef: BuiltModel["backend_refs"][number] =
      model.providerKind === "vllm"
        ? {
            name: endpointName,
            weight: 100,
            endpoint,
            protocol,
            api_key: apiKey,
          }
        : {
            name: endpointName,
            weight: 100,
            protocol,
            base_url: trimmedBaseUrl,
            provider:
              model.providerKind === "anthropic" ? "anthropic" : "openai",
            api_key: apiKey,
          };

    return {
      name: model.name.trim(),
      provider_model_id: model.name.trim(),
      backend_refs: [backendRef],
      api_format: model.providerKind === "anthropic" ? "anthropic" : undefined,
    };
  });

  const defaultModel = builtModels.find((model) => {
    const draft = models.find((item) => item.id === defaultModelId);
    return draft?.name.trim() === model.name;
  });

  if (!defaultModel) {
    throw new Error("Default model selection is invalid.");
  }

  const catchAllDecision = {
    name: "default-route",
    description:
      "Generated during setup to route all requests to the default model.",
    priority: 100,
    rules: {
      operator: "AND",
      conditions: [],
    },
    modelRefs: [
      {
        model: defaultModel.name,
        use_reasoning: false,
      },
    ],
  };

  const config: Record<string, unknown> = {
    providers: {
      models: builtModels,
      defaults: {
        default_model: defaultModel.name,
      },
    },
    routing: {
      modelCards: builtModels.map((model) => ({
        name: model.name,
        modality: "text",
      })),
      decisions: [catchAllDecision],
    },
  };

  return config;
}

export function maskSecrets(config: Record<string, unknown> | null): string {
  if (!config) {
    return "";
  }

  return JSON.stringify(
    config,
    (key, value) => {
      if (
        (key === "api_key" || key === "access_key") &&
        typeof value === "string" &&
        value.length > 0
      ) {
        return "••••••••";
      }
      return value;
    },
    2,
  );
}

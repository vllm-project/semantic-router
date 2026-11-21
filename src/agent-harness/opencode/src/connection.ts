import { DEFAULT_RUNTIME_URL, isRecord, runtimeUrlOf } from "./settings.ts";

export const INTEGRATION = "vsr";
export type Target = { url: string; model: string; token: string | undefined };

const fields = [
  {
    key: "url",
    type: "string",
    title: "Decision runtime URL",
    description:
      "vllm-srun serve on this machine, or a Router listener: everything the enabled apps check is sent here.",
    required: true,
    format: "uri",
    pattern: "^https?://[^?#@]+/v1/decisions$",
    default: DEFAULT_RUNTIME_URL,
  },
  {
    key: "model",
    type: "string",
    title: "Model",
    description:
      "The System One model to ask; leave empty when the runtime serves one.",
  },
] as const;

// `/connect` offers two ways in: a runtime that takes a token, and one that does not.
export const KEY_METHOD = {
  type: "key",
  label: "Runtime with a token",
  form: fields,
} as const;
export const OPEN_METHOD = {
  id: "open",
  type: "external",
  label: "Runtime without a token",
  form: fields,
} as const;

// The runtime a `/connect` credential names; the form's answers are its metadata.
export function targetOf(value: unknown): Target | undefined {
  if (
    !isRecord(value) ||
    (value.type !== "key" && value.type !== "external") ||
    !isRecord(value.metadata)
  )
    return undefined;
  const { url, model } = value.metadata;
  if (
    typeof url !== "string" ||
    (model !== undefined && typeof model !== "string")
  )
    return undefined;
  try {
    return {
      url: runtimeUrlOf(url),
      model: model?.trim() ?? "",
      token:
        value.type === "key" && typeof value.key === "string" && value.key
          ? value.key
          : undefined,
    };
  } catch {
    return undefined;
  }
}

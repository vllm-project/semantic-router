import { describe, expect, it } from "vitest";
import {
  createModelDraft,
  getModelDraftFieldErrors,
  getStepOneErrors,
  type ModelDraft,
} from "./setupWizardSupport";

function modelDraft(overrides: Partial<ModelDraft> = {}): ModelDraft {
  return {
    id: "model-a",
    name: "qwen/qwen3.5-rocm",
    providerKind: "vllm",
    baseUrl: "vllm:8000",
    accessKey: "",
    endpointName: "primary",
    ...overrides,
  };
}

describe("setup wizard model drafts", () => {
  it("keeps the first model prefilled and opens added models as editable drafts", () => {
    const first = createModelDraft(1);
    const second = createModelDraft(2, [first]);

    expect(first.name).toBe("qwen/qwen3.5-rocm");
    expect(first.baseUrl).toBe("vllm:8000");
    expect(second.name).toBe("");
    expect(second.baseUrl).toBe("vllm:8001");
  });

  it("skips local vLLM endpoints that are already used", () => {
    const next = createModelDraft(3, [
      modelDraft({ id: "model-a", baseUrl: "vllm:8000" }),
      modelDraft({
        id: "model-b",
        name: "qwen/qwen3.5-rocm-alt",
        baseUrl: "http://vllm:8001",
      }),
    ]);

    expect(next.baseUrl).toBe("vllm:8002");
  });

  it("reports duplicate model names and local endpoints at field level", () => {
    const models = [
      modelDraft({ id: "model-a", name: "alpha", baseUrl: "vllm:8000" }),
      modelDraft({
        id: "model-b",
        name: "Alpha",
        baseUrl: "http://vllm:8000",
      }),
    ];

    const errors = getModelDraftFieldErrors(models);

    expect(errors["model-a"].name).toBe('Model name "alpha" is duplicated.');
    expect(errors["model-b"].name).toBe('Model name "Alpha" is duplicated.');
    expect(errors["model-a"].baseUrl).toBe(
      'Endpoint "vllm:8000" is already used by another local model.',
    );
    expect(errors["model-b"].baseUrl).toBe(
      'Endpoint "vllm:8000" is already used by another local model.',
    );
  });

  it("includes duplicate local endpoints in step one summary errors", () => {
    const models = [
      modelDraft({ id: "model-a", name: "alpha", baseUrl: "vllm:8000" }),
      modelDraft({
        id: "model-b",
        name: "beta",
        baseUrl: "http://vllm:8000",
      }),
    ];

    expect(getStepOneErrors(models, "model-a")).toEqual([
      'Endpoint "vllm:8000" is already used by another local model.',
    ]);
  });
});
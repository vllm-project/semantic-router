import { describe, expect, it } from "vitest";
import {
  createSetupRequestGuard,
  createModelDraft,
  filterSetupModels,
  getModelDraftFieldErrors,
  getStepOneErrors,
  paginateSetupModels,
  removeSetupModel,
  restoreSetupModel,
  summarizeSetupConfig,
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

  it("filters and bounds large model inventories", () => {
    const models = Array.from({ length: 9 }, (_, index) =>
      modelDraft({
        id: `model-${index}`,
        name: index === 7 ? "anthropic/claude" : `qwen/model-${index}`,
        baseUrl: `vllm:${8000 + index}`,
      }),
    );

    expect(filterSetupModels(models, "CLAUDE").map((model) => model.id)).toEqual([
      "model-7",
    ]);
    expect(paginateSetupModels(models, 3, 4)).toMatchObject({
      page: 3,
      pageCount: 3,
      total: 9,
      startIndex: 8,
    });
    expect(paginateSetupModels(models, 99, 4).items.map((model) => model.id)).toEqual([
      "model-8",
    ]);
  });

  it("removes and restores a default model without losing its position", () => {
    const models = [
      modelDraft({ id: "model-a", name: "alpha" }),
      modelDraft({ id: "model-b", name: "beta", baseUrl: "vllm:8001" }),
      modelDraft({ id: "model-c", name: "gamma", baseUrl: "vllm:8002" }),
    ];

    const removal = removeSetupModel(models, "model-b", "model-b");
    expect(removal.models.map((model) => model.id)).toEqual(["model-a", "model-c"]);
    expect(removal.defaultModelId).toBe("model-a");

    const restored = restoreSetupModel(
      removal.models,
      removal.removed!,
      removal.defaultModelId,
    );
    expect(restored.models.map((model) => model.id)).toEqual([
      "model-a",
      "model-b",
      "model-c",
    ]);
    expect(restored.defaultModelId).toBe("model-b");
  });

  it("drops stale request generations", () => {
    const guard = createSetupRequestGuard();
    const first = guard.begin();
    const second = guard.begin();

    expect(guard.isCurrent(first)).toBe(false);
    expect(guard.isCurrent(second)).toBe(true);
    guard.invalidate();
    expect(guard.isCurrent(second)).toBe(false);
  });

  it("summarizes nested provider and routing collections", () => {
    expect(
      summarizeSetupConfig({
        providers: { models: [{ name: "alpha" }, { name: "beta" }] },
        routing: {
          decisions: [{ name: "default" }],
          signals: { domains: [{ name: "code" }], keywords: [{ name: "fast" }] },
        },
      }),
    ).toEqual({ models: 2, decisions: 1, signals: 2, canActivate: true });
  });
});

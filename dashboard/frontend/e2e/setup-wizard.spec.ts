import { expect, test, type Page } from "@playwright/test";
import { mockAuthenticatedAppShell } from "./support/auth";

const importedConfig = {
  version: "v0.3",
  providers: {
    default_model: "remote-model-primary",
    models: [
      {
        name: "remote-model-primary",
        backend_refs: [
          {
            name: "primary",
            endpoint: "remote-primary.example.com",
            protocol: "https",
            weight: 100,
          },
        ],
      },
      {
        name: "remote-model-backup",
        backend_refs: [
          {
            name: "backup",
            endpoint: "remote-backup.example.com",
            protocol: "https",
            weight: 100,
          },
        ],
      },
    ],
  },
  routing: {
    modelCards: [
      { name: "remote-model-primary", modality: "text" },
      { name: "remote-model-backup", modality: "text" },
    ],
    signals: {
      domains: [{ name: "remote-domain-a" }, { name: "remote-domain-b" }],
      keywords: [{ name: "remote-keyword-a" }, { name: "remote-keyword-b" }],
    },
    decisions: [
      {
        name: "remote-route-primary",
        priority: 900,
        rules: { operator: "AND", conditions: [] },
        modelRefs: [{ model: "remote-model-primary", use_reasoning: false }],
      },
      {
        name: "remote-route-secondary",
        priority: 800,
        rules: { operator: "AND", conditions: [] },
        modelRefs: [{ model: "remote-model-backup", use_reasoning: false }],
      },
      {
        name: "remote-default",
        priority: 100,
        rules: { operator: "AND", conditions: [] },
        modelRefs: [{ model: "remote-model-primary", use_reasoning: false }],
      },
    ],
  },
};

async function mockFirstRunSetup(page: Page) {
  await mockAuthenticatedAppShell(page, {
    setupState: {
      setupMode: true,
      listenerPort: 8700,
      models: 0,
      decisions: 0,
      hasModels: false,
      hasDecisions: false,
      canActivate: false,
    },
    settings: {
      readonlyMode: false,
      setupMode: true,
      platform: "",
      envoyUrl: "",
    },
  });
  await page.route(/\/api\/setup\/presets$/, async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
  });
}

test.describe("Setup wizard routing import", () => {
  test("validates the default from-scratch config once and reaches ready", async ({
    page,
  }) => {
    let validateCallCount = 0;
    let validatePayload: Record<string, unknown> | null = null;

    await mockFirstRunSetup(page);

    await page.route("**/api/setup/validate", async (route) => {
      validateCallCount += 1;
      validatePayload = route.request().postDataJSON() as Record<
        string,
        unknown
      >;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          valid: true,
          config: validatePayload?.config,
          models: 1,
          decisions: 1,
          signals: 0,
          canActivate: true,
        }),
      });
    });

    await page.goto("/setup");

    await expect(
      page.getByRole("heading", { name: "Connect your first model" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Next" }).click();

    await expect(
      page.getByRole("heading", { name: "Choose how routing should begin" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Next" }).click();

    await expect(
      page.getByRole("heading", { name: "Review and activate" }),
    ).toBeVisible();
    await expect
      .poll(() => validatePayload)
      .toEqual({
        config: {
          providers: {
            defaults: {
              default_model: "qwen/qwen3.5-rocm",
            },
            models: [
              {
                name: "qwen/qwen3.5-rocm",
                provider_model_id: "qwen/qwen3.5-rocm",
                backend_refs: [
                  {
                    name: "primary",
                    weight: 100,
                    endpoint: "vllm:8000",
                    protocol: "http",
                  },
                ],
              },
            ],
          },
          routing: {
            modelCards: [
              {
                name: "qwen/qwen3.5-rocm",
                modality: "text",
              },
            ],
            decisions: [
              {
                name: "default-route",
                description:
                  "Generated during setup to route all requests to the default model.",
                priority: 100,
                rules: { operator: "AND", conditions: [] },
                modelRefs: [
                  { model: "qwen/qwen3.5-rocm", use_reasoning: false },
                ],
              },
            ],
          },
        },
      });
    await expect(page.getByText("Ready")).toBeVisible();
    await page.waitForTimeout(300);
    await expect.poll(() => validateCallCount).toBe(1);
  });

  test("imports a remote config and carries it into review", async ({
    page,
  }) => {
    const remoteURL =
      "https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/recipes/balance.yaml";

    let importPayload: Record<string, unknown> | null = null;
    let validatePayload: Record<string, unknown> | null = null;

    await mockFirstRunSetup(page);

    await page.route("**/api/setup/import-remote", async (route) => {
      importPayload = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config: importedConfig,
          models: 2,
          decisions: 3,
          signals: 4,
          canActivate: true,
          sourceUrl: remoteURL,
        }),
      });
    });

    await page.route("**/api/setup/validate", async (route) => {
      validatePayload = route.request().postDataJSON() as Record<
        string,
        unknown
      >;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          valid: true,
          config: importedConfig,
          models: 2,
          decisions: 3,
          signals: 4,
          canActivate: true,
        }),
      });
    });

    await page.goto("/setup");

    await expect(
      page.getByRole("heading", { name: "Connect your first model" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Next" }).click();

    await expect(
      page.getByRole("heading", { name: "Choose how routing should begin" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Starter routing/i }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: /Safety baseline/i }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: /Coding \+ general/i }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: /From scratch/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /From remote/i }),
    ).toBeVisible();

    await page.getByRole("button", { name: /From remote/i }).click();
    await expect(page.getByLabel("Remote config URL")).toHaveValue(remoteURL);
    await page.getByRole("button", { name: "Import", exact: true }).click();

    await expect.poll(() => importPayload).toEqual({ url: remoteURL });
    await expect(
      page.getByText("2 models · 3 decisions · 4 signals"),
    ).toBeVisible();

    await page.getByRole("button", { name: "Next" }).click();

    await expect(
      page.getByRole("heading", { name: "Review and activate" }),
    ).toBeVisible();
    await expect
      .poll(() => validatePayload)
      .toEqual({ config: importedConfig });
    await expect(page.getByText("remote-default")).toBeVisible();
  });

  test("keeps reduced-motion setup static without creating a WebGL canvas", async ({
    page,
  }) => {
    await mockFirstRunSetup(page);
    await page.route(/\/api\/setup\/presets$/, async (route) => {
      await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
    });
    await page.emulateMedia({ reducedMotion: "reduce" });

    await page.goto("/setup");

    await expect(page.getByTestId("setup-motion-background")).toHaveAttribute(
      "data-motion",
      "reduced",
    );
    await expect(page.locator("canvas")).toHaveCount(0);
    await expect(
      page.getByRole("heading", { name: "Build your first Mixture-of-Models." }),
    ).toBeVisible();
  });

  test("validates forward stepper navigation and supports directional keyboard focus", async ({
    page,
  }) => {
    await mockFirstRunSetup(page);
    await page.route(/\/api\/setup\/presets$/, async (route) => {
      await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
    });
    await page.route("**/api/setup/validate", async (route) => {
      const payload = route.request().postDataJSON() as { config: Record<string, unknown> };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          valid: true,
          config: payload.config,
          models: 1,
          decisions: 1,
          signals: 0,
          canActivate: true,
        }),
      });
    });

    await page.goto("/setup");

    const modelName = page.getByLabel("Model name");
    await modelName.fill("");
    const firstStep = page.getByRole("button", { name: "Step 1: Connect model" });
    await expect(firstStep).toHaveAttribute("aria-current", "step");
    await firstStep.focus();
    await page.keyboard.press("End");
    const reviewStep = page.getByRole("button", { name: "Step 3: Review & activate" });
    await expect(reviewStep).toBeFocused();
    await page.keyboard.press("Enter");

    await expect(firstStep).toHaveAttribute("aria-current", "step");
    await expect(modelName).toHaveAttribute("aria-invalid", "true");
    await expect(modelName).toBeFocused();

    await modelName.fill("qwen/qwen3.5-rocm");
    await firstStep.focus();
    await page.keyboard.press("End");
    await page.keyboard.press("Enter");
    await expect(
      page.getByRole("heading", { name: "Review and activate" }),
    ).toBeVisible();
    await expect(reviewStep).toHaveAttribute("aria-current", "step");
  });

  test("bounds model cards and makes removal confirmable and undoable", async ({ page }) => {
    await mockFirstRunSetup(page);
    await page.route(/\/api\/setup\/presets$/, async (route) => {
      await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
    });

    await page.goto("/setup");

    const addModel = page.getByRole("button", { name: "Add model" });
    for (let index = 0; index < 4; index += 1) {
      await addModel.click();
    }
    await expect(page.getByText("5 models", { exact: true })).toBeVisible();
    await expect(page.getByText("Page 2 of 2")).toBeVisible();

    await page.getByRole("button", { name: "Remove model 5" }).click();
    const dialog = page.getByRole("alertdialog", { name: "Remove this model?" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("button", { name: "Remove model", exact: true }).click();

    await expect(page.getByText("4 models", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Undo" })).toBeVisible();
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.getByText("5 models", { exact: true })).toBeVisible();

    await page.getByLabel("Find a model").fill("8004");
    await expect(page.getByText("1 of 5 models")).toBeVisible();
  });

  test("shows retry states for preset catalog and remote import failures", async ({ page }) => {
    await mockFirstRunSetup(page);
    let presetCalls = 0;
    await page.route(/\/api\/setup\/presets$/, async (route) => {
      presetCalls += 1;
      if (presetCalls <= 2) {
        await route.fulfill({ status: 503, body: "preset service unavailable" });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([
          {
            id: "balanced",
            label: "Balanced Mixture",
            summary: "Compose a balanced heterogeneous model fleet.",
            required_models: [{ name: "qwen/qwen3.5-rocm", role: "general" }],
            recipe_url: "https://example.com/balanced.yaml",
          },
        ]),
      });
    });
    let importCalls = 0;
    await page.route("**/api/setup/import-remote", async (route) => {
      importCalls += 1;
      if (importCalls === 1) {
        await route.fulfill({ status: 502, body: "remote temporarily unavailable" });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          config: importedConfig,
          models: 2,
          decisions: 3,
          signals: 4,
          canActivate: true,
          sourceUrl: "https://example.com/config.yaml",
        }),
      });
    });

    await page.goto("/setup");
    await page.getByRole("button", { name: "Next" }).click();

    await expect(page.getByText("Starter architectures unavailable")).toBeVisible();
    await page.getByRole("button", { name: "Retry presets" }).click();
    await expect(page.getByRole("button", { name: /Balanced Mixture/ })).toBeVisible();

    await page.getByRole("button", { name: /From remote/ }).click();
    await page.getByLabel("Remote config URL").fill("https://example.com/config.yaml");
    await page.getByRole("button", { name: "Import", exact: true }).click();
    await expect(page.getByText("remote temporarily unavailable")).toBeVisible();
    await page.getByRole("button", { name: "Retry import" }).click();
    await expect(page.getByText("Remote config ready")).toBeVisible();
  });
});

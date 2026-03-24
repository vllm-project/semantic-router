import { expect, test } from "@playwright/test";
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

test.describe("Setup wizard routing import", () => {
  test("validates the default from-scratch config once and reaches ready", async ({
    page,
  }) => {
    let validateCallCount = 0;
    let validatePayload: Record<string, unknown> | null = null;

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
            models: [
              {
                name: "openai/gpt-oss-120b",
                provider_model_id: "openai/gpt-oss-120b",
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
            default_model: "openai/gpt-oss-120b",
          },
          routing: {
            modelCards: [
              {
                name: "openai/gpt-oss-120b",
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
                  { model: "openai/gpt-oss-120b", use_reasoning: false },
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
});

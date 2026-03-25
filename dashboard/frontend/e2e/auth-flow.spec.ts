import { expect, test } from "@playwright/test";
import {
  mockAuthenticatedAppShell,
  mockAuthenticatedSession,
} from "./support/auth";

const baseSetupState = {
  setupMode: false,
  listenerPort: 8700,
  models: 1,
  decisions: 1,
  hasModels: true,
  hasDecisions: true,
  canActivate: true,
};

const transitionCopyPattern =
  /A Symbolic Analysis of Relay and Switching Circuits/i;

test.describe("Dashboard auth flow", () => {
  test("redirects unauthenticated protected routes to login", async ({
    page,
  }) => {
    await page.route("**/api/setup/state", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(baseSetupState),
      });
    });

    await page.route("**/api/settings", async (route) => {
      await route.fulfill({ status: 401, body: "Unauthorized" });
    });

    await page.route("**/api/auth/bootstrap/can-register", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canRegister: false }),
      });
    });

    await page.goto("/playground", { waitUntil: "domcontentloaded" });

    await expect(page).toHaveURL(/\/login$/);
    await expect(
      page.getByRole("heading", { name: "Sign in", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /create admin/i }),
    ).toHaveCount(0);
  });

  test("login shows the transition loader and reuses the session for protected requests", async ({
    page,
  }) => {
    test.slow();
    const issuedToken = "issued-dashboard-token";
    let settingsAuthHeader = "";
    let statusAuthHeader = "";

    await page.route("**/api/setup/state", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(baseSetupState),
      });
    });

    await page.route("**/api/auth/bootstrap/can-register", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canRegister: false }),
      });
    });

    await page.route("**/api/auth/login", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: issuedToken,
          user: {
            id: "user-admin-1",
            email: "admin@example.com",
            name: "Admin User",
            role: "admin",
          },
        }),
      });
    });

    await page.route("**/api/auth/me", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user: {
            id: "user-admin-1",
            email: "admin@example.com",
            name: "Admin User",
            role: "admin",
          },
        }),
      });
    });

    await page.route("**/api/settings", async (route) => {
      settingsAuthHeader =
        route.request().headers().authorization ?? settingsAuthHeader;
      if (!route.request().headers().authorization) {
        await route.fulfill({ status: 401, body: "Unauthorized" });
        return;
      }

      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          readonlyMode: false,
          setupMode: false,
          platform: "",
          envoyUrl: "",
        }),
      });
    });

    await page.route("**/api/status", async (route) => {
      statusAuthHeader = route.request().headers().authorization ?? "";
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          overall: "healthy",
          deployment_type: "local",
          services: [],
        }),
      });
    });

    await page.route("**/api/router/config/all", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signals: {},
          decisions: [],
          providers: { models: [] },
          plugins: {},
        }),
      });
    });

    await page.route("**/api/mcp/servers", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([]),
      });
    });

    await page.route("**/api/mcp/tools", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tools: [] }),
      });
    });

    await page.goto("/login");
    await page.getByPlaceholder("you@example.com").fill("admin@example.com");
    await page.getByPlaceholder("••••••••").fill("secret-password");
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page).toHaveURL(/\/auth\/transition\?to=%2Fdashboard$/);
    await expect(page.getByText(transitionCopyPattern)).toBeVisible();
    await expect(page).toHaveURL(/\/dashboard$/, { timeout: 12000 });
    await expect.poll(() => settingsAuthHeader).toBe(`Bearer ${issuedToken}`);
    await expect.poll(() => statusAuthHeader).toBe(`Bearer ${issuedToken}`);
  });

  test("bootstrap registration passes through the transition loader", async ({
    page,
  }) => {
    test.slow();
    const issuedToken = "bootstrap-dashboard-token";
    let registerPayload: Record<string, unknown> | null = null;
    let setupStateRequestCount = 0;

    await page.route("**/api/setup/state", async (route) => {
      setupStateRequestCount += 1;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          setupStateRequestCount === 1
            ? baseSetupState
            : {
                ...baseSetupState,
                setupMode: true,
                models: 0,
                decisions: 0,
                hasModels: false,
                hasDecisions: false,
                canActivate: false,
              },
        ),
      });
    });

    await page.route("**/api/auth/bootstrap/can-register", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canRegister: true }),
      });
    });

    await page.route("**/api/auth/bootstrap/register", async (route) => {
      registerPayload = route.request().postDataJSON() as Record<
        string,
        unknown
      >;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: issuedToken,
          user: {
            id: "user-admin-1",
            email: "ada@example.com",
            name: "Ada Router",
            role: "admin",
          },
        }),
      });
    });

    await page.route("**/api/auth/me", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user: {
            id: "user-admin-1",
            email: "ada@example.com",
            name: "Ada Router",
            role: "admin",
          },
        }),
      });
    });

    await page.route("**/api/settings", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          readonlyMode: false,
          setupMode: false,
          platform: "",
          envoyUrl: "",
        }),
      });
    });

    await page.route("**/api/status", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          overall: "healthy",
          deployment_type: "local",
          services: [],
        }),
      });
    });

    await page.route("**/api/router/config/all", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signals: {},
          decisions: [],
          providers: { models: [] },
          plugins: {},
        }),
      });
    });

    await page.goto("/login");

    await expect(
      page.getByRole("heading", {
        name: "Who are you? What should we call you?",
      }),
    ).toBeVisible();
    await expect(page.getByRole("heading", { name: "Sign in" })).toHaveCount(0);

    await page.getByLabel("What should we call you?").fill("Ada Router");
    await page.getByRole("button", { name: "Next" }).click();

    await expect(
      page.getByRole("heading", {
        name: "Where should your future admin sign in?",
      }),
    ).toBeVisible();
    await page.getByLabel("Admin email").fill("ada@example.com");
    await page.getByRole("button", { name: "Next" }).click();

    await expect(
      page.getByRole("heading", {
        name: "Set the key, then step into the future.",
      }),
    ).toBeVisible();
    await page.getByLabel("Password").fill("future-password");
    await page.getByRole("button", { name: "Enter Future" }).click();

    await expect
      .poll(() => registerPayload)
      .toEqual({
        email: "ada@example.com",
        password: "future-password",
        name: "Ada Router",
      });
    await expect.poll(() => setupStateRequestCount).toBeGreaterThanOrEqual(2);
    await expect(page).toHaveURL(/\/auth\/transition\?to=%2Fsetup$/);
    await expect(page.getByText(transitionCopyPattern)).toBeVisible();
    await expect(page).toHaveURL(/\/setup$/, { timeout: 12000 });
  });

  test("login preserves the original protected route through the transition page", async ({
    page,
  }) => {
    test.slow();

    await page.route("**/api/setup/state", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(baseSetupState),
      });
    });

    await page.route("**/api/auth/bootstrap/can-register", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canRegister: false }),
      });
    });

    await page.route("**/api/auth/login", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: "status-flow-token",
          user: {
            id: "user-admin-1",
            email: "admin@example.com",
            name: "Admin User",
            role: "admin",
          },
        }),
      });
    });

    await page.route("**/api/auth/me", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user: {
            id: "user-admin-1",
            email: "admin@example.com",
            name: "Admin User",
            role: "admin",
          },
        }),
      });
    });

    await page.route("**/api/settings", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          readonlyMode: false,
          setupMode: false,
          platform: "",
          envoyUrl: "",
        }),
      });
    });

    await page.route("**/api/status", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          overall: "healthy",
          deployment_type: "local",
          services: [],
        }),
      });
    });

    await page.route("**/api/mcp/servers", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([]),
      });
    });

    await page.route("**/api/mcp/tools", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tools: [] }),
      });
    });

    await page.goto("/status", { waitUntil: "domcontentloaded" });
    await expect(page).toHaveURL(/\/login$/);

    await page.getByPlaceholder("you@example.com").fill("admin@example.com");
    await page.getByPlaceholder("••••••••").fill("secret-password");
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page).toHaveURL(/\/auth\/transition\?to=%2Fstatus$/);
    await expect(page.getByText(transitionCopyPattern)).toBeVisible();
    await expect(page).toHaveURL(/\/status$/, { timeout: 12000 });
    await expect(
      page.getByRole("heading", { name: "System Status", exact: true }),
    ).toBeVisible();
  });

  test("transition route rejects login targets and falls back to dashboard", async ({
    page,
  }) => {
    test.slow();

    await mockAuthenticatedSession(page, { token: "transition-fallback-token" });

    await page.route("**/api/setup/state", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(baseSetupState),
      });
    });

    await page.route("**/api/settings", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          readonlyMode: false,
          setupMode: false,
          platform: "",
          envoyUrl: "",
        }),
      });
    });

    await page.route("**/api/status", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          overall: "healthy",
          deployment_type: "local",
          services: [],
        }),
      });
    });

    await page.route("**/api/router/config/all", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signals: {},
          decisions: [],
          providers: { models: [] },
          plugins: {},
        }),
      });
    });

    await page.route("**/api/mcp/servers", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([]),
      });
    });

    await page.route("**/api/mcp/tools", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tools: [] }),
      });
    });

    await page.goto("/auth/transition?to=%2Flogin%3Fnext%3Dusers");
    await expect(page).toHaveURL(/\/dashboard$/, { timeout: 12000 });
  });

  test("setup mode keeps authenticated users on the setup wizard", async ({
    page,
  }) => {
    await mockAuthenticatedSession(page, { token: "setup-mode-token" });

    await page.route("**/api/setup/state", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...baseSetupState,
          setupMode: true,
          canActivate: false,
        }),
      });
    });

    await page.route("**/api/settings", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          readonlyMode: false,
          setupMode: true,
          platform: "",
          envoyUrl: "",
        }),
      });
    });

    await page.goto("/dashboard", { waitUntil: "domcontentloaded" });

    await expect(page).toHaveURL(/\/setup$/);
  });

  test("read users inherit the readonly shell, keep ClawRoom access, and lose admin-only actions", async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page, {
      user: {
        id: "user-read-1",
        email: "reader@example.com",
        name: "Read User",
        role: "read",
      },
      settings: {
        readonlyMode: true,
        setupMode: false,
        platform: "",
        envoyUrl: "",
      },
    });

    await page.route("**/api/router/config/all", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signals: {},
          decisions: [],
          providers: { models: [] },
          plugins: {},
        }),
      });
    });

    await page.route("**/api/router/config/global", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    });

    await page.route("**/api/router/config/yaml", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "text/plain" },
        body: "signals: {}\ndecisions: []\nproviders:\n  models: []\nplugins: {}\n",
      });
    });

    await page.goto("/config");
    await expect(page).toHaveURL(/\/config$/);
    await expect(page.getByRole("link", { name: "Users" })).toHaveCount(0);

    await page.goto("/playground");
    await expect(
      page.getByRole("button", { name: /Enable HireClaw|Disable HireClaw/i }),
    ).toBeEnabled();
    await expect(
      page.getByRole("button", { name: /Open ClawRoom view|Exit ClawRoom view/i }),
    ).toHaveCount(0);

    await page.getByRole("button", { name: /Enable HireClaw/i }).click();
    await expect(
      page.getByRole("button", { name: /Open ClawRoom view|Exit ClawRoom view/i }),
    ).toBeEnabled();

    await page.goto("/builder");
    const deployButton = page.getByRole("button", { name: "Deploy" });
    await expect(deployButton).toBeDisabled();
    await expect(deployButton).toHaveAttribute(
      "title",
      "Deploy is unavailable in read-only mode",
    );
  });

  test("authenticated admins can open the users page", async ({ page }) => {
    await mockAuthenticatedAppShell(page);

    await page.route("**/api/admin/users", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          users: [
            {
              id: "user-admin-1",
              email: "admin@example.com",
              name: "Admin User",
              role: "admin",
              status: "active",
            },
          ],
        }),
      });
    });

    await page.goto("/users");

    await expect(page).toHaveURL(/\/users$/);
    await expect(page.getByRole("heading", { name: "Users" })).toBeVisible();
  });

  test("users page manages accounts through centered dialogs", async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page);

    let createPayload: Record<string, unknown> | null = null;
    let patchPayload: Record<string, unknown> | null = null;
    let passwordPayload: Record<string, unknown> | null = null;

    await page.route("**/api/admin/users", async (route) => {
      if (route.request().method() === "POST") {
        createPayload = route.request().postDataJSON() as Record<
          string,
          unknown
        >;
        await route.fulfill({
          status: 200,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: "user-2",
            email: createPayload.email,
            name: createPayload.name,
            role: createPayload.role,
            status: "active",
          }),
        });
        return;
      }

      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          users: [
            {
              id: "user-admin-1",
              email: "admin@example.com",
              name: "Admin User",
              role: "admin",
              status: "active",
            },
          ],
        }),
      });
    });

    await page.route("**/api/admin/users/user-admin-1", async (route) => {
      patchPayload = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: "user-admin-1",
          email: "admin@example.com",
          name: "Admin User",
          role: patchPayload.role,
          status: patchPayload.status,
        }),
      });
    });

    await page.route("**/api/admin/users/password", async (route) => {
      passwordPayload = route.request().postDataJSON() as Record<
        string,
        unknown
      >;
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ok: true }),
      });
    });

    await page.goto("/users");
    await expect(page).toHaveURL(/\/users$/);

    await page.getByRole("button", { name: "Create user" }).click();
    const createDialog = page.getByRole("dialog", { name: "Create user" });
    await expect(createDialog).toBeVisible();
    await createDialog
      .locator("#create-user-email")
      .fill("writer@example.com");
    await createDialog.locator("#create-user-name").fill("Writer User");
    await createDialog.locator("#create-user-role").selectOption("write");
    await createDialog
      .locator("#create-user-password")
      .fill("writer-password");
    await createDialog.getByRole("button", { name: "Create user" }).click();

    await expect
      .poll(() => createPayload)
      .toEqual({
        email: "writer@example.com",
        name: "Writer User",
        password: "writer-password",
        role: "write",
      });
    await expect(page.getByText("User created.")).toBeVisible();

    await page.getByRole("button", { name: "Edit" }).click();
    const editDialog = page.getByRole("dialog", { name: "Edit user" });
    await expect(editDialog).toBeVisible();
    await editDialog.locator("#edit-user-role").selectOption("admin");
    await editDialog.locator("#edit-user-status").selectOption("inactive");
    await editDialog.locator("#edit-user-password").fill("rotated-password");
    await editDialog.getByRole("button", { name: "Save changes" }).click();

    await expect
      .poll(() => patchPayload)
      .toEqual({
        role: "admin",
        status: "inactive",
      });
    await expect
      .poll(() => passwordPayload)
      .toEqual({
        userId: "user-admin-1",
        password: "rotated-password",
      });
    await expect(
      page.getByText("User updated and password rotated."),
    ).toBeVisible();
  });

  test("protected browser transports append auth query tokens", async ({
    page,
  }) => {
    const { token } = await mockAuthenticatedAppShell(page, {
      token: "transport-auth-token",
    });

    await page.route("**/api/admin/users", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ users: [] }),
      });
    });

    await page.goto("/users");
    await expect(page).toHaveURL(/\/users$/);

    const urls = await page.evaluate(() => {
      const iframe = document.createElement("iframe");
      iframe.src = "/embedded/openclaw/demo/";

      const source = new EventSource("/api/openclaw/rooms/room-1/stream");
      const sourceUrl = source.url;
      source.close();

      const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(
        `${wsProtocol}://${window.location.host}/api/openclaw/rooms/room-1/ws`,
      );
      const socketUrl = socket.url;
      socket.close();

      return {
        cookie: document.cookie,
        iframeUrl: iframe.src,
        sourceUrl,
        socketUrl,
      };
    });

    expect(urls.cookie).toContain(`vsr_session=${token}`);
    expect(urls.iframeUrl).toContain(`authToken=${token}`);
    expect(urls.sourceUrl).toContain(`authToken=${token}`);
    expect(urls.socketUrl).toContain(`authToken=${token}`);
  });
});

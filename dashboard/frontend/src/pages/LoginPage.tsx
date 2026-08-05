import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { useSetup } from "../contexts/SetupContext";
import ColorBends from "../components/ColorBends";
import {
  DASHBOARD_COLOR_BENDS_MOTION,
  DASHBOARD_MOTION_COLORS,
} from "../components/dashboardMotionTheme";
import {
  buildAuthTransitionPath,
  resolvePostAuthTarget,
} from "./authTransitionSupport";
import styles from "./LoginPage.module.css";

interface LocationState {
  from?: string;
}

type BootstrapStatus = "checking" | "enabled" | "disabled";

type BootstrapFormState = {
  name: string;
  email: string;
  password: string;
};

type BootstrapStep = {
  key: "name" | "email" | "password";
  label: string;
  eyebrow: string;
  title: string;
  description: string;
};

const BOOTSTRAP_STEPS: BootstrapStep[] = [
  {
    key: "name",
    label: "Identity",
    eyebrow: "Step 1",
    title: "Name your first administrator.",
    description:
      "This name identifies who will configure the first model system.",
  },
  {
    key: "email",
    label: "Access",
    eyebrow: "Step 2",
    title: "Choose the administrator email.",
    description:
      "Use the email that will own activation, setup, and future access management.",
  },
  {
    key: "password",
    label: "Security",
    eyebrow: "Step 3",
    title: "Secure the workspace.",
    description:
      "Create a password, then continue directly to model setup.",
  },
];

const LoginPage: React.FC = () => {
  const { setupState, isLoading: setupLoading, refreshSetupState } = useSetup();
  const { isAuthenticated, isLoading, login, setSession } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as LocationState | null)?.from ?? null;

  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [bootstrapForm, setBootstrapForm] = useState<BootstrapFormState>({
    name: "",
    email: "",
    password: "",
  });

  const [bootstrapStatus, setBootstrapStatus] =
    useState<BootstrapStatus>("checking");
  const [bootstrapStepIndex, setBootstrapStepIndex] = useState(0);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const isFirstServe = Boolean(setupState?.setupMode);
  const targetAfterLogin = resolvePostAuthTarget(isFirstServe, from);
  const isBootstrapMode = bootstrapStatus === "enabled";
  const currentStep = BOOTSTRAP_STEPS[bootstrapStepIndex] ?? BOOTSTRAP_STEPS[0];

  const navigateAfterAuth = async (fallbackSetupMode: boolean) => {
    const nextSetupState = await refreshSetupState();
    const nextTarget = resolvePostAuthTarget(
      nextSetupState?.setupMode ?? fallbackSetupMode,
      from,
    );
    navigate(buildAuthTransitionPath(nextTarget), { replace: true });
  };

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch("/api/auth/bootstrap/can-register", {
          method: "GET",
        });
        if (!response.ok) {
          setBootstrapStatus("disabled");
          return;
        }
        const payload = (await response.json()) as { canRegister: boolean };
        setBootstrapStatus(payload?.canRegister ? "enabled" : "disabled");
      } catch {
        setBootstrapStatus("disabled");
      }
    };

    void load();
  }, []);

  const validateBootstrapStep = () => {
    if (currentStep.key === "name" && !bootstrapForm.name.trim()) {
      setError(
        "Tell us what the workspace should call you before we continue.",
      );
      return false;
    }

    if (currentStep.key === "email" && !bootstrapForm.email.trim()) {
      setError("Add the admin email for this workspace.");
      return false;
    }

    if (currentStep.key === "password" && !bootstrapForm.password.trim()) {
      setError("Set a password before entering the workspace.");
      return false;
    }

    setError("");
    return true;
  };

  const onSubmitLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setPending(true);
    try {
      await login(loginEmail.trim(), loginPassword);
      await navigateAfterAuth(isFirstServe);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Login failed. Please check credentials.",
      );
    } finally {
      setPending(false);
    }
  };

  const onSubmitBootstrap = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (bootstrapStepIndex < BOOTSTRAP_STEPS.length - 1) {
      if (validateBootstrapStep()) {
        setBootstrapStepIndex((current) => current + 1);
      }
      return;
    }

    if (!validateBootstrapStep()) {
      return;
    }

    setError("");
    setPending(true);
    try {
      const response = await fetch("/api/auth/bootstrap/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: bootstrapForm.email.trim(),
          password: bootstrapForm.password,
          name: bootstrapForm.name.trim(),
        }),
      });
      if (!response.ok) {
        const message = await response.text();
        if (response.status === 409) {
          setBootstrapStatus("disabled");
          setLoginEmail(bootstrapForm.email.trim());
          throw new Error(
            "The first admin is already registered. Sign in to continue.",
          );
        }
        throw new Error(message || `Request failed: ${response.status}`);
      }
      const payload = (await response.json()) as {
        token: string;
        user?: { id: string; email: string; name: string; role?: string };
      };
      setSession(payload.token, payload.user ?? null);
      await navigateAfterAuth(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Register failed.");
    } finally {
      setPending(false);
    }
  };

  const bootstrapProgress = useMemo(
    () =>
      BOOTSTRAP_STEPS.map((step, index) => ({
        ...step,
        active: index === bootstrapStepIndex,
        complete: index < bootstrapStepIndex,
      })),
    [bootstrapStepIndex],
  );

  if (isAuthenticated && !isLoading && !setupLoading && !pending) {
    return <Navigate to={targetAfterLogin} replace />;
  }

  return (
    <div className={styles.container}>
      <div
        className={styles.backgroundEffect}
        data-testid="login-motion-background"
      >
        <ColorBends
          colors={DASHBOARD_MOTION_COLORS}
          {...DASHBOARD_COLOR_BENDS_MOTION}
          transparent
        />
      </div>

      <main className={styles.mainContent}>
        <div className={styles.shell}>
          <section className={styles.storyPanel}>
            <div className={styles.heroBadge}>
              <img
                src="/vllm.png"
                alt="vLLM logo"
                className={styles.badgeLogo}
              />
              <span>
                {isBootstrapMode ? "First activation" : "Welcome back"}
              </span>
            </div>

            <div className={styles.storyCopy}>
              <p className={styles.storyEyebrow}>
                {bootstrapStatus === "checking"
                  ? "Preparing workspace"
                  : isBootstrapMode
                    ? "Workspace activation"
                    : "Dashboard access"}
              </p>
              <h1 className={styles.storyTitle}>
                {bootstrapStatus === "checking"
                  ? "Preparing your workspace."
                  : isBootstrapMode
                    ? "Create the first administrator."
                    : "Continue building your Mixture-of-Models."}
              </h1>
              <p className={styles.storyDescription}>
                {bootstrapStatus === "checking"
                  ? "We are checking whether this workspace needs an administrator or is ready for sign-in."
                  : isBootstrapMode
                    ? "Create the account that will connect heterogeneous LLMs and define their model paths."
                    : "Sign in to inspect model paths, tune decisions, and operate the running system."}
              </p>
            </div>

            {isBootstrapMode ? (
              <div className={styles.progressRail}>
                {bootstrapProgress.map((step, index) => (
                  <div
                    key={step.key}
                    className={`${styles.progressStep} ${step.active ? styles.progressStepActive : ""} ${step.complete ? styles.progressStepComplete : ""}`}
                  >
                    <span className={styles.progressIndex}>{index + 1}</span>
                    <div>
                      <div className={styles.progressLabel}>{step.label}</div>
                      <div className={styles.progressCaption}>
                        {step.complete
                          ? "Complete"
                          : step.active
                            ? "In focus"
                            : "Ahead"}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.storyMetricRow}>
                <div className={styles.storyMetric}>
                  <span className={styles.metricLabel}>Surface</span>
                  <strong className={styles.metricValue}>
                    {isFirstServe ? "Setup wizard" : "Dashboard"}
                  </strong>
                </div>
                <div className={styles.storyMetric}>
                  <span className={styles.metricLabel}>Workspace mode</span>
                  <strong className={styles.metricValue}>
                    {setupLoading
                      ? "Loading"
                      : isFirstServe
                        ? "First serve"
                        : "Live"}
                  </strong>
                </div>
              </div>
            )}
          </section>

          {bootstrapStatus === "checking" ? (
            <section className={styles.card}>
              <div className={styles.stageHeader}>
                <p className={styles.stageEyebrow}>Bootstrap status</p>
                <h2 className={styles.stageTitle}>
                  Preparing your entry point...
                </h2>
                <p className={styles.stageDescription}>
                  The dashboard is deciding whether this visit should create the
                  first admin or open sign-in.
                </p>
              </div>
            </section>
          ) : isBootstrapMode ? (
            <form
              className={styles.card}
              onSubmit={onSubmitBootstrap}
              autoComplete="on"
            >
              <div className={styles.stageHeader}>
                <p className={styles.stageEyebrow}>{currentStep.eyebrow}</p>
                <h2 className={styles.stageTitle}>{currentStep.title}</h2>
                <p className={styles.stageDescription}>
                  {currentStep.description}
                </p>
              </div>

              {currentStep.key === "name" ? (
                <div className={styles.inputBlock}>
                  <label className={styles.label} htmlFor="bootstrap-name">
                    What should we call you?
                  </label>
                  <input
                    id="bootstrap-name"
                    className={styles.input}
                    type="text"
                    name="name"
                    autoComplete="name"
                    value={bootstrapForm.name}
                    onChange={(event) =>
                      setBootstrapForm((current) => ({
                        ...current,
                        name: event.target.value,
                      }))
                    }
                    placeholder="Ada, Alex, Team Router..."
                    autoFocus
                    required
                  />
                </div>
              ) : null}

              {currentStep.key === "email" ? (
                <div className={styles.inputBlock}>
                  <label className={styles.label} htmlFor="bootstrap-email">
                    Admin email
                  </label>
                  <input
                    id="bootstrap-email"
                    className={styles.input}
                    type="email"
                    name="email"
                    autoComplete="username"
                    value={bootstrapForm.email}
                    onChange={(event) =>
                      setBootstrapForm((current) => ({
                        ...current,
                        email: event.target.value,
                      }))
                    }
                    placeholder="you@example.com"
                    autoFocus
                    required
                  />
                </div>
              ) : null}

              {currentStep.key === "password" ? (
                <div className={styles.finalStage}>
                  <div className={styles.inputBlock}>
                    <label
                      className={styles.label}
                      htmlFor="bootstrap-password"
                    >
                      Password
                    </label>
                    <input
                      id="bootstrap-password"
                      className={styles.input}
                      type="password"
                      name="new-password"
                      autoComplete="new-password"
                      value={bootstrapForm.password}
                      onChange={(event) =>
                        setBootstrapForm((current) => ({
                          ...current,
                          password: event.target.value,
                        }))
                      }
                      placeholder="Choose a strong password"
                      autoFocus
                      required
                    />
                  </div>

                  <div className={styles.summaryCard}>
                    <span className={styles.summaryLabel}>
                      Ready to launch as
                    </span>
                    <strong className={styles.summaryValue}>
                      {bootstrapForm.name || "Your first admin"}
                    </strong>
                    <span className={styles.summaryDetail}>
                      {bootstrapForm.email || "you@example.com"}
                    </span>
                  </div>
                </div>
              ) : null}

              {error ? <div className={styles.error}>{error}</div> : null}

              <div className={styles.footerActions}>
                {bootstrapStepIndex > 0 ? (
                  <button
                    className={styles.secondaryButton}
                    type="button"
                    onClick={() => {
                      setError("");
                      setBootstrapStepIndex((current) =>
                        Math.max(0, current - 1),
                      );
                    }}
                  >
                    Back
                  </button>
                ) : (
                  <button
                    className={styles.secondaryButton}
                    type="button"
                    onClick={() => navigate("/")}
                  >
                    Back to landing
                  </button>
                )}

                <button
                  className={styles.button}
                  type="submit"
                  disabled={pending || setupLoading || isLoading}
                >
                  {bootstrapStepIndex === BOOTSTRAP_STEPS.length - 1
                    ? pending
                      ? "Creating administrator..."
                      : "Create admin and continue"
                    : "Next"}
                </button>
              </div>
            </form>
          ) : (
            <form
              className={styles.card}
              onSubmit={onSubmitLogin}
              autoComplete="on"
            >
              <div className={styles.stageHeader}>
                <p className={styles.stageEyebrow}>Account access</p>
                <h2 className={styles.stageTitle}>Sign in</h2>
                <p className={styles.stageDescription}>
                  Bootstrap is complete. Sign in with your existing dashboard
                  account to continue.
                </p>
              </div>

              <div className={styles.inputBlock}>
                <label className={styles.label} htmlFor="login-email">
                  Email
                </label>
                <input
                  id="login-email"
                  className={styles.input}
                  type="email"
                  name="email"
                  autoComplete="username"
                  value={loginEmail}
                  onChange={(event) => setLoginEmail(event.target.value)}
                  placeholder="you@example.com"
                  autoFocus
                  required
                />
              </div>

              <div className={styles.inputBlock}>
                <label className={styles.label} htmlFor="login-password">
                  Password
                </label>
                <input
                  id="login-password"
                  className={styles.input}
                  type="password"
                  name="password"
                  autoComplete="current-password"
                  value={loginPassword}
                  onChange={(event) => setLoginPassword(event.target.value)}
                  placeholder="••••••••"
                  required
                />
              </div>

              {error ? <div className={styles.error}>{error}</div> : null}

              <div className={styles.footerActions}>
                <button
                  className={styles.secondaryButton}
                  type="button"
                  onClick={() => navigate("/")}
                >
                  Back to landing
                </button>
                <button
                  className={styles.button}
                  type="submit"
                  disabled={pending || setupLoading || isLoading}
                >
                  {isLoading ? "Signing in..." : "Continue"}
                </button>
              </div>
            </form>
          )}
        </div>
      </main>
    </div>
  );
};

export default LoginPage;

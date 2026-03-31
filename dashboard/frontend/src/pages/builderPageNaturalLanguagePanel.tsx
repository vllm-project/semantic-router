import React, { useMemo, useState } from "react";

import type {
  BuilderNLConnectionMode,
  BuilderNLGenerateRequest,
  BuilderNLProviderKind,
  BuilderNLReview,
  Diagnostic,
  EditorMode,
} from "@/types/dsl";

import styles from "./builderPageNaturalLanguagePanel.module.css";

interface BuilderNaturalLanguagePanelProps {
  currentDsl: string;
  diagnostics: Diagnostic[];
  generating: boolean;
  error: string | null;
  summary: string;
  suggestedTestQuery: string;
  review: BuilderNLReview | null;
  lastPrompt: string;
  onGenerate: (input: BuilderNLGenerateRequest) => Promise<void>;
  onClearResult: () => void;
  onModeSwitch: (mode: EditorMode) => void;
}

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

const PROMPT_PRESETS = [
  "Route urgent billing issues to a higher-priority route, then send everything else to MoM.",
  "Create separate routes for code debugging, math tutoring, and general chat.",
  "Add multilingual routing so Chinese and English prompts get their own routes before a general fallback.",
];

const BuilderNaturalLanguagePanel: React.FC<BuilderNaturalLanguagePanelProps> = ({
  currentDsl,
  diagnostics,
  generating,
  error,
  summary,
  suggestedTestQuery,
  review,
  lastPrompt,
  onGenerate,
  onClearResult,
  onModeSwitch,
}) => {
  const [prompt, setPrompt] = useState("");
  const [connectionMode, setConnectionMode] =
    useState<BuilderNLConnectionMode>("default");
  const [useCurrentDslContext, setUseCurrentDslContext] = useState(true);
  const [providerKind, setProviderKind] =
    useState<BuilderNLProviderKind>("openai-compatible");
  const [modelName, setModelName] = useState("gpt-4o-mini");
  const [baseUrl, setBaseUrl] = useState("");
  const [accessKey, setAccessKey] = useState("");
  const [endpointName, setEndpointName] = useState("nl-custom");

  const errorCount = diagnostics.filter((item) => item.level === "error").length;
  const warningCount = diagnostics.filter(
    (item) => item.level === "warning",
  ).length;
  const hasContextDsl = currentDsl.trim().length > 0;
  const activeProvider = PROVIDER_OPTIONS.find((item) => item.id === providerKind)!;
  const generatedPreview = useMemo(() => {
    if (!lastPrompt || !currentDsl.trim()) {
      return "";
    }
    return currentDsl.trim().split("\n").slice(0, 14).join("\n");
  }, [currentDsl, lastPrompt]);

  const handleGenerate = async () => {
    await onGenerate({
      prompt,
      currentDsl: useCurrentDslContext ? currentDsl : "",
      connectionMode,
      customConnection:
        connectionMode === "custom"
          ? {
              providerKind,
              modelName,
              baseUrl,
              accessKey,
              endpointName,
            }
          : undefined,
    });
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div>
          <div className={styles.eyebrow}>Builder AI</div>
          <h2 className={styles.title}>Natural language → DSL</h2>
          <p className={styles.subtitle}>
            Describe the routing behavior you want, generate a Builder-compatible
            DSL draft, review it, then deploy through the existing preview flow.
          </p>
        </div>
        <div className={styles.badges}>
          <span className={styles.badge}>Default alias: `MoM`</span>
          {hasContextDsl && (
            <span className={styles.badgeMuted}>
              Context loaded: {currentDsl.split("\n").length} lines
            </span>
          )}
        </div>
      </div>

      <div className={styles.grid}>
        <section className={styles.card}>
          <div className={styles.cardHeader}>
            <h3 className={styles.cardTitle}>Describe the routing change</h3>
            <span className={styles.cardHint}>
              Existing Builder DSL can be used as editing context.
            </span>
          </div>

          <div className={styles.presets}>
            {PROMPT_PRESETS.map((item) => (
              <button
                key={item}
                className={styles.presetBtn}
                onClick={() => setPrompt(item)}
                type="button"
              >
                {item}
              </button>
            ))}
          </div>

          <label className={styles.label} htmlFor="builder-nl-prompt">
            Request
          </label>
          <textarea
            id="builder-nl-prompt"
            className={styles.textarea}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Example: create a high-priority route for urgent customer escalations, add a multilingual support route, and keep a general fallback route to MoM."
          />

          <label className={styles.checkboxRow}>
            <input
              checked={useCurrentDslContext}
              disabled={!hasContextDsl}
              onChange={(event) => setUseCurrentDslContext(event.target.checked)}
              type="checkbox"
            />
            <span>
              Use the current Builder draft as context
              {hasContextDsl ? " to modify the existing configuration." : "."}
            </span>
          </label>

          <div className={styles.cardHeader}>
            <h3 className={styles.cardTitle}>Model connection</h3>
            <span className={styles.cardHint}>
              Default mode uses the current router gateway and prefers `MoM`.
            </span>
          </div>

          <div className={styles.segmented}>
            <button
              className={
                connectionMode === "default"
                  ? styles.segmentActive
                  : styles.segment
              }
              onClick={() => setConnectionMode("default")}
              type="button"
            >
              Default `MoM`
            </button>
            <button
              className={
                connectionMode === "custom"
                  ? styles.segmentActive
                  : styles.segment
              }
              onClick={() => setConnectionMode("custom")}
              type="button"
            >
              Custom connection
            </button>
          </div>

          {connectionMode === "default" ? (
            <div className={styles.noteBox}>
              Generated routes will prefer the router's automatic model alias
              `MoM`. This keeps deployment aligned with the current runtime.
            </div>
          ) : (
            <div className={styles.formGrid}>
              <div>
                <label className={styles.label} htmlFor="builder-nl-provider">
                  Provider type
                </label>
                <select
                  id="builder-nl-provider"
                  className={styles.select}
                  value={providerKind}
                  onChange={(event) =>
                    setProviderKind(event.target.value as BuilderNLProviderKind)
                  }
                >
                  {PROVIDER_OPTIONS.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
                <div className={styles.helpText}>{activeProvider.description}</div>
              </div>

              <div>
                <label className={styles.label} htmlFor="builder-nl-model">
                  Model name
                </label>
                <input
                  id="builder-nl-model"
                  className={styles.input}
                  value={modelName}
                  onChange={(event) => setModelName(event.target.value)}
                  placeholder="gpt-4o-mini"
                />
              </div>

              <div>
                <label className={styles.label} htmlFor="builder-nl-baseurl">
                  Base URL
                </label>
                <input
                  id="builder-nl-baseurl"
                  className={styles.input}
                  value={baseUrl}
                  onChange={(event) => setBaseUrl(event.target.value)}
                  placeholder={activeProvider.placeholder}
                />
              </div>

              <div>
                <label className={styles.label} htmlFor="builder-nl-endpoint">
                  Endpoint name
                </label>
                <input
                  id="builder-nl-endpoint"
                  className={styles.input}
                  value={endpointName}
                  onChange={(event) => setEndpointName(event.target.value)}
                  placeholder="nl-custom"
                />
              </div>

              <div className={styles.fullSpan}>
                <label className={styles.label} htmlFor="builder-nl-key">
                  Access key
                </label>
                <input
                  id="builder-nl-key"
                  className={styles.input}
                  type="password"
                  value={accessKey}
                  onChange={(event) => setAccessKey(event.target.value)}
                  placeholder="Optional unless the endpoint requires authentication"
                />
              </div>
            </div>
          )}

          {error && <div className={styles.errorBox}>{error}</div>}

          <div className={styles.actions}>
            <button
              className={styles.primaryBtn}
              disabled={generating || !prompt.trim()}
              onClick={() => {
                void handleGenerate();
              }}
              type="button"
            >
              {generating ? "Generating…" : "Generate DSL draft"}
            </button>
            <button
              className={styles.secondaryBtn}
              onClick={() => onModeSwitch("dsl")}
              type="button"
            >
              Open DSL editor
            </button>
          </div>
        </section>

        <section className={styles.card}>
          <div className={styles.cardHeader}>
            <h3 className={styles.cardTitle}>Review and next steps</h3>
            <span className={styles.cardHint}>
              The generated draft is written into the Builder store immediately.
            </span>
          </div>

          {lastPrompt ? (
            <>
              <div className={styles.resultRow}>
                <div>
                  <div className={styles.resultLabel}>Last request</div>
                  <div className={styles.resultText}>{lastPrompt}</div>
                </div>
                <button
                  className={styles.ghostBtn}
                  onClick={onClearResult}
                  type="button"
                >
                  Clear review
                </button>
              </div>

              <div className={styles.resultBlock}>
                <div className={styles.resultLabel}>Generation summary</div>
                <div className={styles.resultText}>
                  {summary || "DSL draft generated and loaded into Builder."}
                </div>
              </div>

              <div className={styles.statusGrid}>
                <div className={styles.statusCard}>
                  <div className={styles.resultLabel}>Compile status</div>
                  <div
                    className={
                      errorCount === 0 ? styles.statusOk : styles.statusWarn
                    }
                  >
                    {errorCount === 0
                      ? "Builder validation passed"
                      : `${errorCount} validation error${errorCount === 1 ? "" : "s"}`}
                  </div>
                  {warningCount > 0 && (
                    <div className={styles.statusMeta}>
                      {warningCount} warning{warningCount === 1 ? "" : "s"}
                    </div>
                  )}
                </div>

                <div className={styles.statusCard}>
                  <div className={styles.resultLabel}>AI double-check</div>
                  <div
                    className={review?.ready ? styles.statusOk : styles.statusWarn}
                  >
                    {review?.ready ? "Ready for preview" : "Needs manual review"}
                  </div>
                  <div className={styles.statusMeta}>
                    {review?.summary || "No AI review summary returned."}
                  </div>
                </div>
              </div>

              {suggestedTestQuery && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Suggested test prompt</div>
                  <div className={styles.testQuery}>{suggestedTestQuery}</div>
                </div>
              )}

              {review?.warnings?.length ? (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Review warnings</div>
                  <ul className={styles.list}>
                    {review.warnings.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {review?.checks?.length ? (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Checks completed</div>
                  <ul className={styles.listMuted}>
                    {review.checks.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {generatedPreview && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Generated DSL preview</div>
                  <pre className={styles.preview}>{generatedPreview}</pre>
                </div>
              )}

              <div className={styles.noteBox}>
                Use the top toolbar to run compile, validation, deploy preview, and
                deploy. The existing Builder flow is already connected to this
                generated draft.
              </div>
            </>
          ) : (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>🤖</div>
              <div className={styles.emptyTitle}>No generated draft yet</div>
              <div className={styles.emptyText}>
                Generate a draft to see the Builder validation result, AI review
                summary, suggested test prompt, and the live DSL preview.
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export { BuilderNaturalLanguagePanel };

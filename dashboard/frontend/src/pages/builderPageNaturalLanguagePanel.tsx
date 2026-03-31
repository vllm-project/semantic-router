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
  const contextLineCount = hasContextDsl ? currentDsl.split("\n").length : 0;
  const hasGeneratedDraft = lastPrompt.trim().length > 0;
  const activeProvider = PROVIDER_OPTIONS.find((item) => item.id === providerKind)!;
  const generatedPreview = useMemo(() => {
    if (!hasGeneratedDraft || !currentDsl.trim()) {
      return "";
    }
    return currentDsl.trim().split("\n").slice(0, 18).join("\n");
  }, [currentDsl, hasGeneratedDraft]);
  const reviewWarnings = review?.warnings ?? [];
  const reviewChecks = review?.checks ?? [];

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
      <div className={styles.inner}>
        <section className={styles.hero}>
          <div className={styles.heroMain}>
            <div className={styles.eyebrow}>Builder AI</div>
            <h2 className={styles.title}>Natural language → DSL</h2>
            <p className={styles.subtitle}>
              Describe the routing behavior you want, generate a Builder-compatible
              DSL draft, review it, then continue through the existing compile,
              validate, deploy preview, and deploy flow.
            </p>
          </div>

          <div className={styles.heroMeta}>
            <div className={styles.badges}>
              <span className={styles.badge}>Default alias · `MoM`</span>
              <span className={styles.badgeMuted}>
                {hasContextDsl
                  ? `Context loaded · ${contextLineCount} lines`
                  : "Start from a blank draft or reuse the current Builder DSL"}
              </span>
            </div>
            <div className={styles.heroNote}>
              Generated drafts are written into the Builder store immediately, so
              the existing toolbar actions continue to work without switching
              pages or modes.
            </div>
          </div>
        </section>

        <div className={styles.workspaceGrid}>
          <section className={`${styles.card} ${styles.composerCard}`}>
            <div className={styles.sectionHeader}>
              <div>
                <h3 className={styles.cardTitle}>Describe the routing change</h3>
                <p className={styles.sectionHint}>
                  Start with one clear request. You can also include the current
                  Builder DSL as editing context.
                </p>
              </div>
            </div>

            <div className={styles.subsection}>
              <div className={styles.subsectionHeader}>
                <div className={styles.resultLabel}>Prompt ideas</div>
                <div className={styles.subsectionHint}>
                  Pick one to prefill the editor, then tailor it.
                </div>
              </div>
              <div className={styles.presetGrid}>
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
            </div>

            <div className={styles.subsection}>
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

              <div className={styles.contextPanel}>
                <label className={styles.checkboxRow}>
                  <input
                    checked={useCurrentDslContext}
                    disabled={!hasContextDsl}
                    onChange={(event) =>
                      setUseCurrentDslContext(event.target.checked)
                    }
                    type="checkbox"
                  />
                  <span>
                    Use the current Builder draft as context
                    {hasContextDsl
                      ? " to modify the existing configuration."
                      : "."}
                  </span>
                </label>
                <div className={styles.contextMeta}>
                  {hasContextDsl
                    ? `${contextLineCount} lines of Builder DSL will be provided as editing context.`
                    : "No current Builder DSL is loaded yet, so the draft will be generated from scratch."}
                </div>
              </div>
            </div>

            <div className={styles.subsection}>
              <div className={styles.sectionHeader}>
                <div>
                  <h3 className={styles.cardTitle}>Model connection</h3>
                  <p className={styles.sectionHint}>
                    Default mode uses the current router gateway and prefers `MoM`.
                  </p>
                </div>
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
                <div className={styles.infoCard}>
                  <div className={styles.infoTitle}>
                    Use the current router runtime
                  </div>
                  <div className={styles.infoText}>
                    Generated routes will prefer the router&apos;s automatic model
                    alias `MoM`, keeping the generated draft aligned with the
                    current deployment runtime.
                  </div>
                </div>
              ) : (
                <div className={styles.formGrid}>
                  <div className={styles.fieldGroup}>
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
                    <div className={styles.helpText}>
                      {activeProvider.description}
                    </div>
                  </div>

                  <div className={styles.fieldGroup}>
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

                  <div className={styles.fieldGroup}>
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

                  <div className={styles.fieldGroup}>
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

                  <div className={`${styles.fieldGroup} ${styles.fullSpan}`}>
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
            </div>

            {error && <div className={styles.errorBox}>{error}</div>}

            <div className={styles.actionBar}>
              <button
                className={styles.primaryBtn}
                disabled={generating || !prompt.trim()}
                onClick={() => {
                  void handleGenerate();
                }}
                type="button"
              >
                {generating ? "Generating draft…" : "Generate DSL draft"}
              </button>
              <button
                className={styles.secondaryBtn}
                onClick={() => onModeSwitch("dsl")}
                type="button"
              >
                Open DSL editor
              </button>
              <div className={styles.actionHint}>
                The generated DSL is loaded directly into Builder, so you can
                immediately compile or open the DSL editor for manual polishing.
              </div>
            </div>
          </section>

          <aside className={styles.resultsRail}>
            <section className={styles.card}>
              <div className={styles.sectionHeader}>
                <div>
                  <h3 className={styles.cardTitle}>Review and next steps</h3>
                  <p className={styles.sectionHint}>
                    Check Builder validation, AI review, and the suggested test
                    prompt before deploying.
                  </p>
                </div>
                {hasGeneratedDraft && (
                  <button
                    className={styles.ghostBtn}
                    onClick={onClearResult}
                    type="button"
                  >
                    Clear review
                  </button>
                )}
              </div>

              {hasGeneratedDraft ? (
                <>
                  <div className={styles.summaryCard}>
                    <div className={styles.resultLabel}>Last request</div>
                    <div className={styles.requestPreview}>{lastPrompt}</div>
                    <div className={styles.summaryDivider} />
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
                      <div className={styles.statusMeta}>
                        {warningCount > 0
                          ? `${warningCount} warning${warningCount === 1 ? "" : "s"} still need attention.`
                          : "No Builder warnings were reported for the current draft."}
                      </div>
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

                  {reviewWarnings.length > 0 && (
                    <div className={styles.resultBlock}>
                      <div className={styles.resultLabel}>Review warnings</div>
                      <ul className={styles.list}>
                        {reviewWarnings.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {reviewChecks.length > 0 && (
                    <div className={styles.resultBlock}>
                      <div className={styles.resultLabel}>Checks completed</div>
                      <ul className={styles.listMuted}>
                        {reviewChecks.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className={styles.noteBox}>
                    Use the top toolbar to run compile, validation, deploy preview,
                    and deploy. The generated draft is already connected to the
                    existing Builder flow.
                  </div>
                </>
              ) : (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>🤖</div>
                  <div className={styles.emptyTitle}>No generated draft yet</div>
                  <div className={styles.emptyText}>
                    Generate a draft to populate validation status, AI review notes,
                    suggested prompts, and the DSL preview panel.
                  </div>
                  <div className={styles.emptyChecklist}>
                    <div className={styles.emptyChecklistItem}>
                      1. Describe the routing behavior you want.
                    </div>
                    <div className={styles.emptyChecklistItem}>
                      2. Choose whether to reuse the current Builder DSL.
                    </div>
                    <div className={styles.emptyChecklistItem}>
                      3. Generate, review, then continue with preview or deploy.
                    </div>
                  </div>
                </div>
              )}
            </section>

            <section className={styles.card}>
              <div className={styles.sectionHeader}>
                <div>
                  <h3 className={styles.cardTitle}>Generated DSL preview</h3>
                  <p className={styles.sectionHint}>
                    A compact live preview of the draft currently loaded into Builder.
                  </p>
                </div>
              </div>

              {generatedPreview ? (
                <pre className={styles.preview}>{generatedPreview}</pre>
              ) : (
                <div className={styles.previewEmpty}>
                  Generate a draft to inspect the live DSL preview here.
                </div>
              )}
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
};

export { BuilderNaturalLanguagePanel };

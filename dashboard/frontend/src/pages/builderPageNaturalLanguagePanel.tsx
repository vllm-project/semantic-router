import React, { useCallback, useEffect, useMemo, useState } from "react";

import { wasmBridge } from "@/lib/wasm";

import type {
  BuilderNLConnectionMode,
  BuilderNLGenerateRequest,
  BuilderNLProgressEvent,
  BuilderNLProviderKind,
  BuilderNLStagedDraft,
  BuilderNLVerifyResponse,
  EditorMode,
} from "@/types/dsl";

import { BuilderNaturalLanguageProgress } from "./builderPageNaturalLanguageProgress";
import styles from "./builderPageNaturalLanguagePanel.module.css";

interface BuilderNaturalLanguagePanelProps {
  currentDsl: string;
  baseConfigYaml: string;
  currentModelNames: string[];
  wasmReady: boolean;
  generating: boolean;
  error: string | null;
  progressEvents: BuilderNLProgressEvent[];
  stagedDraft: BuilderNLStagedDraft | null;
  onGenerate: (input: BuilderNLGenerateRequest) => Promise<void>;
  onApplyDraft: () => void;
  onDiscardDraft: () => void;
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

const DEFAULT_GENERATION_MODEL = "MoM";
const FALLBACK_TARGET_MODEL = DEFAULT_GENERATION_MODEL;

function extractDefaultModelName(baseConfigYaml: string): string | null {
  const match = baseConfigYaml.match(
    /^\s*default_model:\s*(?:"([^"]+)"|'([^']+)'|([^\n#]+))/m,
  );
  const candidate = match?.[1] ?? match?.[2] ?? match?.[3] ?? "";
  const trimmed = candidate.trim();
  return trimmed || null;
}

function buildPromptPresets(targetModelName: string): string[] {
  return [
    `Route urgent billing issues to a higher-priority route, then send everything else to ${targetModelName}.`,
    `Create separate routes for code debugging, math tutoring, and general chat, with a general fallback to ${targetModelName}.`,
    `Add multilingual routing so Chinese and English prompts get their own routes before a general fallback to ${targetModelName}.`,
  ];
}

const BuilderNaturalLanguagePanel: React.FC<
  BuilderNaturalLanguagePanelProps
> = ({
  currentDsl,
  baseConfigYaml,
  currentModelNames,
  wasmReady,
  generating,
  error,
  progressEvents,
  stagedDraft,
  onGenerate,
  onApplyDraft,
  onDiscardDraft,
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
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<BuilderNLVerifyResponse | null>(
    null,
  );
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [previewTab, setPreviewTab] = useState<"dsl" | "yaml" | "crd" | "baseYaml">(
    "dsl",
  );
  const [copiedPreview, setCopiedPreview] = useState<string | null>(null);

  const hasContextDsl = currentDsl.trim().length > 0;
  const contextLineCount = hasContextDsl ? currentDsl.split("\n").length : 0;
  const activeProvider =
    PROVIDER_OPTIONS.find((item) => item.id === providerKind) ??
    PROVIDER_OPTIONS[0];
  const stagedDiagnostics = stagedDraft?.validation.diagnostics ?? [];
  const stagedErrorCount = stagedDraft?.validation.errorCount ?? 0;
  const stagedWarningCount = stagedDiagnostics.filter(
    (item) => item.level !== "error",
  ).length;
  const stagedReviewWarnings = stagedDraft?.review.warnings ?? [];
  const stagedReviewChecks = stagedDraft?.review.checks ?? [];
  const stagedCompiledOutput = useMemo(() => {
    if (!stagedDraft) {
      return { yaml: "", crd: "", error: "" };
    }
    if (!wasmReady) {
      return {
        yaml: "",
        crd: "",
        error: "The Builder compiler is still loading, so YAML and CRD previews are not ready yet.",
      };
    }

    try {
      const result = wasmBridge.compile(stagedDraft.dsl);
      return {
        yaml: result.yaml || "",
        crd: result.crd || "",
        error: result.error || "",
      };
    } catch (err) {
      return {
        yaml: "",
        crd: "",
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }, [stagedDraft, wasmReady]);
  const liveModelCards = useMemo(
    () =>
      Array.from(
        new Set(currentModelNames.map((name) => name.trim()).filter(Boolean)),
      ),
    [currentModelNames],
  );
  const preferredTargetModelName = useMemo(() => {
    return (
      extractDefaultModelName(baseConfigYaml) ??
      liveModelCards[0] ??
      FALLBACK_TARGET_MODEL
    );
  }, [baseConfigYaml, liveModelCards]);
  const promptPresets = useMemo(
    () => buildPromptPresets(preferredTargetModelName),
    [preferredTargetModelName],
  );
  const activeGeneratorModelName =
    connectionMode === "default" ? DEFAULT_GENERATION_MODEL : modelName.trim();
  const connectionFingerprint = useMemo(
    () =>
      JSON.stringify({
        connectionMode,
        providerKind,
        modelName: modelName.trim(),
        baseUrl: baseUrl.trim(),
        endpointName: endpointName.trim(),
        accessKey: accessKey.trim(),
      }),
    [accessKey, baseUrl, connectionMode, endpointName, modelName, providerKind],
  );

  useEffect(() => {
    setVerifyResult(null);
    setVerifyError(null);
  }, [connectionFingerprint]);

  useEffect(() => {
    setPreviewTab("dsl");
    setCopiedPreview(null);
  }, [stagedDraft?.prompt, stagedDraft?.dsl, stagedDraft?.baseYaml]);

  const draftPreviewOptions = useMemo(() => {
    if (!stagedDraft) {
      return [];
    }

    return [
      {
        key: "dsl" as const,
        label: "DSL",
        content: stagedDraft.dsl,
        emptyText: "The staged DSL draft is empty.",
      },
      {
        key: "yaml" as const,
        label: "YAML",
        content: stagedCompiledOutput.yaml,
        emptyText:
          stagedCompiledOutput.error ||
          "No compiled YAML is available for this staged draft yet.",
      },
      {
        key: "crd" as const,
        label: "CRD",
        content: stagedCompiledOutput.crd,
        emptyText:
          stagedCompiledOutput.error ||
          "No CRD output is available for this staged draft yet.",
      },
      {
        key: "baseYaml" as const,
        label: "Base YAML",
        content: stagedDraft.baseYaml,
        emptyText: "No deploy base YAML is available for this staged draft.",
      },
    ];
  }, [stagedCompiledOutput, stagedDraft]);
  const activeDraftPreview =
    draftPreviewOptions.find((option) => option.key === previewTab) ??
    draftPreviewOptions[0];

  const handleCopyPreview = useCallback(async () => {
    if (!activeDraftPreview?.content) {
      return;
    }

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(activeDraftPreview.content);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = activeDraftPreview.content;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopiedPreview(activeDraftPreview.key);
      window.setTimeout(() => {
        setCopiedPreview((current) =>
          current === activeDraftPreview.key ? null : current,
        );
      }, 1800);
    } catch {
      setCopiedPreview(null);
    }
  }, [activeDraftPreview]);

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

  const handleVerify = async () => {
    setVerifying(true);
    setVerifyError(null);
    setVerifyResult(null);

    try {
      const response = await fetch("/api/router/config/nl/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
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
        }),
      });

      const body = await response.text();
      let payload: BuilderNLVerifyResponse | { message?: string; error?: string } =
        { summary: "" } as BuilderNLVerifyResponse;
      try {
        payload = body ? (JSON.parse(body) as typeof payload) : payload;
      } catch {
        payload = { message: body || "Connection verification failed." };
      }

      if (!response.ok) {
        throw new Error(
          "message" in payload && payload.message
            ? payload.message
            : "error" in payload && payload.error
              ? payload.error
              : `HTTP ${response.status}: ${response.statusText}`,
        );
      }

      setVerifyResult(payload as BuilderNLVerifyResponse);
    } catch (err) {
      setVerifyError(err instanceof Error ? err.message : String(err));
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className={styles.container}>
      <section className={styles.headerCard}>
        <div>
          <div className={styles.kicker}>Builder AI</div>
          <h2 className={styles.title}>Natural language drafts</h2>
          <p className={styles.subtitle}>
            Generate a staged DSL draft, inspect repository validation results,
            then apply it into Builder only when you are ready.
          </p>
        </div>
        <div className={styles.headerMeta}>
          <span className={styles.badge}>
            Preferred target: `{preferredTargetModelName}`
          </span>
          {liveModelCards.length > 0 ? (
            <div className={styles.modelCardGroup}>
              <span className={styles.badgeMuted}>
                Live model cards: {liveModelCards.length}
              </span>
              <div className={styles.modelCardList}>
                {liveModelCards.map((modelName) => (
                  <span className={styles.modelCardChip} key={modelName}>
                    `{modelName}`
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <span className={styles.badgeMuted}>
              No live router model cards are loaded yet, so Builder will fall
              back to `{FALLBACK_TARGET_MODEL}`.
            </span>
          )}
          <span className={styles.badgeMuted}>
            {hasContextDsl
              ? `Live Builder context: ${contextLineCount} lines`
              : "No live Builder DSL loaded yet"}
          </span>
          <span className={styles.badgeMuted}>
            Live Builder state stays untouched until you apply the staged draft.
          </span>
        </div>
      </section>

      <BuilderNaturalLanguageProgress
        generating={generating}
        progressEvents={progressEvents}
      />

      <div className={styles.contentStack}>
        <section className={styles.card}>
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.cardTitle}>Describe the routing change</h3>
              <p className={styles.sectionHint}>
                Keep the request concrete. Mention routes, signals, priorities,
                and fallback behavior.
              </p>
            </div>
          </div>

          <div className={styles.presetGrid}>
            {promptPresets.map((item) => (
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
            placeholder={`Example: create a high-priority route for urgent customer escalations, add a multilingual support route, and keep a general fallback route to ${preferredTargetModelName}.`}
          />

          <div className={styles.contextPanel}>
            <label className={styles.checkboxRow}>
              <input
                checked={useCurrentDslContext}
                disabled={!hasContextDsl}
                onChange={(event) => setUseCurrentDslContext(event.target.checked)}
                type="checkbox"
              />
              <span>
                Reuse the live Builder DSL as editing context
                {hasContextDsl ? "." : " (no live Builder DSL is loaded yet)."}
              </span>
            </label>
            <div className={styles.contextMeta}>
              {hasContextDsl
                ? `${contextLineCount} lines from the live Builder draft will be sent as context, but the live draft will not be replaced until you apply the staged result.`
                : "Generation will start from an empty Builder draft."}
            </div>
          </div>

          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.cardTitle}>Generation connection</h3>
              <p className={styles.sectionHint}>
                This controls which model generates the draft. It does not
                silently rewrite deploy-time provider config.
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
              Default runtime
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
              <div className={styles.infoTitle}>Use the current router runtime</div>
              <div className={styles.infoText}>
                Builder will call the configured runtime gateway with
                `{DEFAULT_GENERATION_MODEL}` to create the draft. The generated
                DSL still reuses the current router model cards and prefers
                `{preferredTargetModelName}` for route references.
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
                <div className={styles.helpText}>{activeProvider.description}</div>
              </div>

              <div className={styles.fieldGroup}>
                <label className={styles.label} htmlFor="builder-nl-model">
                  Generation model
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

          {(error || verifyError) && (
            <div className={styles.errorBox}>{error || verifyError}</div>
          )}

          {verifyResult && (
            <div className={styles.verifyBox}>
              <div className={styles.verifyTitle}>Connection verified</div>
              <div className={styles.verifyText}>{verifyResult.summary}</div>
              <div className={styles.verifyMeta}>
                {verifyResult.modelName
                  ? `Generator: ${verifyResult.modelName}`
                  : `Generator: ${activeGeneratorModelName || DEFAULT_GENERATION_MODEL}`}
                {verifyResult.targetModelName
                  ? ` · Draft target: ${verifyResult.targetModelName}`
                  : ""}
                {verifyResult.endpoint ? ` · Endpoint: ${verifyResult.endpoint}` : ""}
              </div>
            </div>
          )}

          <div className={styles.actionRow}>
            <button
              className={styles.secondaryBtn}
              disabled={verifying}
              onClick={() => {
                void handleVerify();
              }}
              type="button"
            >
              {verifying ? "Verifying…" : "Verify connection"}
            </button>
            <button
              className={styles.primaryBtn}
              disabled={generating || !prompt.trim()}
              onClick={() => {
                void handleGenerate();
              }}
              type="button"
            >
              {generating ? "Generating draft…" : "Generate staged draft"}
            </button>
            <button
              className={styles.ghostBtn}
              onClick={() => onModeSwitch("dsl")}
              type="button"
            >
              Open live DSL editor
            </button>
          </div>
        </section>

        <section className={styles.card}>
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.cardTitle}>Staged draft</h3>
              <p className={styles.sectionHint}>
                Review the generated draft before it replaces the live Builder
                DSL.
              </p>
            </div>
            {stagedDraft && (
              <button
                className={styles.ghostBtn}
                onClick={onDiscardDraft}
                type="button"
              >
                Discard
              </button>
            )}
          </div>

          {stagedDraft ? (
            <>
              <div className={styles.summaryCard}>
                <div className={styles.resultLabel}>Request</div>
                <div className={styles.requestPreview}>{stagedDraft.prompt}</div>
                <div className={styles.summaryDivider} />
                <div className={styles.resultLabel}>Summary</div>
                <div className={styles.resultText}>
                  {stagedDraft.summary || "Draft generated."}
                </div>
              </div>

              <div className={styles.statusGrid}>
                <div className={styles.statusCard}>
                  <div className={styles.resultLabel}>Draft validation</div>
                  <div
                    className={
                      stagedDraft.validation.ready
                        ? styles.statusOk
                        : styles.statusWarn
                    }
                  >
                    {stagedDraft.validation.ready
                      ? "Repository validation passed"
                      : `${stagedErrorCount} validation error${stagedErrorCount === 1 ? "" : "s"}`}
                  </div>
                  <div className={styles.statusMeta}>
                    {stagedDraft.validation.ready
                      ? stagedWarningCount > 0
                        ? `${stagedWarningCount} warning${stagedWarningCount === 1 ? "" : "s"} still deserve a review.`
                        : "No validation blockers were reported for this staged draft."
                      : "The staged draft is preserved here for inspection and optional manual repair."}
                  </div>
                </div>

                <div className={styles.statusCard}>
                  <div className={styles.resultLabel}>Readiness review</div>
                  <div
                    className={
                      stagedDraft.review.ready
                        ? styles.statusOk
                        : styles.statusWarn
                    }
                  >
                    {stagedDraft.review.ready
                      ? "Ready for Builder apply"
                      : "Manual review recommended"}
                  </div>
                  <div className={styles.statusMeta}>
                    {stagedDraft.review.summary}
                  </div>
                </div>
              </div>

              {stagedDraft.validation.compileError && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Compile error</div>
                  <div className={styles.errorText}>
                    {stagedDraft.validation.compileError}
                  </div>
                </div>
              )}

              {stagedDiagnostics.length > 0 && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Validation findings</div>
                  <ul className={styles.list}>
                    {stagedDiagnostics.slice(0, 6).map((item, index) => (
                      <li key={`${item.message}-${index}`}>
                        <strong>{item.level}</strong>: {item.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {stagedReviewWarnings.length > 0 && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Review warnings</div>
                  <ul className={styles.list}>
                    {stagedReviewWarnings.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {stagedReviewChecks.length > 0 && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Checks completed</div>
                  <ul className={styles.listMuted}>
                    {stagedReviewChecks.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {stagedDraft.suggestedTestQuery && (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Suggested test prompt</div>
                  <div className={styles.testQuery}>
                    {stagedDraft.suggestedTestQuery}
                  </div>
                </div>
              )}

              <div className={styles.resultBlock}>
                <div className={styles.previewHeader}>
                  <div>
                    <div className={styles.resultLabel}>Draft outputs</div>
                    <div className={styles.previewCaption}>
                      Inspect the staged DSL and generated outputs before you
                      apply anything into the live Builder state.
                    </div>
                  </div>
                  <div className={styles.previewActions}>
                    <div className={styles.previewTabRow}>
                      {draftPreviewOptions.map((option) => (
                        <button
                          className={
                            option.key === activeDraftPreview?.key
                              ? styles.segmentActive
                              : styles.segment
                          }
                          key={option.key}
                          onClick={() => setPreviewTab(option.key)}
                          type="button"
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                    <button
                      className={styles.ghostBtn}
                      disabled={!activeDraftPreview?.content}
                      onClick={handleCopyPreview}
                      type="button"
                    >
                      {copiedPreview === activeDraftPreview?.key
                        ? "Copied"
                        : `Copy ${activeDraftPreview?.label ?? "output"}`}
                    </button>
                  </div>
                </div>

                {activeDraftPreview?.content ? (
                  <pre className={styles.preview}>{activeDraftPreview.content}</pre>
                ) : (
                  <div className={styles.previewEmpty}>
                    {activeDraftPreview?.emptyText ||
                      "No staged draft output is available yet."}
                  </div>
                )}
              </div>

              <div className={styles.applyRow}>
                <button
                  className={styles.primaryBtn}
                  onClick={onApplyDraft}
                  type="button"
                >
                  {stagedDraft.validation.ready
                    ? "Apply draft to Builder"
                    : "Open draft in Builder for repair"}
                </button>
                <button
                  className={styles.secondaryBtn}
                  onClick={() => onModeSwitch("dsl")}
                  type="button"
                >
                  Open live DSL editor
                </button>
              </div>
            </>
          ) : (
            <div className={styles.emptyState}>
              <div className={styles.emptyTitle}>No staged draft yet</div>
              <div className={styles.emptyText}>
                Generate a draft to inspect repository validation and the staged
                readiness review before you touch the live Builder state.
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export { BuilderNaturalLanguagePanel };

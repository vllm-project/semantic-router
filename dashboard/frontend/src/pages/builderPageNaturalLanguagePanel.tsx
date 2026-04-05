import React, { useCallback, useEffect, useMemo, useState } from "react";

import { wasmBridge } from "@/lib/wasm";

import type {
  BuilderNLConnectionMode,
  BuilderNLGenerateRequest,
  BuilderNLProgressEvent,
  BuilderNLProviderKind,
  BuilderNLStagedDraft,
  BuilderNLVerifyRequest,
  BuilderNLVerifyResponse,
  EditorMode,
} from "@/types/dsl";

import DashboardSurfaceHero from "../components/DashboardSurfaceHero";
import { BuilderNaturalLanguageProgress } from "./builderPageNaturalLanguageProgress";
import { BuilderNaturalLanguageResultCard } from "./builderPageNaturalLanguageResultCard";
import {
  DEFAULT_GENERATION_MODEL,
  DEFAULT_GENERATION_TEMPERATURE,
  DEFAULT_REPAIR_BUDGET,
  DEFAULT_TIMEOUT_SECONDS,
  FALLBACK_TARGET_MODEL,
  PROVIDER_OPTIONS,
  buildPromptPresets,
  extractDefaultModelName,
  normalizeOptionalFloat,
  normalizeOptionalInteger,
} from "./builderPageNaturalLanguagePanelSupport";
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
  const [providerKind, setProviderKind] =
    useState<BuilderNLProviderKind>("openai-compatible");
  const [modelName, setModelName] = useState("gpt-4o-mini");
  const [baseUrl, setBaseUrl] = useState("");
  const [accessKey, setAccessKey] = useState("");
  const [endpointName, setEndpointName] = useState("nl-custom");
  const [temperature, setTemperature] = useState(
    String(DEFAULT_GENERATION_TEMPERATURE),
  );
  const [maxRetries, setMaxRetries] = useState(String(DEFAULT_REPAIR_BUDGET));
  const [timeoutSeconds, setTimeoutSeconds] = useState(
    String(DEFAULT_TIMEOUT_SECONDS),
  );
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
  const activeProvider =
    PROVIDER_OPTIONS.find((item) => item.id === providerKind) ??
    PROVIDER_OPTIONS[0];
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
  const normalizedTemperature = normalizeOptionalFloat(temperature);
  const normalizedMaxRetries = normalizeOptionalInteger(maxRetries);
  const normalizedTimeoutSeconds = normalizeOptionalInteger(timeoutSeconds);
  const connectionFingerprint = useMemo(
    () =>
      JSON.stringify({
        connectionMode,
        providerKind,
        modelName: modelName.trim(),
        baseUrl: baseUrl.trim(),
        endpointName: endpointName.trim(),
        accessKey: accessKey.trim(),
        timeoutSeconds: normalizedTimeoutSeconds,
      }),
    [
      accessKey,
      baseUrl,
      connectionMode,
      endpointName,
      modelName,
      normalizedTimeoutSeconds,
      providerKind,
    ],
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
  const heroMeta = useMemo(
    () => [
      { label: "Current surface", value: "NL→DSL" },
      {
        label: "Generator",
        value: activeGeneratorModelName || DEFAULT_GENERATION_MODEL,
      },
      { label: "Draft target", value: preferredTargetModelName },
    ],
    [activeGeneratorModelName, preferredTargetModelName],
  );
  const heroPills = useMemo(
    () => [
      { label: "Prompt", active: true },
      { label: "Process" },
      { label: "Staged Draft" },
    ],
    [],
  );

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
      currentDsl: hasContextDsl ? currentDsl : "",
      connectionMode,
      temperature: normalizedTemperature,
      maxRetries: normalizedMaxRetries,
      timeoutSeconds: normalizedTimeoutSeconds,
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
      const verifyRequest: BuilderNLVerifyRequest = {
        connectionMode,
        timeoutSeconds: normalizedTimeoutSeconds,
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
      };
      const response = await fetch("/api/router/config/nl/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(verifyRequest),
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
      <div className={styles.heroShell}>
        <DashboardSurfaceHero
          eyebrow="Builder"
          title="Natural language"
          description="Describe the routing change. Builder stages the draft only after the shared generation call and repository checks finish."
          meta={heroMeta}
          panelEyebrow="Shared nlgen"
          panelTitle="Schema-grounded NL→DSL"
          panelDescription="Run the shared generation pipeline first, then review the staged Builder draft before applying it to the live config."
          pills={heroPills}
          panelFooter={
            liveModelCards.length > 0 ? (
              <div className={styles.modelCardGroup}>
                <span className={styles.badgeMuted}>Live route models</span>
                <div className={styles.modelCardList}>
                  {liveModelCards.map((liveModelName) => (
                    <span className={styles.modelCardChip} key={liveModelName}>
                      `{liveModelName}`
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <span className={styles.badgeMuted}>
                No live router model cards are loaded yet, so Builder will fall back
                to `{FALLBACK_TARGET_MODEL}` for draft route references.
              </span>
            )
          }
        />
      </div>

      <div className={styles.bodyColumn}>
        <BuilderNaturalLanguageProgress
          generating={generating}
          progressEvents={progressEvents}
          stagedDraft={stagedDraft}
        />

        <div className={styles.contentStack}>
        <section className={styles.card}>
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.cardTitle}>Prompt and runtime</h3>
              <p className={styles.sectionHint}>
                Describe the routing change, then choose how the shared model
                call should run.
              </p>
            </div>
          </div>

          <div className={styles.composerGrid}>
            <div className={styles.composerPrimary}>
              <div>
                <div className={styles.resultLabel}>Routing request</div>
                <div className={styles.composerTitle}>
                  Describe the routing intent
                </div>
                <div className={styles.composerText}>
                  Write the same kind of request you would pass to `sr-dsl
                  generate`: what should route where, how it should match, and
                  what fallback behavior should remain.
                </div>
              </div>

              <div className={styles.requestPanel}>
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

                <div className={styles.requestMain}>
                  <label className={styles.label} htmlFor="builder-nl-prompt">
                    Request
                  </label>
                  <textarea
                    id="builder-nl-prompt"
                    className={styles.textarea}
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder={`Example: add domain routes for computer science and math, then keep a general fallback to ${preferredTargetModelName}.`}
                  />
                  {!hasContextDsl ? (
                    <div className={styles.inlineHint}>
                      No live Builder DSL is loaded yet, so generation will
                      start from an empty draft.
                    </div>
                  ) : null}
                </div>
              </div>
            </div>

            <div className={styles.composerSecondary}>
              <div>
                <div className={styles.resultLabel}>Generation runtime</div>
                <div className={styles.composerTitle}>
                  Choose how the shared model call runs
                </div>
                <div className={styles.composerText}>
                  This changes only the generation runtime. It does not rewrite
                  deploy-time provider config or the live Builder state.
                </div>
              </div>

              <div className={styles.runtimeSummaryGrid}>
                <div className={styles.runtimeSummaryCard}>
                  <div className={styles.resultLabel}>Generator</div>
                  <div className={styles.runtimeSummaryValue}>
                    `{activeGeneratorModelName || DEFAULT_GENERATION_MODEL}`
                  </div>
                  <div className={styles.runtimeSummaryText}>
                    The model used for the actual NL→DSL generation call.
                  </div>
                </div>
                <div className={styles.runtimeSummaryCard}>
                  <div className={styles.resultLabel}>Draft target</div>
                  <div className={styles.runtimeSummaryValue}>
                    `{preferredTargetModelName}`
                  </div>
                  <div className={styles.runtimeSummaryText}>
                    The preferred live router model referenced by generated
                    routes.
                  </div>
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
                  <div className={styles.infoTitle}>Use the current runtime</div>
                  <div className={styles.infoText}>
                    Builder calls the configured runtime gateway with
                    `{DEFAULT_GENERATION_MODEL}` and still prefers
                    `{preferredTargetModelName}` for generated route references.
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

              <div className={styles.formGrid}>
                <div className={styles.fieldGroup}>
                  <label
                    className={styles.label}
                    htmlFor="builder-nl-temperature"
                  >
                    Temperature
                  </label>
                  <input
                    id="builder-nl-temperature"
                    className={styles.input}
                    inputMode="decimal"
                    type="number"
                    min="0"
                    max="2"
                    step="0.05"
                    value={temperature}
                    onChange={(event) => setTemperature(event.target.value)}
                  />
                  <div className={styles.helpText}>
                    Default {DEFAULT_GENERATION_TEMPERATURE}. Lower is more
                    deterministic.
                  </div>
                </div>

                <div className={styles.fieldGroup}>
                  <label
                    className={styles.label}
                    htmlFor="builder-nl-max-retries"
                  >
                    Repair budget
                  </label>
                  <input
                    id="builder-nl-max-retries"
                    className={styles.input}
                    inputMode="numeric"
                    type="number"
                    min="0"
                    max="4"
                    step="1"
                    value={maxRetries}
                    onChange={(event) => setMaxRetries(event.target.value)}
                  />
                  <div className={styles.helpText}>
                    Default {DEFAULT_REPAIR_BUDGET}. Applies to shared parse
                    retries and staged-draft repair rounds.
                  </div>
                </div>

                <div className={styles.fieldGroup}>
                  <label
                    className={styles.label}
                    htmlFor="builder-nl-timeout-seconds"
                  >
                    Timeout (seconds)
                  </label>
                  <input
                    id="builder-nl-timeout-seconds"
                    className={styles.input}
                    inputMode="numeric"
                    type="number"
                    min="30"
                    max="600"
                    step="5"
                    value={timeoutSeconds}
                    onChange={(event) => setTimeoutSeconds(event.target.value)}
                  />
                  <div className={styles.helpText}>
                    Default {DEFAULT_TIMEOUT_SECONDS}s per model call.
                  </div>
                </div>
              </div>

              {(error || verifyError) && (
                <div className={styles.errorBox}>{error || verifyError}</div>
              )}

              {verifyResult && (
                <div className={styles.verifyBox}>
                  <div className={styles.verifyTitle}>Runtime verified</div>
                  <div className={styles.verifyText}>{verifyResult.summary}</div>
                  <div className={styles.verifyMeta}>
                    {verifyResult.modelName
                      ? `Generator: ${verifyResult.modelName}`
                      : `Generator: ${activeGeneratorModelName || DEFAULT_GENERATION_MODEL}`}
                    {verifyResult.targetModelName
                      ? ` · Draft target: ${verifyResult.targetModelName}`
                      : ""}
                    {verifyResult.endpoint
                      ? ` · Endpoint: ${verifyResult.endpoint}`
                      : ""}
                  </div>
                </div>
              )}

              <div className={styles.actionRow}>
                <button
                  className={styles.primaryBtn}
                  disabled={generating || !prompt.trim()}
                  onClick={() => {
                    void handleGenerate();
                  }}
                  type="button"
                >
                  {generating ? "Generating DSL…" : "Generate DSL"}
                </button>
                <button
                  className={styles.secondaryBtn}
                  disabled={verifying}
                  onClick={() => {
                    void handleVerify();
                  }}
                  type="button"
                >
                  {verifying ? "Verifying runtime…" : "Verify runtime"}
                </button>
              </div>
            </div>
          </div>
        </section>

        <BuilderNaturalLanguageResultCard
          stagedDraft={stagedDraft}
          draftPreviewOptions={draftPreviewOptions}
          activeDraftPreview={activeDraftPreview}
          copiedPreview={copiedPreview}
          onPreviewTabChange={setPreviewTab}
          onCopyPreview={handleCopyPreview}
          onApplyDraft={onApplyDraft}
          onDiscardDraft={onDiscardDraft}
          onModeSwitch={onModeSwitch}
        />
      </div>
      </div>
    </div>
  );
};

export { BuilderNaturalLanguagePanel };

import React, { useEffect, useMemo, useRef, useState } from "react";

import type { BuilderNLProgressEvent, BuilderNLStagedDraft } from "@/types/dsl";

import {
  ATTEMPT_PHASES,
  type BuilderNLProcessStatus,
  PhaseIcon,
  attemptPhaseEvent,
  buildAttemptSummary,
  compactProgressMessage,
  compactProgressText,
  deriveSyntheticStatus,
  formatDuration,
  formatPhaseLabel,
  formatTimestamp,
  isHeartbeatEvent,
  latestEventForPhase,
  latestEventForPhases,
  resolveEventStatus,
  summarizeAttempt,
} from "./builderPageNaturalLanguageProgressSupport";
import styles from "./builderPageNaturalLanguagePanel.module.css";

interface ProcessNode {
  key: string;
  label: string;
  description: string;
  phase: string;
  status: BuilderNLProcessStatus;
  message: string;
  event?: BuilderNLProgressEvent;
  note?: string;
}

function processStatusLabel(status: BuilderNLProcessStatus): string {
  switch (status) {
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "warning":
      return "Needs review";
    case "error":
      return "Failed";
    default:
      return "Pending";
  }
}

function processStatusClassName(status: BuilderNLProcessStatus): string {
  switch (status) {
    case "running":
      return styles.processStatusRunning;
    case "completed":
      return styles.processStatusCompleted;
    case "warning":
      return styles.processStatusWarning;
    case "error":
      return styles.processStatusError;
    default:
      return styles.processStatusPending;
  }
}

function processNodeClassName(status: BuilderNLProcessStatus): string {
  switch (status) {
    case "running":
      return styles.runningNode;
    case "completed":
      return styles.completedNode;
    case "warning":
      return styles.warningNode;
    case "error":
      return styles.errorNode;
    default:
      return styles.pendingNode;
  }
}

function eventLevelClassName(level: BuilderNLProgressEvent["level"]): string {
  switch (level) {
    case "success":
      return styles.processEventSuccess;
    case "warning":
      return styles.processEventWarning;
    case "error":
      return styles.processEventError;
    default:
      return styles.processEventInfo;
  }
}

interface BuilderNaturalLanguageProgressProps {
  generating: boolean;
  progressEvents: BuilderNLProgressEvent[];
  stagedDraft: BuilderNLStagedDraft | null;
}

const BuilderNaturalLanguageProgress: React.FC<
  BuilderNaturalLanguageProgressProps
> = ({ generating, progressEvents, stagedDraft }) => {
  const consoleRef = useRef<HTMLDivElement | null>(null);
  const [showRawEvents, setShowRawEvents] = useState(false);

  useEffect(() => {
    if (!showRawEvents || !consoleRef.current) {
      return;
    }
    consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [progressEvents, showRawEvents]);

  const latestProgressEvent =
    progressEvents.length > 0 ? progressEvents[progressEvents.length - 1] : null;
  const latestStableEvent = useMemo(() => {
    return [...progressEvents].reverse().find((event) => !isHeartbeatEvent(event)) ?? null;
  }, [progressEvents]);

  const attemptSummaries = useMemo(() => {
    const byAttempt = new Map<number, BuilderNLProgressEvent[]>();
    for (const event of progressEvents) {
      const attempt = event.attempt ?? 0;
      if (attempt <= 0) {
        continue;
      }
      const events = byAttempt.get(attempt);
      if (events) {
        events.push(event);
      } else {
        byAttempt.set(attempt, [event]);
      }
    }

    return Array.from(byAttempt.entries())
      .sort(([left], [right]) => left - right)
      .map(([attempt, events]) => buildAttemptSummary(attempt, events));
  }, [progressEvents]);

  const contextEvent = latestEventForPhase(progressEvents, "context");
  const validateEvent = latestEventForPhase(progressEvents, "validate");
  const reviewEvent = latestEventForPhase(progressEvents, "review");
  const completeEvent = latestEventForPhase(progressEvents, "complete");
  const promptEvent = latestEventForPhases(progressEvents, [
    "generate",
    "context",
    "request",
  ]);
  const modelCallEvent = attemptPhaseEvent(attemptSummaries, "model_call");
  const parseEvent = attemptPhaseEvent(attemptSummaries, "parse");
  const repairEvent = attemptPhaseEvent(attemptSummaries, "repair");

  const firstEvent = progressEvents.length > 0 ? progressEvents[0] : null;
  const elapsedSeconds =
    firstEvent && latestProgressEvent
      ? Math.max(0, Math.round((latestProgressEvent.timestamp - firstEvent.timestamp) / 1000))
      : null;

  const attemptCount = attemptSummaries.length;
  const currentAttemptLabel =
    attemptCount > 0
      ? `Attempt ${attemptSummaries[attemptCount - 1].attempt}`
      : generating
        ? "Preparing attempt 1"
        : "No model attempt yet";

  const currentActivityEvent = latestProgressEvent ?? latestStableEvent;
  const currentActivityLabel = currentActivityEvent
    ? formatPhaseLabel(currentActivityEvent.phase)
    : "Idle";
  const currentActivityMessage = compactProgressMessage(
    currentActivityEvent ?? undefined,
    "Send a request to start generation.",
  );

  const finalDraftReady = Boolean(stagedDraft);
  const repositoryChecksBlocked = Boolean(
    stagedDraft && (!stagedDraft.validation.ready || !stagedDraft.review.ready),
  );
  const stagedDraftValidationReady = stagedDraft?.validation.ready ?? false;
  const repairNeeded = attemptCount > 1 || Boolean(repairEvent);

  const coreNodes: ProcessNode[] = [
    {
      key: "prompt",
      label: "Ground prompt",
      description: "Build the shared prompt.",
      phase: "generate",
      event: promptEvent,
      status: deriveSyntheticStatus({
        event: promptEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["request", "context", "generate"],
        completed: attemptCount > 0 || Boolean(promptEvent && !generating),
      }),
      message: compactProgressMessage(
        promptEvent,
        "Prompt not started.",
      ),
      note: attemptCount > 0 ? `Attempt ${attemptSummaries[attemptCount - 1].attempt}.` : undefined,
    },
    {
      key: "model_call",
      label: "Call generator model",
      description: "Run the generator call.",
      phase: "model_call",
      event: modelCallEvent,
      status: deriveSyntheticStatus({
        event: modelCallEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["model_call"],
        completed: Boolean(modelCallEvent && !generating),
      }),
      message: compactProgressMessage(modelCallEvent, "No model call yet."),
    },
    {
      key: "parse",
      label: "Sanitize + parse",
      description: "Clean and parse DSL.",
      phase: "parse",
      event: parseEvent,
      status: deriveSyntheticStatus({
        event: parseEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["parse"],
        completed: Boolean(parseEvent && !generating),
      }),
      message: compactProgressMessage(parseEvent, "Waiting for model output."),
    },
    {
      key: "repair",
      label: "Repair loop",
      description: "Retry only if needed.",
      phase: "repair",
      event: repairEvent,
      status: repairNeeded
        ? deriveSyntheticStatus({
            event: repairEvent,
            latestEvent: latestProgressEvent,
            generating,
            activePhases: ["repair"],
            completed: Boolean(repairEvent && !generating),
          })
        : finalDraftReady || Boolean(completeEvent)
          ? "completed"
          : "pending",
      message: repairNeeded
        ? compactProgressMessage(
            repairEvent,
            `${attemptCount} attempts recorded.`,
          )
        : finalDraftReady || Boolean(completeEvent)
          ? "No repair needed."
          : "No repair yet.",
      note:
        repairNeeded && attemptCount > 1
          ? `${attemptCount} attempts total.`
          : undefined,
    },
    {
      key: "draft",
      label: "Final DSL draft",
      description: "Stage the final draft.",
      phase: "complete",
      event: completeEvent ?? reviewEvent ?? parseEvent,
      status: deriveSyntheticStatus({
        event: completeEvent ?? reviewEvent ?? parseEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["complete", "review", "format"],
        completed: finalDraftReady || Boolean(completeEvent),
        warning: repositoryChecksBlocked,
      }),
      message: finalDraftReady
        ? stagedDraftValidationReady
          ? "Draft staged."
          : "Draft staged with findings."
        : compactProgressMessage(
            completeEvent ?? reviewEvent ?? parseEvent,
            "No draft yet.",
          ),
      note: undefined,
    },
  ];

  const handoffNodes: ProcessNode[] = [
    {
      key: "context",
      label: "Builder context",
      description: "Resolve Builder context.",
      phase: "context",
      event: contextEvent,
      status: deriveSyntheticStatus({
        event: contextEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["context"],
        completed: Boolean(contextEvent && (attemptCount > 0 || !generating)),
      }),
      message: compactProgressMessage(contextEvent, "Context pending."),
    },
    {
      key: "validate",
      label: "Repository checks",
      description: "Validate and compile.",
      phase: "validate",
      event: validateEvent ?? reviewEvent,
      status: deriveSyntheticStatus({
        event: validateEvent ?? reviewEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["validate", "review"],
        completed: Boolean(stagedDraft) || Boolean(validateEvent && !generating),
        warning: repositoryChecksBlocked,
      }),
      message: stagedDraft
        ? stagedDraft.validation.ready
          ? "Checks passed."
          : `${stagedDraft.validation.errorCount} validation error${stagedDraft.validation.errorCount === 1 ? "" : "s"}.`
        : compactProgressMessage(validateEvent ?? reviewEvent, "Checks pending."),
      note:
        stagedDraft?.validation.compileError
          ? compactProgressText(stagedDraft.validation.compileError, "Compile error.")
          : undefined,
    },
    {
      key: "complete",
      label: "Stage for Builder",
      description: "Hold draft until apply.",
      phase: "complete",
      event: completeEvent,
      status: deriveSyntheticStatus({
        event: completeEvent,
        latestEvent: latestProgressEvent,
        generating,
        activePhases: ["complete"],
        completed: finalDraftReady || Boolean(completeEvent),
        warning: repositoryChecksBlocked,
      }),
      message: compactProgressMessage(completeEvent, 
        (finalDraftReady
          ? "Staged draft ready."
          : "No staged draft yet.")),
      note: finalDraftReady
        ? "Live Builder stays unchanged."
        : undefined,
    },
  ];

  const latestStatus = latestProgressEvent
    ? resolveEventStatus(latestProgressEvent, latestProgressEvent, generating)
    : "pending";

  return (
    <section className={styles.consoleCard}>
      <div className={styles.sectionHeader}>
        <div>
          <h3 className={styles.cardTitle}>Live process</h3>
          <p className={styles.sectionHint}>
            Follow the shared NL→DSL pipeline first, then the Builder-specific
            handoff that stages the result for apply.
          </p>
        </div>
        <div className={styles.consoleMeta}>
          <span
            className={`${styles.processStatusPill} ${processStatusClassName(
              latestStatus,
            )}`}
          >
            {generating ? "Streaming" : latestProgressEvent ? "Completed" : "Idle"}
          </span>
          {latestProgressEvent ? (
            <span className={styles.consoleTimestamp}>
              {formatTimestamp(latestProgressEvent.timestamp)}
            </span>
          ) : null}
        </div>
      </div>

      <div className={styles.processSummaryGrid}>
        <div className={styles.processSummaryCard}>
          <div className={styles.resultLabel}>Current activity</div>
          <div className={styles.processSummaryValue}>{currentActivityLabel}</div>
          <div className={styles.processSummaryText}>{currentActivityMessage}</div>
        </div>
        <div className={styles.processSummaryCard}>
          <div className={styles.resultLabel}>Shared loop</div>
          <div className={styles.processSummaryValue}>{currentAttemptLabel}</div>
          <div className={styles.processSummaryText}>
            {attemptCount > 1
              ? `${attemptCount} attempts have been recorded across the shared repair loop.`
              : attemptCount === 1
                ? "The first shared generation attempt has been recorded."
                : "No model attempt has started yet."}
          </div>
        </div>
        <div className={styles.processSummaryCard}>
          <div className={styles.resultLabel}>Elapsed</div>
          <div className={styles.processSummaryValue}>
            {formatDuration(elapsedSeconds)}
          </div>
          <div className={styles.processSummaryText}>
            {generating
              ? "The dashboard is streaming live progress from the backend."
              : finalDraftReady
                ? "The latest result is staged below for review."
                : "Waiting for the next request."}
          </div>
        </div>
      </div>

      {generating ? (
        <div className={styles.processLoading}>
          <div className={styles.processLoadingBar} />
        </div>
      ) : null}

      <div className={styles.progressSection}>
        <div className={styles.progressSectionLead}>
          <div className={styles.resultLabel}>Shared NL→DSL core</div>
          <div className={styles.progressSectionTitle}>
            Schema-grounded generation pipeline
          </div>
          <div className={styles.progressSectionText}>
            Shared prompt, one model call, parse, then repair only if needed.
          </div>
        </div>
        <div className={styles.progressTrack}>
          {coreNodes.map((node) => (
            <div
              className={`${styles.flowNode} ${processNodeClassName(node.status)}`}
              key={node.key}
            >
              <div className={styles.flowNodeIcon}>
                <PhaseIcon phase={node.phase} />
              </div>
              <div className={styles.flowNodeBody}>
                <div className={styles.flowNodeHeader}>
                  <span className={styles.flowNodeTitle}>{node.label}</span>
                  <span
                    className={`${styles.processStatusPill} ${processStatusClassName(
                      node.status,
                    )}`}
                  >
                    {processStatusLabel(node.status)}
                  </span>
                </div>
                <div className={styles.flowNodeDescription}>{node.description}</div>
                <div className={styles.flowNodeMessage}>{node.message}</div>
                {node.note ? (
                  <div className={styles.flowNodeNote}>{node.note}</div>
                ) : null}
                <div className={styles.flowNodeMeta}>
                  {node.event?.attempt ? (
                    <span className={styles.consoleAttempt}>
                      attempt {node.event.attempt}
                    </span>
                  ) : null}
                  {node.event?.elapsedSeconds ? (
                    <span className={styles.consoleAttempt}>
                      {formatDuration(node.event.elapsedSeconds)}
                    </span>
                  ) : null}
                  {node.event ? (
                    <span className={styles.consoleTimestamp}>
                      {formatTimestamp(node.event.timestamp)}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {attemptSummaries.length > 0 ? (
        <div className={styles.progressSection}>
          <div className={styles.progressSectionLead}>
            <div className={styles.resultLabel}>Attempts</div>
          <div className={styles.progressSectionTitle}>Repair loop history</div>
          <div className={styles.progressSectionText}>
            One card per shared nlgen attempt.
          </div>
        </div>
          <div className={styles.attemptStack}>
            {attemptSummaries.map((summary) => {
              const attemptStatus = resolveEventStatus(
                summary.latestEvent,
                latestProgressEvent,
                generating,
              );

              return (
                <div
                  className={`${styles.attemptCard} ${processNodeClassName(attemptStatus)}`}
                  key={summary.attempt}
                >
                  <div className={styles.attemptCardHeader}>
                    <div>
                      <div className={styles.resultLabel}>Attempt {summary.attempt}</div>
                      <div className={styles.attemptCardTitle}>
                        {formatPhaseLabel(summary.latestStableEvent.phase)}
                      </div>
                    </div>
                    <div className={styles.processJourneyMeta}>
                      <span
                        className={`${styles.processStatusPill} ${processStatusClassName(
                          attemptStatus,
                        )}`}
                      >
                        {processStatusLabel(attemptStatus)}
                      </span>
                      <span className={styles.consoleTimestamp}>
                        {formatTimestamp(summary.latestEvent.timestamp)}
                      </span>
                    </div>
                  </div>

                  <div className={styles.attemptCardText}>
                    {summarizeAttempt(summary)}
                  </div>

                  <div className={styles.attemptPhaseList}>
                    {ATTEMPT_PHASES.map((phase) => {
                      const event = summary.phaseEvents.get(phase.key);
                      const status = resolveEventStatus(
                        event,
                        latestProgressEvent,
                        generating,
                      );
                      return (
                        <div
                          className={`${styles.attemptPhaseRow} ${processNodeClassName(status)}`}
                          key={`${summary.attempt}-${phase.key}`}
                        >
                          <div className={styles.attemptPhaseIcon}>
                            <PhaseIcon phase={phase.key} />
                          </div>
                          <div className={styles.attemptPhaseBody}>
                            <div className={styles.attemptPhaseHeader}>
                              <span className={styles.attemptPhaseTitle}>
                                {phase.label}
                              </span>
                              <span
                                className={`${styles.processStatusPill} ${processStatusClassName(
                                  status,
                                )}`}
                              >
                                {processStatusLabel(status)}
                              </span>
                            </div>
                            <div className={styles.attemptPhaseText}>
                              {compactProgressMessage(event, phase.description)}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className={styles.progressSection}>
        <div className={styles.progressSectionLead}>
          <div className={styles.resultLabel}>Builder handoff</div>
          <div className={styles.progressSectionTitle}>Stage before apply</div>
          <div className={styles.progressSectionText}>
            After shared nlgen returns a draft, Builder validates it and stages it for apply.
          </div>
        </div>
        <div className={styles.handoffGrid}>
          {handoffNodes.map((node) => (
            <div
              className={`${styles.handoffNode} ${processNodeClassName(node.status)}`}
              key={node.key}
            >
              <div className={styles.handoffNodeIcon}>
                <PhaseIcon phase={node.phase} />
              </div>
              <div className={styles.handoffNodeBody}>
                <div className={styles.handoffNodeHeader}>
                  <span className={styles.flowNodeTitle}>{node.label}</span>
                  <span
                    className={`${styles.processStatusPill} ${processStatusClassName(
                      node.status,
                    )}`}
                  >
                    {processStatusLabel(node.status)}
                  </span>
                </div>
                <div className={styles.flowNodeDescription}>{node.description}</div>
                <div className={styles.flowNodeMessage}>{node.message}</div>
                {node.note ? (
                  <div className={styles.flowNodeNote}>{node.note}</div>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.consoleToggleRow}>
        <div>
          <div className={styles.resultLabel}>Raw events</div>
          <div className={styles.consoleLogHint}>
            Use the low-level stream only when you need to debug stalls or verify
            timing details.
          </div>
        </div>
        <button
          className={styles.ghostBtn}
          onClick={() => setShowRawEvents((current) => !current)}
          type="button"
        >
          {showRawEvents ? "Hide raw event stream" : "Show raw event stream"}
        </button>
      </div>

      {showRawEvents ? (
        <div className={styles.consoleViewport} ref={consoleRef}>
          {progressEvents.length > 0 ? (
            <div className={styles.consoleList}>
              {progressEvents.map((event, index) => (
                <div className={styles.consoleRow} key={`${event.timestamp}-${index}`}>
                  <div className={styles.consoleTime}>
                    {formatTimestamp(event.timestamp)}
                  </div>
                  <div className={styles.consoleMessageBlock}>
                    <div className={styles.consoleLine}>
                      <span className={styles.consolePhase}>
                        {formatPhaseLabel(event.phase)}
                      </span>
                      <span
                        className={`${styles.processEventLevel} ${eventLevelClassName(
                          event.level,
                        )}`}
                      >
                        {event.kind === "heartbeat" ? "heartbeat" : event.level}
                      </span>
                      {event.attempt ? (
                        <span className={styles.consoleAttempt}>
                          attempt {event.attempt}
                        </span>
                      ) : null}
                      {event.elapsedSeconds ? (
                        <span className={styles.consoleAttempt}>
                          {formatDuration(event.elapsedSeconds)}
                        </span>
                      ) : null}
                    </div>
                    <div className={styles.consoleMessage}>{event.message}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.consoleEmpty}>
              Send a Builder request to stream generation progress here.
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
};

export { BuilderNaturalLanguageProgress };

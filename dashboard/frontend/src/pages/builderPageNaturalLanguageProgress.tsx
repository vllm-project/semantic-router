import React, { useEffect, useMemo, useRef } from "react";

import type { BuilderNLProgressEvent } from "@/types/dsl";

import styles from "./builderPageNaturalLanguagePanel.module.css";

type BuilderNLProcessStatus =
  | "pending"
  | "running"
  | "completed"
  | "warning"
  | "error";

type BuilderNLPhaseKey =
  | "request"
  | "context"
  | "generate"
  | "model_call"
  | "parse"
  | "validate"
  | "format"
  | "repair"
  | "review"
  | "complete";

interface BuilderNLPhaseMeta {
  key: BuilderNLPhaseKey;
  label: string;
  description: string;
}

const PROCESS_PHASES: BuilderNLPhaseMeta[] = [
  {
    key: "request",
    label: "Request intake",
    description: "Validate the incoming request and open a staged draft loop.",
  },
  {
    key: "context",
    label: "Router context",
    description: "Load the live router config and resolve the draft target model.",
  },
  {
    key: "generate",
    label: "Draft plan",
    description: "Prepare a generation or repair attempt.",
  },
  {
    key: "model_call",
    label: "Model call",
    description: "Send the prompt to the selected draft-generation model.",
  },
  {
    key: "parse",
    label: "Parse output",
    description: "Convert the model response into a candidate Builder DSL draft.",
  },
  {
    key: "validate",
    label: "Repository validate",
    description: "Run the repository DSL parse, validate, and compile checks.",
  },
  {
    key: "format",
    label: "Format draft",
    description: "Normalize the draft after a successful validation pass.",
  },
  {
    key: "repair",
    label: "Repair loop",
    description: "Feed repository findings back into the next model attempt.",
  },
  {
    key: "review",
    label: "Readiness review",
    description: "Summarize whether the staged draft is ready to apply.",
  },
  {
    key: "complete",
    label: "Stage result",
    description: "Persist the final staged draft outcome for Builder review.",
  },
];

function formatPhaseLabel(phase: string): string {
  return phase
    .split("_")
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString();
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

function resolvePhaseStatus(
  event: BuilderNLProgressEvent | undefined,
  latestEvent: BuilderNLProgressEvent | null,
  generating: boolean,
): BuilderNLProcessStatus {
  if (!event) {
    return "pending";
  }
  if (event.level === "error") {
    return "error";
  }
  if (latestEvent?.phase === event.phase && generating) {
    return "running";
  }
  if (event.level === "warning") {
    return "warning";
  }
  if (event.level === "success") {
    return "completed";
  }
  return generating && latestEvent != null ? "completed" : "pending";
}

function PhaseIcon({
  phase,
  className,
}: {
  phase: string;
  className?: string;
}): React.ReactElement {
  const sharedProps = {
    className,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.8",
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  switch (phase) {
    case "request":
      return (
        <svg {...sharedProps}>
          <path d="M6 4.75h7.5L18 9.25V19a1.25 1.25 0 0 1-1.25 1.25h-10.5A1.25 1.25 0 0 1 5 19V6A1.25 1.25 0 0 1 6.25 4.75Z" />
          <path d="M13 4.75V9.5h4.75" />
          <path d="M8.5 13h7" />
          <path d="M8.5 16.25H13" />
        </svg>
      );
    case "context":
      return (
        <svg {...sharedProps}>
          <ellipse cx="12" cy="6.25" rx="6.5" ry="2.75" />
          <path d="M5.5 6.25V11c0 1.52 2.91 2.75 6.5 2.75s6.5-1.23 6.5-2.75V6.25" />
          <path d="M5.5 11v4.75c0 1.52 2.91 2.75 6.5 2.75s6.5-1.23 6.5-2.75V11" />
        </svg>
      );
    case "generate":
      return (
        <svg {...sharedProps}>
          <path d="M12 3.75 14.3 8.4l5.15.75-3.72 3.62.88 5.12L12 15.47 7.39 17.9l.88-5.12-3.72-3.62 5.15-.75L12 3.75Z" />
        </svg>
      );
    case "model_call":
      return (
        <svg {...sharedProps}>
          <rect x="5.5" y="6" width="13" height="10.5" rx="2.25" />
          <path d="M9 18.5h6" />
          <path d="M10 3.75h4" />
          <path d="M9 10h6" />
          <path d="M12 7.75v4.5" />
        </svg>
      );
    case "parse":
      return (
        <svg {...sharedProps}>
          <path d="M9.25 6.25 5.75 12l3.5 5.75" />
          <path d="m14.75 6.25 3.5 5.75-3.5 5.75" />
          <path d="m13.5 5-3 14" />
        </svg>
      );
    case "validate":
      return (
        <svg {...sharedProps}>
          <path d="M12 3.75 18 6v5.62c0 3.63-2.48 6.96-6 7.88-3.52-.92-6-4.25-6-7.88V6l6-2.25Z" />
          <path d="m9.25 11.75 1.9 1.9 3.6-3.85" />
        </svg>
      );
    case "format":
      return (
        <svg {...sharedProps}>
          <path d="M6 7.5h12" />
          <path d="M6 11.25h8.5" />
          <path d="M6 15h12" />
          <path d="M6 18.75h8.5" />
        </svg>
      );
    case "repair":
      return (
        <svg {...sharedProps}>
          <path d="m14.75 5.25 4 4" />
          <path d="m7 18.5-2.25.75.75-2.25L13.5 9l1.75 1.75Z" />
          <path d="m12.75 9.75 4-4a1.41 1.41 0 0 1 2 2l-4 4" />
        </svg>
      );
    case "review":
      return (
        <svg {...sharedProps}>
          <rect x="6" y="4.5" width="12" height="15" rx="2" />
          <path d="M9 8.25h6" />
          <path d="M9 12h6" />
          <path d="m9.25 15.5 1.5 1.5 3-3.25" />
        </svg>
      );
    case "complete":
      return (
        <svg {...sharedProps}>
          <circle cx="12" cy="12" r="8.25" />
          <path d="m8.75 12.25 2.1 2.1 4.4-4.6" />
        </svg>
      );
    default:
      return (
        <svg {...sharedProps}>
          <circle cx="12" cy="12" r="8.25" />
          <path d="M12 8.5v4.25" />
          <path d="M12 15.5h.01" />
        </svg>
      );
  }
}

interface BuilderNaturalLanguageProgressProps {
  generating: boolean;
  progressEvents: BuilderNLProgressEvent[];
}

const BuilderNaturalLanguageProgress: React.FC<
  BuilderNaturalLanguageProgressProps
> = ({ generating, progressEvents }) => {
  const consoleRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!consoleRef.current) {
      return;
    }
    consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [progressEvents]);

  const latestProgressEvent =
    progressEvents.length > 0 ? progressEvents[progressEvents.length - 1] : null;

  const phaseEvents = useMemo(() => {
    const byPhase = new Map<string, BuilderNLProgressEvent>();
    for (const event of progressEvents) {
      byPhase.set(event.phase, event);
    }
    return byPhase;
  }, [progressEvents]);

  const spotlightLabel = latestProgressEvent
    ? formatPhaseLabel(latestProgressEvent.phase)
    : "Waiting for request";
  const spotlightDescription =
    latestProgressEvent?.message ??
    "Send a Builder request to stream generation, validation, repair, and review progress.";

  return (
    <section className={styles.consoleCard}>
      <div className={styles.sectionHeader}>
        <div>
          <h3 className={styles.cardTitle}>Live process</h3>
          <p className={styles.sectionHint}>
            Track the staged Builder draft lifecycle with highlighted phases and
            a raw event log.
          </p>
        </div>
        <div className={styles.consoleMeta}>
          <span
            className={`${styles.processStatusPill} ${processStatusClassName(
              latestProgressEvent
                ? resolvePhaseStatus(
                    latestProgressEvent,
                    latestProgressEvent,
                    generating,
                  )
                : "pending",
            )}`}
          >
            {generating ? "Streaming" : latestProgressEvent ? "Latest event" : "Idle"}
          </span>
          {latestProgressEvent ? (
            <span className={styles.consoleTimestamp}>
              {formatTimestamp(latestProgressEvent.timestamp)}
            </span>
          ) : null}
        </div>
      </div>

      <div className={styles.processSpotlight}>
        <div className={styles.processSpotlightIcon}>
          <PhaseIcon phase={latestProgressEvent?.phase ?? "request"} />
        </div>
        <div className={styles.processSpotlightBody}>
          <div className={styles.processSpotlightEyebrow}>Current stage</div>
          <div className={styles.processSpotlightTitle}>{spotlightLabel}</div>
          <div className={styles.processSpotlightText}>{spotlightDescription}</div>
        </div>
        <div className={styles.processSpotlightMeta}>
          {latestProgressEvent?.attempt ? (
            <span className={styles.consoleAttempt}>
              attempt {latestProgressEvent.attempt}
            </span>
          ) : null}
        </div>
      </div>

      {generating ? (
        <div className={styles.processLoading}>
          <div className={styles.processLoadingBar} />
        </div>
      ) : null}

      <div className={styles.processTimeline}>
        {PROCESS_PHASES.map((phase) => {
          const event = phaseEvents.get(phase.key);
          const status = resolvePhaseStatus(event, latestProgressEvent, generating);

          return (
            <div
              className={`${styles.processNode} ${processNodeClassName(status)}`}
              key={phase.key}
            >
              <div className={styles.processNodeIcon}>
                <PhaseIcon phase={phase.key} />
              </div>
              <div className={styles.processNodeBody}>
                <div className={styles.processNodeHeader}>
                  <span className={styles.processNodeTitle}>{phase.label}</span>
                  <span
                    className={`${styles.processStatusPill} ${processStatusClassName(status)}`}
                  >
                    {processStatusLabel(status)}
                  </span>
                </div>
                <div className={styles.processNodeText}>
                  {event?.message ?? phase.description}
                </div>
                <div className={styles.processNodeMeta}>
                  {event?.attempt ? (
                    <span className={styles.consoleAttempt}>
                      attempt {event.attempt}
                    </span>
                  ) : null}
                  {event ? (
                    <span className={styles.consoleTimestamp}>
                      {formatTimestamp(event.timestamp)}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className={styles.consoleDivider} />

      <div className={styles.consoleLogHeader}>
        <span className={styles.resultLabel}>Recent events</span>
        <span className={styles.consoleLogHint}>
          Raw event stream for debugging stalls or retries.
        </span>
      </div>

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
                      {event.level}
                    </span>
                    {event.attempt ? (
                      <span className={styles.consoleAttempt}>
                        attempt {event.attempt}
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
            Send a Builder request to stream generation, validation, repair, and
            review progress here.
          </div>
        )}
      </div>
    </section>
  );
};

export { BuilderNaturalLanguageProgress };

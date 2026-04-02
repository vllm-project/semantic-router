import React from "react";

import type { BuilderNLProgressEvent } from "@/types/dsl";

export type BuilderNLProcessStatus =
  | "pending"
  | "running"
  | "completed"
  | "warning"
  | "error";

export type BuilderNLPhaseKey =
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

export interface BuilderNLPhaseMeta {
  key: BuilderNLPhaseKey;
  label: string;
  description: string;
}

export interface BuilderNLAttemptSummary {
  attempt: number;
  events: BuilderNLProgressEvent[];
  latestEvent: BuilderNLProgressEvent;
  latestStableEvent: BuilderNLProgressEvent;
  latestHeartbeatEvent: BuilderNLProgressEvent | null;
  phaseEvents: Map<string, BuilderNLProgressEvent>;
}

export const ATTEMPT_PHASES: BuilderNLPhaseMeta[] = [
  {
    key: "generate",
    label: "Ground prompt",
    description: "Build the shared prompt.",
  },
  {
    key: "model_call",
    label: "Model call",
    description: "Run the generator call.",
  },
  {
    key: "parse",
    label: "Sanitize + parse",
    description: "Clean and parse DSL.",
  },
  {
    key: "repair",
    label: "Repair",
    description: "Retry only if needed.",
  },
];

export function formatPhaseLabel(phase: string): string {
  return phase
    .split("_")
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}

export function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString();
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || Number.isNaN(seconds)) {
    return "Not started";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder}s`;
}

export function isHeartbeatEvent(event: BuilderNLProgressEvent | undefined): boolean {
  return event?.kind === "heartbeat";
}

export function resolveEventStatus(
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
  if (isHeartbeatEvent(event)) {
    return "running";
  }
  if (
    generating &&
    latestEvent &&
    latestEvent.phase === event.phase &&
    latestEvent.attempt === event.attempt &&
    latestEvent.timestamp === event.timestamp
  ) {
    return "running";
  }
  if (event.level === "warning") {
    return "warning";
  }
  if (event.level === "success" || event.level === "info") {
    return "completed";
  }
  return "pending";
}

export function buildAttemptSummary(
  attempt: number,
  events: BuilderNLProgressEvent[],
): BuilderNLAttemptSummary {
  const phaseEvents = new Map<string, BuilderNLProgressEvent>();
  let latestHeartbeatEvent: BuilderNLProgressEvent | null = null;

  for (const event of events) {
    phaseEvents.set(event.phase, event);
    if (isHeartbeatEvent(event)) {
      latestHeartbeatEvent = event;
    }
  }

  const latestEvent = events[events.length - 1];
  const latestStableEvent =
    [...events].reverse().find((event) => !isHeartbeatEvent(event)) ?? latestEvent;

  return {
    attempt,
    events,
    latestEvent,
    latestStableEvent,
    latestHeartbeatEvent,
    phaseEvents,
  };
}

export function latestEventForPhase(
  progressEvents: BuilderNLProgressEvent[],
  phase: string,
): BuilderNLProgressEvent | undefined {
  return [...progressEvents].reverse().find((event) => event.phase === phase);
}

export function latestEventForPhases(
  progressEvents: BuilderNLProgressEvent[],
  phases: string[],
): BuilderNLProgressEvent | undefined {
  return [...progressEvents].reverse().find((event) => phases.includes(event.phase));
}

export function attemptPhaseEvent(
  attemptSummaries: BuilderNLAttemptSummary[],
  phase: BuilderNLPhaseKey,
): BuilderNLProgressEvent | undefined {
  for (let index = attemptSummaries.length - 1; index >= 0; index -= 1) {
    const event = attemptSummaries[index].phaseEvents.get(phase);
    if (event) {
      return event;
    }
  }
  return undefined;
}

export function deriveSyntheticStatus({
  event,
  latestEvent,
  generating,
  activePhases,
  completed,
  warning,
  error,
}: {
  event?: BuilderNLProgressEvent;
  latestEvent: BuilderNLProgressEvent | null;
  generating: boolean;
  activePhases: string[];
  completed: boolean;
  warning?: boolean;
  error?: boolean;
}): BuilderNLProcessStatus {
  if (error) {
    return "error";
  }
  if (event?.level === "error") {
    return "error";
  }
  if (warning) {
    return "warning";
  }
  if (
    generating &&
    latestEvent &&
    (activePhases.includes(latestEvent.phase) || isHeartbeatEvent(latestEvent))
  ) {
    return "running";
  }
  if (completed) {
    return "completed";
  }
  return event ? resolveEventStatus(event, latestEvent, generating) : "pending";
}

export function summarizeAttempt(summary: BuilderNLAttemptSummary): string {
  if (isHeartbeatEvent(summary.latestEvent) && summary.latestHeartbeatEvent) {
    return compactProgressText(summary.latestHeartbeatEvent.message, "Waiting on model.");
  }
  return compactProgressMessage(summary.latestStableEvent);
}

export function compactProgressMessage(
  event: BuilderNLProgressEvent | undefined,
  fallback?: string,
): string {
  if (!event) {
    return fallback ?? "Pending.";
  }

  if (event.kind === "heartbeat") {
    if (event.elapsedSeconds) {
      return `Waiting on model (${formatDuration(event.elapsedSeconds)}).`;
    }
    return "Waiting on model.";
  }

  switch (event.phase) {
    case "request":
      return event.level === "error" ? compactProgressText(event.message, "Request failed.") : "Request accepted.";
    case "context":
      return "Context ready.";
    case "generate":
      return event.attempt ? `Prompt ready for attempt ${event.attempt}.` : "Prompt ready.";
    case "model_call":
      if (event.level === "error") {
        return compactProgressText(event.message, "Model call failed.");
      }
      if (event.elapsedSeconds) {
        return `Response in ${formatDuration(event.elapsedSeconds)}.`;
      }
      return "Model call finished.";
    case "parse":
      return event.level === "error" ? compactProgressText(event.message, "Parse failed.") : "DSL parsed.";
    case "repair":
      return event.level === "error"
        ? compactProgressText(event.message, "Repair failed.")
        : event.attempt
          ? `Repair attempt ${event.attempt}.`
          : "Repair finished.";
    case "validate":
      return event.level === "error"
        ? compactProgressText(event.message, "Checks failed.")
        : "Checks passed.";
    case "review":
      return event.level === "warning"
        ? compactProgressText(event.message, "Review needs attention.")
        : "Review ready.";
    case "format":
      return "Draft formatted.";
    case "complete":
      return event.level === "error" ? compactProgressText(event.message, "Draft staging failed.") : "Draft staged.";
    default:
      return compactProgressText(event.message, fallback ?? formatPhaseLabel(event.phase));
  }
}

export function compactProgressText(text: string | undefined, fallback: string): string {
  if (!text) {
    return fallback;
  }

  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return fallback;
  }

  const withoutDetail = normalized
    .replace(/^Accepted Builder NL request and prepared staged draft generation\.?\s*/i, "")
    .replace(/^Resolved \d+ current router model card\(s\);?\s*/i, "")
    .replace(/Builder will still use ".*?" to generate the draft while route references prefer ".*?"\.?/i, "")
    .replace(/^Generated a staged Builder DSL draft for .*?\.\s*/i, "")
    .trim();

  const firstSentenceMatch = withoutDetail.match(/^(.+?[.!?])(?:\s|$)/);
  const firstSentence = firstSentenceMatch?.[1] ?? withoutDetail;
  const collapsed = firstSentence.replace(/\s+/g, " ").trim();
  if (!collapsed) {
    return fallback;
  }
  if (collapsed.length <= 88) {
    return collapsed;
  }
  return `${collapsed.slice(0, 85).trimEnd()}...`;
}

export function PhaseIcon({
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

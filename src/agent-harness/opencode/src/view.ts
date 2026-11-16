import type { Snapshot, Feedback, Decision } from "./decisions.ts";
import { isRecord } from "./settings.ts";
import type { Settings } from "./settings.ts";
import type { RuntimeStatus } from "./runtime.ts";
import type { ControlResult } from "./controls.ts";
export interface ViewSnapshot extends Snapshot {
  apps: Settings["apps"];
  waits: Settings["waits"];
  controls: {
    revision: number;
    persistence: "durable" | "unavailable";
    goal: Settings["routing"]["goal"];
  };
  runtime: RuntimeStatus;
}
export type ControlPatch = {
  apps?: Partial<Settings["apps"]>;
  waits?: Partial<Settings["waits"]>;
  goal?: Settings["routing"]["goal"];
  follow?: true;
};
export interface ViewService {
  snapshot(sessionID: string, signal: AbortSignal): Promise<ViewSnapshot>;
  controls(
    input: ControlPatch & { instanceID: string; expectedRevision: number },
  ): Promise<ControlResult>;
  feedback(
    sessionID: string,
    decisionID: string,
    feedback: Feedback,
  ): Promise<unknown>;
  diagnose(): Promise<{ ok: boolean }>;
}
export interface ViewState {
  sessionID: string | undefined;
  snapshot: ViewSnapshot | undefined;
  loading: boolean;
  busy: boolean;
  error: string | undefined;
}
export function emptyView(): ViewState {
  return {
    sessionID: undefined,
    snapshot: undefined,
    loading: false,
    busy: false,
    error: undefined,
  };
}
export class DecisionView {
  private readonly service: ViewService;
  private readonly publish: (state: ViewState) => void;
  private state = emptyView();
  private request = 0;
  private operation = 0;
  private abort: AbortController | undefined;
  private closed = false;
  constructor(service: ViewService, publish: (state: ViewState) => void) {
    this.service = service;
    this.publish = publish;
  }
  private emit(): void {
    if (!this.closed) this.publish({ ...this.state });
  }
  select(sessionID: string | undefined): void {
    if (this.closed || this.state.sessionID === sessionID) return;
    this.abort?.abort();
    this.request++;
    this.operation++;
    this.state = { ...emptyView(), sessionID };
    this.emit();
    if (sessionID) void this.refresh();
  }
  async refresh(): Promise<void> {
    const id = this.state.sessionID;
    if (!id || this.closed) return;
    const request = ++this.request;
    this.abort?.abort();
    this.abort = new AbortController();
    const signal = this.abort.signal,
      timer = setTimeout(
        () => this.abort?.signal === signal && this.abort.abort(),
        3000,
      );
    this.state.loading = true;
    this.emit();
    try {
      const snapshot = await this.service.snapshot(id, signal);
      if (
        this.closed ||
        request !== this.request ||
        id !== this.state.sessionID
      )
        return;
      if (
        snapshot.sessionID !== id ||
        !snapshot.instanceID ||
        !Number.isSafeInteger(snapshot.revision) ||
        !snapshot.controls
      )
        throw new Error("Wrong session snapshot");
      const previous = this.state.snapshot;
      if (
        !previous ||
        snapshot.instanceID !== previous.instanceID ||
        snapshot.revision >= previous.revision
      )
        this.state.snapshot = snapshot;
      this.state.error = undefined;
    } catch {
      if (!this.closed && request === this.request)
        this.state.error = "Snapshot unavailable; displayed data may be stale.";
    } finally {
      clearTimeout(timer);
      if (!this.closed && request === this.request) {
        this.state.loading = false;
        this.emit();
      }
    }
  }
  hint(event: unknown): void {
    if (
      !isRecord(event) ||
      typeof event.instanceID !== "string" ||
      typeof event.revision !== "number" ||
      !Number.isSafeInteger(event.revision) ||
      event.revision < 0 ||
      (event.sessionID !== undefined && typeof event.sessionID !== "string")
    )
      return;
    if (event.sessionID && event.sessionID !== this.state.sessionID) return;
    const snapshot = this.state.snapshot;
    if (
      snapshot &&
      event.instanceID === snapshot.instanceID &&
      event.revision <= snapshot.revision
    )
      return;
    void this.refresh();
  }
  async change(patch: ControlPatch): Promise<boolean> {
    const snapshot = this.state.snapshot;
    if (this.closed || this.state.busy || !snapshot) return false;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let saved = false,
      failure: string | undefined;
    try {
      const result = await this.service.controls({
        ...patch,
        instanceID: snapshot.instanceID,
        expectedRevision: snapshot.controls.revision,
      });
      saved = result.saved;
      if (!result.saved)
        failure =
          result.reason === "stale"
            ? "Settings changed elsewhere; refresh and retry."
            : "Settings were not saved.";
    } catch {
      failure = "Could not confirm the settings change; refresh to reconcile.";
    }
    if (this.closed || operation !== this.operation) return saved;
    await this.refresh();
    if (this.closed || operation !== this.operation) return saved;
    if (!saved) this.state.error = failure;
    this.state.busy = false;
    this.emit();
    return saved;
  }
  async feedback(decision: Decision, feedback: Feedback): Promise<boolean> {
    const id = this.state.sessionID;
    if (!id || decision.sessionID !== id || this.state.busy || this.closed)
      return false;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let saved = false,
      failure: string | undefined;
    try {
      await this.service.feedback(id, decision.id, feedback);
      saved = true;
    } catch {
      failure = "Could not confirm feedback; refresh to reconcile.";
    }
    if (this.closed || operation !== this.operation) return saved;
    await this.refresh();
    if (this.closed || operation !== this.operation) return saved;
    if (!saved) this.state.error = failure;
    this.state.busy = false;
    this.emit();
    return saved;
  }
  async diagnose(): Promise<boolean> {
    if (this.state.busy || this.closed) return false;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let ok = false;
    try {
      ok = (await this.service.diagnose()).ok;
    } catch {
      /* Transport status is shown by refresh. */
    }
    if (this.closed || operation !== this.operation) return ok;
    await this.refresh();
    if (this.closed || operation !== this.operation) return ok;
    this.state.busy = false;
    this.emit();
    return ok;
  }
  close(): void {
    this.closed = true;
    this.request++;
    this.operation++;
    this.abort?.abort();
  }
}
export function summary(decisions: readonly Decision[]) {
  const seen = new Set<string>(),
    children = new Set<string>();
  const outcomes: Record<string, number> = {};
  let masked = 0,
    input = 0,
    output = 0,
    cached = 0,
    reasoning = 0,
    cost = 0,
    priced = 0,
    unpriced = 0;
  for (const decision of decisions) {
    if (seen.has(decision.id)) continue;
    seen.add(decision.id);
    outcomes[decision.outcome] = (outcomes[decision.outcome] ?? 0) + 1;
    masked += Object.values(decision.counts ?? {}).reduce<number>(
      (a, b) => a + (b ?? 0),
      0,
    );
    if (
      !decision.usage ||
      !decision.observedModel ||
      !decision.childSessionID ||
      children.has(decision.childSessionID)
    )
      continue;
    children.add(decision.childSessionID);
    input += decision.usage.input;
    output += decision.usage.output;
    cached += decision.usage.cachedInput;
    reasoning += decision.usage.reasoning;
    if (decision.costKnown) {
      cost += decision.usage.cost;
      priced++;
    } else unpriced++;
  }
  return {
    total: seen.size,
    outcomes,
    masked,
    children: children.size,
    input,
    output,
    cached,
    reasoning,
    cost,
    priced,
    unpriced,
  };
}

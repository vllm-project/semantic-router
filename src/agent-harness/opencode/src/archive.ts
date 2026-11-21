import type { DecisionStore } from "./decisions.ts";
import type { LogStore } from "./log.ts";
import { isRecord } from "./settings.ts";

const PREFIX = "sessions/";
// Changes in a burst, a check's several windows or a subagent's usage, become one write.
const WRITE_AFTER_MS = 250;

// Keeps each session's decisions in plugin storage, one key per session, so a restart or /reload shows what was
// decided before it, as the Claude Code plugin keeps them in its session state. Only what the session view holds is
// kept: subjects, scores and outcomes, never output text.
export class SessionArchive {
  private readonly pending = new Map<string, boolean>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private tail: Promise<void> = Promise.resolve();
  private readonly store: LogStore;
  private readonly decisions: DecisionStore;
  constructor(store: LogStore, decisions: DecisionStore) {
    this.store = store;
    this.decisions = decisions;
  }
  // Restores the newest sessions first written last, so the store's own limit keeps the most recent ones, then
  // follows the store's changes.
  async load(): Promise<void> {
    try {
      const found: { sessionID: string; at: number; value: unknown }[] = [];
      let after: string | undefined;
      do {
        const page = await this.store.scan({
          prefix: PREFIX,
          limit: 1000,
          ...(after ? { after } : {}),
        });
        for (const entry of page.entries)
          if (isRecord(entry.value) && typeof entry.value.at === "number")
            found.push({
              sessionID: entry.key.slice(PREFIX.length),
              at: entry.value.at,
              value: entry.value,
            });
        after = page.next;
      } while (after);
      found.sort((a, b) => a.at - b.at);
      for (const one of found) this.decisions.restore(one.sessionID, one.value);
      // Sessions the store's limit left out are dropped from storage too.
      for (const one of found)
        if (!this.decisions.saved(one.sessionID))
          this.pending.set(one.sessionID, true);
    } catch {
      /* Storage unavailable: the view starts empty, as before. */
    }
    this.decisions.onChange = (sessionID, removed) => {
      this.pending.set(sessionID, removed);
      this.timer ??= setTimeout(() => void this.flush(), WRITE_AFTER_MS);
    };
    if (this.pending.size)
      this.timer ??= setTimeout(() => void this.flush(), WRITE_AFTER_MS);
  }
  // Writes what changed since the last write, in order; a failed write is retried by the next change.
  flush(): Promise<void> {
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    const batch = [...this.pending];
    this.pending.clear();
    this.tail = this.tail.then(async () => {
      for (const [sessionID, removed] of batch) {
        const saved = removed ? undefined : this.decisions.saved(sessionID);
        try {
          if (saved)
            await this.store.set(PREFIX + sessionID, {
              at: Date.now(),
              ...saved,
            });
          else await this.store.remove(PREFIX + sessionID);
        } catch {
          /* Kept in memory; the next change writes it again. */
        }
      }
    });
    return this.tail;
  }
  async close(): Promise<void> {
    this.decisions.onChange = undefined;
    await this.flush();
  }
}

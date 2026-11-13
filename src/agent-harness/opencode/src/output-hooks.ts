import { createHash } from "node:crypto";
import type { Plugin } from "@opencode/plugin";
import type { DecisionStore } from "./decisions.ts";
import type { Settings } from "./settings.ts";
import { isRecord } from "./settings.ts";
import type { RuntimeClient } from "./runtime.ts";
import {
  inspectOutput,
  applyAttachmentPolicy,
  ATTACHMENT_NOTE,
  ATTACHMENT_WITHHELD,
} from "./output-policy.ts";
import type { Inspection, OutputTool } from "./output-policy.ts";
import type { Tag } from "./masking.ts";

const covered = (tool: string): tool is OutputTool =>
  ["read", "shell", "webfetch"].includes(tool);
interface State {
  tags: Set<Tag>;
  controller: AbortController;
}
interface Rendered {
  text: string;
  files: { type: "file"; uri: string; mime: string; name?: string }[];
}
function render(content: unknown, output?: unknown): Rendered {
  if (typeof content === "string") return { text: content, files: [] };
  if (Array.isArray(content))
    return {
      text: content
        .filter(
          (part) =>
            isRecord(part) &&
            part.type === "text" &&
            typeof part.text === "string",
        )
        .map((part) => part.text)
        .join("\n"),
      files: content.filter((part) => isRecord(part) && part.type === "file"),
    };
  return {
    text: typeof output === "string" ? output : (JSON.stringify(output) ?? ""),
    files: [],
  };
}
const marker = /\[vsr:(name|email|phone|card|ip|ssn|iban|address|id|secret)\]/g;
// Inspect only additions. Existing documentation tags in an edit/patch may remain unchanged.
export function addedPlaceholders(
  tool: string,
  input: unknown,
  tags: ReadonlySet<Tag>,
): boolean {
  if (!isRecord(input)) return false;
  const pairs: { added: string; removed: string }[] = [];
  if (tool === "write")
    pairs.push({
      added:
        typeof input.content === "string"
          ? input.content
          : typeof input.text === "string"
            ? input.text
            : "",
      removed: "",
    });
  else if (tool === "edit")
    pairs.push({
      added: String(input.new_string ?? input.newString ?? input.new_str ?? ""),
      removed: String(
        input.old_string ?? input.oldString ?? input.old_str ?? "",
      ),
    });
  else if (tool === "apply_patch" || tool === "patch") {
    let added = "",
      removed = "";
    const flush = () => {
      pairs.push({ added, removed });
      added = "";
      removed = "";
    };
    for (const line of String(
      input.patchText ?? input.patch ?? input.text ?? "",
    ).split("\n")) {
      if (/^(?:@@|diff --git |\*\*\* |--- |\+\+\+ )/.test(line)) flush();
      else if (line.startsWith("+")) added += line + "\n";
      else if (line.startsWith("-")) removed += line + "\n";
    }
    flush();
  } else return false;
  for (const { added, removed } of pairs)
    for (const tag of tags) {
      const token = `[vsr:${tag}]`;
      if (added.split(token).length > removed.split(token).length) return true;
    }
  return false;
}
export async function registerOutput(
  ctx: Plugin.Context,
  settings: Settings,
  runtime: RuntimeClient,
  decisions: DecisionStore,
  notify: (sessionID: string) => Promise<void>,
): Promise<() => void> {
  const sessions = new Map<string, State>(),
    cache = new Map<string, Inspection>();
  const stop = new AbortController();
  let bytes = 0;
  function state(sessionID: string): State {
    const value = sessions.get(sessionID) ?? {
      tags: new Set<Tag>(),
      controller: new AbortController(),
    };
    sessions.delete(sessionID);
    sessions.set(sessionID, value);
    while (sessions.size > settings.maxSessions) {
      const first = sessions.keys().next().value!;
      sessions.get(first)?.controller.abort();
      sessions.delete(first);
    }
    return value;
  }
  function remember(key: string, result: Inspection) {
    const prior = cache.get(key);
    if (prior) bytes -= prior.text.length * 2;
    cache.delete(key);
    cache.set(key, result);
    bytes += result.text.length * 2;
    while (cache.size > 128 || bytes > 8 * 1024 * 1024) {
      const first = cache.keys().next().value!;
      bytes -= cache.get(first)!.text.length * 2;
      cache.delete(first);
    }
  }
  async function inspect(
    sessionID: string,
    callID: string,
    identity: string,
    tool: OutputTool,
    rendered: Rendered,
  ): Promise<Inspection> {
    const current = state(sessionID),
      config = structuredClone(settings);
    const hash = (text: string) =>
      createHash("sha256").update(text).digest("hex");
    const prefix = `${sessionID}:${identity}:${tool}:${JSON.stringify([config.apps.redact, config.apps.shield, config.waits.redact, config.waits.shield, config.redact, config.shield])}:`;
    const key =
      prefix + hash(rendered.text) + (rendered.files.length ? ":files" : "");
    const cached = cache.get(key);
    if (cached) {
      cache.delete(key);
      cache.set(key, cached);
      if (!cached.withheld)
        for (const tag of cached.tags) current.tags.add(tag);
      return cached;
    }
    const result = await inspectOutput(
      tool,
      rendered.text,
      config,
      runtime,
      AbortSignal.any([stop.signal, current.controller.signal]),
    );
    // Projection can append an image-omission explanation and remove the file
    // part. Retain its incomplete coverage across that text-only representation.
    applyAttachmentPolicy(
      result,
      tool,
      config,
      rendered.files.length > 0 ||
        rendered.text.startsWith(ATTACHMENT_NOTE) ||
        rendered.text.startsWith(ATTACHMENT_WITHHELD),
    );
    if (!result.withheld) for (const tag of result.tags) current.tags.add(tag);
    for (const decision of result.decisions)
      decisions.record({ ...decision, sessionID, toolCallID: callID });
    if (result.decisions.length) await notify(sessionID);
    remember(key, result);
    // The host can lift attachments into separate user-message parts while
    // retaining the tool's text. Its coverage must remain partial in that form.
    remember(prefix + hash(result.text), result);
    remember(
      prefix +
        hash(result.text) +
        (rendered.files.length && !result.withheld ? ":files" : ""),
      result,
    );
    return result;
  }
  void (async () => {
    for await (const event of ctx.event.subscribe({
      signal: stop.signal,
      types: [
        "session.execution.started",
        "session.execution.interrupted",
        "session.deleted",
      ],
    })) {
      const id = event.data.sessionID;
      if (event.type === "session.execution.started") {
        const value = sessions.get(id);
        if (value?.controller.signal.aborted)
          value.controller = new AbortController();
      } else {
        sessions.get(id)?.controller.abort();
        if (event.type === "session.deleted") {
          sessions.delete(id);
          for (const [key, value] of cache)
            if (key.startsWith(id + ":")) {
              bytes -= value.text.length * 2;
              cache.delete(key);
            }
        }
      }
    }
  })().catch(() => {});
  await ctx.tool.hook("execute.before", async (event) => {
    if (
      !settings.apps.redact ||
      !["write", "edit", "apply_patch", "patch"].includes(event.tool)
    )
      return;
    const current = state(event.sessionID);
    const session = await ctx.session.get({ sessionID: event.sessionID });
    if (session.parentID)
      for (const tag of sessions.get(session.parentID)?.tags ?? [])
        current.tags.add(tag);
    if (!addedPlaceholders(event.tool, event.input, current.tags)) return;
    decisions.record({
      sessionID: event.sessionID,
      toolCallID: event.id,
      app: "redact",
      outcome: "withheld",
      reason: "placeholder-write",
    });
    await notify(event.sessionID);
    throw new Error(
      "vSR redact: this write would insert a masking placeholder in place of real data. Edit only the intended lines and preserve masked values.",
    );
  });
  await ctx.tool.hook("execute.after", async (event) => {
    if (!covered(event.tool) || event.status !== "completed") return;
    const rendered = render(event.result.content, event.result.output);
    const result = await inspect(
      event.sessionID,
      event.id,
      `${event.messageID}:${event.id}`,
      event.tool,
      rendered,
    );
    if (result.text === rendered.text && !result.withheld) return;
    event.result = {
      ...event.result,
      content: [
        ...(result.withheld ? [] : rendered.files),
        { type: "text", text: result.text },
      ],
    };
  });
  // Last content boundary: also covers native error results and synthetic background completions.
  await ctx.session.hook("context", async (event) => {
    const assistantIDs = new Map<string, string>();
    const messages: typeof event.messages = [];
    for (const message of event.messages) {
      if (message.role === "assistant") {
        messages.push(message);
        for (const part of message.content)
          if (part.type === "tool-call")
            assistantIDs.set(part.id, message.id ?? "context");
      } else if (message.role === "tool") {
        const content = [...message.content];
        for (const [index, part] of content.entries()) {
          if (part.type !== "tool-result" || !covered(part.name)) continue;
          const rendered =
            part.result.type === "content"
              ? render(part.result.value)
              : render(undefined, part.result.value);
          const result = await inspect(
            event.sessionID,
            part.id,
            `${assistantIDs.get(part.id) ?? "context"}:${part.id}`,
            part.name,
            rendered,
          );
          if (!result.withheld)
            for (const match of result.text.matchAll(marker))
              state(event.sessionID).tags.add(match[1] as Tag);
          if (result.text === rendered.text && !result.withheld) continue;
          content[index] = {
            ...part,
            result:
              rendered.files.length && !result.withheld
                ? {
                    type: "content",
                    value: [
                      ...rendered.files,
                      { type: "text", text: result.text },
                    ],
                  }
                : {
                    type: part.result.type === "error" ? "error" : "text",
                    value: result.text,
                  },
          };
        }
        messages.push({ ...message, content });
      } else if (message.role === "user") {
        const content = [...message.content];
        for (const [index, part] of content.entries())
          if (part.type === "text" && /^<shell\s/.test(part.text)) {
            const id = `background:${message.id ?? createHash("sha256").update(part.text).digest("hex")}`;
            const result = await inspect(event.sessionID, id, id, "shell", {
              text: part.text,
              files: [],
            });
            content[index] = { ...part, text: result.text };
          }
        messages.push({ ...message, content });
      } else messages.push(message);
    }
    event.messages = messages;
  });
  return () => {
    stop.abort();
    for (const value of sessions.values()) value.controller.abort();
    sessions.clear();
    cache.clear();
    bytes = 0;
  };
}

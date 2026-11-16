import { Rpc } from "@opencode/plugin/rpc";
import { APPS } from "./settings.ts";
import { FEEDBACK, OUTCOMES, REASONS } from "./decisions.ts";
const id = { type: "string", minLength: 1, maxLength: 256 } as const;
const session = {
  type: "object",
  properties: { sessionID: id },
  required: ["sessionID"],
  additionalProperties: false,
} as const;
const models = {
  type: "array",
  items: {
    type: "object",
    properties: {
      id: { type: "string" },
      questionTypes: { type: "array", items: { type: "string" } },
    },
    required: ["id", "questionTypes"],
    additionalProperties: false,
  },
} as const;
const decision = {
  type: "object",
  properties: {
    messageID: id,
    costKnown: { type: "boolean" },
    childOutcome: { enum: ["succeeded", "failed", "interrupted"] },
    counts: {
      type: "object",
      // A tag or a listener rule's name; DecisionStore.record checks its form (OpenCode RPC refuses `pattern`).
      propertyNames: { type: "string", minLength: 1, maxLength: 128 },
      additionalProperties: { type: "integer", minimum: 0 },
    },
    coverage: {
      type: "object",
      properties: {
        checked: { type: "integer", minimum: 0 },
        total: { type: "integer", minimum: 0 },
      },
      required: ["checked", "total"],
      additionalProperties: false,
    },
    reason: { type: "string", enum: [...REASONS] },
    observedModel: { type: "string" },
    childSessionID: id,
    usage: {
      type: "object",
      properties: Object.fromEntries(
        ["input", "output", "cachedInput", "reasoning", "cost"].map((key) => [
          key,
          { type: "number", minimum: 0 },
        ]),
      ),
      required: ["input", "output", "cachedInput", "reasoning", "cost"],
      additionalProperties: false,
    },
    id,
    sessionID: id,
    turnID: id,
    toolCallID: id,
    app: { type: "string", enum: [...APPS] },
    outcome: { type: "string", enum: [...OUTCOMES] },
    at: { type: "number" },
    elapsedMs: { type: "number", minimum: 0 },
    score: { type: "number" },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    model: { type: "string" },
    labels: { type: "array", items: { type: "string" } },
  },
  required: ["id", "sessionID", "app", "outcome", "at"],
  additionalProperties: false,
} as const;
const appFlags = {
  type: "object",
  properties: Object.fromEntries(APPS.map((app) => [app, { type: "boolean" }])),
  required: [...APPS],
  additionalProperties: false,
} as const;
const wait = {
  anyOf: [
    { type: "number", enum: [1000, 2000, 3000] },
    { type: "string", const: "done" },
  ],
} as const;
export const Vsr = Rpc.define({
  id: "vsr.opencode.v1",
  methods: {
    snapshot: {
      input: session,
      output: {
        type: "object",
        properties: {
          version: { const: 1 },
          instanceID: id,
          revision: { type: "integer", minimum: 0 },
          sessionID: id,
          decisions: { type: "array", items: decision },
          feedback: {
            type: "object",
            additionalProperties: { type: "string", enum: [...FEEDBACK] },
          },
          apps: appFlags,
          controls: {
            type: "object",
            properties: {
              revision: { type: "integer", minimum: 0 },
              persistence: { enum: ["durable", "unavailable"] },
              goal: { enum: ["balanced", "quality", "save"] },
            },
            required: ["revision", "persistence", "goal"],
            additionalProperties: false,
          },
          waits: {
            type: "object",
            properties: { redact: wait, shield: wait, guard: wait },
            required: ["redact", "shield", "guard"],
            additionalProperties: false,
          },
          runtime: {
            type: "object",
            properties: {
              reachable: { type: ["boolean", "null"] },
              retryAt: { type: "number" },
              models,
            },
            required: ["reachable", "retryAt", "models"],
            additionalProperties: false,
          },
        },
        required: [
          "version",
          "instanceID",
          "revision",
          "sessionID",
          "decisions",
          "feedback",
          "apps",
          "controls",
          "waits",
          "runtime",
        ],
        additionalProperties: false,
      },
    },
    diagnose: {
      input: { type: "object", additionalProperties: false },
      output: {
        anyOf: [
          {
            type: "object",
            properties: {
              ok: { const: true },
              value: models,
              elapsedMs: { type: "number" },
            },
            required: ["ok", "value", "elapsedMs"],
            additionalProperties: false,
          },
          {
            type: "object",
            properties: {
              ok: { const: false },
              code: { type: "string" },
              elapsedMs: { type: "number" },
              status: { type: "number" },
            },
            required: ["ok", "code", "elapsedMs"],
            additionalProperties: false,
          },
        ],
      },
    },
    controls: {
      input: {
        type: "object",
        properties: {
          instanceID: id,
          expectedRevision: { type: "integer", minimum: 0 },
          apps: { ...appFlags, required: [] },
          waits: {
            type: "object",
            properties: { redact: wait, shield: wait, guard: wait },
            additionalProperties: false,
          },
          goal: { enum: ["balanced", "quality", "save"] },
          follow: { const: true },
        },
        required: ["instanceID", "expectedRevision"],
        additionalProperties: false,
      },
      output: {
        anyOf: [
          {
            type: "object",
            properties: {
              saved: { const: true },
              revision: { type: "integer", minimum: 0 },
            },
            required: ["saved", "revision"],
            additionalProperties: false,
          },
          {
            type: "object",
            properties: {
              saved: { const: false },
              reason: { enum: ["stale", "invalid", "storage"] },
            },
            required: ["saved", "reason"],
            additionalProperties: false,
          },
        ],
      },
    },
    feedback: {
      input: {
        type: "object",
        properties: {
          sessionID: id,
          decisionID: id,
          feedback: { type: "string", enum: [...FEEDBACK] },
        },
        required: ["sessionID", "decisionID", "feedback"],
        additionalProperties: false,
      },
      output: {
        type: "object",
        properties: { saved: { const: true } },
        required: ["saved"],
        additionalProperties: false,
      },
    },
  },
  events: {
    stateChanged: {
      schema: {
        type: "object",
        properties: {
          instanceID: id,
          revision: { type: "integer", minimum: 0 },
        },
        required: ["instanceID", "revision"],
        additionalProperties: false,
      },
    },
    updated: {
      schema: {
        type: "object",
        properties: {
          instanceID: id,
          sessionID: id,
          revision: { type: "integer", minimum: 0 },
        },
        required: ["instanceID", "sessionID", "revision"],
        additionalProperties: false,
      },
    },
  },
});

import { Rpc } from "@opencode/plugin/rpc";
import { APPS } from "./settings.ts";
import { FEEDBACK, OUTCOMES } from "./decisions.ts";
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

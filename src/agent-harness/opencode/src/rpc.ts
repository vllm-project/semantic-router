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
    subject: { type: "string", maxLength: 200 },
    features: {
      type: "array",
      items: { type: "number", minimum: 0, maximum: 1 },
      maxItems: 2,
    },
  },
  required: ["id", "sessionID", "app", "outcome", "at"],
  additionalProperties: false,
} as const;
const app = { type: "string", enum: [...APPS] } as const;
const numbers = { type: "array", items: { type: "number" } } as const;
const goal = {
  enum: ["balanced", "fewer-false-alarms", "catch-more"],
} as const;
const agreement = {
  type: "object",
  properties: {
    agree: { type: "integer", minimum: 0 },
    total: { type: "integer", minimum: 0 },
  },
  required: ["agree", "total"],
  additionalProperties: false,
} as const;
const suggestion = {
  type: "object",
  properties: {
    app,
    labels: { type: "integer", minimum: 0 },
    before: numbers,
    after: numbers,
    changed: { type: "boolean" },
    agreeBefore: agreement,
    agreeAfter: agreement,
    unrouted: { type: "integer", minimum: 0 },
    why: { type: "string" },
  },
  required: [
    "app",
    "labels",
    "before",
    "after",
    "changed",
    "agreeBefore",
    "agreeAfter",
    "why",
  ],
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
          calibration: {
            type: "object",
            properties: {
              at: { type: "number" },
              since: { type: "integer", minimum: 0 },
              remind: { type: "boolean" },
            },
            required: ["since", "remind"],
            additionalProperties: false,
          },
          runtime: {
            type: "object",
            properties: {
              reachable: { type: ["boolean", "null"] },
              retryAt: { type: "number" },
              models,
              source: { enum: ["options", "connect"] },
            },
            required: ["reachable", "retryAt", "models", "source"],
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
          "calibration",
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
    verify: {
      input: { type: "object", additionalProperties: false },
      output: {
        type: "object",
        properties: {
          source: { enum: ["options", "connect"] },
          models: { type: "array", items: { type: "string" } },
          probes: {
            type: "array",
            items: {
              type: "object",
              properties: {
                app: { type: "string", enum: [...APPS] },
                verdict: { enum: ["acts", "passes", "no answer", "off"] },
                expected: { const: "acts" },
                elapsedMs: { type: "number", minimum: 0 },
                detail: { type: "string" },
              },
              required: ["app", "verdict", "expected", "elapsedMs", "detail"],
              additionalProperties: false,
            },
          },
        },
        required: ["source", "models", "probes"],
        additionalProperties: false,
      },
    },
    // Calibration: what the log holds per app, questions about the decisions nearest each line, the person's
    // answers, and new lines fitted to them, applied only on request.
    calibrationReport: {
      input: { type: "object", additionalProperties: false },
      output: {
        type: "object",
        properties: {
          persistence: { enum: ["durable", "unavailable"] },
          at: { type: "number" },
          since: { type: "integer", minimum: 0 },
          remind: { type: "boolean" },
          apps: {
            type: "array",
            items: {
              type: "object",
              properties: {
                app,
                enabled: { type: "boolean" },
                decisions: { type: "integer", minimum: 0 },
                answered: { type: "integer", minimum: 0 },
                thresholds: numbers,
                calibrated: { type: "boolean" },
              },
              required: [
                "app",
                "enabled",
                "decisions",
                "answered",
                "thresholds",
                "calibrated",
              ],
              additionalProperties: false,
            },
          },
        },
        required: ["persistence", "since", "remind", "apps"],
        additionalProperties: false,
      },
    },
    calibrationSample: {
      input: {
        type: "object",
        properties: { app },
        required: ["app"],
        additionalProperties: false,
      },
      output: {
        type: "array",
        items: {
          type: "object",
          properties: {
            id,
            app,
            subject: { type: "string" },
            acted: { type: "boolean" },
            scores: numbers,
            level: { type: "number" },
            tier: { type: "integer", minimum: 0, maximum: 2 },
            thresholds: numbers,
          },
          required: ["id", "app", "subject", "acted", "thresholds"],
          additionalProperties: false,
        },
      },
    },
    calibrationLabel: {
      input: {
        type: "object",
        properties: {
          id,
          label: { enum: ["yes", "no", "too-low", "right", "too-high"] },
        },
        required: ["id", "label"],
        additionalProperties: false,
      },
      output: {
        type: "object",
        properties: { saved: { type: "boolean" } },
        required: ["saved"],
        additionalProperties: false,
      },
    },
    calibrationSuggest: {
      input: {
        type: "object",
        properties: { app, goal },
        required: ["app"],
        additionalProperties: false,
      },
      output: suggestion,
    },
    calibrationApply: {
      input: {
        type: "object",
        properties: { app, goal },
        required: ["app"],
        additionalProperties: false,
      },
      output: {
        type: "object",
        properties: {
          applied: { type: "boolean" },
          suggestion,
        },
        required: ["applied", "suggestion"],
        additionalProperties: false,
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

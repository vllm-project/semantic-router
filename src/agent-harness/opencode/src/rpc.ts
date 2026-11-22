import { Rpc } from "@opencode/plugin/rpc";
import { APPS } from "./settings.ts";
import { FEEDBACK, OUTCOMES, REASONS } from "./decisions.ts";
const id = { type: "string", minLength: 1, maxLength: 256 } as const;
const verdict = {
  enum: ["acts", "passes", "light", "medium", "strong", "kept"],
} as const;
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
      presets: { type: "array", items: { type: "string" } },
      modalities: { type: "array", items: { type: "string" } },
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
    claims: {
      type: "array",
      maxItems: 3,
      items: { type: "string", minLength: 1, maxLength: 160 },
    },
    estimate: {
      type: "object",
      properties: {
        actual: { type: "number", minimum: 0 },
        baseline: { type: "number", minimum: 0 },
        on: { type: "string", minLength: 1, maxLength: 256 },
      },
      required: ["actual", "baseline", "on"],
      additionalProperties: false,
    },
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
    thresholds: {
      type: "array",
      items: { type: "number", minimum: 0 },
      maxItems: 3,
    },
    scope: { const: "turn" },
    reread: {
      type: "object",
      properties: {
        tokens: { type: "integer", minimum: 0 },
        cost: { type: "number", minimum: 0 },
        saving: { type: "number" },
      },
      required: ["tokens", "cost", "saving"],
      additionalProperties: false,
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
    decisions: { type: "integer", minimum: 0 },
    acted: {
      type: "object",
      properties: {
        before: { type: "integer", minimum: 0 },
        after: { type: "integer", minimum: 0 },
      },
      required: ["before", "after"],
      additionalProperties: false,
    },
    moves: {
      type: "array",
      items: {
        type: "object",
        properties: {
          from: { enum: ["light", "medium", "strong", "kept"] },
          to: { enum: ["light", "medium", "strong", "kept"] },
          n: { type: "integer", minimum: 1 },
        },
        required: ["from", "to", "n"],
        additionalProperties: false,
      },
    },
    recommend: { enum: ["apply", "keep", "more"] },
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
    "decisions",
    "recommend",
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
          answerable: { type: "array", items: id },
          apps: appFlags,
          controls: {
            type: "object",
            properties: {
              revision: { type: "integer", minimum: 0 },
              persistence: { enum: ["durable", "unavailable"] },
              goal: { enum: ["balanced", "quality", "save"] },
              turns: { type: "boolean" },
              calibrateEvery: { type: "integer", minimum: 0 },
              downgradeConfidence: { type: "number", minimum: 0, maximum: 1 },
              stepsLow: { type: "number", minimum: 0, maximum: 2 },
              stepsHigh: { type: "number", minimum: 0, maximum: 2 },
              stepsMinConfidence: { type: "number", minimum: 0, maximum: 1 },
              // What a Router listener's profile changes from the person's settings; empty when nothing.
              note: { type: "string" },
              // Whether the runtime is a Router listener with a vsr profile to follow.
              listener: { type: "boolean" },
            },
            required: [
              "revision",
              "persistence",
              "goal",
              "turns",
              "calibrateEvery",
              "downgradeConfidence",
              "stepsLow",
              "stepsHigh",
              "stepsMinConfidence",
              "note",
              "listener",
            ],
            additionalProperties: false,
          },
          waits: {
            type: "object",
            properties: { redact: wait, shield: wait, guard: wait },
            required: ["redact", "shield", "guard"],
            additionalProperties: false,
          },
          // Your own turns: whether the router routes them, the tier it moved them to, and the model you picked when
          // that paused it.
          turns: {
            type: "object",
            properties: {
              on: { type: "boolean" },
              model: { type: "string" },
              paused: { type: "string" },
              spend: {
                type: "object",
                properties: {
                  actual: { type: "number", minimum: 0 },
                  baseline: { type: "number", minimum: 0 },
                  turns: { type: "integer", minimum: 0 },
                  on: { type: "string" },
                },
                required: ["actual", "baseline", "turns", "on"],
                additionalProperties: false,
              },
            },
            required: ["on"],
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
              // The guard case scored under the review line by this runtime's model, said once in a toast.
              guardNote: { type: "string" },
              // Plain http to another machine: what the apps check, and the token, travel unencrypted.
              cleartext: { type: "boolean" },
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
          "turns",
          "waits",
          "calibration",
          "runtime",
          "answerable",
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
          model: { type: "string" },
          cleartext: { type: "boolean" },
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
          // The decision model now, and the one the last calibration was made with: when they differ, probe.
          model: { type: "string" },
          calibratedModel: { type: "string" },
          // When the oldest and newest logged decisions were made.
          first: { type: "number" },
          last: { type: "number" },
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
                acted: { type: "integer", minimum: 0 },
                explicit: { type: "integer", minimum: 0 },
                implied: { type: "integer", minimum: 0 },
                // Router: light, medium, strong, then kept.
                routes: {
                  type: "array",
                  items: { type: "integer", minimum: 0 },
                  minItems: 4,
                  maxItems: 4,
                },
              },
              required: [
                "app",
                "enabled",
                "decisions",
                "answered",
                "thresholds",
                "calibrated",
                "acted",
                "explicit",
                "implied",
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
        properties: {
          app,
          skip: { type: "array", items: { type: "string" }, maxItems: 1000 },
        },
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
          // null takes an answer back, when the person changes it to "not sure".
          label: { enum: ["yes", "no", "too-low", "right", "too-high", null] },
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
    // /vsr-config: each app's thresholds now, the options' own, and whose they are; a change sets them as the
    // person's, or with no values brings the options' back.
    thresholds: {
      input: {
        type: "object",
        properties: {
          app,
          values: { type: "array", items: { type: "number" } },
        },
        additionalProperties: false,
      },
      output: {
        type: "array",
        items: {
          type: "object",
          properties: {
            app,
            values: numbers,
            options: numbers,
            source: { enum: ["options", "calibration", "you"] },
          },
          required: ["app", "values", "options", "source"],
          additionalProperties: false,
        },
      },
    },
    // Known cases asked again, as the Claude Code plugin's `calibrate probe`: what the decision model now decides.
    calibrationProbe: {
      input: {
        type: "object",
        properties: { apps: { type: "array", items: app, maxItems: 6 } },
        additionalProperties: false,
      },
      output: {
        type: "object",
        properties: {
          model: { type: "string" },
          calibratedModel: { type: "string" },
          rows: {
            type: "array",
            items: {
              type: "object",
              properties: {
                app,
                repeat: { const: true },
                name: { type: "string" },
                measured: { type: "number" },
                score: { type: "number" },
                expect: verdict,
                now: verdict,
                yours: verdict,
                status: { enum: ["ok", "drift", "flip", "error"] },
                drift: { type: "number" },
                error: { type: "string" },
                elapsedMs: { type: "number", minimum: 0 },
              },
              required: ["app", "name", "expect", "status", "elapsedMs"],
              additionalProperties: false,
            },
          },
        },
        required: ["rows"],
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
          turns: { type: "boolean" },
          calibrateEvery: { type: "integer", minimum: 0, maximum: 100000 },
          downgradeConfidence: { type: "number", minimum: 0, maximum: 1 },
          stepsLow: { type: "number", minimum: 0, maximum: 2 },
          stepsHigh: { type: "number", minimum: 0, maximum: 2 },
          stepsMinConfidence: { type: "number", minimum: 0, maximum: 1 },
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

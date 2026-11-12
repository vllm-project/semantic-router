import { Rpc } from "@opencode/plugin/rpc";
export const Probe = Rpc.define({
  id: "vsr-gate0",
  methods: {
    snapshot: {
      input: { type: "object", additionalProperties: false },
      output: {
        type: "object",
        properties: {
          enabled: { type: "boolean" },
          decisions: { type: "number" },
        },
        required: ["enabled", "decisions"],
        additionalProperties: false,
      },
    },
    setEnabled: {
      input: {
        type: "object",
        properties: { enabled: { type: "boolean" } },
        required: ["enabled"],
        additionalProperties: false,
      },
      output: {
        type: "object",
        properties: { enabled: { type: "boolean" } },
        required: ["enabled"],
        additionalProperties: false,
      },
    },
  },
  events: {
    updated: {
      schema: {
        type: "object",
        properties: {
          enabled: { type: "boolean" },
          decisions: { type: "number" },
        },
        required: ["enabled", "decisions"],
        additionalProperties: false,
      },
    },
  },
});

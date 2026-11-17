package config

import (
	"strings"
	"testing"
)

const promptLimitDocument = `
version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
global:
  plugins:
    - type: router_replay
      configuration: GLOBAL
recipes:
  - name: default
    plugins:
      - type: router_replay
        configuration: RECIPE
    decisions:
      - name: inherits
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: general}]
      - name: own
        priority: 50
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: general}]
        plugins:
          - type: router_replay
            configuration: DECISION
`

// The prompt's limit is max_prompt_bytes from the most specific layer that
// sets it, 0 for no limit; where no layer does, max_tool_trace_bytes when that
// is above 0, so a configuration that bounded the prompt with it keeps that
// bound; else 64 KiB, in either mode.
func TestRouterReplayPromptByteLimit(t *testing.T) {
	const raw = "{enabled: true, prompt_capture: raw"
	for _, test := range []struct {
		name, global, recipe, decision string
		inherits, own                  int
	}{
		{"neither set, raw", raw + "}", "{}", "{}", 64 << 10, 64 << 10},
		{"neither set, verified", "{enabled: true, prompt_capture: verified}", "{}", "{}", 64 << 10, 64 << 10},
		{"a trace limit of 0 is no fallback", raw + ", max_tool_trace_bytes: 0}", "{}", "{}", 64 << 10, 64 << 10},
		{"a 4096 trace limit keeps its prompt bound", raw + ", max_tool_trace_bytes: 4096}", "{}", "{}", 4096, 4096},
		{"a 32 trace limit keeps its prompt bound", raw + ", max_tool_trace_bytes: 32}", "{}", "{}", 32, 32},
		{"a 262144 trace limit keeps 256 KiB", raw + ", max_tool_trace_bytes: 262144}", "{}", "{}", 256 << 10, 256 << 10},
		{"explicit above the trace limit", raw + ", max_tool_trace_bytes: 4096, max_prompt_bytes: 65536}", "{}", "{}", 64 << 10, 64 << 10},
		{"explicit below the trace limit", raw + ", max_tool_trace_bytes: 262144, max_prompt_bytes: 8192}", "{}", "{}", 8192, 8192},
		{"explicit 0 over a trace limit", raw + ", max_tool_trace_bytes: 4096, max_prompt_bytes: 0}", "{}", "{}", 0, 0},
		{"a decision's value over a global fallback", raw + ", max_tool_trace_bytes: 262144}", "{}", "{max_prompt_bytes: 4096}", 256 << 10, 4096},
		{"a decision's 0 over a global fallback", raw + ", max_tool_trace_bytes: 4096}", "{}", "{max_prompt_bytes: 0}", 4096, 0},
		{"a global value over a decision's trace limit", raw + ", max_prompt_bytes: 8192}", "{}", "{max_tool_trace_bytes: 1048576}", 8192, 8192},
		{"a decision's trace limit where no layer sets one", raw + "}", "{}", "{max_tool_trace_bytes: 4096}", 64 << 10, 4096},
		{"recipe over global", raw + ", max_prompt_bytes: 1048576}", "{max_prompt_bytes: 262144}", "{}", 256 << 10, 256 << 10},
		{"decision over recipe", raw + "}", "{max_prompt_bytes: 262144}", "{max_prompt_bytes: 4096}", 256 << 10, 4096},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes([]byte(strings.NewReplacer("GLOBAL", test.global, "RECIPE", test.recipe, "DECISION", test.decision).Replace(promptLimitDocument)))
			if err != nil {
				t.Fatal(err)
			}
			limits := map[string]int{}
			for i := range cfg.Decisions {
				limits[cfg.Decisions[i].Name] = cfg.EffectiveRouterReplayConfig(&cfg.Decisions[i]).PromptByteLimit()
			}
			if limits["inherits"] != test.inherits || limits["own"] != test.own {
				t.Fatalf("prompt limits %d and %d, want %d and %d", limits["inherits"], limits["own"], test.inherits, test.own)
			}
			if len(cfg.FormWarnings) != 0 {
				t.Fatalf("unexpected warnings: %+v", cfg.FormWarnings)
			}
		})
	}
}

// A negative max_prompt_bytes is refused wherever it is written.
func TestRouterReplayRefusesANegativeMaxPromptBytes(t *testing.T) {
	for _, layer := range []string{"GLOBAL", "RECIPE", "DECISION"} {
		replacements := map[string]string{"GLOBAL": "{enabled: true, prompt_capture: raw}", "RECIPE": "{}", "DECISION": "{}"}
		replacements[layer] = "{enabled: true, max_prompt_bytes: -1}"
		document := strings.NewReplacer("GLOBAL", replacements["GLOBAL"], "RECIPE", replacements["RECIPE"], "DECISION", replacements["DECISION"]).Replace(promptLimitDocument)
		if _, err := ParseYAMLBytes([]byte(document)); err == nil || !strings.Contains(err.Error(), "max_prompt_bytes cannot be negative") {
			t.Fatalf("%s: a negative max_prompt_bytes loaded: %v", layer, err)
		}
	}
}

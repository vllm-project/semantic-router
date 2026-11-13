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

// max_prompt_bytes layers field by field like the other capture settings:
// unset is 64 KiB, in every mode, a recipe or decision overrides it, and 0
// removes the limit.
func TestRouterReplayMaxPromptBytesLayersFieldByField(t *testing.T) {
	for _, test := range []struct {
		name, global, recipe, decision string
		inherits, own                  int
	}{
		{"unset, raw", "{enabled: true, prompt_capture: raw}", "{}", "{}", 64 << 10, 64 << 10},
		{"unset, verified", "{enabled: true, prompt_capture: verified}", "{}", "{}", 64 << 10, 64 << 10},
		{"global", "{enabled: true, prompt_capture: raw, max_prompt_bytes: 1048576}", "{}", "{}", 1 << 20, 1 << 20},
		{"recipe over global", "{enabled: true, prompt_capture: raw, max_prompt_bytes: 1048576}", "{max_prompt_bytes: 262144}", "{}", 256 << 10, 256 << 10},
		{"decision over recipe", "{enabled: true, prompt_capture: raw}", "{max_prompt_bytes: 262144}", "{max_prompt_bytes: 4096}", 256 << 10, 4096},
		{"0 removes the limit", "{enabled: true, prompt_capture: raw, max_prompt_bytes: 4096}", "{}", "{max_prompt_bytes: 0}", 4096, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes([]byte(strings.NewReplacer("GLOBAL", test.global, "RECIPE", test.recipe, "DECISION", test.decision).Replace(promptLimitDocument)))
			if err != nil {
				t.Fatal(err)
			}
			limits := map[string]int{}
			for i := range cfg.Decisions {
				limits[cfg.Decisions[i].Name] = cfg.EffectiveRouterReplayConfig(&cfg.Decisions[i]).MaxPromptBytes
			}
			if limits["inherits"] != test.inherits || limits["own"] != test.own {
				t.Fatalf("max_prompt_bytes %d and %d, want %d and %d", limits["inherits"], limits["own"], test.inherits, test.own)
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

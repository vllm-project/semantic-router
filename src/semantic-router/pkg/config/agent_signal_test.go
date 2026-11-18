package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

// A recipe routes a subagent type to its own model and effort, and blocks a
// subagent that sends personal data: the agent rules load, decisions name
// them, and they survive the canonical export.
func TestAgentSignalLoadsAndRoundTrips(t *testing.T) {
	input := []byte(`version: v0.5
providers:
  models:
    - name: cheap
      backend_refs: [{name: local, endpoint: 127.0.0.1:8000}]
recipes:
  - name: default
    signals:
      agent:
        - name: explore
          role: subagent
          types: [Explore, collab_spawn]
        - name: codex_main
          role: main
          harness: codex
        - name: housekeeping
          request_classes: [compaction, auxiliary]
      pii:
        - name: personal_data
    decisions:
      - name: block_subagent_pii
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: agent, name: explore}
            - {type: pii, name: personal_data}
        plugins:
          - type: fast_response
            configuration: {message: Subagents may not send personal data., block: true}
      - name: explore
        priority: 100
        rules: {type: agent, name: explore}
        modelRefs: [{model: cheap, use_reasoning: true, reasoning_effort: low}]
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.AgentRules) != 3 || cfg.AgentRules[0].Types[0] != "Explore" || cfg.AgentRules[1].Harness != AgentHarnessCodex ||
		len(cfg.AgentRules[2].RequestClasses) != 2 {
		t.Fatalf("agent rules = %+v", cfg.AgentRules)
	}
	if block := cfg.Decisions[0].GetFastResponseConfig(); block == nil || !block.Block {
		t.Fatalf("blocking fast_response = %+v", block)
	}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	again, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if len(again.AgentRules) != 3 || again.AgentRules[0].Role != AgentRoleSubagent || again.AgentRules[2].RequestClasses[1] != AgentRequestClassAuxiliary {
		t.Fatalf("agent rules after export = %+v", again.AgentRules)
	}
	if entry, ok := LookupSignalCatalog(SignalTypeAgent); !ok || !entry.DecisionReferenceable {
		t.Fatalf("catalog entry = %+v, %v", entry, ok)
	}
}

func TestValidateAgentContracts(t *testing.T) {
	for _, tc := range []struct {
		name string
		rule AgentRule
		want string
	}{
		{name: "role", rule: AgentRule{Name: "sub", Role: AgentRoleSubagent}},
		{name: "harness", rule: AgentRule{Name: "cc", Harness: AgentHarnessClaudeCode}},
		{name: "opencode harness", rule: AgentRule{Name: "oc", Harness: AgentHarnessOpenCode}},
		{name: "types", rule: AgentRule{Name: "review", Types: []string{"review"}}},
		{name: "request classes", rule: AgentRule{Name: "housekeeping", RequestClasses: []string{AgentRequestClassCompaction, AgentRequestClassAuxiliary}}},
		{name: "opencode request classes", rule: AgentRule{Name: "oc-titles", Harness: AgentHarnessOpenCode, RequestClasses: []string{AgentRequestClassAuxiliary}}},
		{name: "nothing to match", rule: AgentRule{Name: "any"}, want: "set at least one of role, harness, types or request_classes"},
		{name: "unknown request class", rule: AgentRule{Name: "x", RequestClasses: []string{"title"}}, want: "request_classes[0] must be one of main, subagent, workflow, compaction, auxiliary"},
		{name: "codex request class", rule: AgentRule{Name: "x", Harness: AgentHarnessCodex, RequestClasses: []string{"compaction"}}, want: "request_classes cannot go with harness codex"},
		{name: "deepseek harness request class", rule: AgentRule{Name: "x", Harness: AgentHarnessDeepSeekHarness, RequestClasses: []string{"auxiliary"}}, want: "request_classes cannot go with harness deepseek_harness"},
		{name: "unknown role", rule: AgentRule{Name: "x", Role: "child"}, want: "role must be one of main, subagent"},
		{name: "unknown harness", rule: AgentRule{Name: "x", Harness: "gemini_cli"}, want: "harness must be one of claude_code, codex, opencode"},
		{name: "main with types", rule: AgentRule{Name: "x", Role: AgentRoleMain, Types: []string{"Explore"}}, want: "types name a subagent's type"},
		{name: "blank type", rule: AgentRule{Name: "x", Types: []string{" "}}, want: "types[0] must be nonempty and trimmed"},
		{name: "untrimmed name", rule: AgentRule{Name: " x ", Role: AgentRoleMain}, want: "name must not contain surrounding whitespace"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{Signals: Signals{AgentRules: []AgentRule{tc.rule}}}}
			err := validateAgentContracts(cfg)
			if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("validateAgentContracts() = %v, want %q", err, tc.want)
			}
		})
	}
	duplicate := &RouterConfig{IntelligentRouting: IntelligentRouting{Signals: Signals{AgentRules: []AgentRule{
		{Name: "sub", Role: AgentRoleSubagent}, {Name: "Sub", Role: AgentRoleMain},
	}}}}
	if err := validateAgentContracts(duplicate); err == nil || !strings.Contains(err.Error(), "duplicate name") {
		t.Fatalf("duplicate names: %v", err)
	}
}

func TestDecisionLeafRejectsUnknownAgentReference(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`version: v0.5
providers:
  models:
    - name: cheap
      backend_refs: [{name: local, endpoint: 127.0.0.1:8000}]
recipes:
  - name: default
    decisions:
      - name: explore
        rules: {type: agent, name: missing}
        modelRefs: [{model: cheap}]
`))
	if err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("unknown agent reference: %v", err)
	}
}

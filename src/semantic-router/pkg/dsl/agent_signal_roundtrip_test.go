package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// An agent signal and a blocking fast_response compile, decompile and
// compile again unchanged, and an agent rule that matches nothing is refused.
func TestAgentSignalAndBlockingFastResponseRoundTrip(t *testing.T) {
	input := `
SIGNAL agent "explore" {
  description: "Exploration subagents."
  role: "subagent"
  harness: "claude_code"
  types: ["Explore"]
  request_classes: ["subagent"]
}

ROUTE "explore" {
  PRIORITY 100
  WHEN agent("explore")
  MODEL "cheap" (reasoning = true, effort = "low")
}

ROUTE "block_main" {
  PRIORITY 200
  WHEN agent("explore") AND keyword("secret")
  PLUGIN fast_response {
    message: "Blocked by policy."
    block: true
  }
}

SIGNAL keyword "secret" {
  operator: "any"
  keywords: ["secret"]
}`
	cfg := mustCompilePolicyDSL(t, input)
	want := []config.AgentRule{{
		Name: "explore", Description: "Exploration subagents.", Role: config.AgentRoleSubagent,
		Harness: config.AgentHarnessClaudeCode, Types: []string{"Explore"}, RequestClasses: []string{"subagent"},
	}}
	if !reflect.DeepEqual(cfg.AgentRules, want) {
		t.Fatalf("agent rules = %+v", cfg.AgentRules)
	}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(source, `SIGNAL agent explore {`) || !strings.Contains(source, "block: true") {
		t.Fatalf("decompiled source lost the agent signal or block:\n%s", source)
	}
	again := mustCompilePolicyDSL(t, source)
	if !reflect.DeepEqual(again.AgentRules, want) {
		t.Fatalf("round-trip agent rules = %+v", again.AgentRules)
	}
	for _, decision := range again.Decisions {
		if decision.Name == "block_main" && (decision.GetFastResponseConfig() == nil || !decision.GetFastResponseConfig().Block) {
			t.Fatalf("round-trip lost block: %+v", decision.GetFastResponseConfig())
		}
	}

	if _, errs := Compile(`SIGNAL agent "any" { description: "matches nothing" }`); len(errs) == 0 {
		t.Fatal("an agent rule without role, harness, types or request_classes compiled")
	}
	if _, errs := Compile(`SIGNAL agent "titles" { request_classes: ["title"] }`); len(errs) == 0 {
		t.Fatal("an agent rule with an unknown request class compiled")
	}
}

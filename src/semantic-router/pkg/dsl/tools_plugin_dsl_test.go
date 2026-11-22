package dsl

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompileToolsPlugin(t *testing.T) {
	input := `
ROUTE security_containment {
  PRIORITY 300
  PLUGIN tools { enabled: true, policy: { mode: "none", strip_tool_history: true } }
  MODEL "local-guard"
}

ROUTE frontier_reasoning {
  PRIORITY 200
  PLUGIN tools {
    enabled: true
    policy: { mode: "filtered", allow_tools: ["search_web"], block_tools: ["exec_cmd"] }
    select: { mode: "add", tools_db_path: "config/tools_db.json", match: { judge: { preset: "tool_rank", top_k: 5, min_ratio: 0.2 } }, need_threshold: 0.6 }
  }
  MODEL "frontier-model"
}

ROUTE local_standard {
  PRIORITY 100
  PLUGIN tools { enabled: true, select: { mode: "filter", preserve_count: 2, input: { history: true } } }
  MODEL "default-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 3 {
		t.Fatalf("expected 3 decisions, got %d", len(cfg.Decisions))
	}
	containment := cfg.Decisions[0].GetToolsConfig()
	if containment.Policy.EffectiveMode() != config.ToolsPolicyModeNone || !containment.Policy.StripToolHistory || containment.Select != nil {
		t.Fatalf("security_containment = %+v", containment)
	}
	frontier := cfg.Decisions[1].GetToolsConfig()
	if frontier.Policy.EffectiveMode() != config.ToolsPolicyModeFiltered || frontier.Select.Mode != config.ToolsSelectModeAdd ||
		frontier.Select.Match.Judge.TopK != 5 || *frontier.Select.Match.Judge.MinRatio != 0.2 || frontier.Select.EffectiveNeedThreshold() != 0.6 {
		t.Fatalf("frontier_reasoning = %+v / %+v", frontier.Policy, frontier.Select)
	}
	standard := cfg.Decisions[2].GetToolsConfig()
	if standard.Select.Mode != config.ToolsSelectModeFilter || standard.Select.PreserveCount != 2 || !standard.Select.Input.History {
		t.Fatalf("local_standard = %+v", standard.Select)
	}
}

// The DSL refuses the tool forms the Router no longer reads and names what
// replaced them.
func TestCompileRefusesRemovedToolForms(t *testing.T) {
	for input, want := range map[string]string{
		`ROUTE r { PRIORITY 1 PLUGIN tool_selection { enabled: true, mode: "add" } MODEL "m" }`:     `plugin "tool_selection" was removed; write a select block in the decision's tools plugin`,
		`ROUTE r { PRIORITY 1 PLUGIN tools { enabled: true, mode: "none" } MODEL "m" }`:             `unknown field "mode"`,
		`ROUTE r { PRIORITY 1 PLUGIN tools { enabled: true, semantic_selection: true } MODEL "m" }`: `unknown field "semantic_selection"`,
	} {
		_, errs := Compile(input)
		if len(errs) == 0 || !strings.Contains(fmt.Sprint(errs), want) {
			t.Errorf("%s: errors %v, want %q", input, errs, want)
		}
	}
}

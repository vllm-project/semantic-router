package classification

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func multiObjectiveEngine(t *testing.T, recipeName string) *decision.DecisionEngine {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "multi-objective", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	recipe, ok := cfg.RecipeByName(config.RecipeName(recipeName))
	if !ok {
		t.Fatalf("%s recipe missing", recipeName)
	}
	cfg = cfg.ConfigForRecipe(recipe)
	return decision.NewDecisionEngine(nil, nil, nil, cfg.Decisions, config.RoutingStrategyPriority)
}

func TestMultiObjectiveObjectivesRouteOnDecisionQuestions(t *testing.T) {
	tests := []struct {
		recipe, name string
		signals      decision.SignalMatches
		want         string
	}{
		{"balanced", "an ordinary request stays on the balanced route", decision.SignalMatches{}, "unified_balance_route"},
		{"balanced", "a reported wrong answer recovers", decision.SignalMatches{DecisionRules: []string{"correction"}}, "unified_balance_recovery"},
		{"balanced", "a repeated request recovers", decision.SignalMatches{ReaskRules: []string{"unified_balance_reask"}}, "unified_balance_recovery"},
		{"balanced", "careful reasoning is deliberate", decision.SignalMatches{DecisionRules: []string{"deliberate"}}, "unified_balance_deliberate_route"},
		{"balanced", "explicit verification is deliberate", decision.SignalMatches{DecisionRules: []string{"verification"}}, "unified_balance_deliberate_route"},
		{"balanced", "a long input is deliberate", decision.SignalMatches{ContextRules: []string{"unified_balance_long_context"}}, "unified_balance_deliberate_route"},
		{"balanced", "an unanswered deliberate question keeps quality", decision.SignalMatches{SignalErrors: map[string]string{"decision:deliberate": "timeout"}}, "unified_balance_deliberate_route"},
		{"balanced", "an image outranks a correction", decision.SignalMatches{ConversationRules: []string{"unified_balance_has_images"}, DecisionRules: []string{"correction"}}, "omni"},
		{"speed-first", "light work takes the fastest route", decision.SignalMatches{}, "unified_speed_first_route"},
		{"speed-first", "heavy work takes the heavy route", decision.SignalMatches{DecisionRules: []string{"heavy"}}, "unified_speed_heavy_route"},
		{"speed-first", "a long input takes the heavy route", decision.SignalMatches{ContextRules: []string{"unified_speed_long_context"}}, "unified_speed_heavy_route"},
		{"cost-first", "ordinary work stays on the economy route", decision.SignalMatches{}, "unified_cost_first_route"},
		{"cost-first", "a hard problem enables local reasoning", decision.SignalMatches{DecisionRules: []string{"depth"}}, "unified_cost_local_reasoning"},
		{"accuracy-first", "ordinary work stays direct", decision.SignalMatches{}, "unified_frontier_direct"},
		{"accuracy-first", "a staged task uses Router Flow", decision.SignalMatches{DecisionRules: []string{"workflow"}}, "unified_frontier_workflow"},
		{"accuracy-first", "a quoted workflow phrase stays direct", decision.SignalMatches{DecisionRules: []string{"workflow", "quoted"}}, "unified_frontier_direct"},
		{"accuracy-first", "an expert panel is not a workflow", decision.SignalMatches{DecisionRules: []string{"workflow", "mode:experts"}}, "unified_frontier_fusion"},
		{"accuracy-first", "an active tool loop is synthesized directly", decision.SignalMatches{ConversationRules: []string{"unified_frontier_active_tool_loop"}}, "unified_frontier_tool_result_synthesis"},
		{"accuracy-first", "a staged task owns its tool loop", decision.SignalMatches{ConversationRules: []string{"unified_frontier_active_tool_loop"}, DecisionRules: []string{"workflow"}}, "unified_frontier_workflow"},
		{"accuracy-first", "requested verification uses confidence", decision.SignalMatches{DecisionRules: []string{"verification", "depth"}}, "unified_frontier_verified_answer"},
		{"accuracy-first", "a deep problem uses ReMoM", decision.SignalMatches{DecisionRules: []string{"depth"}}, "unified_frontier_remom"},
		{"accuracy-first", "an unanswered mode never fans out", decision.SignalMatches{SignalErrors: map[string]string{"decision:mode": "timeout", "decision:workflow": "timeout"}}, "unified_frontier_direct"},
		{"privacy-first", "ordinary work stays on the local default", decision.SignalMatches{}, "unified_privacy_local_default"},
		{"privacy-first", "an attack is contained", decision.SignalMatches{DecisionRules: []string{"attack", "confidential"}}, "unified_privacy_security_containment"},
		{"privacy-first", "personal data is sensitive", decision.SignalMatches{DecisionRules: []string{"personal_data"}}, "unified_privacy_sensitive_route"},
		{"privacy-first", "an unanswered question stays on the sensitive route", decision.SignalMatches{SignalErrors: map[string]string{"decision:confidential": "timeout", "decision:personal_data": "timeout"}}, "unified_privacy_sensitive_route"},
	}
	engines := map[string]*decision.DecisionEngine{}
	for _, tt := range tests {
		t.Run(tt.recipe+"/"+tt.name, func(t *testing.T) {
			engine := engines[tt.recipe]
			if engine == nil {
				engine = multiObjectiveEngine(t, tt.recipe)
				engines[tt.recipe] = engine
			}
			signals := tt.signals
			got, err := engine.EvaluateDecisionsWithSignals(&signals)
			if err != nil {
				t.Fatal(err)
			}
			if got == nil || got.Decision == nil || got.Decision.Name != tt.want {
				t.Fatalf("got %+v, want %s", got, tt.want)
			}
		})
	}
}

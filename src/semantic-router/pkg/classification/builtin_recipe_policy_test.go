package classification

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func builtinPolicyClassifier(t *testing.T, name string) *Classifier {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "built-in", "latest", "mom-v1", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	recipe, ok := cfg.RecipeByName(config.RecipeName(name))
	if !ok {
		t.Fatalf("built-in recipe %q is missing", name)
	}
	return &Classifier{Config: cfg.ConfigForRecipe(recipe)}
}

// balanceAnswers supplies the decision-model answers Balance reads: the task
// label, the 0-4 difficulty score and the deliberation probability. They are
// controlled inputs to the shipped projections and rules, not model output.
func balanceAnswers(task string, difficulty, deliberation float64, matched ...string) SignalResults {
	in := SignalResults{
		MatchedDecisionRules: append([]string{"task:" + task}, matched...),
		SignalValues: map[string]float64{
			"decision:difficulty":         difficulty,
			"decision:needs:deliberation": deliberation,
			"decision:needs:verification": 0,
			"decision:needs:tools":        0,
		},
	}
	if difficulty >= 2 {
		in.MatchedDecisionRules = append(in.MatchedDecisionRules, "difficulty")
	}
	return in
}

func balanceWith(in SignalResults, conversation, keywords, feedback []string) SignalResults {
	in.MatchedConversationRules = conversation
	in.MatchedKeywordRules = keywords
	in.MatchedUserFeedbackRules = feedback
	return in
}

// These are policy counterexamples, not classifier accuracy measurements. Read
// the shipped recipe and run its actual projections and decision evaluator.
func TestBuiltinBalanceEffortLanes(t *testing.T) {
	c := builtinPolicyClassifier(t, "balance")
	unknown := SignalResults{SignalErrors: map[string]string{}}
	for _, name := range []string{"task", "difficulty", "precise_facts", "needs", "correction"} {
		unknown.SignalErrors["decision:"+name] = "unavailable"
	}
	tools := balanceAnswers("chat", 0.5, 0)
	tools.SignalValues["decision:needs:tools"] = 0.7
	tests := []struct {
		name string
		in   SignalResults
		want string
	}{
		{"unknown answers keep the middle pool", unknown, "medium"},
		{"no reasoning needed", balanceAnswers("chat", 0.5, 0), "simple"},
		{"some reasoning", balanceAnswers("analysis", 1.8, 0), "medium"},
		{"brief request lowers effort", balanceWith(balanceAnswers("analysis", 1.8, 0), nil, []string{"brief_request"}, nil), "simple"},
		{"high effort", balanceAnswers("analysis", 2.5, 0.2), "reasoning"},
		{"maximum effort", balanceAnswers("stem", 3.5, 1), "reasoning"},
		{"brief STEM problem rated multi-step", balanceWith(balanceAnswers("stem", 2, 0), nil, []string{"brief_request"}, nil), "reasoning"},
		{"code at high effort keeps its lane", balanceAnswers("code", 2.5, 0.2), "medium"},
		{"code at maximum effort", balanceAnswers("code", 3.5, 1), "reasoning"},
		{"exact facts at high effort keep their lane", balanceAnswers("facts", 2.5, 0.2, "precise_facts"), "medium"},
		{"exact facts are not simple", balanceAnswers("facts", 0.5, 0, "precise_facts"), "medium"},
		{"exact facts at maximum effort", balanceAnswers("facts", 3.5, 1, "precise_facts"), "reasoning"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &tt.in, tt.want) })
	}
	for _, tt := range []struct {
		name string
		in   SignalResults
		want string
	}{
		{"active tool loop", balanceWith(balanceAnswers("chat", 0.5, 0), []string{"tool_loop"}, nil, nil), "reasoning"},
		{"tool loop with unknown answers", balanceWith(unknown, []string{"tool_loop"}, nil, nil), "reasoning"},
		{"declared tools for agentic work", balanceWith(balanceAnswers("agentic", 0.5, 0), []string{"has_tools"}, nil, nil), "reasoning"},
		{"declared tools the answer needs", balanceWith(tools, []string{"has_tools"}, nil, nil), "reasoning"},
		{"tools disabled by the client", balanceWith(balanceAnswers("agentic", 0.5, 0), []string{"has_tools", "tool_disabled"}, nil, nil), "simple"},
		{"declared tools the request does not need", balanceWith(balanceAnswers("writing", 0.5, 0), []string{"has_tools"}, nil, nil), "simple"},
		{"correction and feedback after an answer", balanceWith(balanceAnswers("chat", 0.5, 0, "correction"), []string{"has_prior_answer"}, nil, []string{"wrong_answer"}), "reasoning"},
		{"correction without the feedback judgment", balanceWith(balanceAnswers("chat", 0.5, 0, "correction"), []string{"has_prior_answer"}, nil, nil), "simple"},
		{"feedback without an earlier answer", balanceWith(balanceAnswers("chat", 0.5, 0, "correction"), nil, nil, []string{"wrong_answer"}), "simple"},
	} {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &tt.in, tt.want) })
	}
}

// Decision priority cannot substitute for the named tool's capability demand.
// Models here are synthetic metadata, not claims about a deployed checkpoint.
func TestBuiltinHardNamedToolRequiresCapableCandidate(t *testing.T) {
	for _, recipe := range []string{"balance", "speed", "cost", "accuracy"} {
		t.Run(recipe, func(t *testing.T) {
			c := builtinPolicyClassifier(t, recipe)
			in := &SignalResults{MatchedDecisionRules: []string{"depth"}, MatchedConversationRules: []string{"has_tools", "tool_required"}}
			if recipe == "balance" {
				answers := balanceWith(balanceAnswers("agentic", 2.5, 0.2), []string{"has_tools"}, nil, nil)
				in = &answers
			}
			result, err := c.EvaluateDecisionWithEngine(c.applyProjections(in))
			if err != nil || result == nil || result.Decision == nil || result.Decision.Name != "reasoning" {
				t.Fatalf("hard named-tool decision=%+v err=%v", result, err)
			}
			request := &llmprotocol.Request{
				Tools:      []llmprotocol.Tool{{Name: "read_state", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)}},
				ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "read_state"},
			}
			before, _ := json.Marshal(request)
			demand, err := selection.EffectiveCandidateDemand(request, result.Decision)
			if err != nil {
				t.Fatal(err)
			}
			after, _ := json.Marshal(request)
			if string(before) != string(after) || !demand.ModelCapabilities.Supports(llmprotocol.CapabilityTools) {
				t.Fatal("reasoning changed the ingress request or lost its tools requirement")
			}
			for _, capable := range []bool{false, true} {
				model := config.ModelParams{Capabilities: []string{"chat", "reasoning"}, ContextWindowSize: 32768, MaxOutputTokens: 8192}
				if capable {
					model.Capabilities = append(model.Capabilities, "tools")
				}
				err := selection.ValidateCandidateRequirements(c.Config.CandidateRequirements, "synthetic", model, demand)
				if capable && err != nil || !capable && !errors.Is(err, selection.ErrNoEligibleCandidates) {
					t.Fatalf("tools capability=%v admission error=%v", capable, err)
				}
			}
		})
	}
}

func TestBuiltinRecipesReplayDefaultsRespectOperatorOptOut(t *testing.T) {
	for _, name := range []string{"balance", "speed", "cost", "accuracy", "vault"} {
		t.Run(name, func(t *testing.T) {
			cfg := builtinPolicyClassifier(t, name).Config
			if !cfg.RouterReplay.Enabled || cfg.RouterReplay.StoreBackend != "postgres" || cfg.RouterReplay.TTLSeconds != 604800 {
				t.Fatal("built-in recipes must default to seven-day PostgreSQL Replay")
			}
			for _, route := range cfg.Decisions {
				if cfg.EffectiveRouterReplayConfig(&route) == nil {
					t.Fatalf("decision %q must inherit Replay capture", route.Name)
				}
			}
			cfg.RouterReplay.Enabled = false
			for _, route := range cfg.Decisions {
				if cfg.EffectiveRouterReplayConfig(&route) != nil {
					t.Fatalf("global opt-out must prevent default capture for decision %q", route.Name)
				}
			}
			if cfg.EffectiveRouterReplayConfig(nil) != nil {
				t.Fatal("operator opt-out must also prevent capture before decision selection")
			}
			for _, route := range cfg.Decisions {
				if cfg.EffectiveRouterReplayConfig(&route) != nil {
					t.Fatalf("operator opt-out must prevent capture for decision %q", route.Name)
				}
			}
		})
	}
}

func TestBuiltinVaultDataControlsAreIndependentOfVerdict(t *testing.T) {
	cfg := builtinPolicyClassifier(t, "vault").Config
	for _, route := range cfg.Decisions {
		t.Run(route.Name, func(t *testing.T) {
			tools := route.GetToolsConfig()
			memory := route.GetMemoryConfig()
			cache := route.GetResponseCacheConfig()
			if tools == nil || !tools.Enabled || tools.EffectiveMode() != config.ToolsPluginModeNone || !tools.StripToolHistory {
				t.Fatal("every Vault verdict must prevent client tool dispatch")
			}
			if memory == nil || memory.Enabled || cache == nil || cache.Enabled {
				t.Fatal("every Vault verdict must disable memory and response caching")
			}
			drop := false
			for _, emit := range route.Emits {
				if emit.Kind == "retention" && emit.Retention != nil && emit.Retention.Drop != nil && *emit.Retention.Drop {
					drop = true
				}
			}
			if !drop || route.Adaptations.EffectiveMode() != config.DecisionAdaptationModeBypass {
				t.Fatal("every Vault verdict must suppress content writes and adaptation")
			}
		})
	}
}

func assertBuiltinPolicy(t *testing.T, c *Classifier, input *SignalResults, want string) {
	t.Helper()
	result, err := c.EvaluateDecisionWithEngine(c.applyProjections(input))
	if want == "" {
		if result != nil || !errors.Is(err, decision.ErrDecisionUnresolved) {
			t.Fatalf("wanted unresolved policy, got result=%+v err=%v", result, err)
		}
		return
	}
	if err != nil || result == nil || result.Decision == nil || result.Decision.Name != want {
		t.Fatalf("wanted %q, got result=%+v err=%v", want, result, err)
	}
}

package classification

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
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

// These are policy counterexamples, not classifier accuracy measurements. Read
// the shipped recipe and run its actual projections and decision evaluator.
func TestBuiltinBalanceIntentAndRecovery(t *testing.T) {
	c := builtinPolicyClassifier(t, "balance")
	tests := []struct {
		name string
		in   SignalResults
		want string
	}{
		{"uncertain task keeps middle pool", SignalResults{}, "medium"},
		{"positive simple evidence", SignalResults{MatchedComplexityRules: []string{"difficulty:easy"}}, "simple"},
		{"hard task", SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}}, "reasoning"},
		{"concise answer to a hard task", SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}, MatchedKeywordRules: []string{"deliberate", "no_analysis"}}, "reasoning"},
		{"quoted execution vocabulary", SignalResults{MatchedKeywordRules: []string{"deliberate", "verify"}, MatchedStructureRules: []string{"quoted_request"}}, "medium"},
		{"topic is not difficulty", SignalResults{MatchedComplexityRules: []string{"difficulty:easy"}, MatchedDomainRules: []string{"health"}, MatchedFactCheckRules: []string{"needs_fact_check"}}, "simple"},
		{"consequential advice combines evidence", SignalResults{MatchedComplexityRules: []string{"difficulty:easy"}, MatchedDomainRules: []string{"health"}, MatchedFactCheckRules: []string{"needs_fact_check"}, SignalValues: map[string]float64{"embedding:consequential": 0.6, "embedding:informational": 0.4}}, "reasoning"},
		{"financial stakes include economics", SignalResults{MatchedDomainRules: []string{"economics"}, MatchedFactCheckRules: []string{"needs_fact_check"}, SignalValues: map[string]float64{"embedding:consequential": 0.6, "embedding:informational": 0.4}}, "reasoning"},
		{"repeat alone is not a failed answer", SignalResults{MatchedConversationRules: []string{"has_answer"}, MatchedReaskRules: []string{"repeat"}}, "medium"},
		{"negative feedback after answer", SignalResults{MatchedConversationRules: []string{"has_answer"}, MatchedUserFeedbackRules: []string{"wrong_answer"}}, "reasoning"},
		{"feedback cannot replace missing assistant history", SignalResults{MatchedKeywordRules: []string{"correction"}, MatchedUserFeedbackRules: []string{"wrong_answer"}}, "medium"},
		{"repeat with correction", SignalResults{MatchedConversationRules: []string{"has_answer"}, MatchedReaskRules: []string{"repeat"}, MatchedKeywordRules: []string{"correction"}}, "reasoning"},
		{"quoted feedback is not recovery", SignalResults{MatchedConversationRules: []string{"has_answer"}, MatchedUserFeedbackRules: []string{"wrong_answer"}, MatchedStructureRules: []string{"quoted_request"}}, "medium"},
		{"classifier error does not fabricate low complexity", builtinComplexityUnavailable(c, SignalResults{}), "medium"},
		{"failed feedback propagates through projection", SignalResults{MatchedConversationRules: []string{"has_answer"}, MatchedReaskRules: []string{"repeat"}, MatchedKeywordRules: []string{"correction"}, SignalErrors: map[string]string{"user_feedback:wrong_answer": "unavailable"}}, "medium"},
		{"active tool conversation is not simple", SignalResults{MatchedComplexityRules: []string{"difficulty:easy"}, MatchedConversationRules: []string{"tool_loop"}}, "medium"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &tt.in, tt.want) })
	}
}

func TestBuiltinSpeedAndCostToolIntent(t *testing.T) {
	for _, profile := range []struct{ name, fallback string }{{"speed", "fast"}, {"cost", "economy"}} {
		t.Run(profile.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, profile.name)
			for _, tt := range []struct {
				name string
				in   SignalResults
				want string
			}{
				{"tools merely available", SignalResults{MatchedConversationRules: []string{"has_tools"}}, profile.fallback},
				{"explicit execution with definitions", SignalResults{MatchedConversationRules: []string{"has_tools"}, MatchedKeywordRules: []string{"tool_intent"}}, "tools"},
				{"tool intent without definitions", SignalResults{MatchedKeywordRules: []string{"tool_intent"}}, profile.fallback},
				{"required named tool", SignalResults{MatchedConversationRules: []string{"has_tools", "tool_required"}}, "tools"},
				{"tools disabled", SignalResults{MatchedConversationRules: []string{"has_tools", "tool_disabled"}, MatchedKeywordRules: []string{"tool_intent"}}, profile.fallback},
				{"quoted tool action", SignalResults{MatchedConversationRules: []string{"has_tools"}, MatchedKeywordRules: []string{"tool_intent"}, MatchedStructureRules: []string{"quoted_request"}}, profile.fallback},
				{"quoted correction is not answer recovery", SignalResults{MatchedConversationRules: []string{"has_answer"}, MatchedReaskRules: []string{"repeat"}, MatchedKeywordRules: []string{"correction"}, MatchedStructureRules: []string{"quoted_request"}}, profile.fallback},
				{"continue ordinary tool loop", SignalResults{MatchedConversationRules: []string{"tool_loop"}}, "tools"},
				{"hard task in a tool loop", SignalResults{MatchedConversationRules: []string{"tool_loop"}, MatchedComplexityRules: []string{"difficulty:hard"}}, "reasoning"},
				{"unknown difficulty preserves quality", builtinComplexityUnavailable(c, SignalResults{}), "reasoning"},
				{"concise answer preserves unknown policy", builtinComplexityUnavailable(c, SignalResults{MatchedKeywordRules: []string{"no_analysis"}}), "reasoning"},
			} {
				t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &tt.in, tt.want) })
			}
		})
	}
}

func TestBuiltinHardTasksPreservePresentationAndToolBoundaries(t *testing.T) {
	for _, profile := range []struct{ name, fallback string }{
		{"balance", "medium"}, {"speed", "fast"}, {"cost", "economy"}, {"accuracy", "simple"},
	} {
		t.Run(profile.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, profile.name)
			for _, test := range []struct {
				name string
				in   SignalResults
				want string
			}{
				{"concise hard task", SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}, MatchedKeywordRules: []string{"no_analysis"}}, "reasoning"},
				{"hard task with named tool", SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}, MatchedKeywordRules: []string{"no_analysis"}, MatchedConversationRules: []string{"has_tools", "tool_required"}}, "reasoning"},
				{"quoted instruction alone", SignalResults{MatchedKeywordRules: []string{"deliberate", "verify"}, MatchedStructureRules: []string{"quoted_request"}}, profile.fallback},
				{"hard analysis of quoted material", SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}, MatchedKeywordRules: []string{"deliberate", "verify"}, MatchedStructureRules: []string{"quoted_request"}}, "reasoning"},
				{"medium retains ordinary pool", SignalResults{MatchedComplexityRules: []string{"difficulty:medium"}, MatchedKeywordRules: []string{"no_analysis"}}, profile.fallback},
			} {
				t.Run(test.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &test.in, test.want) })
			}
		})
	}
}

// Decision priority cannot substitute for the named tool's capability demand.
// Models here are synthetic metadata, not claims about a deployed checkpoint.
func TestBuiltinHardNamedToolRequiresCapableCandidate(t *testing.T) {
	for _, recipe := range []string{"balance", "speed", "cost", "accuracy"} {
		t.Run(recipe, func(t *testing.T) {
			c := builtinPolicyClassifier(t, recipe)
			in := &SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}, MatchedKeywordRules: []string{"no_analysis"}, MatchedConversationRules: []string{"has_tools", "tool_required"}}
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

func TestBuiltinAccuracyRequiresExecutionIntent(t *testing.T) {
	c := builtinPolicyClassifier(t, "accuracy")
	for _, tt := range []struct {
		name string
		in   SignalResults
		want string
	}{
		{"strong single default", SignalResults{}, "simple"},
		{"hard does not imply fanout", SignalResults{MatchedComplexityRules: []string{"difficulty:hard"}}, "reasoning"},
		{"review keyword alone", SignalResults{MatchedKeywordRules: []string{"review"}}, "simple"},
		{"review semantics alone", SignalResults{MatchedEmbeddingRules: []string{"review_intent"}}, "simple"},
		{"explicit independent review", SignalResults{MatchedKeywordRules: []string{"plural_workers", "review_action", "independent_work"}}, "review"},
		{"single model vetoes review", SignalResults{MatchedKeywordRules: []string{"plural_workers", "review_action", "independent_work", "single"}}, "simple"},
		{"quoted workflow does not execute", SignalResults{MatchedKeywordRules: []string{"distributed_execution"}, MatchedStructureRules: []string{"quoted_request"}}, "simple"},
		{"explicit distribution", SignalResults{MatchedKeywordRules: []string{"distributed_execution", "plural_workers"}}, "agent"},
		{"distribution clue without plurality", SignalResults{MatchedKeywordRules: []string{"distributed_execution"}}, "simple"},
		{"client tool loop stays single", SignalResults{MatchedKeywordRules: []string{"distributed_execution", "plural_workers", "review_action", "independent_work"}, MatchedConversationRules: []string{"tool_loop"}}, "reasoning"},
		{"Router flow state resumes", SignalResults{MatchedConversationRules: []string{"flow", "tool_loop"}}, "agent"},
		{"workflow semantics unavailable does not fanout", SignalResults{MatchedKeywordRules: []string{"delegated_executors"}, SignalErrors: map[string]string{"embedding:workflow_intent": "unavailable"}}, "simple"},
		{"unknown difficulty stays strong single", builtinComplexityUnavailable(c, SignalResults{}), "reasoning"},
	} {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &tt.in, tt.want) })
	}
}

func TestBuiltinVaultUnknownTriageFailsClosed(t *testing.T) {
	c := builtinPolicyClassifier(t, "vault")
	for _, tt := range []struct {
		name string
		in   SignalResults
		want string
	}{
		{"ordinary private request", SignalResults{}, "private"},
		{"personal data", SignalResults{MatchedPIIRules: []string{"personal_data"}}, "sensitive"},
		{"explicit confidential request", SignalResults{MatchedKeywordRules: []string{"confidential"}}, "sensitive"},
		{"security outranks sensitivity", SignalResults{MatchedPIIRules: []string{"personal_data"}, MatchedJailbreakRules: []string{"prompt_attack"}}, "guard"},
		{"unsafe content receives responsible handling", SignalResults{MatchedSafetyRules: []string{"unsafe"}}, "sensitive"},
		{"unavailable Guard", SignalResults{SignalErrors: map[string]string{"jailbreak:prompt_attack": "unavailable"}}, ""},
		{"unavailable Safety", SignalResults{SignalErrors: map[string]string{"safety:unsafe": "unavailable"}}, ""},
		{"unavailable PII", SignalResults{SignalErrors: map[string]string{"pii:personal_data": "unavailable"}}, ""},
		{"content risk cannot resolve an unknown prompt attack", SignalResults{MatchedSafetyRules: []string{"unsafe"}, SignalErrors: map[string]string{"jailbreak:prompt_attack": "unavailable"}}, ""},
		{"hazard can identify care when binary safety is negative", SignalResults{MatchedClassifierRules: []string{"content-risk:self_harm"}}, "sensitive"},
		{"hazard failure is not evidence for ordinary handling", SignalResults{SignalErrors: map[string]string{"classifier:content-risk": "unavailable"}}, ""},
	} {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, &tt.in, tt.want) })
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
			cfg.RouterReplay.Enabled = true
			denyReplay := false
			cfg.DataPolicy = &config.RoutingDataPolicy{Replay: &denyReplay}
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

func builtinComplexityUnavailable(c *Classifier, input SignalResults) SignalResults {
	var mu sync.Mutex
	c.recordComplexityFailure(&input, &mu)
	return input
}

// These inputs exercise the shipped heuristic pipeline. Learned values are
// controlled component inputs, not predictions or claims about model accuracy.
func TestBuiltinAccuracyWorkflowAuthorizationScope(t *testing.T) {
	c := builtinPolicyClassifier(t, "accuracy")
	attachBuiltinPolicyHeuristics(t, c)
	for _, tt := range []struct {
		name, text, want, fact string
		margin                 float64
		unknown                bool
	}{
		{name: "explicit coordination", text: "Coordinate a workflow with multiple agents.", want: "agent"},
		{name: "explicit distribution", text: "Let two workers split the tasks.", want: "agent"},
		{name: "delegation needs semantic direction", text: "Assign three agents to the work.", margin: .1, want: "agent"},
		{name: "delegation tied evidence", text: "Assign three agents to the work.", want: "simple"},
		{name: "delegation unavailable evidence", text: "Assign three agents to the work.", unknown: true, want: "simple"},
		{name: "explaining is not executing", text: "Explain how two agents can divide the work.", margin: .1, want: "simple"},
		{name: "plan without execution", text: "Assign the work to two agents, but only describe the plan; do not run them.", margin: .1, want: "simple"},
		{name: "quotation is inert", text: "Quote: Coordinate a workflow with multiple agents.", margin: .1, want: "simple"},
		{name: "unrelated worker mention", text: "Several agents are available. Assign the work to one worker.", margin: .1, want: "simple"},
		{name: "active tool ownership", text: "Coordinate a workflow with multiple agents.", fact: "tool_loop", want: "reasoning"},
		{name: "required client tool", text: "Coordinate a workflow with multiple agents.", fact: "tool_required", want: "reasoning"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			in := evaluateBuiltinPolicyHeuristics(c, tt.text)
			in.SignalValues["embedding:workflow_intent"] = .3 + tt.margin
			in.SignalValues["embedding:informational"] = .3
			if tt.unknown {
				in.SignalErrors = map[string]string{"embedding:workflow_intent": "unavailable"}
			}
			if tt.fact != "" {
				in.MatchedConversationRules = []string{tt.fact}
			}
			if tt.unknown && c.applyProjections(in).SignalErrors["projection:execution_direction"] == "" {
				t.Fatal("unavailable learned input must remain unknown through projection")
			}
			assertBuiltinPolicy(t, c, in, tt.want)
		})
	}
}

func TestBuiltinAccuracyTranslationPreservesOuterScope(t *testing.T) {
	c := builtinPolicyClassifier(t, "accuracy")
	attachBuiltinPolicyHeuristics(t, c)
	for _, tt := range []struct{ language, prefix, instruction string }{
		{"en", "Translate: ", "Coordinate a workflow with multiple agents."},
		{"zh", "翻译：", "协调多个智能体完成工作。"},
		{"ar", "ترجم: ", "نسق العمل بين وكيلين."},
		{"de", "Übersetze: ", "Koordiniere die Arbeit mit zwei Agenten."},
		{"es", "Traduce: ", "Coordina un flujo de trabajo con dos agentes."},
		{"ja", "翻訳してください：", "複数のエージェントを連携させてください。"},
	} {
		t.Run(tt.language, func(t *testing.T) {
			assertBuiltinPolicy(t, c, evaluateBuiltinPolicyHeuristics(c, tt.instruction), "agent")
			in := evaluateBuiltinPolicyHeuristics(c, tt.prefix+tt.instruction)
			if !slices.Contains(in.MatchedStructureRules, "quoted_request") {
				t.Fatal("actual structure evaluator must identify the outer translation scope")
			}
			assertBuiltinPolicy(t, c, in, "simple")
		})
	}
}

func TestBuiltinCareAndSupportRequireGrounding(t *testing.T) {
	for _, tt := range []struct {
		name, recipe, want string
		in                 SignalResults
	}{
		{"care with either corroborator", "balance", "reasoning", SignalResults{MatchedFactCheckRules: []string{"needs_fact_check"}, SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:informational": .4}}},
		{"care direction without corroboration", "balance", "medium", SignalResults{SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:informational": .4}}},
		{"informational topic", "balance", "medium", SignalResults{MatchedDomainRules: []string{"health"}, SignalValues: map[string]float64{"embedding:consequential": .4, "embedding:informational": .6}}},
		{"care unavailable direction", "balance", "medium", SignalResults{MatchedDomainRules: []string{"health"}, SignalValues: map[string]float64{"embedding:consequential": .6}, SignalErrors: map[string]string{"embedding:informational": "unavailable"}}},
		{"weak support direction alone", "vault", "private", SignalResults{SignalValues: map[string]float64{"embedding:supportive_intent": .2, "embedding:informational": .1}}},
		{"support cue plus direction", "vault", "sensitive", SignalResults{MatchedKeywordRules: []string{"personal_support"}, SignalValues: map[string]float64{"embedding:supportive_intent": .6, "embedding:informational": .5}}},
		{"support unavailable direction", "vault", "", SignalResults{MatchedKeywordRules: []string{"personal_support"}, SignalErrors: map[string]string{"embedding:supportive_intent": "unavailable", "embedding:informational": "unavailable"}}},
		{"vault care direction with domain", "vault", "sensitive", SignalResults{MatchedDomainRules: []string{"health"}, SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:informational": .4}}},
		{"vault explicit care with domain", "vault", "sensitive", SignalResults{MatchedKeywordRules: []string{"care_request"}, MatchedDomainRules: []string{"law"}}},
		{"vault care direction without domain", "vault", "private", SignalResults{SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:informational": .4}}},
		{"vault informational domain", "vault", "private", SignalResults{MatchedDomainRules: []string{"health"}, SignalValues: map[string]float64{"embedding:consequential": .4, "embedding:informational": .6}}},
		{"vault quoted care wording", "vault", "private", SignalResults{MatchedKeywordRules: []string{"care_request"}, MatchedDomainRules: []string{"law"}, MatchedStructureRules: []string{"quoted_request"}, SignalValues: map[string]float64{"embedding:consequential": .4, "embedding:informational": .6}}},
		{"vault actionable care with quoted material", "vault", "sensitive", SignalResults{MatchedDomainRules: []string{"law"}, MatchedStructureRules: []string{"quoted_request"}, SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:informational": .4}}},
		{"vault personal action with both directions", "vault", "sensitive", SignalResults{MatchedKeywordRules: []string{"care_request"}, MatchedDomainRules: []string{"other"}, SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:supportive_intent": .6, "embedding:informational": .4}}},
		{"vault personal action alongside quotation", "vault", "sensitive", SignalResults{MatchedKeywordRules: []string{"care_request"}, MatchedDomainRules: []string{"other"}, MatchedStructureRules: []string{"quoted_request"}, SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:supportive_intent": .6, "embedding:informational": .4}}},
		{"vault both directions without personal action", "vault", "private", SignalResults{SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:supportive_intent": .6, "embedding:informational": .4}}},
		{"vault personal action without support direction", "vault", "private", SignalResults{MatchedKeywordRules: []string{"care_request"}, SignalValues: map[string]float64{"embedding:consequential": .6, "embedding:supportive_intent": .3, "embedding:informational": .4}}},
		{"vault personal action without care direction", "vault", "private", SignalResults{MatchedKeywordRules: []string{"care_request"}, SignalValues: map[string]float64{"embedding:consequential": .3, "embedding:supportive_intent": .6, "embedding:informational": .4}}},
		{"vault quoted personal action without directions", "vault", "private", SignalResults{MatchedKeywordRules: []string{"care_request"}, MatchedStructureRules: []string{"quoted_request"}, SignalValues: map[string]float64{"embedding:consequential": .3, "embedding:supportive_intent": .3, "embedding:informational": .6}}},
		{"vault care unavailable direction", "vault", "", SignalResults{MatchedDomainRules: []string{"health"}, SignalErrors: map[string]string{"embedding:consequential": "unavailable", "embedding:informational": "unavailable"}}},
		{"vault quoted safety still protected", "vault", "sensitive", SignalResults{MatchedStructureRules: []string{"quoted_request"}, MatchedSafetyRules: []string{"unsafe"}}},
		{"vault quoted hazard still protected", "vault", "sensitive", SignalResults{MatchedStructureRules: []string{"quoted_request"}, MatchedClassifierRules: []string{"content-risk:weapons"}}},
		{"vault prompt attack retains priority", "vault", "guard", SignalResults{MatchedKeywordRules: []string{"care_request"}, MatchedDomainRules: []string{"health"}, MatchedStructureRules: []string{"quoted_request"}, MatchedSafetyRules: []string{"unsafe"}, MatchedJailbreakRules: []string{"prompt_attack"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assertBuiltinPolicy(t, builtinPolicyClassifier(t, tt.recipe), &tt.in, tt.want)
		})
	}
}

func TestBuiltinConsequentialDomainsWithOptionalFactCheck(t *testing.T) {
	for _, recipe := range []string{"balance", "accuracy"} {
		c := builtinPolicyClassifier(t, recipe)
		for _, domain := range []string{"health", "law", "business", "economics"} {
			for _, evidence := range []string{"domain", "domain_and_fact_check"} {
				t.Run(recipe+"/"+domain+"/"+evidence, func(t *testing.T) {
					in := SignalResults{
						MatchedComplexityRules: []string{"difficulty:easy"},
						MatchedDomainRules:     []string{domain},
						SignalValues: map[string]float64{
							"embedding:consequential": .6,
							"embedding:informational": .4,
						},
					}
					if evidence == "domain_and_fact_check" {
						in.MatchedFactCheckRules = []string{"needs_fact_check"}
					}
					informational := in
					informational.SignalValues = map[string]float64{
						"embedding:consequential": .4,
						"embedding:informational": .6,
					}
					assertBuiltinPolicy(t, c, &in, "reasoning")
					assertBuiltinPolicy(t, c, &informational, "simple")
				})
			}
		}
	}
}

func TestBuiltinSpeedReasoningWinsMatchedToolBranch(t *testing.T) {
	c := builtinPolicyClassifier(t, "speed")
	for _, cue := range []string{"difficulty:hard", "deliberate"} {
		t.Run(cue, func(t *testing.T) {
			in := SignalResults{
				MatchedKeywordRules:      []string{"tool_intent"},
				MatchedConversationRules: []string{"has_tools"},
			}
			if cue == "difficulty:hard" {
				in.MatchedComplexityRules = []string{cue}
			} else {
				in.MatchedKeywordRules = append(in.MatchedKeywordRules, cue)
			}
			result, traces, err := c.EvaluateDecisionWithEngineAndTrace(c.applyProjections(&in))
			if err != nil || result == nil || result.Decision == nil || result.Decision.Name != "reasoning" {
				t.Fatalf("reasoning must win the tool collision: result=%+v err=%v", result, err)
			}
			for _, branch := range []string{"tools", "reasoning"} {
				if !slices.ContainsFunc(traces, func(trace decision.DecisionTrace) bool {
					return trace.DecisionName == branch && trace.Matched
				}) {
					t.Errorf("expected matching %s branch in %+v", branch, traces)
				}
			}
		})
	}
}

func TestBuiltinVaultActionWordingPreservesQuotationScope(t *testing.T) {
	c := builtinPolicyClassifier(t, "vault")
	attachBuiltinPolicyHeuristics(t, c)
	for _, tt := range []struct{ language, prefix, text string }{
		{"en", "Translate: ", "Guide me on how to proceed before I accept this settlement."},
		{"zh", "翻译：", "请帮我判断下一步是否应该签署这份协议。"},
	} {
		t.Run(tt.language, func(t *testing.T) {
			for _, quoted := range []bool{false, true} {
				text, want := tt.text, "sensitive"
				if quoted {
					text, want = tt.prefix+text, "private"
				}
				in := evaluateBuiltinPolicyHeuristics(c, text)
				if !slices.Contains(in.MatchedKeywordRules, "care_request") {
					t.Fatalf("action wording was not recognized: %q", text)
				}
				if slices.Contains(in.MatchedStructureRules, "quoted_request") != quoted {
					t.Fatalf("quotation scope mismatch: %q", text)
				}
				// Domain evidence is supplied; this exercises actual wording and
				// projection/decision logic without measuring a domain model.
				in.MatchedDomainRules = []string{"law"}
				assertBuiltinPolicy(t, c, in, want)
			}
		})
	}
}

func TestBuiltinRecoveryUsesAssistantHistory(t *testing.T) {
	for _, profile := range []struct{ name, fallback string }{{"balance", "medium"}, {"cost", "economy"}, {"accuracy", "simple"}} {
		t.Run(profile.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, profile.name)
			if len(c.Config.UserFeedbackRules) == 0 {
				t.Fatal("answer recovery requires the configured Feedback signal")
			}
			attachBuiltinPolicyHeuristics(t, c)
			for _, tt := range []struct {
				name, text, want  string
				history, feedback bool
			}{
				{name: "negative correctness feedback", history: true, feedback: true, want: "reasoning"},
				{name: "feedback cannot create history", feedback: true, want: profile.fallback},
				{name: "self-contained correction", text: "The answer is incorrect. Correct the answer.", want: "reasoning"},
				{name: "ordinary editing", text: "Rewrite the answer in a warmer tone.", want: profile.fallback},
				{name: "quoted correction", text: "Translate: The answer is incorrect. Correct the answer.", want: profile.fallback},
			} {
				t.Run(tt.name, func(t *testing.T) {
					in := evaluateBuiltinPolicyHeuristics(c, tt.text)
					if tt.history {
						in.MatchedConversationRules = []string{"has_answer"}
					}
					if tt.feedback {
						in.MatchedUserFeedbackRules = []string{"wrong_answer"}
					}
					assertBuiltinPolicy(t, c, in, tt.want)
				})
			}
		})
	}
}

func attachBuiltinPolicyHeuristics(t *testing.T, c *Classifier) {
	t.Helper()
	var err error
	c.keywordClassifier, err = NewKeywordClassifier(c.Config.KeywordRules)
	if err != nil {
		t.Fatal(err)
	}
	c.structureClassifier, err = NewStructureClassifier(c.Config.StructureRules)
	if err != nil {
		t.Fatal(err)
	}
}

func evaluateBuiltinPolicyHeuristics(c *Classifier, text string) *SignalResults {
	in := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}}
	var mu sync.Mutex
	c.evaluateKeywordSignal(in, &mu, text)
	c.evaluateStructureSignal(in, &mu, text)
	return in
}

func TestBuiltinVaultCareConfidenceRequiresDirectionsAndEitherStrongIntent(t *testing.T) {
	c := builtinPolicyClassifier(t, "vault")
	if len(c.Config.SafetyRules) != 1 || c.Config.SafetyRules[0].Threshold != .5 {
		t.Fatal("care confidence must preserve the strict unsafe operating point")
	}
	for _, tt := range []struct {
		name, want, unavailable      string
		support, consequential       float64
		care, quoted, unsafe, attack bool
	}{
		{name: "both absolute signals below boundary", support: .599, consequential: .599, care: true, want: "private"},
		{name: "support boundary is sufficient", support: .6, consequential: .5, care: true, want: "sensitive"},
		{name: "consequence boundary is sufficient", support: .5, consequential: .6, care: true, want: "sensitive"},
		{name: "strong support still needs consequence direction", support: .6, consequential: .3, care: true, want: "private"},
		{name: "strong consequence still needs support direction", support: .3, consequential: .6, care: true, want: "private"},
		{name: "confidence still requires action cue", support: .6, consequential: .6, want: "private"},
		{name: "action and quotation can coexist", support: .5, consequential: .6, care: true, quoted: true, want: "sensitive"},
		{name: "strict unsafe does not require care confidence", support: .3, consequential: .3, unsafe: true, want: "sensitive"},
		{name: "support error remains unknown", support: .5, consequential: .6, care: true, unavailable: "supportive_intent", want: ""},
		{name: "consequence error remains unknown", support: .6, consequential: .5, care: true, unavailable: "consequential", want: ""},
		{name: "Guard keeps priority", support: .5, consequential: .6, care: true, attack: true, want: "guard"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			in := &SignalResults{SignalValues: map[string]float64{"embedding:consequential": tt.consequential, "embedding:informational": .4, "embedding:supportive_intent": tt.support}}
			if tt.care {
				in.MatchedKeywordRules = []string{"care_request"}
			}
			if tt.quoted {
				in.MatchedStructureRules = []string{"quoted_request"}
			}
			if tt.unsafe {
				in.MatchedSafetyRules = []string{"unsafe"}
				in.SignalValues["safety:unsafe"] = .5
			}
			if tt.attack {
				in.MatchedJailbreakRules = []string{"prompt_attack"}
			}
			if tt.unavailable != "" {
				in.SignalErrors = map[string]string{"embedding:" + tt.unavailable: "unavailable"}
			}
			assertBuiltinPolicy(t, c, in, tt.want)
		})
	}
}

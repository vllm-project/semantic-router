package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestApplySemanticReasoningMode(t *testing.T) {
	router := newReasoningRouterForTest()
	decision := router.Config.GetDecisionByName("analysis")

	t.Run("selected effort mutates neutral request", func(t *testing.T) {
		request := testNeutralRequest("local/mistral-small-4", "analyze")
		if changed := router.applySemanticReasoningMode(request, true, decision); !changed {
			t.Fatal("enabled reasoning did not mutate the neutral request")
		}
		if request.ReasoningEffort != "high" {
			t.Fatalf("reasoning effort = %q", request.ReasoningEffort)
		}
	})

	t.Run("disabled preserves caller semantic effort", func(t *testing.T) {
		request := testNeutralRequest("local/mistral-small-4", "answer")
		request.ReasoningEffort = "low"
		if changed := router.applySemanticReasoningMode(request, false, decision); changed {
			t.Fatal("disabled reasoning unexpectedly mutated the request")
		}
		if request.ReasoningEffort != "low" {
			t.Fatalf("caller effort = %q", request.ReasoningEffort)
		}
	})

	t.Run("model without declared capability is unchanged", func(t *testing.T) {
		request := testNeutralRequest("plain-model", "answer")
		if changed := router.applySemanticReasoningMode(request, true, decision); changed {
			t.Fatal("reasoning was applied without a declared model family")
		}
		if request.ReasoningEffort != "" {
			t.Fatalf("unexpected reasoning effort %q", request.ReasoningEffort)
		}
	})
}

func TestModifyRequestBodyForEntrypointRoutingEncodesSemanticReasoning(t *testing.T) {
	router := newReasoningRouterForTest()
	decision := router.Config.GetDecisionByName("analysis")
	request := testNeutralRequest("entrypoint", "analyze")
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		SemanticRequest:     request,
		VSRSelectedDecision: decision,
	}

	body, err := router.modifyRequestBodyForEntrypointRouting(
		request, "local/mistral-small-4", decision.Name, true, ctx,
	)
	if err != nil {
		t.Fatalf("modify neutral routing request: %v", err)
	}
	engine, err := router.protocolEngine()
	if err != nil {
		t.Fatalf("construct protocol engine: %v", err)
	}
	decoded, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	if decoded.Model != "local/mistral-small-4" || decoded.ReasoningEffort != "high" {
		t.Fatalf("unexpected routed semantics: model=%q effort=%q", decoded.Model, decoded.ReasoningEffort)
	}
}

func TestGetReasoningEffortPrefersDecisionThenDefault(t *testing.T) {
	router := newReasoningRouterForTest()
	decision := router.Config.GetDecisionByName("analysis")
	if got := router.getReasoningEffort(decision, "local/mistral-small-4"); got != "high" {
		t.Fatalf("decision effort = %q", got)
	}
	if got := router.getReasoningEffort(router.Config.GetDecisionByName("defaulted"), "plain-model"); got != "medium" {
		t.Fatalf("default effort = %q", got)
	}
	if got := (&OpenAIRouter{}).getReasoningEffort(nil, "model"); got != "medium" {
		t.Fatalf("nil-config effort = %q", got)
	}
}

func TestGetModelReasoningFamily(t *testing.T) {
	router := newReasoningRouterForTest()
	family := router.getModelReasoningFamily("local/mistral-small-4")
	if family == nil || family.Type != config.ReasoningFamilyTypeTopLevelReasoningEffort {
		t.Fatalf("unexpected reasoning family: %#v", family)
	}
	if family := router.getModelReasoningFamily("plain-model"); family != nil {
		t.Fatalf("plain model unexpectedly has a reasoning family: %#v", family)
	}
}

func newReasoningRouterForTest() *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{
		RoutingScope: "test-recipe",
		IntelligentRouting: config.IntelligentRouting{
			ReasoningConfig: config.ReasoningConfig{
				DefaultReasoningEffort: "medium",
				ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
					"mistral": {
						Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort",
					},
				},
			},
			Decisions: []config.Decision{
				reasoningDecision("analysis", "local/mistral-small-4", "high"),
				{Name: "defaulted", ModelRefs: []config.ModelRef{{Model: "plain-model"}}},
			},
		},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"local/mistral-small-4": {ReasoningFamily: "mistral"},
			"plain-model":           {},
		}},
	}}
}

func reasoningDecision(name, model, effort string) config.Decision {
	return config.Decision{
		Name: name,
		ModelRefs: []config.ModelRef{{
			Model: model,
			ModelReasoningControl: config.ModelReasoningControl{
				UseReasoning: boolPtr(true), ReasoningEffort: effort,
			},
		}},
	}
}

func boolPtr(value bool) *bool { return &value }

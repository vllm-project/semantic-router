package extproc

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestLateResponsePreservesToolContinuationOwner(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "built-in", "latest", "mom-v1", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	recipe, ok := cfg.RecipeByName("speed")
	if !ok {
		t.Fatal("speed recipe is missing")
	}
	cfg = cfg.ConfigForRecipe(recipe)
	cfg.ModelConfig = map[string]config.ModelParams{"cheap": {}, "frontier": {}}
	registry := selection.NewRegistry()
	router := &OpenAIRouter{Config: cfg, ModelSelector: registry}
	selectRequest := func(decision *config.Decision, proposal, conversation string, tool bool) *RequestContext {
		t.Helper()
		ctx := routerLearningRequestContext("overlapping-session", conversation)
		ctx.Routing.SelectRecipe(recipe)
		ctx.VSRSelectedDecision = decision
		ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: tool}
		registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
			SelectedModel: proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
			AllScores: map[string]float64{proposal: 1},
		}})
		selected, _, selectErr := router.selectModelFromCandidates(&selection.SelectionContext{
			SessionID: ctx.SessionID, RecipeName: "speed", DecisionName: decision.Name,
			CandidateModels: []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}},
		}, nil, ctx)
		if selectErr != nil || selected == nil {
			t.Fatalf("selection failed: %v", selectErr)
		}
		// Model an accepted dispatch before delivering out-of-order responses.
		if err := commitAgenticSessionDecision(ctx); err != nil {
			t.Fatal(err)
		}
		ctx.RequestModel = selected.Model
		return ctx
	}
	// The new portable request changes the decision before the old response completes.
	old := selectRequest(&cfg.Decisions[0], "cheap", "conversation", false)
	latest := selectRequest(&cfg.Decisions[1], "frontier", "conversation", false)
	if old.RequestModel != "cheap" || latest.RequestModel != "frontier" {
		t.Fatalf("decision change did not establish the new owner: old=%s new=%s", old.RequestModel, latest.RequestModel)
	}
	recordSessionTurn(latest, responseUsageMetrics{promptTokens: 20}, sessiontelemetry.TurnPricing{})
	recordSessionTurn(old, responseUsageMetrics{promptTokens: 10}, sessiontelemetry.TurnPricing{})
	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(config.RoutingNamespaceKey("speed", latest.VSRLearningSessionID), time.Now())
	if !ok || snapshot.CurrentModel != "frontier" || snapshot.CumulativePromptTokens != 30 {
		t.Errorf("response completion changed ownership or lost usage: %+v, found=%t", snapshot, ok)
	}
	continuation := selectRequest(&cfg.Decisions[1], "cheap", "conversation", true)
	if continuation.RequestModel != "frontier" {
		t.Errorf("tool continuation switched to late response model %q", continuation.RequestModel)
	}
	fresh := selectRequest(&cfg.Decisions[1], "cheap", "new-conversation", false)
	if fresh.RequestModel != "cheap" {
		t.Errorf("new CID inherited the old owner: %q", fresh.RequestModel)
	}
}

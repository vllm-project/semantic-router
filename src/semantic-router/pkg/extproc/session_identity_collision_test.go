package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestRouterSessionIdentitySeparatesDecisionWriters(t *testing.T) {
	for _, tc := range []struct {
		name               string
		scope              string
		firstRecipe        config.RecipeName
		firstSession       string
		firstConversation  string
		secondRecipe       config.RecipeName
		secondSession      string
		secondConversation string
	}{
		{
			name: "missing_conversation_writer", scope: config.RouterLearningScopeConversation,
			firstSession: "team", firstConversation: "job", secondSession: "team/job",
		},
		{
			name: "default_and_named_recipe", scope: config.RouterLearningScopeSession,
			firstRecipe: config.DefaultRecipeName, firstSession: "speed::client",
			secondRecipe: "speed", secondSession: "client",
		},
		{
			name: "conversation_component_boundary", scope: config.RouterLearningScopeConversation,
			firstRecipe: "speed", firstSession: "team/run", firstConversation: "job",
			secondRecipe: "speed", secondSession: "team", secondConversation: "run/job",
		},
		{
			name: "literal_percent_escape", scope: config.RouterLearningScopeConversation,
			firstSession: "team/run", firstConversation: "job",
			secondSession: "team%2Frun", secondConversation: "job",
		},
		{
			name: "unicode_component_boundary", scope: config.RouterLearningScopeConversation,
			firstSession: "团队/运行", firstConversation: "任务",
			secondSession: "团队", secondConversation: "运行/任务",
		},
		{
			name: "recipe_component_boundary", scope: config.RouterLearningScopeConversation,
			firstRecipe: "speed::team", firstSession: "client", firstConversation: "job",
			secondRecipe: "speed", secondSession: "team::client", secondConversation: "job",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			registry := selection.NewRegistry()
			router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(tc.scope), ModelSelector: registry}
			selectRequest := func(recipe config.RecipeName, sid, cid, proposal string, tool bool, candidates []config.ModelRef) string {
				t.Helper()
				ctx := routerLearningRequestContext(sid, cid)
				ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: recipe})
				ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
				ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: tool}
				registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
					SelectedModel: proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
					AllScores: map[string]float64{proposal: 1},
				}})
				selCtx := router.buildSelectionContext(candidates, "choice", "", nil, "", nil, ctx)
				selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
				if err != nil || selected == nil {
					t.Fatalf("selection failed for recipe=%q SID=%q CID=%q: selected=%+v err=%v", recipe, sid, cid, selected, err)
				}
				// This identity test models successful dispatches, not just proposals.
				if err := commitAgenticSessionDecision(ctx); err != nil {
					t.Fatal(err)
				}
				if ctx.Headers["x-session-id"] != sid || ctx.Headers["x-conversation-id"] != cid {
					t.Fatal("storage identity changed public headers")
				}
				return selected.Model
			}
			both := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			if model := selectRequest(tc.firstRecipe, tc.firstSession, tc.firstConversation, "frontier", false, both); model != "frontier" {
				t.Fatalf("first request did not establish frontier: %q", model)
			}
			if model := selectRequest(tc.firstRecipe, tc.firstSession, tc.firstConversation, "cheap", true, both); model != "frontier" {
				t.Fatalf("same identity did not retain its tool owner before B: %q", model)
			}
			// B admits only cheap. Its legitimate decision must not overwrite A.
			if model := selectRequest(tc.secondRecipe, tc.secondSession, tc.secondConversation, "cheap", false, []config.ModelRef{{Model: "cheap"}}); model != "cheap" {
				t.Fatalf("second request did not select its admitted model: %q", model)
			}
			if model := selectRequest(tc.firstRecipe, tc.firstSession, tc.firstConversation, "cheap", true, both); model != "frontier" {
				t.Errorf("distinct request overwrote the first tool owner: got %q, want frontier", model)
			}
		})
	}
}

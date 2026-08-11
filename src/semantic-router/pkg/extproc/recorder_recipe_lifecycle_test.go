package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestStartRouterReplayPersistsResolvedRecipe(t *testing.T) {
	namedRecorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	otherRecorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{
			config.RoutingDecisionKey("named-recipe", "shared-route"): namedRecorder,
			config.RoutingDecisionKey("other-recipe", "shared-route"): otherRecorder,
		},
	}

	recipe := &config.RoutingRecipe{
		Name: "named-recipe",
		Profile: config.RoutingProfile{Decisions: []config.Decision{
			{Name: "shared-route", ModelRefs: []config.ModelRef{{Model: "model-a"}}},
		}},
	}
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	ctx := &RequestContext{
		RequestID:                "normal-recipe-request",
		OriginalRequestBody:      []byte(`{"model":"named-entrypoint","messages":[{"role":"user","content":"hello"}]}`),
		VSRSelectedDecision:      &recipe.Profile.Decisions[0],
		RouterReplayPluginConfig: &replayConfig,
	}
	ctx.Routing.SelectRecipe(recipe)

	router.startRouterReplay(ctx, "named-entrypoint", "model-a", "shared-route")

	if ctx.RouterReplayID == "" {
		t.Fatal("expected normal request lifecycle to create a replay record")
	}
	record, found := namedRecorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatalf("replay record %q not found in named recipe recorder", ctx.RouterReplayID)
	}
	if record.Recipe != "named-recipe" {
		t.Fatalf("record recipe = %q, want %q", record.Recipe, "named-recipe")
	}
	if record.Decision != "shared-route" {
		t.Fatalf("record decision = %q, want %q", record.Decision, "shared-route")
	}
	if records := otherRecorder.ListAllRecords(); len(records) != 0 {
		t.Fatalf("other recipe recorder received %d records, want 0", len(records))
	}
}

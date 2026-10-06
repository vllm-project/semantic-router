package extproc

import (
	"reflect"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestReplayStartupRespectsEachRecipeDataPolicy(t *testing.T) {
	yes, no := true, false
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"}}
	cfg.DataPolicy = &config.RoutingDataPolicy{Replay: &no}
	cfg.Recipes = []config.RoutingRecipe{
		replayPolicyRecipe(t, config.DefaultRecipeName, &no),
		replayPolicyRecipe(t, "inherit", nil),
		replayPolicyRecipe(t, "unrestricted", &yes),
		replayPolicyRecipe(t, "restricted", &no),
	}
	recorders, err := initializeIsolatedReplayRecorders(cfg, "memory")
	if err != nil {
		t.Fatal(err)
	}
	for _, recorder := range recorders {
		t.Cleanup(func() { _ = recorder.Close() })
	}
	if len(recorders) != 2 {
		t.Fatalf("initialized %d recorders, want only the two allowed recipes", len(recorders))
	}
	for _, recipe := range cfg.Recipes {
		_, exists := recorders[config.RoutingDecisionKey(recipe.Name, "ordinary")]
		if exists != recipe.Profile.DataPolicy.ReplayAllowed() {
			t.Fatalf("recipe %q recorder eligibility = %v", recipe.Name, exists)
		}
	}
}

func TestReplayDeniedRecipesDoNotInitializeSharedStorage(t *testing.T) {
	no := false
	cfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "redis"},
		Recipes:      []config.RoutingRecipe{replayPolicyRecipe(t, "restricted", &no)},
	}
	// Redis is intentionally unconfigured: a denied recipe must not attempt
	// service initialization even if its decision explicitly enables replay.
	recorders, fallback, err := initializeSharedReplayRecorders(cfg, "redis")
	if err != nil || len(recorders) != 0 || fallback != nil {
		t.Fatalf("denied recipe initialized shared storage: recorders=%d fallback=%v err=%v", len(recorders), fallback, err)
	}
}

func TestReplayDataPolicyCoversUnresolvedEntrypoint(t *testing.T) {
	yes, no := true, false
	for _, preference := range []*bool{nil, &yes, &no} {
		t.Run(replayPreferenceName(preference), func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(8, 60))
			t.Cleanup(func() { _ = recorder.Close() })
			recipe := replayPolicyRecipe(t, "isolated", preference)
			router := &OpenAIRouter{
				Config: &config.RouterConfig{
					RouterReplay: config.RouterReplayConfig{Enabled: true},
					Recipes:      []config.RoutingRecipe{recipe},
					Entrypoints:  []config.EntrypointMapping{{ModelNames: []string{"public-entry"}, Recipe: recipe.Name}},
				},
				ReplayRecorder: recorder,
			}
			ctx := replayPolicyRequest(t)
			router.resolveEntrypointForRequest("public-entry", ctx)
			response := router.respondDecisionUnresolved(ctx, "public-entry", &decision.DecisionUnresolvedError{Decision: "ordinary"})
			if response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_ServiceUnavailable {
				t.Fatal("standing replay policy changed the fail_request result")
			}
			allowed := preference == nil || *preference
			if (ctx.RouterReplayID != "") != allowed || (len(recorder.ListAllRecords()) > 0) != allowed {
				t.Fatalf("unresolved request replay: id=%q records=%d allowed=%v", ctx.RouterReplayID, len(recorder.ListAllRecords()), allowed)
			}
			if !allowed && (ctx.SessionID != "" || ctx.RouterReplayPluginConfig != nil) {
				t.Fatal("denied request reached replay session/policy preparation")
			}
			if ctx.VSRSelectedDecision != nil || !router.Config.RouterReplay.Enabled {
				t.Fatal("replay policy fabricated a selected decision or changed global configuration")
			}
		})
	}
}

func TestReplayCaptureGateCannotUseAnotherRecipesRecorder(t *testing.T) {
	no := false
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(8, 60))
	t.Cleanup(func() { _ = recorder.Close() })
	router := &OpenAIRouter{ReplayRecorder: recorder}
	recipe := replayPolicyRecipe(t, "restricted", &no)
	ctx := replayPolicyRequest(t)
	ctx.Routing.SelectRecipe(&recipe)
	// Even a previously resolved enabled plugin and an available shared
	// recorder cannot override the selected recipe's standing denial.
	enabled := config.DefaultRouterReplayPluginConfig()
	ctx.RouterReplayPluginConfig = &enabled
	router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
	if ctx.RouterReplayID != "" || ctx.SessionID != "" || len(recorder.ListAllRecords()) != 0 {
		t.Fatal("capture crossed the recipe policy before a record was constructed")
	}
}

func TestReplayRequestPolicyDoesNotInheritAnotherRecipe(t *testing.T) {
	no := false
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true}}
	cfg.DataPolicy = &config.RoutingDataPolicy{Replay: &no}
	router := &OpenAIRouter{Config: cfg}
	ctx := replayPolicyRequest(t)
	allowed := replayPolicyRecipe(t, "other", nil)
	ctx.Routing.SelectRecipe(&allowed)
	if router.effectiveReplayConfigForRequest(ctx, &allowed.Profile.Decisions[0]) == nil {
		t.Fatal("named recipe inherited the default recipe's denial")
	}
	ctx.Routing.SelectPassthrough()
	if router.effectiveReplayConfigForRequest(ctx, nil) == nil {
		t.Fatal("concrete backend request inherited an unselected recipe's policy")
	}
	ctx.Routing.SelectRecipe(nil)
	if router.effectiveReplayConfigForRequest(ctx, nil) != nil {
		t.Fatal("programmatic scoped router lost its standing default policy")
	}
}

func TestReplayDataPolicyBlocksResponseCaptureWithExistingBinding(t *testing.T) {
	no := false
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(8, 60))
	t.Cleanup(func() { _ = recorder.Close() })
	router := &OpenAIRouter{ReplayRecorder: recorder}
	ctx := replayPolicyRequest(t)
	enabled := config.DefaultRouterReplayPluginConfig()
	ctx.RouterReplayPluginConfig = &enabled
	router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
	if ctx.RouterReplayID == "" {
		t.Fatal("test did not establish an existing recorder binding")
	}
	before, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		t.Fatal("initial record missing")
	}
	recipe := replayPolicyRecipe(t, "restricted", &no)
	ctx.Routing.SelectRecipe(&recipe)
	router.attachRouterReplayResponse(ctx, []byte(`{"text":"Meeting summary."}`), true)
	after, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok || !reflect.DeepEqual(before, after) {
		t.Fatal("standing denial allowed response capture through an existing binding")
	}
}

func replayPolicyRecipe(t *testing.T, name config.RecipeName, replay *bool) config.RoutingRecipe {
	t.Helper()
	payload, err := config.NewStructuredPayload(config.RouterReplayPluginConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	return config.RoutingRecipe{
		Name: name,
		Profile: config.RoutingProfile{
			DataPolicy: &config.RoutingDataPolicy{Replay: replay},
			Decisions: []config.Decision{{Name: "ordinary", Plugins: []config.DecisionPlugin{{
				Type: config.DecisionPluginRouterReplay, Configuration: payload,
			}}}},
		},
	}
}

func replayPolicyRequest(t *testing.T) *RequestContext {
	t.Helper()
	return &RequestContext{
		TraceContext: t.Context(), RequestID: "synthetic-replay-policy", SourceFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public-entry", Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Summarize the meeting agenda."}}},
		}},
	}
}

func replayPreferenceName(value *bool) string {
	if value == nil {
		return "unset"
	}
	if *value {
		return "true"
	}
	return "false"
}

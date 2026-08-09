package extproc

import (
	"slices"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func TestAuthenticatedLooperReplayUsesRecipeScopedDuplicateDecision(t *testing.T) {
	cfg := looperReplayRecipeConfig()
	recorders := initializeReplayRecorders(cfg)
	router := &OpenAIRouter{
		Config:          cfg,
		Cache:           &spyCache{},
		ReplayRecorders: recorders,
	}
	ctx := &RequestContext{
		Headers: map[string]string{},
	}

	requestHeaders := newRequestHeaders("POST", "/v1/chat/completions")
	requestHeaders.RequestHeaders.Headers.Headers = append(
		requestHeaders.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: headers.RequestID, Value: "looper-internal-recipe"},
		&core.HeaderValue{Key: headers.VSRLooperRequest, Value: "true"},
		&core.HeaderValue{Key: headers.VSRInternalAuth, Value: internalauth.Token()},
		&core.HeaderValue{Key: headers.VSRLooperDecision, Value: "shared-route"},
		&core.HeaderValue{Key: headers.VSRSelectedRecipe, Value: "second-recipe"},
	)
	headerResponse, err := router.handleRequestHeaders(requestHeaders, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders: %v", err)
	}
	if !ctx.LooperRequest {
		t.Fatal("valid internal token did not authenticate the looper request")
	}
	for _, internalHeader := range looperInternalContextHeaders {
		if removed := headerResponse.GetRequestHeaders().Response.HeaderMutation.RemoveHeaders; !slices.Contains(removed, internalHeader) {
			t.Fatalf("request header mutation retained internal header %q", internalHeader)
		}
	}

	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body: []byte(`{"model":"model-a","messages":[{"role":"user","content":"hello"}]}`),
		},
	}, ctx)
	if err != nil {
		t.Fatalf("handleRequestBody: %v", err)
	}
	if response == nil || response.GetRequestBody() == nil {
		t.Fatal("expected looper internal continue response")
	}
	for _, internalHeader := range looperInternalContextHeaders {
		if removed := response.GetRequestBody().Response.HeaderMutation.RemoveHeaders; !slices.Contains(removed, internalHeader) {
			t.Fatalf("physical backend dispatch retained internal header %q", internalHeader)
		}
	}

	assertLooperInternalReplayScope(
		t,
		ctx,
		recorders[config.RoutingDecisionKey("second-recipe", "shared-route")],
		recorders[config.RoutingDecisionKey("first-recipe", "shared-route")],
	)
}

func TestSpoofedLooperContextCannotExecuteRecipePlugins(t *testing.T) {
	cfg := looperReplayRecipeConfig()
	cfg.RouterReplay.Enabled = false
	cfg.Recipes[1].Profile.Decisions[0].Plugins = []config.DecisionPlugin{{
		Type: config.DecisionPluginFastResponse,
		Configuration: config.MustStructuredPayload(map[string]interface{}{
			"message": "internal-only response",
		}),
	}}
	cfg.BackendModels = config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"model-a": {PreferredEndpoints: []string{"model-backend"}},
		},
		VLLMEndpoints: []config.VLLMEndpoint{{
			Name:    "model-backend",
			Address: "127.0.0.1",
			Port:    8000,
			Type:    "vllm",
			Weight:  1,
		}},
	}
	router := &OpenAIRouter{
		Config: cfg,
		Cache:  &spyCache{},
	}
	router.CredentialResolver = newTestCredentialResolver(cfg)
	ctx := &RequestContext{Headers: map[string]string{}}

	requestHeaders := newRequestHeaders("POST", "/v1/chat/completions")
	requestHeaders.RequestHeaders.Headers.Headers = append(
		requestHeaders.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: headers.VSRLooperRequest, Value: "true"},
		&core.HeaderValue{Key: headers.VSRLooperDecision, Value: "shared-route"},
		&core.HeaderValue{Key: headers.VSRSelectedRecipe, Value: "second-recipe"},
	)
	headerResponse, err := router.handleRequestHeaders(requestHeaders, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders: %v", err)
	}
	if ctx.LooperRequest {
		t.Fatal("unauthenticated looper marker was trusted")
	}
	if got := ctx.Routing.SelectedRecipe(); got != nil {
		t.Fatalf("spoofed recipe hydrated routing context: %+v", got)
	}
	for _, internalHeader := range looperInternalContextHeaders {
		if removed := headerResponse.GetRequestHeaders().Response.HeaderMutation.RemoveHeaders; !slices.Contains(removed, internalHeader) {
			t.Fatalf("request header mutation retained internal header %q", internalHeader)
		}
	}

	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body: []byte(`{"model":"model-a","messages":[{"role":"user","content":"hello"}]}`),
		},
	}, ctx)
	if err != nil {
		t.Fatalf("handleRequestBody: %v", err)
	}
	if response.GetImmediateResponse() != nil {
		t.Fatal("spoofed looper context executed the recipe fast-response plugin")
	}
	if ctx.VSRSelectedDecision != nil {
		t.Fatalf("spoofed looper context selected decision %+v", ctx.VSRSelectedDecision)
	}
}

func assertLooperInternalReplayScope(
	t *testing.T,
	ctx *RequestContext,
	secondRecorder *routerreplay.Recorder,
	firstRecorder *routerreplay.Recorder,
) {
	t.Helper()
	if got := ctx.Routing.RecipeName(); got != "second-recipe" {
		t.Fatalf("hydrated recipe = %q, want %q", got, "second-recipe")
	}
	if ctx.VSRSelectedDecision == nil || ctx.VSRSelectedDecision.Tier != 2 {
		t.Fatalf("selected decision = %+v, want second recipe decision", ctx.VSRSelectedDecision)
	}
	if ctx.RouterReplayID == "" {
		t.Fatal("expected looper internal request to start router replay")
	}

	record, found := secondRecorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatalf("replay record %q not found in second recipe recorder", ctx.RouterReplayID)
	}
	if record.Recipe != "second-recipe" || record.Decision != "shared-route" {
		t.Fatalf("replay scope = recipe %q decision %q", record.Recipe, record.Decision)
	}
	if record.DecisionTier != 2 {
		t.Fatalf("replay decision tier = %d, want 2", record.DecisionTier)
	}

	if records := firstRecorder.ListAllRecords(); len(records) != 0 {
		t.Fatalf("first recipe recorder received %d records, want 0", len(records))
	}
}

func TestHydrateLooperRoutingContextPreservesResolvedRouting(t *testing.T) {
	cfg := looperReplayRecipeConfig()
	router := &OpenAIRouter{Config: cfg}
	firstRecipe, ok := cfg.RecipeByName("first-recipe")
	if !ok {
		t.Fatal("first recipe not found")
	}
	ctx := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			headers.VSRSelectedRecipe: "second-recipe",
		},
	}
	ctx.Routing.SelectRecipe(firstRecipe)

	router.hydrateLooperRoutingContext(ctx)

	if got := ctx.Routing.SelectedRecipe(); got != firstRecipe {
		t.Fatalf("resolved routing was overwritten: got %+v, want %+v", got, firstRecipe)
	}
}

func looperReplayRecipeConfig() *config.RouterConfig {
	return &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{
			Enabled:      true,
			StoreBackend: "memory",
		},
		Recipes: []config.RoutingRecipe{
			{
				Name: "first-recipe",
				Profile: config.RoutingProfile{Decisions: []config.Decision{
					{
						Name:      "shared-route",
						Tier:      1,
						ModelRefs: []config.ModelRef{{Model: "model-a"}},
					},
				}},
			},
			{
				Name: "second-recipe",
				Profile: config.RoutingProfile{Decisions: []config.Decision{
					{
						Name:      "shared-route",
						Tier:      2,
						ModelRefs: []config.ModelRef{{Model: "model-a"}},
					},
				}},
			},
		},
	}
}

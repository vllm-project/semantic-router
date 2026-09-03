package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestHandleLooperExecutionFailureIsVisibleAndTerminalInReplay(t *testing.T) {
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: "http://127.0.0.1:1"},
		},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
	}
	decision := &config.Decision{
		Name: "confidence-route",
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
		Algorithm: &config.AlgorithmConfig{Type: "confidence"},
	}
	request := testNeutralRequest("router-entrypoint", "hello")
	ctx := &RequestContext{
		RequestID:                "replay-looper-failure",
		Headers:                  map[string]string{},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RouterReplayPluginConfig: &replayConfig,
		VSRSelectedDecision:      decision,
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
	if err != nil {
		t.Fatalf("handleLooperExecution: %v", err)
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_InternalServerError {
		t.Fatalf("looper failure status = %v, want %v", got, typev3.StatusCode_InternalServerError)
	}
	if ctx.RouterReplayID == "" {
		t.Fatal("looper failure was not captured in replay")
	}
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)
	if headerMap[headers.RouterReplayID] != ctx.RouterReplayID {
		t.Fatalf("replay response header = %q, want %q", headerMap[headers.RouterReplayID], ctx.RouterReplayID)
	}
	record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatalf("replay record %q not found", ctx.RouterReplayID)
	}
	if record.ResponseStatus != 500 || record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("replay terminal state = status %d / %q, want 500 / %q", record.ResponseStatus, record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.SelectedModel != "" {
		t.Fatalf("failed selection recorded false final model %q", record.SelectedModel)
	}
	if record.EndedAt == nil || record.TerminalReason != "looper_execution_failed" {
		t.Fatalf("replay terminal metadata = ended_at %v / reason %q", record.EndedAt, record.TerminalReason)
	}
}

func TestHandleConfidencePartialFailurePersistsAccountingWithoutCandidateBody(t *testing.T) {
	server := newConfidencePartialFailureServer(t)
	defer server.Close()

	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: server.URL},
			BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
				"small": {ParamSize: "1b"},
				"large": {ParamSize: "10b"},
			}},
		},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
	}
	decision := &config.Decision{
		Name:      "confidence-partial-route",
		ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
		Algorithm: &config.AlgorithmConfig{
			Type: "confidence",
			Confidence: &config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: "avg_logprob",
				OnError:          "fail",
			},
		},
	}
	request := testNeutralRequest("router-entrypoint", "hello")
	ctx := &RequestContext{
		RequestID:                "replay-confidence-partial",
		Headers:                  map[string]string{},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RouterReplayPluginConfig: &replayConfig,
		VSRSelectedDecision:      decision,
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
	if err != nil {
		t.Fatalf("handleLooperExecution: %v", err)
	}
	record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	assertConfidencePartialFailureReplay(t, response, record, found, ctx.RouterReplayID)
}

func newConfidencePartialFailureServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var requestBody struct {
			Model    string `json:"model"`
			Logprobs bool   `json:"logprobs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if !requestBody.Logprobs {
			t.Errorf("confidence call did not request logprobs")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chatcmpl-partial-confidence", "object": "chat.completion", "created": 1,
			"model": requestBody.Model,
			"choices": []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role": "assistant", "content": "candidate private answer must not escape",
				},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
		})
	}))
}

func assertConfidencePartialFailureReplay(
	t *testing.T,
	response *ext_proc.ProcessingResponse,
	record store.Record,
	found bool,
	replayID string,
) {
	t.Helper()
	assertConfidenceFailureResponse(t, response)
	assertPartialConfidenceRecord(t, record, found, replayID)
}

func assertConfidenceFailureResponse(t *testing.T, response *ext_proc.ProcessingResponse) {
	t.Helper()
	immediate := response.GetImmediateResponse()
	if immediate.GetStatus().GetCode() != typev3.StatusCode_InternalServerError ||
		strings.Contains(string(immediate.Body), "candidate private answer") {
		t.Fatalf("unsafe confidence failure response: status=%v body=%s", immediate.GetStatus().GetCode(), immediate.Body)
	}
}

func assertPartialConfidenceRecord(t *testing.T, record store.Record, found bool, replayID string) {
	t.Helper()
	if !found || record.SelectedModel != "" {
		t.Fatalf("replay record %q missing or recorded false final model %q", replayID, record.SelectedModel)
	}
	assertPartialConfidenceUsage(t, record)
	assertPartialConfidenceTrace(t, record)
}

func assertPartialConfidenceUsage(t *testing.T, record store.Record) {
	t.Helper()
	if record.PromptTokens == nil || *record.PromptTokens != 5 ||
		record.CompletionTokens == nil || *record.CompletionTokens != 2 ||
		record.TotalTokens == nil || *record.TotalTokens != 7 {
		t.Fatalf("replay usage = prompt %v completion %v total %v, want 5/2/7", record.PromptTokens, record.CompletionTokens, record.TotalTokens)
	}
}

func assertPartialConfidenceTrace(t *testing.T, record store.Record) {
	t.Helper()
	if record.SelectionMethod != "confidence" || record.RouteDiagnostics == nil ||
		!strings.Contains(record.RouteDiagnostics.SelectionReasoning, "1 call attempts") ||
		!strings.Contains(record.RouteDiagnostics.SelectionReasoning, "small") {
		t.Fatalf("replay confidence trace = method %q diagnostics %+v", record.SelectionMethod, record.RouteDiagnostics)
	}
	if strings.Contains(record.ResponseBody, "candidate private answer") {
		t.Fatalf("replay response body leaked candidate answer: %q", record.ResponseBody)
	}
}

func TestHandleLooperExecutionRecordsFinalModelInReplay(t *testing.T) {
	server := newRatingsReplayServer()
	defer server.Close()

	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: server.URL},
		},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
	}
	decision := &config.Decision{
		Name: "ratings-route",
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
		Algorithm: &config.AlgorithmConfig{Type: "ratings"},
	}
	request := testNeutralRequest("router-entrypoint", "hello")
	ctx := &RequestContext{
		RequestID:                "replay-final-model",
		Headers:                  map[string]string{},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RouterReplayPluginConfig: &replayConfig,
		VSRSelectedDecision:      decision,
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
	if err != nil {
		t.Fatalf("handleLooperExecution: %v", err)
	}
	record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	assertFinalLooperReplay(t, response, ctx, record, found)
}

func newRatingsReplayServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chatcmpl-replay-model", "object": "chat.completion", "created": 1, "model": "backend-model",
			"choices": []map[string]interface{}{{
				"index": 0, "message": map[string]interface{}{"role": "assistant", "content": "ok"}, "finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
		})
	}))
}

func assertFinalLooperReplay(
	t *testing.T,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
	record store.Record,
	found bool,
) {
	t.Helper()
	assertLooperResponseFinalModel(t, response, ctx)
	assertReplayFinalModel(t, record, found)
	assertReplayAggregateUsage(t, record)
}

func assertLooperResponseFinalModel(t *testing.T, response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	t.Helper()
	if response == nil || response.GetImmediateResponse() == nil || ctx.RouterReplayID == "" {
		t.Fatalf("expected immediate response and replay record: response=%+v replay_id=%q", response, ctx.RouterReplayID)
	}
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)
	if ctx.VSRSelectedModel != "model-b" || headerMap[headers.VSRSelectedModel] != "model-b" {
		t.Fatalf("final model mismatch: context=%q headers=%q", ctx.VSRSelectedModel, headerMap[headers.VSRSelectedModel])
	}
}

func assertReplayFinalModel(t *testing.T, record store.Record, found bool) {
	t.Helper()
	if !found || record.SelectedModel != "model-b" || record.RouteDiagnostics == nil ||
		record.RouteDiagnostics.SelectedModel != "model-b" {
		t.Fatalf("replay selected-model contract failed: found=%v record=%+v", found, record)
	}
}

func assertReplayAggregateUsage(t *testing.T, record store.Record) {
	t.Helper()
	if record.PromptTokens == nil || *record.PromptTokens != 6 ||
		record.CompletionTokens == nil || *record.CompletionTokens != 4 ||
		record.TotalTokens == nil || *record.TotalTokens != 10 {
		t.Fatalf("replay aggregate usage = prompt %v completion %v total %v, want 6/4/10", record.PromptTokens, record.CompletionTokens, record.TotalTokens)
	}
	if record.ActualCost != nil || record.BaselineCost != nil || record.CostSavings != nil {
		t.Fatalf("looper aggregate usage invented cross-model cost: actual %v baseline %v savings %v", record.ActualCost, record.BaselineCost, record.CostSavings)
	}
	if record.LifecycleState != routerreplay.LifecycleCompleted || record.EndedAt == nil {
		t.Fatalf("replay lifecycle = %q / %v, want completed with end time", record.LifecycleState, record.EndedAt)
	}
}

func TestAuthenticatedLooperReplayUsesRecipeScopedDuplicateDecision(t *testing.T) {
	cfg := looperReplayRecipeConfig()
	recorders := initializeReplayRecorders(cfg)
	router := &OpenAIRouter{
		Config:             cfg,
		Cache:              &spyCache{},
		ReplayRecorders:    recorders,
		CredentialResolver: newTestCredentialResolver(cfg),
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
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {PreferredEndpoints: []string{"model-backend"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "model-backend", Address: "127.0.0.1", Port: 8000, Type: "vllm", Weight: 1,
			}},
		},
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

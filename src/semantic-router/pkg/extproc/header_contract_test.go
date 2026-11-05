package extproc

import (
	"maps"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// The registry in pkg/headers gives every header a direction and a surface.
// These tests drive the paths that answer a client and the paths that build
// a provider-bound request, and hold each header they write to its entry.

func headerContractRequest(extra map[string]string) *ext_proc.ProcessingRequest_RequestHeaders {
	request := newRequestHeaders("POST", "/v1/chat/completions")
	for name, value := range extra {
		request.RequestHeaders.Headers.Headers = append(request.RequestHeaders.Headers.Headers, &core.HeaderValue{Key: name, Value: value})
	}
	return request
}

// recordResponseHeaderFacts sets every fact a response header reports, so
// each path writes all the headers it can.
func recordResponseHeaderFacts(ctx *RequestContext) {
	drop, keep, prefer, turns := true, true, false, 3
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "support"})
	ctx.ConfigVersion = 7
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1
	ctx.ProtocolDiagnostics = llmprotocol.Diagnostics{{Field: "metadata", Action: llmprotocol.DiagnosticDropped, Reason: "unsupported"}}
	ctx.VSRSelectedDecisionName = "escalate-extreme"
	ctx.VSRSelectedDecisionConfidence, ctx.VSRSelectedDecisionConfidenceScored = 0.91, true
	ctx.VSRSelectionMethod = "static"
	ctx.VSRSelectedModel = "reasoning-model"
	ctx.FallbackRecord = &fallback.ExecutionRecord{FinalStatus: "succeeded", Attempts: make([]fallback.AttemptOutcome, 2)}
	ctx.RoutingLatency = 412 * time.Microsecond
	ctx.VSRDecisionDiagnostics = decision.EvaluationDiagnostics{
		AppliedUnknownPolicies: map[string]string{"guarded": "no_match"},
		Ranking:                &decision.RankingTrace{Winner: "escalate-extreme", RunnerUp: "escalate-hard", Reason: "higher priority"},
	}
	ctx.RouterReplayID = "replay-1"
	ctx.RequestCost, ctx.RequestCostCurrency, ctx.RequestCostPriced = 0.000054, "USD", true
	ctx.EmittedRetention = &config.RetentionDirective{Drop: &drop, TTLTurns: &turns, KeepCurrentModel: &keep, PreferPrefixRetention: &prefer}
	ctx.VSRSelectedCategory = "math"
	ctx.ModalityClassification = &ModalityClassificationResult{Modality: ModalityAR, Method: "classifier"}
	ctx.VSRReasoningMode = "on"
	ctx.VSRLearningPolicies.Set(routerLearningPolicy{
		Method: routerLearningMethodProtection, Scope: "conversation", Action: routerLearningActionAllowSwitch, Reason: "switch_allowed",
	})
	ctx.VSRInjectedSystemPrompt = true
	ctx.PromptCacheAction, ctx.PromptCacheReason = "skipped", "caller_markers"
	ctx.PromptCacheInserted, ctx.PromptCachePreserved = 1, 1
	ctx.VSRCacheSimilarity = 0.97
	for _, matched := range []*[]string{
		&ctx.VSRMatchedKeywords, &ctx.VSRMatchedEmbeddings, &ctx.VSRMatchedDomains, &ctx.VSRMatchedFactCheck,
		&ctx.VSRMatchedUserFeedback, &ctx.VSRMatchedReask, &ctx.VSRMatchedPreference, &ctx.VSRMatchedLanguage,
		&ctx.VSRMatchedAction, &ctx.VSRMatchedContext, &ctx.VSRMatchedStructure, &ctx.VSRMatchedComplexity,
		&ctx.VSRMatchedModality, &ctx.VSRMatchedAuthz, &ctx.VSRMatchedJailbreak, &ctx.VSRMatchedSafety,
		&ctx.VSRMatchedPII, &ctx.VSRMatchedKB, &ctx.VSRMatchedConversation, &ctx.VSRMatchedEvent,
		&ctx.VSRMatchedInputModality, &ctx.VSRMatchedDecisionModel, &ctx.VSRMatchedProjection,
		&ctx.VSRMatchedResponseJailbreak, &ctx.VSRMatchedHallucination,
	} {
		*matched = []string{"matched-rule"}
	}
	ctx.VSRContextTokenCount = 1500
}

// responseHeaderPaths are the ways the Router answers a client, each with the
// headers it writes; debug marks the paths that have debug headers to write.
var responseHeaderPaths = []struct {
	name  string
	debug bool
	write func(t *testing.T, router *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption
}{
	{name: "provider response", debug: true, write: func(t *testing.T, router *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		response, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":status", Value: "200"},
				{Key: "content-type", Value: "application/json"},
			}}},
		}, ctx)
		if err != nil {
			t.Fatalf("handleResponseHeaders() error = %v", err)
		}
		return response.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	}},
	{name: "response body", debug: true, write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ResponseBody{ResponseBody: &ext_proc.BodyResponse{}}}
		addResponseCostHeaders(ctx, response)
		setResponseWarningsHeader(response, []string{headers.ResponseWarningHallucination})
		addResponseStageSignalHeaders(ctx, response)
		return response.GetResponseBody().GetResponse().GetHeaderMutation().GetSetHeaders()
	}},
	{name: "cache hit", debug: true, write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		category, keywords, similarity := cacheDetailForSurface(ctx, ctx.VSRSelectedCategory)
		response := httputil.CreateCacheHitResponseWithBody([]byte(`{}`), "application/json", category, ctx.VSRSelectedDecisionName, keywords, similarity)
		appendRecipeHeaderToImmediateResponse(response, ctx)
		appendConfigVersionToImmediateResponse(response, ctx)
		return response.GetImmediateResponse().GetHeaders().GetSetHeaders()
	}},
	{name: "fast response", write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		response := httputil.CreateFastResponseWithBody([]byte(`{}`), "application/json", ctx.VSRSelectedDecisionName)
		appendRecipeHeaderToImmediateResponse(response, ctx)
		appendConfigVersionToImmediateResponse(response, ctx)
		return response.GetImmediateResponse().GetHeaders().GetSetHeaders()
	}},
	{name: "looper", debug: true, write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		return buildLooperResponseHeaders(looperResponseWithLatencyAndUsage(), ctx)
	}},
	{name: "rate limited", write: func(_ *testing.T, router *OpenAIRouter, _ *RequestContext) []*core.HeaderValueOption {
		response := router.createRateLimitResponse(&ratelimit.Decision{Limit: 10, ResetAt: time.Unix(1_700_000_000, 0)})
		return response.GetImmediateResponse().GetHeaders().GetSetHeaders()
	}},
	{name: "error", write: func(_ *testing.T, router *OpenAIRouter, _ *RequestContext) []*core.HeaderValueOption {
		return router.createErrorResponse(503, "unavailable").GetImmediateResponse().GetHeaders().GetSetHeaders()
	}},
}

// Every path writes only public headers, plus debug headers when the caller
// is allowed them, and never an internal or secret one. Each debug emitter
// asks debugHeadersRequested, so whether a caller is allowed them is what it
// reports once handleRequestHeaders has applied global.router.headers.debug.
func TestResponseHeadersStayOnTheirRegisteredSurfaces(t *testing.T) {
	allowAny := &config.RouterConfig{}
	allowAny.Headers.Debug = config.DebugHeadersAny
	for _, tt := range []struct {
		name    string
		router  *OpenAIRouter
		request map[string]string
		debug   bool
	}{
		{name: "no debug opt-in", router: &OpenAIRouter{Config: allowAny}},
		{name: "debug opt-in the policy refuses", router: &OpenAIRouter{Config: &config.RouterConfig{}}, request: map[string]string{headers.VSRDebug: "true"}},
		{name: "debug opt-in the policy allows", router: &OpenAIRouter{Config: allowAny}, request: map[string]string{headers.VSRDebug: "true"}, debug: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: map[string]string{}}
			if _, err := tt.router.handleRequestHeaders(headerContractRequest(tt.request), ctx); err != nil {
				t.Fatalf("handleRequestHeaders() error = %v", err)
			}
			if got := debugHeadersRequested(ctx); got != tt.debug {
				t.Fatalf("caller allowed the debug headers = %t, want %t", got, tt.debug)
			}
			recordResponseHeaderFacts(ctx)
			for _, path := range responseHeaderPaths {
				wroteDebug := false
				for _, option := range path.write(t, tt.router, ctx) {
					name := option.GetHeader().GetKey()
					header, registered := headers.Lookup(name)
					switch {
					case !registered && strings.HasPrefix(strings.ToLower(name), "x-"):
						t.Errorf("%s: %s is not a registered header", path.name, name)
					case !registered:
					case header.Direction&headers.ToClient == 0 || (header.Surface != headers.SurfacePublic && header.Surface != headers.SurfaceDebug):
						t.Errorf("%s: %s is a %s header, which never reaches clients", path.name, name, header.Surface)
					case header.Surface == headers.SurfaceDebug && !tt.debug:
						t.Errorf("%s: debug header %s reached a caller not allowed the debug headers", path.name, name)
					case header.Surface == headers.SurfaceDebug:
						wroteDebug = true
					}
				}
				if tt.debug && path.debug && !wroteDebug {
					t.Errorf("%s wrote no debug header for a caller allowed them", path.name)
				}
			}
		})
	}
}

// A client request can carry any header the registry keeps from providers:
// the Router's control headers (x-vsr-debug among them), internal context,
// identity, per-user provider keys and forged internal secrets. None reaches a
// provider: the Router removes them from the provider-bound request, and every
// header it sets there is registered to reach providers.
func TestProviderBoundRequestsCarryNoHeaderKeptFromProviders(t *testing.T) {
	kept := make(map[string]string)
	for _, name := range headers.KeptFromProviders() {
		// A config-bound evaluation request is refused unless these match the
		// active configuration; benchmark_config_binding_test covers their removal.
		if name == headers.SRBenchExpectedConfigHash || name == headers.SRBenchMaxInferenceCalls {
			continue
		}
		kept[name] = "client-" + name
	}
	for _, name := range []string{headers.VSRDebug, headers.VSRCacheControl, headers.AuthzUserID, headers.UserOpenAIKey} {
		if _, ok := kept[name]; !ok {
			t.Fatalf("KeptFromProviders() leaves out %s", name)
		}
	}
	for _, tt := range []struct {
		name string
		skip bool
	}{
		{name: "routed request"},
		{name: "skipped request", skip: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			router.Config.SkipProcessing.Enabled = tt.skip
			request := maps.Clone(kept)
			if tt.skip {
				request[headers.VSRSkipProcessing] = "true"
			}
			ctx := &RequestContext{Headers: map[string]string{}}
			headerResponse, err := router.handleRequestHeaders(headerContractRequest(request), ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders() error = %v", err)
			}
			if ctx.SkipProcessing != tt.skip {
				t.Fatalf("SkipProcessing = %t, want %t", ctx.SkipProcessing, tt.skip)
			}
			mutations := []*ext_proc.HeaderMutation{headerResponse.GetRequestHeaders().GetResponse().GetHeaderMutation()}
			if !tt.skip {
				semantic, immediate := router.prepareProtocolRequest([]byte(`{"model":"`+model+`","messages":[{"role":"user","content":"hello"}]}`), ctx)
				if immediate != nil {
					t.Fatalf("prepareProtocolRequest() answered immediately: %v", immediate)
				}
				bodyResponse, err := router.handleSpecifiedModelRouting(semantic, model, "", ctx)
				if err != nil {
					t.Fatalf("handleSpecifiedModelRouting() error = %v", err)
				}
				mutations = append(mutations, bodyResponse.GetRequestBody().GetResponse().GetHeaderMutation())
			}

			removed := make(map[string]bool)
			for _, mutation := range mutations {
				for _, name := range mutation.GetRemoveHeaders() {
					removed[strings.ToLower(name)] = true
				}
				for _, option := range mutation.GetSetHeaders() {
					name := option.GetHeader().GetKey()
					header, registered := headers.Lookup(name)
					if !registered && strings.HasPrefix(strings.ToLower(name), "x-") {
						t.Errorf("the Router set %s, which is not a registered header", name)
					}
					if registered && header.Direction&headers.ToProvider == 0 {
						t.Errorf("the Router set %s (%s, %s) on a provider-bound request", name, header.Surface, header.Sensitivity)
					}
				}
			}
			for name := range kept {
				if !removed[name] {
					t.Errorf("%s reaches the provider", name)
				}
			}
		})
	}
}

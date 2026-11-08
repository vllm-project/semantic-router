package extproc

import (
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
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
	ctx.Retention = &config.DecisionRetention{Drop: &drop, TTLTurns: &turns, KeepCurrentModel: &keep, PreferPrefixRetention: &prefer}
	ctx.VSRSelectedCategory = "math"
	ctx.VSRSelectedDecision = &config.Decision{Name: "escalate-extreme", Output: &config.DecisionOutput{Kind: config.OutputKinds{config.OutputKindText, config.OutputKindImage}}}
	ctx.VSRReasoningMode = "on"
	ctx.VSRLearningPolicies.Set(routerLearningPolicy{
		Method: routerLearningMethodProtection, Scope: "conversation", Action: routerLearningActionAllowSwitch, Reason: "switch_allowed",
	})
	ctx.VSRInjectedSystemPrompt = true
	ctx.PromptCacheAction, ctx.PromptCacheReason = "skipped", "caller_markers"
	ctx.PromptCacheInserted, ctx.PromptCachePreserved = 1, 1
	ctx.VSRCacheSimilarity = 0.97
	for _, matched := range []*[]string{
		&ctx.VSRMatchedKeywords, &ctx.VSRMatchedDomains, &ctx.VSRMatchedFactCheck,
		&ctx.VSRMatchedUserFeedback, &ctx.VSRMatchedTurnRelation, &ctx.VSRMatchedLanguage,
		&ctx.VSRMatchedContext, &ctx.VSRMatchedComplexity,
		&ctx.VSRMatchedModality, &ctx.VSRMatchedAuthz, &ctx.VSRMatchedJailbreak, &ctx.VSRMatchedSafety,
		&ctx.VSRMatchedPII, &ctx.VSRMatchedConversation,
		&ctx.VSRMatchedInputModality, &ctx.VSRMatchedDecisionModel, &ctx.VSRMatchedProjection,
	} {
		*matched = []string{"matched-rule"}
	}
	responseMatches := map[string][]string{
		config.SignalTypeJailbreak:     {"matched-rule"},
		config.SignalTypeHallucination: {"matched-rule"},
		config.SignalTypeSystemOne:     {"matched-rule"},
	}
	ctx.ResponseGuard = &responseGuardRun{conditions: observedConditions{observed: &classification.ResponseSignal{Matched: responseMatches, Reported: responseMatches}}}
	ctx.VSRContextTokenCount = 1500
}

// immediateHeaders finishes a response the Router answers itself at the reply
// boundary, as processor_phase does, and returns the headers it carries.
func immediateHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) []*core.HeaderValueOption {
	appendConfigVersionToImmediateResponse(response, ctx)
	appendRoutedResponseHeaders(response, ctx)
	return response.GetImmediateResponse().GetHeaders().GetSetHeaders()
}

// responseHeaderPaths are the ways the Router answers a client, each with the
// headers it writes; debug marks the paths that have a trace to write and
// routing the ones that report the routing facts.
var responseHeaderPaths = []struct {
	name    string
	debug   bool
	routing bool
	write   func(t *testing.T, router *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption
}{
	{name: "provider response", debug: true, routing: true, write: func(t *testing.T, router *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
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
		if warnings, ok := bufferedWarnings(ctx); ok {
			setResponseBodyHeaderOverwrite(response, headers.VSRWarnings, warnings)
		}
		addResponseStageTrace(ctx, response)
		return response.GetResponseBody().GetResponse().GetHeaderMutation().GetSetHeaders()
	}},
	{name: "cache hit", debug: true, routing: true, write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		return immediateHeaders(httputil.CreateCacheHitResponseWithBody([]byte(`{}`), "application/json"), ctx)
	}},
	{name: "fast response", debug: true, routing: true, write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		return immediateHeaders(httputil.CreateFastResponseWithBody([]byte(`{}`), "application/json"), ctx)
	}},
	{name: "looper", debug: true, routing: true, write: func(_ *testing.T, _ *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		ctx.LooperTrace = newLooperTrace(looperResponseWithLatencyAndUsage())
		response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ImmediateResponse{ImmediateResponse: &ext_proc.ImmediateResponse{
			Headers: &ext_proc.HeaderMutation{SetHeaders: buildLooperResponseHeaders("application/json")},
		}}}
		return immediateHeaders(response, ctx)
	}},
	{name: "rate limited", debug: true, routing: true, write: func(_ *testing.T, router *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		return immediateHeaders(router.createRateLimitResponse(&ratelimit.Decision{Limit: 10, ResetAt: time.Unix(1_700_000_000, 0)}), ctx)
	}},
	{name: "error", debug: true, routing: true, write: func(_ *testing.T, router *OpenAIRouter, ctx *RequestContext) []*core.HeaderValueOption {
		return immediateHeaders(router.createErrorResponse(503, "unavailable"), ctx)
	}},
}

// publicSet is the default surface of a routed response, beside the external
// contracts the Router keeps, such as x-request-id and x-ratelimit-*.
var publicSet = []string{
	headers.VSRSchemaVersion, headers.VSRConfigVersion, headers.VSRResponsePath, headers.VSRSelectedRecipe,
	headers.VSRSelectedDecision, headers.VSRSelectedModel, headers.RouterReplayID, headers.VSRWarnings,
	headers.VSRCost, headers.VSRCostCurrency,
}

// Every path writes only public headers, plus the trace when the caller is
// allowed the debug headers, and never an internal or secret one. The trace
// asks debugHeadersRequested, so whether a caller is allowed it is what it
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
					case strings.HasPrefix(name, headers.VSRPrefix) && !slices.Contains(publicSet, name):
						t.Errorf("%s: %s is not in the public set of a routed response", path.name, name)
					}
				}
				if tt.debug && path.debug && !wroteDebug {
					t.Errorf("%s wrote no trace for a caller allowed it", path.name)
				}
			}
		})
	}
}

// Cache hits and fast responses report the same routing facts as any routed
// response, and the trace to a caller allowed it.
func TestEveryRoutedResponseCarriesThePublicRoutingFacts(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.Config.Headers.Debug = config.DebugHeadersAny
	for _, path := range responseHeaderPaths {
		if !path.routing {
			continue
		}
		ctx := debugRequestContext()
		recordResponseHeaderFacts(ctx)
		written := headerValuesByName(path.write(t, router, ctx))
		for name, want := range map[string]string{
			headers.VSRSchemaVersion:    headers.SchemaVersionValue,
			headers.VSRConfigVersion:    "7",
			headers.VSRSelectedRecipe:   "support",
			headers.VSRSelectedDecision: "escalate-extreme",
			headers.VSRSelectedModel:    "reasoning-model",
			headers.RouterReplayID:      "replay-1",
		} {
			if got := written[name]; got != want {
				t.Errorf("%s: %s = %q, want %q", path.name, name, got, want)
			}
		}
		if written[headers.VSRWarnings] == "" || written[headers.VSRTrace] == "" {
			t.Errorf("%s: warnings %q, trace %q; want both", path.name, written[headers.VSRWarnings], written[headers.VSRTrace])
		}
	}
}

// A fast response answers for the decision whose plugin produced it, so the
// public set names that decision.
func TestFastResponseNamesTheDecisionThatAnswered(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.FastResponsePluginConfig{Message: "blocked"})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name:    "block_on_classifier_error",
		Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginFastResponse, Configuration: payload}},
	}
	ctx := immediateResponseContext(t, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, false)
	ctx.VSRSelectedDecision = decision
	response := (&OpenAIRouter{}).handleFastResponse(ctx, decision.Name)
	written := headerValuesByName(immediateHeaders(response, ctx))
	if written[headers.VSRResponsePath] != headers.ResponsePathFastResponse || written[headers.VSRSelectedDecision] != decision.Name {
		t.Fatalf("fast response path %q, decision %q; want %q, %q",
			written[headers.VSRResponsePath], written[headers.VSRSelectedDecision], headers.ResponsePathFastResponse, decision.Name)
	}
}

// A provider cannot write the Router's headers: the response mutation
// removes every x-vsr- header the provider sent before the Router sets its
// own.
func TestProviderSentRouterHeadersNeverReachTheClient(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{}, VSRSelectedModel: "model-a"}
	response, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"},
			{Key: "X-VSR-Selected-Model", Value: "forged"},
			{Key: headers.VSRTrace, Value: "algorithm=forged"},
			{Key: headers.VSRUpstreamHost, Value: "10.0.0.1:8000"},
		}}},
	}, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders() error = %v", err)
	}
	mutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
	for _, name := range []string{headers.VSRSelectedModel, headers.VSRTrace, headers.VSRUpstreamHost} {
		if !slices.Contains(mutation.GetRemoveHeaders(), name) {
			t.Errorf("the provider's %s reaches the client", name)
		}
	}
	if got := headerValuesByName(mutation.GetSetHeaders())[headers.VSRSelectedModel]; got != "model-a" {
		t.Errorf("x-vsr-selected-model = %q, want the Router's model-a", got)
	}
}

// A client may send only the Router's control headers. Any other x-vsr-
// header it sends, the route key, internal authentication and KV hints among
// them, is dropped before anything reads it and never reaches a provider.
func TestClientSentRouterHeadersAreDropped(t *testing.T) {
	router, _ := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	sent := map[string]string{}
	for _, name := range headers.DroppedFromClients() {
		sent[name] = "client-" + name
	}
	ctx := &RequestContext{Headers: map[string]string{}}
	response, err := router.handleRequestHeaders(headerContractRequest(sent), ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	removed := response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
	for name := range sent {
		if headerValueCI(ctx, name) != "" {
			t.Errorf("the Router kept the client's %s", name)
		}
		if !slices.Contains(removed, name) {
			t.Errorf("the client's %s reaches the provider", name)
		}
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
		if name == headers.VSRBenchExpectedConfigHash || name == headers.VSRBenchMaxInferenceCalls {
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

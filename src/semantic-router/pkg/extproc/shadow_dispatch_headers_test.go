package extproc

import (
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/baggage"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// These tests pin the shadow trust boundary for headers. A decision's
// header_mutation runs for the primary backend and may carry its credential,
// so none of its mutations reach the shadow. Only configuredProviderAuthorizer
// sets the shadow's key.
// Client headers never reach the shadow either, including baggage the trace
// propagator extracts from the client: only the span context crosses.

const shadowTestPrimaryAuthHeader = "x-primary-token"

// shadowCredentialHeaderNames are the mutation headers a shadow must never
// receive: known credential carriers plus the primary profile's auth header.
var shadowCredentialHeaderNames = []string{
	"Authorization", "Proxy-Authorization", "Cookie", "x-api-key", "api-key",
	shadowTestPrimaryAuthHeader, headers.UserOpenAIKey,
}

// shadowCredentialMutations configures the decision with what a primary
// backend might need: known credential carriers, the primary profile's custom
// auth header, an unknown custom secret, and one harmless routing header.
func shadowCredentialMutations(ctx *RequestContext) {
	ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{{
		Type: config.DecisionPluginHeaderMutation,
		Configuration: config.MustStructuredPayload(map[string]interface{}{
			"add": []map[string]string{
				{"name": "Authorization", "value": "Bearer primary-secret"},
				{"name": "X-Api-Key", "value": "primary-secret"},
				{"name": "api-key", "value": "primary-secret"},
				{"name": shadowTestPrimaryAuthHeader, "value": "primary-secret"},
				{"name": headers.UserOpenAIKey, "value": "primary-secret"},
				{"name": "X-Internal-Token", "value": "primary-secret"},
				{"name": "x-tenant", "value": "acme"},
			},
			"update": []map[string]string{
				{"name": "PROXY-AUTHORIZATION", "value": "Basic primary-secret"},
				{"name": "Cookie", "value": "session=primary-secret"},
			},
		}),
	}}
}

// newShadowHeaderTestRouter gives the primary backend a custom auth header,
// as a provider profile may declare with auth_header.
func newShadowHeaderTestRouter(t *testing.T) (*shadowTestBackend, *OpenAIRouter, string) {
	t.Helper()
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	primaryProfile := router.Config.ProviderProfiles["provider"]
	primaryProfile.AuthHeader = shadowTestPrimaryAuthHeader
	router.Config.ProviderProfiles["provider"] = primaryProfile
	return backend, router, primaryModel
}

func assertNoShadowCredentialHeaders(t *testing.T, wire http.Header) {
	t.Helper()
	for _, name := range shadowCredentialHeaderNames {
		if got := wire.Get(name); got != "" {
			t.Fatalf("shadow received primary credential header %s=%q", name, got)
		}
	}
}

func TestShadowDispatchForwardsNoDecisionHeadersByDefault(t *testing.T) {
	backend, router, primaryModel := newShadowHeaderTestRouter(t)

	run := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), shadowCredentialMutations)
	waitForShadow(t, router)
	if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("shadow verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	if got := backend.requestCount(); got != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", got)
	}
	wire := backend.headers[0]
	assertNoShadowCredentialHeaders(t, wire)
	for _, name := range []string{"X-Internal-Token", "x-tenant"} {
		if got := wire.Get(name); got != "" {
			t.Fatalf("decision header %s=%q reached the shadow", name, got)
		}
	}
	if wire.Get(headers.RequestID) == "" {
		t.Fatal("shadow request lacks its own request id")
	}
}

func TestShadowDispatchForwardsAllowlistedDecisionHeaders(t *testing.T) {
	backend, router, primary := newShadowHeaderTestRouter(t)
	cfg := shadowTestPluginConfig()
	// Include credential names to exercise runtime filtering independently of config validation.
	cfg.ForwardHeaders = []string{"X-Tenant", "Authorization", shadowTestPrimaryAuthHeader}
	run := runShadowRequest(t, router, primary, cfg, shadowCredentialMutations)
	waitForShadow(t, router)
	if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("shadow outcome: %+v", outcome)
	}
	wire := backend.headers[0]
	assertNoShadowCredentialHeaders(t, wire)
	if wire.Get("x-tenant") != "acme" || wire.Get("x-internal-token") != "" {
		t.Fatalf("unexpected forwarded headers: %v", wire)
	}
}

func TestShadowDispatchUsesOnlyCandidateCredential(t *testing.T) {
	backend, router, primary := newShadowHeaderTestRouter(t)
	params := router.Config.ModelConfig[shadowTestModel]
	params.AccessKeys = map[string]string{"openai": "shadow-key"}
	router.Config.ModelConfig[shadowTestModel] = params
	runShadowRequest(t, router, primary, shadowTestPluginConfig(), shadowCredentialMutations)
	waitForShadow(t, router)
	wire := backend.headers[0]
	if wire.Get("Authorization") != "Bearer shadow-key" {
		t.Fatal("wrong candidate credential")
	}
	for _, name := range []string{"X-Internal-Token", "x-tenant", shadowTestPrimaryAuthHeader} {
		if wire.Get(name) != "" {
			t.Fatalf("primary header leaked: %s", name)
		}
	}
}

// installTracingPropagator installs the propagator InitTracing sets, W3C
// trace context plus baggage, so the request phase extracts client baggage
// exactly as it does on a router with tracing enabled.
func installTracingPropagator(t *testing.T) {
	t.Helper()
	previous := otel.GetTextMapPropagator()
	otel.SetTextMapPropagator(tracing.DefaultPropagator())
	t.Cleanup(func() { otel.SetTextMapPropagator(previous) })
}

const shadowTestClientTraceID = "4bf92f3577b34da6a3ce929d0e0e4736"

// shadowClientTraceHeaders runs the request-header phase with the trace
// headers a client sent, so ctx.TraceContext holds the client's span context
// and baggage the same way it does on a live request.
func shadowClientTraceHeaders(t *testing.T) func(ctx *RequestContext) {
	t.Helper()
	return func(ctx *RequestContext) {
		request := newRequestHeaders("POST", "/v1/chat/completions")
		request.RequestHeaders.Headers.Headers = append(
			request.RequestHeaders.Headers.Headers,
			&core.HeaderValue{Key: "traceparent", Value: "00-" + shadowTestClientTraceID + "-00f067aa0ba902b7-01"},
			&core.HeaderValue{Key: "tracestate", Value: "vendor=abc"},
			&core.HeaderValue{Key: "baggage", Value: "token=client-secret,tenant=acme"},
		)
		startRequestHeaderSpan(request, ctx).End()
		if got := baggage.FromContext(ctx.TraceContext).Member("token").Value(); got != "client-secret" {
			t.Fatalf("request phase did not extract client baggage (token=%q), so nothing could leak", got)
		}
	}
}

func TestShadowDispatchNeverForwardsClientBaggage(t *testing.T) {
	installTracingPropagator(t)
	backend, router, primaryModel := newShadowHeaderTestRouter(t)

	run := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), shadowClientTraceHeaders(t))
	waitForShadow(t, router)
	if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("shadow verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	wire := backend.headers[0]
	if got := wire.Values("baggage"); len(got) != 0 {
		t.Fatalf("client baggage reached the shadow: %q", got)
	}
	if got := wire.Get("traceparent"); !strings.Contains(got, shadowTestClientTraceID) {
		t.Fatalf("traceparent = %q, want the client's trace %s still propagated", got, shadowTestClientTraceID)
	}
	if got := wire.Get("tracestate"); got != "vendor=abc" {
		t.Fatalf("tracestate = %q, want the client's trace state kept", got)
	}
	assertNoShadowCredentialHeaders(t, wire)
}

func TestShadowCallHeadersDropsShadowAuthHeaderMutation(t *testing.T) {
	job := &shadowJob{
		shadowRequestID: "shadow-req-1",
		extraHeaders: map[string]string{
			"x-shadow-token": "primary-secret",
			"X-SHADOW-TOKEN": "primary-secret",
			"x-team":         "platform",
			"traceparent":    "00-abc-def-01",
		},
	}
	target := &shadowTarget{profile: &config.ProviderProfile{
		Type:         "openai",
		AuthHeader:   "x-shadow-token",
		ExtraHeaders: map[string]string{"x-static": "from-profile"},
	}}

	got := shadowCallHeaders(job, target)
	for key := range got {
		if strings.EqualFold(key, "x-shadow-token") {
			t.Fatalf("shadow auth header %q forwarded from a decision mutation: %v", key, got)
		}
	}
	want := map[string]string{
		"x-team":          "platform",
		"traceparent":     "00-abc-def-01",
		"x-static":        "from-profile",
		headers.RequestID: "shadow-req-1",
	}
	for key, value := range want {
		if got[key] != value {
			t.Fatalf("header %s = %q, want %q (all: %v)", key, got[key], value, got)
		}
	}
	if len(got) != len(want) {
		t.Fatalf("unexpected extra headers: %v", got)
	}
}

func TestShadowHeaderIsSensitive(t *testing.T) {
	cases := map[string]bool{
		"Authorization":        true,
		"authorization":        true,
		"Proxy-Authorization":  true,
		"Cookie":               true,
		"x-api-key":            true,
		"X-API-KEY":            true,
		"api-key":              true,
		"x-goog-api-key":       true,
		headers.UserOpenAIKey:  true,
		headers.UserBedrockKey: true,
		//nolint:gocritic // mapKey: the surrounding whitespace is the case under test.
		" x-api-key ":          true,
		"x-tenant":             false,
		"X-Internal-Token":     false,
		"traceparent":          false,
		"x-authorization-hint": false,
		headers.RequestID:      false,
	}
	for name, want := range cases {
		if got := shadowHeaderIsSensitive(name); got != want {
			t.Errorf("shadowHeaderIsSensitive(%q) = %v, want %v", name, got, want)
		}
	}
	if !shadowHeaderIsSensitive("X-Custom-Token", "x-custom-token") {
		t.Error("profile auth header must be sensitive regardless of case")
	}
	if shadowHeaderIsSensitive("x-custom-token", "") {
		t.Error("an empty profile auth header must not match")
	}
}

// Default primary header mutations must not cross into a shadow backend.
func TestShadowDoesNotForwardDefaultHeaderMutations(t *testing.T) {
	ctx := &RequestContext{RequestID: "shadow-defaults"}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{DefaultPlugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginHeaderMutation,
		Configuration: config.MustStructuredPayload(map[string]interface{}{"add": []map[string]string{{"name": "x-tenant", "value": "acme"}}}),
	}}}})
	extra := (&OpenAIRouter{}).shadowExtraHeaders(ctx, nil, nil)
	if extra["x-tenant"] != "" {
		t.Fatalf("shadow headers = %v, default x-tenant must stay on the primary", extra)
	}
}

func TestShadowDispatchSendsNoKeyToANoAuthProvider(t *testing.T) {
	for _, authHeader := range []string{"", "Authorization"} {
		t.Run("auth.header="+authHeader, func(t *testing.T) {
			backend := newShadowTestBackend(t)
			router, primaryModel := newShadowTestRouter(t, backend)
			for index := range router.Config.VLLMEndpoints {
				if router.Config.VLLMEndpoints[index].Name == "shadow-backend" {
					router.Config.VLLMEndpoints[index].Type = "ollama"
					router.Config.VLLMEndpoints[index].ProviderProfileName = "shadow-ollama"
				}
			}
			router.Config.ProviderProfiles["shadow-ollama"] = config.ProviderProfile{
				Type: "ollama", BaseURL: backend.server.URL, AuthHeader: authHeader,
			}
			shadowParams := router.Config.ModelConfig[shadowTestModel]
			shadowParams.AccessKeys = map[string]string{"ollama": "ollama"}
			router.Config.ModelConfig[shadowTestModel] = shadowParams

			run := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
			waitForShadow(t, router)
			if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
				t.Fatalf("shadow verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
			}
			if got := backend.headers[0].Values("Authorization"); len(got) != 0 {
				t.Fatalf("Authorization = %q, want no key sent to a no-auth provider", got)
			}
		})
	}
}

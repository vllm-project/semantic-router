package extproc

import (
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// These tests pin the shadow trust boundary for headers. A decision's
// header_mutation runs for the primary backend and may carry its credential,
// so nothing it sets reaches the shadow unless forward_headers names it, and
// credential carriers never do. Only shadowAuthorizer sets the shadow's key.

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
			t.Fatalf("decision header %s=%q reached the shadow without forward_headers", name, got)
		}
	}
	if wire.Get(headers.RequestID) == "" {
		t.Fatal("shadow request lacks its own request id")
	}
}

func TestShadowDispatchForwardsOnlyAllowlistedDecisionHeaders(t *testing.T) {
	backend, router, primaryModel := newShadowHeaderTestRouter(t)
	pluginCfg := shadowTestPluginConfig()
	pluginCfg.ForwardHeaders = []string{"X-TENANT"}

	runShadowRequest(t, router, primaryModel, pluginCfg, shadowCredentialMutations)
	waitForShadow(t, router)
	wire := backend.headers[0]
	if got := wire.Get("x-tenant"); got != "acme" {
		t.Fatalf("x-tenant = %q, want the allowlisted decision header forwarded", got)
	}
	if got := wire.Get("X-Internal-Token"); got != "" {
		t.Fatalf("custom secret outside forward_headers reached the shadow: %q", got)
	}
	assertNoShadowCredentialHeaders(t, wire)
}

func TestShadowDispatchNeverForwardsCredentialsEvenWhenListed(t *testing.T) {
	backend, router, primaryModel := newShadowHeaderTestRouter(t)
	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.AccessKey = "shadow-key"
	router.Config.ModelConfig[shadowTestModel] = shadowParams
	// Config validation rejects these names; bypass it here to prove the
	// runtime floor holds on its own.
	pluginCfg := shadowTestPluginConfig()
	pluginCfg.ForwardHeaders = []string{"Authorization", shadowTestPrimaryAuthHeader, "x-api-key", "x-tenant"}

	runShadowRequest(t, router, primaryModel, pluginCfg, shadowCredentialMutations)
	waitForShadow(t, router)
	wire := backend.headers[0]
	if got := wire.Values("Authorization"); len(got) != 1 || got[0] != "Bearer shadow-key" {
		t.Fatalf("Authorization = %q, want only the shadow model's static key", got)
	}
	for _, name := range shadowCredentialHeaderNames {
		if name == "Authorization" {
			continue
		}
		if got := wire.Get(name); got != "" {
			t.Fatalf("listed credential header %s=%q still reached the shadow", name, got)
		}
	}
	if got := wire.Get("x-tenant"); got != "acme" {
		t.Fatalf("x-tenant = %q, want the harmless allowlisted header kept", got)
	}
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

package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// These tests pin the shadow trust boundary for headers: a decision's
// header_mutation may carry the primary backend's credential, and none of it
// may reach the shadow backend. Only shadowAuthorizer sets the shadow's key.

func TestShadowDispatchStripsCredentialHeaderMutations(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	// The primary backend authenticates through a custom header, as a
	// provider profile may declare with auth_header.
	primaryProfile := router.Config.ProviderProfiles["provider"]
	primaryProfile.AuthHeader = "x-primary-token"
	router.Config.ProviderProfiles["provider"] = primaryProfile

	credentialMutations := func(ctx *RequestContext) {
		ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{{
			Type: config.DecisionPluginHeaderMutation,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"add": []map[string]string{
					{"name": "Authorization", "value": "Bearer primary-secret"},
					{"name": "X-Api-Key", "value": "primary-secret"},
					{"name": "api-key", "value": "primary-secret"},
					{"name": "x-primary-token", "value": "primary-secret"},
					{"name": headers.UserOpenAIKey, "value": "primary-secret"},
					{"name": "x-team", "value": "platform"},
				},
				"update": []map[string]string{
					{"name": "PROXY-AUTHORIZATION", "value": "Basic primary-secret"},
					{"name": "Cookie", "value": "session=primary-secret"},
				},
			}),
		}}
	}

	keyless := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), credentialMutations)
	waitForShadow(t, router)
	if outcome := singleShadowOutcome(t, keyless); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("shadow verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	if got := backend.requestCount(); got != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", got)
	}
	wire := backend.headers[0]
	for _, name := range []string{
		"Authorization", "Proxy-Authorization", "Cookie", "x-api-key", "api-key",
		"x-primary-token", headers.UserOpenAIKey,
	} {
		if got := wire.Get(name); got != "" {
			t.Fatalf("keyless shadow received primary credential header %s=%q", name, got)
		}
	}
	if got := wire.Get("x-team"); got != "platform" {
		t.Fatalf("x-team = %q, want the harmless decision mutation kept", got)
	}
	if got := wire.Get(headers.RequestID); got == "" {
		t.Fatal("shadow request lacks its own request id")
	}

	// With a shadow credential configured, the shadow's static key is the
	// only Authorization value on the wire; the primary mutation never wins.
	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.AccessKey = "shadow-key"
	router.Config.ModelConfig[shadowTestModel] = shadowParams
	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), credentialMutations)
	waitForShadow(t, router)
	if got := backend.headers[1].Values("Authorization"); len(got) != 1 || got[0] != "Bearer shadow-key" {
		t.Fatalf("Authorization = %q, want only the shadow model's static key", got)
	}
	if got := backend.headers[1].Get("x-primary-token"); got != "" {
		t.Fatalf("shadow with its own key still received primary credential: %q", got)
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

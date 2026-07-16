package extproc

import (
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// callerAuthCanary is a distinctive caller credential value used to prove the
// router never forwards a caller Authorization to an upstream unless a backend
// explicitly opts into forward_authorization_header (issue #2375).
const callerAuthCanary = "Bearer caller-virtual-key-canary-2375"

// stripAuthTestRouter builds a router whose single backend "plain" does NOT opt
// into forward_authorization_header. withStaticKey controls whether the backend
// carries a static service key: without one, the default/no-provider path is
// exercised; with one, the static-key injection path is exercised. stripEnabled
// sets global.router.strip_inbound_authorization (the opt-in hardening).
func stripAuthTestRouter(t *testing.T, withStaticKey, stripEnabled bool) *OpenAIRouter {
	t.Helper()
	staticKey := ""
	if withStaticKey {
		staticKey = "service-key-canary"
	}
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{StripInboundAuthorization: stripEnabled},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {
					PreferredEndpoints: []string{"plain"},
					AccessKey:          staticKey,
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:                "plain",
					ProviderProfileName: "plain",
					APIKey:              staticKey,
					Weight:              1,
				},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"plain": {
					Type:       "openai",
					BaseURL:    "https://upstream.example.com/v1",
					AuthHeader: "Authorization",
					AuthPrefix: "Bearer",
					// ForwardAuthorizationHeader intentionally left false.
				},
			},
		},
	}
	return &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
}

// allSetHeaderValues collects every set-header value so canary assertions can
// scan the whole mutation, not just the last value under a given key.
func allSetHeaderValues(opts []*core.HeaderValueOption) []string {
	values := make([]string, 0, len(opts))
	for _, opt := range opts {
		values = append(values, string(opt.GetHeader().GetRawValue()))
		values = append(values, opt.GetHeader().GetValue())
	}
	return values
}

// TestBuildRouteHeaderStateStripsCallerAuthorizationOnDefaultPath is the core
// #2375 regression: a backend with no static key and no forward opt-in must
// strip the caller's Authorization rather than preserve it upstream.
func TestBuildRouteHeaderStateStripsCallerAuthorizationOnDefaultPath(t *testing.T) {
	router := stripAuthTestRouter(t, false, true)
	ctx := &RequestContext{Headers: map[string]string{
		"authorization": callerAuthCanary,
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "plain", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	// Authorization is scheduled for removal before the upstream.
	assert.Contains(t, state.removeHeaders, forwardedAuthorizationHeaderName,
		"default path must strip the caller Authorization")
	// And it is never re-set: no Authorization survives in the mutation.
	headerMap := headerValuesByName(state.setHeaders)
	_, present := headerMap[forwardedAuthorizationHeaderName]
	assert.False(t, present, "default path must not set any Authorization header")
	// Canary: the caller's credential value appears nowhere in the outbound set.
	for _, v := range allSetHeaderValues(state.setHeaders) {
		assert.NotContains(t, v, "caller-virtual-key-canary",
			"caller credential must never be forwarded on the default path")
	}
}

// TestBuildRouteHeaderStatePreservesCallerAuthorizationByDefault proves the
// strip is opt-in: with global.router.strip_inbound_authorization off (the
// default) and a backend that neither injects a static key nor forwards, the
// caller's inbound Authorization is left untouched (not scheduled for removal),
// preserving pre-#2375 passthrough behavior. The internal carrier is still
// stripped unconditionally.
func TestBuildRouteHeaderStatePreservesCallerAuthorizationByDefault(t *testing.T) {
	router := stripAuthTestRouter(t, false, false)
	ctx := &RequestContext{Headers: map[string]string{
		"authorization": callerAuthCanary,
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "plain", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	assert.NotContains(t, state.removeHeaders, forwardedAuthorizationHeaderName,
		"default (opt-in off) must not strip the caller Authorization")
	// The internal carrier is always stripped, independent of the opt-in.
	assert.Contains(t, state.removeHeaders, headers.VSRInboundAuthorization)
}

// TestBuildRouteHeaderStateStaticKeyOverwritesCallerAuthorization proves the
// static-key path replaces (not appends to) the caller Authorization so the
// caller credential cannot ride alongside the injected service key.
func TestBuildRouteHeaderStateStaticKeyOverwritesCallerAuthorization(t *testing.T) {
	router := stripAuthTestRouter(t, true, true)
	ctx := &RequestContext{Headers: map[string]string{
		"authorization": callerAuthCanary,
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "plain", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	// The caller's Authorization is stripped and the service key is injected.
	assert.Contains(t, state.removeHeaders, forwardedAuthorizationHeaderName)

	var authOption *core.HeaderValueOption
	for _, opt := range state.setHeaders {
		if opt.GetHeader().GetKey() == "Authorization" {
			authOption = opt
		}
	}
	require.NotNil(t, authOption, "expected a service-key Authorization header")
	assert.Equal(t, "Bearer service-key-canary", string(authOption.GetHeader().GetRawValue()))
	assert.Equal(t, core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD, authOption.GetAppendAction(),
		"service key must overwrite so the caller credential cannot survive alongside it")

	for _, v := range allSetHeaderValues(state.setHeaders) {
		assert.NotContains(t, v, "caller-virtual-key-canary",
			"caller credential must never be forwarded on the static-key path")
	}
}

// TestBuildRouteHeaderStateStaticKeyOverwritesCallerAuthorizationDefaultOptOut
// proves that even with the strip opt-in OFF (the default), a static-key backend
// still overwrites the caller Authorization so the caller credential cannot ride
// alongside the injected key — OVERWRITE is the sole defense in the default state.
func TestBuildRouteHeaderStateStaticKeyOverwritesCallerAuthorizationDefaultOptOut(t *testing.T) {
	router := stripAuthTestRouter(t, true, false)
	ctx := &RequestContext{Headers: map[string]string{
		"authorization": callerAuthCanary,
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "plain", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	// The strip is off, so Authorization is not scheduled for removal...
	assert.NotContains(t, state.removeHeaders, forwardedAuthorizationHeaderName)
	// ...but the static key is injected with OVERWRITE, so the caller value cannot survive.
	var authOption *core.HeaderValueOption
	for _, opt := range state.setHeaders {
		if opt.GetHeader().GetKey() == "Authorization" {
			authOption = opt
		}
	}
	require.NotNil(t, authOption)
	assert.Equal(t, "Bearer service-key-canary", string(authOption.GetHeader().GetRawValue()))
	assert.Equal(t, core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD, authOption.GetAppendAction())
	for _, v := range allSetHeaderValues(state.setHeaders) {
		assert.NotContains(t, v, "caller-virtual-key-canary")
	}
}

// TestBuildRouteHeaderStateForwardOptInStillReceivesCredential guards against the
// strip regressing the opt-in feature: a forward-enabled backend must still
// receive the caller credential verbatim.
func TestBuildRouteHeaderStateForwardOptInStillReceivesCredential(t *testing.T) {
	router := forwardAuthTestRouter(t)
	// With the opt-in strip enabled, the forward path must still deliver the
	// caller credential (remove_headers is applied before set_headers).
	router.Config.StripInboundAuthorization = true
	ctx := &RequestContext{Headers: map[string]string{
		"authorization": callerAuthCanary,
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "gateway", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	// Envoy applies remove_headers before set_headers, so even though
	// Authorization is scheduled for removal, the forwarded value is re-set and
	// reaches the upstream.
	assert.Contains(t, state.removeHeaders, forwardedAuthorizationHeaderName)
	headerMap := headerValuesByName(state.setHeaders)
	assert.Equal(t, callerAuthCanary, headerMap["Authorization"],
		"forward opt-in must still deliver the caller credential verbatim")
	assert.Contains(t, headerMap["Authorization"], "caller-virtual-key-canary")
}

// TestRouterReplayRecordExcludesCallerAuthorization is the Replay canary: a
// captured routing record must never contain the caller's Authorization value.
// The record is built from structured context fields only; this guards against
// a future regression that starts recording request headers (issue #2375).
func TestRouterReplayRecordExcludesCallerAuthorization(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req-2375",
		SessionID: "sess-2375",
		Headers: map[string]string{
			"authorization": callerAuthCanary,
		},
		// A representative request body that carries no credential.
		OriginalRequestBody: []byte(`{"model":"model-a","messages":[{"role":"user","content":"hi"}]}`),
	}

	record := buildReplayRoutingRecord(ctx, "model-a", "model-a", "decision-a")

	serialized, err := json.Marshal(record)
	require.NoError(t, err)
	assert.NotContains(t, string(serialized), "caller-virtual-key-canary",
		"the caller Authorization must never be captured in a router replay record")
	assert.NotContains(t, string(serialized), callerAuthCanary)
}

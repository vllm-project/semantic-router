package extproc

import (
	"context"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// forwardAuthTestRouter builds a router whose single backend "gateway" opts into
// forward_authorization_header. The endpoint also carries a static APIKey so the
// tests can assert the static key is NOT injected when forwarding is enabled.
func forwardAuthTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {
					PreferredEndpoints: []string{"gateway"},
					AccessKey:          "static-service-key",
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:                "gateway",
					ProviderProfileName: "gateway",
					APIKey:              "static-service-key",
					Weight:              1,
				},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"gateway": {
					Type:                       "openai",
					BaseURL:                    "https://litellm.example.com/v1",
					AuthHeader:                 "Authorization",
					AuthPrefix:                 "Bearer",
					ForwardAuthorizationHeader: true,
				},
			},
		},
	}
	return &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
}

func TestBuildRouteHeaderStateForwardsInboundAuthorizationVerbatim(t *testing.T) {
	router := forwardAuthTestRouter(t)
	ctx := &RequestContext{Headers: map[string]string{
		"authorization": "Bearer user-virtual-key",
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "gateway", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	headerMap := headerValuesByName(state.setHeaders)
	// Forwarded verbatim — same value the caller sent, including the prefix.
	assert.Equal(t, "Bearer user-virtual-key", headerMap["Authorization"])
	// The static service key must not leak through when forwarding is enabled.
	assert.NotEqual(t, "Bearer static-service-key", headerMap["Authorization"])

	// The forwarded header must overwrite the caller's inbound Authorization
	// rather than append a duplicate (the caller always sends one in this mode).
	var authOption *core.HeaderValueOption
	for _, opt := range state.setHeaders {
		if opt.GetHeader().GetKey() == "Authorization" {
			authOption = opt
		}
	}
	require.NotNil(t, authOption, "expected an Authorization header option")
	assert.Equal(t, core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD, authOption.GetAppendAction())
}

func TestBuildRouteHeaderStateForwardsInboundAuthorizationCaseInsensitive(t *testing.T) {
	router := forwardAuthTestRouter(t)
	// Some transports preserve the original header casing.
	ctx := &RequestContext{Headers: map[string]string{
		"Authorization": "Bearer mixed-case-key",
	}}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "gateway", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	headerMap := headerValuesByName(state.setHeaders)
	assert.Equal(t, "Bearer mixed-case-key", headerMap["Authorization"])
}

func TestDecisionForwardsAuthorization(t *testing.T) {
	router := forwardAuthTestRouter(t)

	forwardDecision := &config.Decision{ModelRefs: []config.ModelRef{{Model: "model-a"}}}
	assert.True(t, router.decisionForwardsAuthorization(forwardDecision),
		"decision targeting a forward-enabled backend should require forwarding")

	otherDecision := &config.Decision{ModelRefs: []config.ModelRef{{Model: "unknown-model"}}}
	assert.False(t, router.decisionForwardsAuthorization(otherDecision),
		"decision targeting no forward-enabled backend should not require forwarding")

	assert.False(t, router.decisionForwardsAuthorization(nil))
}

func TestHandleLooperExecutionForwardBackendRequiresAuthorization(t *testing.T) {
	router := forwardAuthTestRouter(t)
	decision := &config.Decision{
		Name:      "panel",
		Algorithm: &config.AlgorithmConfig{Type: "fusion"},
		ModelRefs: []config.ModelRef{{Model: "model-a"}},
	}
	// Caller sent no Authorization; a candidate backend forwards it, so the
	// looper must reject up front rather than fall back to a static key.
	reqCtx := &RequestContext{Headers: map[string]string{}}
	req := &openai.ChatCompletionNewParams{
		Model:    "model-a",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}

	resp, err := router.handleLooperExecution(context.Background(), req, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, resp)
	immediate := resp.GetImmediateResponse()
	require.NotNil(t, immediate, "missing Authorization on a forward-enabled looper route should 401")
	assert.Equal(t, 401, int(immediate.GetStatus().GetCode()))
}

func TestBuildRouteHeaderStateForwardOnLooperLegUsesDedicatedHeader(t *testing.T) {
	router := forwardAuthTestRouter(t)
	// Internal looper leg: Authorization holds the static access key; the caller's
	// identity is on the dedicated header. The forwarded value must be the caller's
	// identity, never the static key.
	ctx := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			"authorization":                 "Bearer static-service-key",
			headers.VSRInboundAuthorization: "Bearer user-virtual-key",
		},
	}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "gateway", ctx, nil)
	require.Nil(t, errorResponse)
	require.NotNil(t, state)

	headerMap := headerValuesByName(state.setHeaders)
	assert.Equal(t, "Bearer user-virtual-key", headerMap["Authorization"])
	assert.NotEqual(t, "Bearer static-service-key", headerMap["Authorization"])
	// The dedicated carrier must be stripped before the upstream.
	assert.Contains(t, state.removeHeaders, headers.VSRInboundAuthorization)
}

func TestBuildRouteHeaderStateForwardOnLooperLegRejectsStaticKeyOnlyRequest(t *testing.T) {
	router := forwardAuthTestRouter(t)
	// Bypass regression: an internal looper leg carrying only the static access
	// key (no caller identity) must 401 rather than forward the static key. This
	// is the case that arises when the caller sent no Authorization but the looper
	// falls back to the static key on a model outside decision.ModelRefs.
	ctx := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			"authorization": "Bearer static-service-key",
		},
	}

	state, errorResponse := router.buildRouteHeaderState("model-a", "", "gateway", ctx, nil)
	require.Nil(t, state)
	require.NotNil(t, errorResponse)
	immediate := errorResponse.GetImmediateResponse()
	require.NotNil(t, immediate)
	assert.Equal(t, 401, int(immediate.GetStatus().GetCode()))
}

func TestBuildRouteHeaderStateForwardMissingAuthorizationReturns401(t *testing.T) {
	router := forwardAuthTestRouter(t)

	for name, inboundHeaders := range map[string]map[string]string{
		"absent":          {},
		"empty":           {"authorization": ""},
		"whitespace-only": {"authorization": "   "},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{Headers: inboundHeaders}

			state, errorResponse := router.buildRouteHeaderState("model-a", "", "gateway", ctx, nil)
			require.Nil(t, state)
			require.NotNil(t, errorResponse)

			immediate := errorResponse.GetImmediateResponse()
			require.NotNil(t, immediate, "missing inbound Authorization should produce an immediate error response")
			assert.Equal(t, 401, int(immediate.GetStatus().GetCode()))
		})
	}
}

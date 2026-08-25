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

// TestHandleLooperExecutionMixedDecisionNotRejectedUpFront proves the
// decision-wide early 401 is gone: a decision whose candidates include a
// forward-auth backend must NOT be rejected before the selected backend is
// known (a mixed decision may still resolve to a static-key backend that needs
// no caller credential). The per-request Authorization requirement is enforced
// per-leg at the selected backend instead — see the buildRouteHeaderState
// forward-path tests below and TestBuildRouteHeaderStateForwardMissingAuthorizationReturns401.
func TestHandleLooperExecutionMixedDecisionNotRejectedUpFront(t *testing.T) {
	router := forwardAuthTestRouter(t)
	// Unreachable endpoint with a short timeout: the looper Execute fails fast,
	// which is fine — we only assert the request is not short-circuited with a
	// 401 before the looper (and thus the selected backend) even runs.
	router.Config.Looper = config.LooperConfig{Endpoint: "http://127.0.0.1:1", TimeoutSeconds: 1}
	decision := &config.Decision{
		Name:      "panel",
		Algorithm: &config.AlgorithmConfig{Type: "fusion"},
		ModelRefs: []config.ModelRef{{Model: "model-a"}},
	}
	// Caller sent no Authorization.
	reqCtx := &RequestContext{Headers: map[string]string{}}
	req := &openai.ChatCompletionNewParams{
		Model:    "model-a",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}

	resp, err := router.handleLooperExecution(context.Background(), req, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, resp)
	immediate := resp.GetImmediateResponse()
	require.NotNil(t, immediate)
	// The request must reach the looper and fail on the unreachable endpoint
	// (500), proving it got PAST the removed decision-wide gate rather than being
	// short-circuited with a 401 before the selected backend is known.
	assert.Equal(t, 500, int(immediate.GetStatus().GetCode()),
		"a forward-auth candidate must not be rejected before the selected backend is known")
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
